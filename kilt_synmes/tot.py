"""Task-decomposed Tree-of-Thought search for multi-entity summarization.

Follows the ToT4ES strategy (https://github.com/dice-group/ToT4ES): thought
generation is decomposed into relatedness, informativeness, and coverage, and a
beam search keeps the best-scoring partial summaries.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

from kilt_synmes.pipeline import (
    GENERIC_RELATIONS,
    NOISE_RELATION_TERMS,
    text_terms,
)


Triple = Tuple[str, str, str]
TASKS = ("relatedness", "informativeness", "coverage")

SEMANTIC_ROLE_PATTERNS = {
    "location": ["place", "location", "country", "city", "region", "hometown"],
    "time": ["date", "year", "time", "born", "died", "founded", "release"],
    "relationship": ["spouse", "parent", "child", "sibling", "member", "partner", "friend"],
    "attribute": ["name", "label", "title", "type", "category", "gender", "description"],
    "work": ["work", "author", "writer", "director", "producer", "film", "book", "album", "song"],
    "organization": ["organization", "company", "institution", "team", "school", "university"],
}

DEFAULT_OBJECTIVE_WEIGHTS = {
    "relatedness": 0.4,
    "informativeness": 0.4,
    "coverage": 0.2,
}


@dataclass
class ToTConfig:
    """Search hyper-parameters, mirroring the ToT4ES runner defaults."""

    max_summary_len: int = 5
    n_candidates_per_task: int = 2
    n_evals: int = 3
    breadth_limit: int = 3
    prune_keep_multiplier: float = 1.5
    thought_temperature: float = 0.8
    eval_temperature: float = 0.3
    objective_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE_WEIGHTS))


@dataclass
class TreeNode:
    state: Tuple[int, ...]
    thought: int | None = None
    task: str | None = None
    value: float = 0.0
    depth: int = 0


def extract_first_int(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def semantic_role(relation: str) -> str:
    name = relation.split("/")[-1].split(":")[-1].lower()
    for role, patterns in SEMANTIC_ROLE_PATTERNS.items():
        if any(pattern in name for pattern in patterns):
            return role
    return "other"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class HeuristicScorer:
    """Training-free R/I/C scores, used for offline runs and for state scoring."""

    def __init__(self, triples: Sequence[Triple], question: str, topic_entities: Sequence[str]):
        self.triples = [tuple(triple) for triple in triples]
        self.question_terms = text_terms(question)
        self.topics = [str(topic).lower() for topic in topic_entities]
        self.predicate_freqs = Counter(relation for _, relation, _ in self.triples)
        self.max_freq = max(self.predicate_freqs.values(), default=1)
        self.roles = {relation: semantic_role(relation) for _, relation, _ in self.triples}

    def _endpoints(self, index: int) -> set[str]:
        head, _, tail = self.triples[index]
        return {head.lower(), tail.lower()}

    def relatedness(self, index: int) -> float:
        head, relation, tail = self.triples[index]
        overlap = len(text_terms(" ".join((head, relation, tail))) & self.question_terms)
        overlap_score = overlap / max(len(self.question_terms), 1)
        touch = 1.0 if self._endpoints(index) & set(self.topics) else 0.0
        centrality = self.predicate_freqs[relation] / self.max_freq
        return clamp(0.45 * min(1.0, 2 * overlap_score) + 0.4 * touch + 0.15 * centrality)

    def informativeness(self, index: int, selected: Sequence[int]) -> float:
        _, relation, _ = self.triples[index]
        rarity = 1.0 - (self.predicate_freqs[relation] - 1) / max(self.max_freq - 1, 1)
        novelty = 0.0 if relation in {self.triples[i][1] for i in selected} else 1.0
        specificity = 1.0
        if relation in GENERIC_RELATIONS:
            specificity -= 0.5
        if text_terms(relation) & NOISE_RELATION_TERMS:
            specificity -= 0.5
        return clamp(0.4 * rarity + 0.35 * novelty + 0.25 * clamp(specificity))

    def coverage(self, index: int, selected: Sequence[int]) -> float:
        covered_topics = {
            topic
            for i in selected
            for topic in self.topics
            if topic in self._endpoints(i)
        }
        covered_roles = {self.roles[self.triples[i][1]] for i in selected}
        new_topics = {topic for topic in self.topics if topic in self._endpoints(index)} - covered_topics
        new_role = self.roles[self.triples[index][1]] not in covered_roles
        topic_gain = len(new_topics) / max(len(self.topics), 1) if self.topics else 0.0
        balance = len(covered_topics | new_topics) / max(len(self.topics), 1) if self.topics else 0.0
        return clamp(0.45 * min(1.0, topic_gain * len(self.topics)) + 0.3 * float(new_role) + 0.25 * balance)

    def objective_score(self, task: str, index: int, selected: Sequence[int]) -> float:
        if task == "relatedness":
            return self.relatedness(index)
        if task == "informativeness":
            return self.informativeness(index, selected)
        return self.coverage(index, selected)

    def state_score(self, state: Sequence[int], weights: Dict[str, float]) -> float:
        if not state:
            return 0.0
        relatedness, informativeness, coverage = [], [], []
        for position, index in enumerate(state):
            prefix = state[:position]
            relatedness.append(self.relatedness(index))
            informativeness.append(self.informativeness(index, prefix))
            coverage.append(self.coverage(index, prefix))
        return clamp(
            weights["relatedness"] * (sum(relatedness) / len(relatedness))
            + weights["informativeness"] * (sum(informativeness) / len(informativeness))
            + weights["coverage"] * (sum(coverage) / len(coverage))
        )


def format_candidates(
    triples: Sequence[Triple],
    owners: Sequence[str],
    selected: Sequence[int],
) -> str:
    lines = [
        f"{index + 1}. [{owners[index]}] {triples[index][0]} | {triples[index][1]} | {triples[index][2]}"
        for index in range(len(triples))
        if index not in set(selected)
    ]
    return "\n".join(lines) if lines else "<no candidates>"


def format_selected(triples: Sequence[Triple], owners: Sequence[str], selected: Sequence[int]) -> str:
    if not selected:
        return "None yet."
    return "\n".join(
        f"{index + 1}. [{owners[index]}] {triples[index][0]} | {triples[index][1]} | {triples[index][2]}"
        for index in selected
    )


class PromptFactory:
    """Builds the three task prompts and the combined state-evaluation prompt."""

    def __init__(
        self,
        question: str,
        topic_entities: Sequence[str],
        triples: Sequence[Triple],
        owners: Sequence[str],
        scorer: HeuristicScorer,
    ):
        self.question = question
        self.topics = [str(topic) for topic in topic_entities]
        self.triples = [tuple(triple) for triple in triples]
        self.owners = list(owners)
        self.scorer = scorer

    def _header(self) -> str:
        return (
            f"Question: {self.question}\n"
            f"Topic entities: {', '.join(self.topics) if self.topics else '(none)'}"
        )

    def _core_predicates(self, top_n: int = 8) -> str:
        ranked = sorted(self.scorer.predicate_freqs.items(), key=lambda item: -item[1])
        return ", ".join(relation for relation, _ in ranked[:top_n]) or "(none)"

    def _rare_predicates(self, top_n: int = 8) -> str:
        ranked = sorted(self.scorer.predicate_freqs.items(), key=lambda item: item[1])
        return ", ".join(relation for relation, _ in ranked[:top_n]) or "(none)"

    def _covered(self, selected: Sequence[int]) -> Tuple[str, str]:
        entities = sorted(
            {
                topic
                for index in selected
                for topic in self.topics
                if topic.lower() in {self.triples[index][0].lower(), self.triples[index][2].lower()}
            }
        )
        roles = sorted({semantic_role(self.triples[index][1]) for index in selected})
        return (", ".join(entities) or "None yet", ", ".join(roles) or "None yet")

    def task_prompt(self, task: str, selected: Sequence[int]) -> str:
        candidates = format_candidates(self.triples, self.owners, selected)
        chosen = format_selected(self.triples, self.owners, selected)
        exclusion = (
            f"\nDO NOT select indices: {', '.join(str(index + 1) for index in selected)}"
            if selected
            else ""
        )
        if task == "relatedness":
            body = (
                "RELATEDNESS DEFINITION:\n"
                "A triple is RELATED if it states a core, defining fact about one of the topic\n"
                "entities and helps answer the question.\n\n"
                f"Core/frequent predicates: {self._core_predicates()}\n\n"
                "Judge candidates ONLY on relatedness. Do NOT trade off informativeness or coverage."
            )
            instruction = "Select ONE triple index that is MOST RELATED to the topic entities and the question."
        elif task == "informativeness":
            covered_predicates = ", ".join(
                sorted({self.triples[index][1] for index in selected})
            ) or "None yet"
            body = (
                "INFORMATIVENESS DEFINITION:\n"
                "A triple is INFORMATIVE if it uses an uncommon predicate, adds a non-trivial fact,\n"
                "and introduces information not already present in the selection.\n\n"
                f"Rare predicates: {self._rare_predicates()}\n"
                f"Already selected predicates: {covered_predicates}\n\n"
                "Judge candidates ONLY on informativeness. Do NOT trade off relatedness or coverage."
            )
            instruction = "Select ONE triple index that is MOST INFORMATIVE (rarity + novelty + specificity)."
        else:
            covered_entities, covered_roles = self._covered(selected)
            missing = [topic for topic in self.topics if topic not in covered_entities.split(", ")]
            body = (
                "COVERAGE DEFINITION:\n"
                "The summary must describe EVERY topic entity, not only one. A triple maximizes\n"
                "coverage if it describes a topic entity that is not yet represented, or adds a\n"
                "semantic role (location, time, relationship, attribute, work, organization) that is missing.\n\n"
                f"Topic entities covered so far: {covered_entities}\n"
                f"Topic entities still missing: {', '.join(missing) if missing else 'None'}\n"
                f"Semantic roles covered so far: {covered_roles}\n\n"
                "Judge candidates ONLY on coverage. Do NOT trade off relatedness or informativeness."
            )
            instruction = "Select ONE triple index that MAXIMIZES multi-entity coverage."

        return (
            f"{self._header()}\n\n"
            f"{body}\n\n"
            f"Already selected:\n{chosen}\n\n"
            f"Remaining candidates:\n{candidates}\n\n"
            f"{instruction}{exclusion}\n\n"
            "Return ONLY the integer index. No reasoning, no lists, no other text."
        ).strip()

    def evaluation_prompt(self, states: Sequence[Sequence[int]]) -> str:
        blocks = []
        for position, state in enumerate(states):
            triples_text = format_selected(self.triples, self.owners, state) if state else "(empty summary)"
            blocks.append(f"SUMMARY {position}:\n{triples_text}")
        n_states = len(states)
        return (
            f"EVALUATE candidate multi-entity summaries.\n\n"
            f"{self._header()}\n\n"
            "Score each summary independently on three criteria in [0.00, 1.00]:\n"
            "1. RELATEDNESS: the triples state defining facts about the topic entities and support the question.\n"
            "2. INFORMATIVENESS: the triples add specific, non-generic facts rather than boilerplate.\n"
            "3. COVERAGE: the triples describe ALL topic entities and varied semantic roles, without redundancy.\n\n"
            "SCORING RUBRIC:\n"
            "0.00-0.20 irrelevant or redundant; 0.21-0.40 weak; 0.41-0.60 moderate; "
            "0.61-0.80 strong; 0.81-1.00 excellent.\n\n"
            f"SUMMARIES TO EVALUATE ({n_states} total):\n\n"
            + "\n\n".join(blocks)
            + "\n\nRESPONSE FORMAT:\n"
            f"Output ONLY a JSON array with exactly {n_states} objects, indices 0 to {n_states - 1}:\n"
            '[{"idx": 0, "relatedness": 0.87, "informativeness": 0.75, "coverage": 0.69}]\n'
            "No markdown, no explanations. Start with '[' and end with ']'."
        ).strip()


def extract_json_objects(raw: str) -> List[dict]:
    """Best-effort JSON extraction from model output, tolerating fences and truncation."""
    cleaned = raw.strip().replace("```json", "```").replace("```JSON", "```").strip("`").strip()
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, list):
                return [entry for entry in parsed if isinstance(entry, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

    objects = []
    depth, block_start = 0, None
    for position, char in enumerate(cleaned):
        if char == "{":
            if depth == 0:
                block_start = position
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and block_start is not None:
                try:
                    obj = json.loads(cleaned[block_start : position + 1])
                    if isinstance(obj, dict):
                        objects.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
                block_start = None
    return objects


def aggregate_votes(
    n_states: int,
    raw_outputs: Sequence[str],
    weights: Dict[str, float],
) -> List[float] | None:
    """Average multi-sample LLM votes into one weighted score per state."""
    totals = [{"relatedness": 0.0, "informativeness": 0.0, "coverage": 0.0} for _ in range(n_states)]
    counts = [0] * n_states
    for raw in raw_outputs:
        for entry in extract_json_objects(raw):
            try:
                index = int(entry["idx"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= index < n_states:
                continue
            for key, short in (("relatedness", "r"), ("informativeness", "i"), ("coverage", "c")):
                try:
                    value = float(entry.get(key, entry.get(short, 0.5)))
                except (TypeError, ValueError):
                    value = 0.5
                totals[index][key] += clamp(value)
            counts[index] += 1

    if not any(counts):
        return None
    scores = []
    for index in range(n_states):
        if counts[index] == 0:
            scores.append(0.0)
            continue
        scores.append(
            clamp(sum(weights[key] * totals[index][key] / counts[index] for key in totals[index]))
        )
    return scores


class TaskDecomposedToT:
    """Beam search over partial summaries driven by three task-specific generators."""

    def __init__(
        self,
        num_candidates: int,
        thought_fn: Callable[[str, Tuple[int, ...]], List[int]],
        eval_fn: Callable[[List[Tuple[int, ...]]], List[float]],
        config: ToTConfig,
    ):
        self.num_candidates = num_candidates
        self.thought_fn = thought_fn
        self.eval_fn = eval_fn
        self.config = config

    def search(self, on_step: Callable[[int, int, float], None] | None = None) -> Tuple[TreeNode, List[dict]]:
        beam = [TreeNode(state=())]
        trace: List[dict] = []
        steps = min(self.config.max_summary_len, self.num_candidates)

        for step in range(1, steps + 1):
            children: List[TreeNode] = []
            seen: set[frozenset[int]] = set()
            for node in beam:
                for task in TASKS:
                    for index in self.thought_fn(task, node.state):
                        if not 0 <= index < self.num_candidates or index in node.state:
                            continue
                        key = frozenset(node.state + (index,))
                        if key in seen:
                            continue
                        seen.add(key)
                        children.append(
                            TreeNode(state=node.state + (index,), thought=index, task=task, depth=step)
                        )
            if not children:
                break

            for child, value in zip(children, self.eval_fn([child.state for child in children])):
                child.value = value

            keep = (
                1
                if step == steps
                else max(1, round(self.config.breadth_limit * self.config.prune_keep_multiplier))
            )
            children.sort(key=lambda node: (-node.value, sum(node.state)))
            beam = children[:keep]
            trace.append(
                {
                    "step": step,
                    "expanded": len(children),
                    "kept": len(beam),
                    "best_value": round(beam[0].value, 4),
                    "best_state": list(beam[0].state),
                    "tasks": sorted({child.task for child in beam if child.task}),
                }
            )
            if on_step is not None:
                on_step(step, steps, beam[0].value)

        best = max(beam, key=lambda node: node.value)
        return best, trace
