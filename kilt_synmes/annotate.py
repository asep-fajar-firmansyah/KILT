"""Annotation stage: LLM-in-the-loop multi-entity summarization over candidate triples.

Each annotator runs a task-decomposed Tree-of-Thought search (ToT4ES style) over the
candidate triples produced by `kilt_synmes.pipeline`, yielding one multi-entity
summary per annotator.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from kilt_synmes.llm_backends import build_backend
from kilt_synmes.pipeline import load_jsonl, triple_to_text
from kilt_synmes.tot import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    TASKS,
    HeuristicScorer,
    PromptFactory,
    ToTConfig,
    TaskDecomposedToT,
    aggregate_votes,
    extract_first_int,
)

MAX_STATES_PER_EVAL_CHUNK = 5


def availability_fn(owners: Sequence[str], num_candidates: int, per_entity_limit: int | None):
    """Hard cap: an entity stops accepting triples once it reaches its quota."""

    def available(state: Tuple[int, ...]) -> List[int]:
        unselected = [index for index in range(num_candidates) if index not in state]
        if per_entity_limit is None:
            return unselected
        used = Counter(owners[index] for index in state)
        return [index for index in unselected if used[owners[index]] < per_entity_limit]

    return available


def heuristic_thought_fn(scorer: HeuristicScorer, config: ToTConfig):
    def generate(task: str, state: Tuple[int, ...], available: Sequence[int]) -> List[int]:
        ranked = sorted(
            available,
            key=lambda index: (-scorer.objective_score(task, index, state), index),
        )
        return ranked[: config.n_candidates_per_task]

    return generate


def llm_thought_fn(backend, prompts: PromptFactory, scorer: HeuristicScorer, config: ToTConfig):
    fallback = heuristic_thought_fn(scorer, config)

    def generate(task: str, state: Tuple[int, ...], available: Sequence[int]) -> List[int]:
        prompt = prompts.task_prompt(task, state, available)
        outputs = backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=config.thought_temperature,
            max_new_tokens=32,
            n=config.n_candidates_per_task,
        )
        allowed = set(available)
        indices = []
        for text in outputs:
            value = extract_first_int(text)
            if value is not None and (value - 1) in allowed:
                indices.append(value - 1)
        unique = list(dict.fromkeys(indices))
        return unique or fallback(task, state, available)

    return generate


def heuristic_eval_fn(scorer: HeuristicScorer, config: ToTConfig):
    def evaluate(states: Sequence[Tuple[int, ...]]) -> List[float]:
        return [scorer.state_score(state, config.objective_weights) for state in states]

    return evaluate


def llm_eval_fn(backend, prompts: PromptFactory, scorer: HeuristicScorer, config: ToTConfig):
    fallback = heuristic_eval_fn(scorer, config)
    cache: Dict[frozenset, float] = {}

    def evaluate(states: Sequence[Tuple[int, ...]]) -> List[float]:
        scores: List[float | None] = [cache.get(frozenset(state)) for state in states]
        pending = [position for position, score in enumerate(scores) if score is None]
        for chunk_start in range(0, len(pending), MAX_STATES_PER_EVAL_CHUNK):
            chunk = pending[chunk_start : chunk_start + MAX_STATES_PER_EVAL_CHUNK]
            chunk_states = [states[position] for position in chunk]
            outputs = backend.chat(
                [{"role": "user", "content": prompts.evaluation_prompt(chunk_states)}],
                temperature=config.eval_temperature,
                max_new_tokens=max(64, len(chunk_states) * 48),
                n=config.n_evals,
            )
            chunk_scores = aggregate_votes(len(chunk_states), outputs, config.objective_weights)
            if chunk_scores is None:
                chunk_scores = fallback(chunk_states)
            for position, score in zip(chunk, chunk_scores):
                scores[position] = score
                cache[frozenset(states[position])] = score
        return [score if score is not None else 0.0 for score in scores]

    return evaluate


def summarize(triples: Sequence[Sequence[str]], selected: Sequence[int]) -> str:
    return " ".join(triple_to_text(tuple(triples[index])) for index in selected)


def entity_coverage(
    triples: Sequence[Sequence[str]],
    topic_entities: Sequence[str],
    selected: Sequence[int],
) -> Dict[str, int]:
    coverage = {str(topic): 0 for topic in topic_entities}
    for index in selected:
        head, _, tail = triples[index]
        endpoints = {head.lower(), tail.lower()}
        for topic in coverage:
            if topic.lower() in endpoints:
                coverage[topic] += 1
    return coverage


def group_by_entity(
    owners: Sequence[str],
    topic_entities: Sequence[str],
    selected: Sequence[int],
) -> List[Tuple[str, List[int]]]:
    """Bucket selected candidates by seed entity, keeping topic-entity order."""
    groups: Dict[str, List[int]] = {}
    for index in selected:
        groups.setdefault(owners[index], []).append(index)
    ordered = [entity for entity in topic_entities if entity in groups]
    ordered += [entity for entity in groups if entity not in ordered]
    return [(entity, groups[entity]) for entity in ordered]


def mean_pairwise_jaccard(selections: Sequence[Sequence[int]]) -> float:
    pairs = [
        (set(selections[i]), set(selections[j]))
        for i in range(len(selections))
        for j in range(i + 1, len(selections))
    ]
    if not pairs:
        return 1.0
    scores = [len(left & right) / len(left | right) if (left | right) else 1.0 for left, right in pairs]
    return round(sum(scores) / len(scores), 4)


def annotate_record(
    record: dict,
    annotators: Sequence[dict],
    config: ToTConfig,
    progress: bool = False,
    per_entity_budget: bool = True,
) -> dict:
    """Run every annotator over one retrieval record and collect its summaries."""
    triples = [tuple(triple) for triple in record["candidate_triples"]]
    provenance = record.get("provenance", [])
    owners = [item.get("topic_entity", "") for item in provenance] or [""] * len(triples)
    topic_entities = [str(topic) for topic in record.get("meta", {}).get("topic_entities", [])]
    question = record["question"]

    scorer = HeuristicScorer(triples, question, topic_entities)
    prompts = PromptFactory(question, topic_entities, triples, owners, scorer)

    per_entity_limit = config.max_summary_len if per_entity_budget else None
    available_fn = availability_fn(owners, len(triples), per_entity_limit)
    buckets = len(set(owners[: len(triples)])) or 1
    max_steps = min(
        config.max_summary_len * buckets if per_entity_budget else config.max_summary_len,
        len(triples),
    )

    outputs = []
    selections = []
    for annotator in annotators:
        backend = annotator["backend"]
        if backend is None:
            thought_fn = heuristic_thought_fn(scorer, config)
            eval_fn = heuristic_eval_fn(scorer, config)
        else:
            thought_fn = llm_thought_fn(backend, prompts, scorer, config)
            eval_fn = llm_eval_fn(backend, prompts, scorer, config)

        search = TaskDecomposedToT(
            len(triples), thought_fn, eval_fn, config, available_fn, max_steps
        )
        with tqdm(
            total=max_steps,
            desc=f"{record['id']} {annotator['name']}",
            unit="triple",
            leave=False,
            disable=not progress,
        ) as bar:
            best, trace = search.search(
                on_step=lambda step, total, value: (
                    bar.update(1),
                    bar.set_postfix(value=f"{value:.3f}"),
                )
            )
        selected = list(best.state)
        selections.append(selected)
        grouped = group_by_entity(owners, topic_entities, selected)
        ordered = [index for _, indices in grouped for index in indices]
        outputs.append(
            {
                "answer": summarize(triples, ordered),
                "provenance": [provenance[index] for index in ordered if index < len(provenance)],
                "summary_by_entity": [
                    {
                        "topic_entity": entity,
                        "answer": summarize(triples, indices),
                        "candidate_indices": indices,
                        "triples": [list(triples[index]) for index in indices],
                    }
                    for entity, indices in grouped
                ],
                "meta": {
                    "annotator": annotator["name"],
                    "model": annotator["model"],
                    "seed": annotator["seed"],
                    "selected_candidate_indices": ordered,
                    "selected_triples": [list(triples[index]) for index in ordered],
                    "selection_order": selected,
                    "value": round(best.value, 4),
                    "entity_coverage": entity_coverage(triples, topic_entities, selected),
                    "triples_per_seed_entity": dict(Counter(owners[index] for index in selected)),
                    "search_trace": trace,
                },
            }
        )

    return {
        "id": record["id"],
        "input": question,
        "output": outputs,
        "meta": {
            "graph_id": record.get("meta", {}).get("graph_id"),
            "topic_entities": topic_entities,
            "candidate_count": len(triples),
            "annotators": [annotator["name"] for annotator in annotators],
            "annotator_models": [annotator["model"] for annotator in annotators],
            "max_summary_len": config.max_summary_len,
            "budget_scope": "per-entity" if per_entity_budget else "total",
            "max_summary_triples": max_steps,
            "objective_weights": config.objective_weights,
            "thought_temperature": config.thought_temperature,
            "eval_temperature": config.eval_temperature,
            "breadth_limit": config.breadth_limit,
            "n_candidates_per_task": config.n_candidates_per_task,
            "n_evals": config.n_evals,
            "tasks": list(TASKS),
            "annotator_agreement": mean_pairwise_jaccard(selections),
        },
    }


def build_annotators(
    count: int,
    default_model: str,
    model_overrides: Sequence[str] | None,
    seed: int,
    ollama_url: str,
) -> List[dict]:
    overrides = list(model_overrides or [])
    if overrides and len(overrides) not in (1, count):
        raise ValueError("--annotator-model must be given once or once per annotator")
    annotators = []
    backends: Dict[Tuple[str, int], object] = {}
    for position in range(count):
        model = overrides[position] if len(overrides) == count else (overrides[0] if overrides else default_model)
        annotator_seed = seed + position
        key = (model, annotator_seed)
        if key not in backends:
            backends[key] = build_backend(model, seed=annotator_seed, ollama_url=ollama_url)
        annotators.append(
            {
                "name": f"annotator-{position + 1}",
                "model": model,
                "seed": annotator_seed,
                "backend": backends[key],
            }
        )
    return annotators


def annotate_dataset(
    input_path: Path,
    output_path: Path,
    annotators: Sequence[dict],
    config: ToTConfig,
    limit: int | None = None,
    progress: bool = False,
    per_entity_budget: bool = True,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Retrieval input does not exist: {input_path}. "
            "Run kilt_synmes.pipeline first to produce candidate-triple records."
        )
    paths = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    if not paths:
        raise FileNotFoundError(f"No retrieval JSONL files found in {input_path}")

    if input_path.is_dir() or output_path.suffix != ".jsonl":
        output_path.mkdir(parents=True, exist_ok=True)
        targets = [output_path / path.name for path in paths]
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        targets = [output_path]

    count = 0
    for path, target in zip(paths, targets):
        records = list(load_jsonl(path))
        if limit is not None:
            records = records[:limit]
        with target.open("w", encoding="utf-8") as handle:
            for record in tqdm(
                records, desc=path.name, unit="record", disable=not progress
            ):
                line = json.dumps(
                    annotate_record(record, annotators, config, progress, per_entity_budget),
                    ensure_ascii=False,
                )
                handle.write(line + "\n")
                handle.flush()
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Retrieval JSONL file or directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument(
        "--model",
        default="heuristic",
        help="Backend spec: 'heuristic', 'ollama:<model>', or '[hf:]<huggingface-model-id>'",
    )
    parser.add_argument(
        "--annotator-model",
        action="append",
        default=None,
        help="Per-annotator backend spec; repeat once per annotator to mix models",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-summary-len", type=int, default=5)
    parser.add_argument(
        "--budget-scope",
        choices=["per-entity", "total"],
        default="per-entity",
        help="Whether --max-summary-len caps triples per topic entity or for the whole summary",
    )
    parser.add_argument("--n-candidates-per-task", type=int, default=2)
    parser.add_argument("--n-evals", type=int, default=3)
    parser.add_argument("--breadth-limit", type=int, default=3)
    parser.add_argument("--prune-keep-multiplier", type=float, default=1.5)
    parser.add_argument("--thought-temperature", type=float, default=0.8)
    parser.add_argument("--eval-temperature", type=float, default=0.3)
    parser.add_argument("--w-relatedness", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["relatedness"])
    parser.add_argument("--w-informativeness", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["informativeness"])
    parser.add_argument("--w-coverage", type=float, default=DEFAULT_OBJECTIVE_WEIGHTS["coverage"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()

    if args.annotators < 1 or args.max_summary_len < 1 or args.breadth_limit < 1:
        parser.error("--annotators, --max-summary-len, and --breadth-limit must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    weight_total = args.w_relatedness + args.w_informativeness + args.w_coverage
    if weight_total <= 0:
        parser.error("objective weights must sum to a positive value")

    config = ToTConfig(
        max_summary_len=args.max_summary_len,
        n_candidates_per_task=args.n_candidates_per_task,
        n_evals=args.n_evals,
        breadth_limit=args.breadth_limit,
        prune_keep_multiplier=args.prune_keep_multiplier,
        thought_temperature=args.thought_temperature,
        eval_temperature=args.eval_temperature,
        objective_weights={
            "relatedness": args.w_relatedness / weight_total,
            "informativeness": args.w_informativeness / weight_total,
            "coverage": args.w_coverage / weight_total,
        },
    )

    annotators = build_annotators(
        args.annotators, args.model, args.annotator_model, args.seed, args.ollama_url
    )
    count = annotate_dataset(
        args.input,
        args.output,
        annotators,
        config,
        args.limit,
        not args.no_progress,
        args.budget_scope == "per-entity",
    )
    print(f"Annotated {count} records with {len(annotators)} annotators into {args.output}")


if __name__ == "__main__":
    main()
