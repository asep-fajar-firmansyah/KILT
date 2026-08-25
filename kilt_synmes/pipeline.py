"""Retrieve candidate triples for SynMeS multi-entity summarization."""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Triple = Tuple[str, str, str]


DEFAULT_RANKING_WEIGHTS = {
    "question_match": 4,
    "source_priority": 1,
    "noise_penalty": -4,
    "generic_relation_penalty": -2,
}
NOISE_RELATION_TERMS = {"article", "image", "reviewed", "type"}
GENERIC_RELATIONS = {"common.topic.subject_of", "common.topic.subjects"}


def positive_int_or_unlimited(value: str) -> int | None:
    if value == "unlimited":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer or 'unlimited'") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer or 'unlimited'")
    return parsed


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error


def load_graphs(path: Path, graph_ids: set[str]) -> Dict[str, List[Triple]]:
    """Load only graph records needed by the selected M3GQA examples."""
    graphs: Dict[str, List[Triple]] = {}
    for record in load_jsonl(path):
        graph_id = str(record.get("id", record.get("graph_id", "")))
        if graph_id in graph_ids:
            graphs[graph_id] = normalize_triples(record.get("graph", record.get("edges", [])))
            if len(graphs) == len(graph_ids):
                break
    missing = graph_ids - set(graphs)
    if missing:
        raise KeyError(f"Missing graph records for graph_id(s): {sorted(missing)[:5]}")
    return graphs


def load_graphs_from_directory(
    graph_dir: Path,
    split: str,
    graph_ids: set[str],
) -> Dict[str, List[Triple]]:
    """Load graph files directly by graph ID without scanning a JSONL file."""
    split_dir = graph_dir / split
    graphs: Dict[str, List[Triple]] = {}
    missing = []
    for graph_id in graph_ids:
        path = split_dir / f"{graph_id}.json"
        if not path.is_file():
            missing.append(graph_id)
            continue
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        file_graph_id = str(record.get("graph_id", record.get("id", graph_id)))
        if file_graph_id != graph_id:
            raise ValueError(f"Graph file {path} contains graph_id {file_graph_id}, expected {graph_id}")
        graphs[graph_id] = normalize_triples(
            record.get("subgraph", record.get("graph", record.get("edges", [])))
        )
    if missing:
        raise KeyError(f"Missing graph file(s) for graph_id(s): {sorted(missing)[:5]}")
    return graphs


def normalize_triples(edges: Sequence[Sequence[object]]) -> List[Triple]:
    triples = []
    for edge in edges:
        if len(edge) != 3:
            raise ValueError(f"Every edge must contain three values: {edge!r}")
        triples.append((str(edge[0]), str(edge[1]), str(edge[2])))
    return triples


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "based", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "what", "which", "who",
    "with",
}


def text_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 1 and term not in STOP_WORDS
    }


def triple_matches_question(triple: Triple, question_terms: set[str]) -> bool:
    return bool(text_terms(" ".join(triple)) & question_terms)


def triple_rank_score(
    triple: Triple,
    question_terms: set[str],
    preferred_triples: set[Triple],
    weights: Dict[str, int],
) -> int:
    """Return a transparent, training-free relevance score for one triple."""
    _, relation, _ = triple
    relation_terms = text_terms(relation)
    score = weights["question_match"] * len(text_terms(" ".join(triple)) & question_terms)
    if triple in preferred_triples:
        score += weights["source_priority"]
    if relation in GENERIC_RELATIONS:
        score += weights["generic_relation_penalty"]
    if relation_terms & NOISE_RELATION_TERMS:
        score += weights["noise_penalty"]
    return score


def reasoning_path_triples(record: dict) -> List[Triple]:
    """Read triples from either supported original reasoning-path field."""
    triples = []
    for field in ("reasoning_path", "original_reasoning_path"):
        value = record.get(field, [])
        if value:
            triples.extend(normalize_triples(value))
    return triples


def bfs_distances(triples: Sequence[Triple], start: str) -> Dict[str, int]:
    """Return undirected shortest-path distances from one topic entity."""
    adjacency: Dict[str, List[str]] = {}
    for head, _, tail in triples:
        adjacency.setdefault(head.lower(), []).append(tail.lower())
        adjacency.setdefault(tail.lower(), []).append(head.lower())
    distances = {start.lower(): 0}
    queue = deque([start.lower()])
    while queue:
        entity = queue.popleft()
        for neighbor in adjacency.get(entity, []):
            if neighbor not in distances:
                distances[neighbor] = distances[entity] + 1
                queue.append(neighbor)
    return distances


def assign_initial_triples(
    triples: Sequence[Triple],
    topic_entities: Sequence[str],
    question: str,
    max_evidence: int | None,
) -> List[Tuple[Triple, str, Tuple[int, int, float]]]:
    """Assign top initial triples to topics by touch, overlap, and BFS distance."""
    topics = [str(topic) for topic in topic_entities]
    distances = {topic: bfs_distances(triples, topic) for topic in topics}
    question_terms = text_terms(question)
    assigned = {topic: [] for topic in topics}

    for index, triple in enumerate(triples):
        head, relation, tail = triple
        endpoints = {head.lower(), tail.lower()}
        best_topic = None
        best_score = None
        for topic in topics:
            distance = min(
                distances[topic].get(head.lower(), float("inf")),
                distances[topic].get(tail.lower(), float("inf")),
            )
            score = (
                int(topic.lower() in endpoints),
                len(text_terms(relation) & question_terms),
                -distance,
            )
            if best_score is None or score > best_score:
                best_topic = topic
                best_score = score
        if best_topic is not None:
            assigned[best_topic].append((best_score, index, triple))

    result = []
    for topic in topics:
        assigned[topic].sort(key=lambda item: (item[0], -item[1]), reverse=True)
        for score, _, triple in assigned[topic][:max_evidence]:
            result.append((triple, topic, score))
    return result


def map_triples_to_topics(
    triples: Sequence[Triple],
    topic_entities: Sequence[str],
) -> List[Tuple[Triple, List[str]]]:
    """Map triples to topic entities occurring at either endpoint."""
    topics = [str(topic) for topic in topic_entities]
    mapped = []
    for triple in triples:
        endpoints = {triple[0].lower(), triple[2].lower()}
        owners = [topic for topic in topics if topic.lower() in endpoints]
        mapped.append((triple, owners))
    return mapped


def map_reasoning_triples_to_topics(
    triples: Sequence[Triple],
    topic_entities: Sequence[str],
) -> List[Tuple[Triple, List[str]]]:
    """Backward-compatible reasoning-path mapping helper."""
    return map_triples_to_topics(triples, topic_entities)


def combine_candidates(*triple_sets: Sequence[Triple]) -> List[Triple]:
    candidates = []
    seen = set()
    for triple_set in triple_sets:
        for triple in triple_set:
            if triple not in seen:
                seen.add(triple)
                candidates.append(triple)
    return candidates


def select_topk_by_entity(
    triples: Sequence[Triple],
    topic_entities: Sequence[str],
    question_terms: set[str],
    top_k: int | None,
    preferred_triples: set[Triple] | None = None,
    ranking_weights: Dict[str, int] | None = None,
) -> List[Tuple[str, List[Tuple[int, Triple]]]]:
    """Rank direct edges independently for each retrieval seed."""
    preferred_triples = preferred_triples or set()
    ranking_weights = ranking_weights or DEFAULT_RANKING_WEIGHTS
    selected = []
    for topic in (str(entity) for entity in topic_entities):
        topic_lower = topic.lower()
        ranked = []
        for index, triple in enumerate(triples):
            head, relation, tail = triple
            endpoints = {head.lower(), tail.lower()}
            direct = int(topic_lower in endpoints)
            score = (direct, triple_rank_score(triple, question_terms, preferred_triples, ranking_weights), -len(relation), -index)
            if direct:
                ranked.append((score, index, triple))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected.append((topic, [(index, triple) for _, index, triple in ranked[:top_k]]))
    return selected


def question_relevant_paths(
    triples: Sequence[Triple],
    start: str,
    targets: set[str],
    question_terms: set[str],
    max_hops: int,
) -> List[List[Triple]]:
    """Find bounded shortest paths to seed or question-relevant graph context."""
    start_lower = start.lower()
    targets = {target.lower() for target in targets} - {start_lower}
    adjacency: Dict[str, List[Tuple[str, Triple]]] = {}
    for triple in triples:
        head, _, tail = triple
        adjacency.setdefault(head.lower(), []).append((tail.lower(), triple))
        adjacency.setdefault(tail.lower(), []).append((head.lower(), triple))

    queue = deque([(start_lower, [])])
    visited = {start_lower}
    paths = []
    seen_paths = set()
    while queue:
        node, path = queue.popleft()
        if len(path) >= max_hops:
            continue
        for neighbor, triple in adjacency.get(node, []):
            next_path = path + [triple]
            path_key = tuple(next_path)
            reaches_target = neighbor in targets
            matches_question = triple_matches_question(triple, question_terms)
            if (reaches_target or matches_question) and path_key not in seen_paths:
                seen_paths.add(path_key)
                paths.append(next_path)
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, next_path))
    return paths


def select_multihop_paths(
    triples: Sequence[Triple],
    seed_entities: Sequence[str],
    question_terms: set[str],
    max_hops: int,
) -> List[Tuple[str, Triple]]:
    """Retrieve bounded paths connecting seeds or reaching question context."""
    seeds = [str(seed) for seed in seed_entities]
    paths = []
    for seed in seeds:
        for path in question_relevant_paths(triples, seed, set(seeds), question_terms, max_hops):
            paths.extend((seed, triple) for triple in path)
    return paths


def triple_to_text(triple: Triple) -> str:
    head, relation, tail = triple
    relation_text = relation.replace("_", " ").replace(".", " ")
    return f"{head} {relation_text.strip()} {tail}."


def build_retrieval_record(
    record: dict,
    max_evidence: int | None,
    full_graph: Sequence[Triple] | None = None,
    max_hops: int = 3,
    ranking_weights: Dict[str, int] | None = None,
) -> dict:
    """Build the candidate-triple retrieval result without summarization."""
    graph_id = str(record["graph_id"])
    original_triples = normalize_triples(record.get("edges", []))
    path_triples = reasoning_path_triples(record)
    triples = combine_candidates(full_graph or original_triples, original_triples, path_triples)
    topics = [str(entity) for entity in record.get("topic_entities", [])]
    initial_triples = combine_candidates(path_triples, original_triples)
    assigned_initial = assign_initial_triples(
        initial_triples,
        topics,
        record["question"],
        max_evidence,
    )
    path_entities = [
        entity
        for triple, owner, _ in assigned_initial
        if triple in path_triples
        for entity in (triple[0], triple[2])
    ]
    expansion_pairs = [(topic, topic) for topic in topics]
    for triple, owner, _ in assigned_initial:
        expansion_pairs.extend((entity, owner) for entity in (triple[0], triple[2]))
    expansion_pairs = list(dict.fromkeys(expansion_pairs))
    seed_entities = list(dict.fromkeys(seed for seed, _ in expansion_pairs))
    seed_owners = {seed: owner for seed, owner in expansion_pairs}
    question_terms = text_terms(record["question"])
    ranking_weights = ranking_weights or DEFAULT_RANKING_WEIGHTS
    selected = []
    for seed, owner in expansion_pairs:
        ranked = select_topk_by_entity(
            triples,
            [seed],
            question_terms,
            len(triples),
            preferred_triples=set(original_triples) | set(path_triples),
            ranking_weights=ranking_weights,
        )
        selected.append((owner, ranked[0][1]))
    multihop_paths = select_multihop_paths(triples, seed_entities, question_terms, max_hops)
    triple_indices = {triple: index for index, triple in enumerate(triples)}

    evidence = []
    seen_triples = set()
    retrieved_count = 0
    initial_by_topic = {topic: 0 for topic in topics}
    retrieved_by_seed = {topic: 0 for topic in topics}

    def add_evidence(
        topic: str,
        triple: Triple,
        mandatory: bool = False,
        initial_topics: Sequence[str] | None = None,
    ) -> bool:
        nonlocal retrieved_count
        if triple in seen_triples:
            return False
        owners = [owner for owner in initial_topics or [topic] if owner in initial_by_topic]
        if not mandatory and max_evidence is not None and retrieved_by_seed[topic] + initial_by_topic[topic] >= max_evidence:
            return False
        seen_triples.add(triple)
        if mandatory:
            for owner in owners:
                initial_by_topic[owner] += 1
        else:
            retrieved_count += 1
            retrieved_by_seed[topic] += 1
        triple_index = triple_indices[triple]
        evidence.append(
            {
                "source_id": f"m3gqa-{graph_id}-edge-{triple_index}",
                "source_type": "structured_graph",
                "graph_id": graph_id,
                "triple_index": triple_index,
                "title": f"M3GQA graph {graph_id}",
                "text": triple_to_text(triple),
                "topic_entity": topic,
                "triple": list(triple),
                "initial_topic_entities": list(initial_topics or []),
            }
        )
        return True

    for triple, owner, score in assigned_initial:
        add_evidence(owner, triple, mandatory=True, initial_topics=[owner])

    for topic, topic_triples in selected:
        for _, triple in topic_triples:
            add_evidence(topic, triple)
            if max_evidence is not None and retrieved_by_seed[topic] >= max_evidence:
                break

    for topic, triple in multihop_paths:
        owner = seed_owners.get(topic)
        if owner is None:
            continue
        add_evidence(owner, triple)

    return {
        "id": f"m3gqa-{graph_id}",
        "question": record["question"],
        "candidate_triples": [item["triple"] for item in evidence],
        "provenance": evidence,
        "meta": {
            "graph_id": record["graph_id"],
            "topic_entities": topics,
            "reasoning_path_entities": path_entities,
            "initial_topic_triples": [
                {"triple": list(triple), "topic_entity": owner, "score": list(score)}
                for triple, owner, score in assigned_initial
            ],
            "question_terms": sorted(question_terms),
            "ranking_weights": ranking_weights,
            "max_evidence_per_topic": max_evidence,
            "max_retrieved_evidence": (
                None if max_evidence is None else max_evidence * len(topics)
            ),
            "initial_evidence_by_topic": initial_by_topic,
            "source_evidence_count": len(assigned_initial),
            "retrieved_evidence_count": retrieved_count,
            "retrieved_evidence_by_seed": retrieved_by_seed,
            "max_hops": max_hops,
            "retrieved_from_full_graph": full_graph is not None,
            "candidate_sources": {
                "source_edges": len(original_triples),
                "reasoning_path": len(path_triples),
                "knowledge_graph": len(full_graph or []),
            },
        },
    }


def export_dataset(
    data_dir: Path,
    split: str,
    output: Path,
    max_evidence: int | None,
    source_file: str | None = None,
    limit: int | None = None,
    graph_path: Path | None = None,
    graph_dir: Path | None = None,
    max_hops: int = 3,
    ranking_weights: Dict[str, int] | None = None,
) -> int:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"M3GQA split directory does not exist: {split_dir}")
    paths = [split_dir / source_file] if source_file else sorted(split_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No JSONL files found in {split_dir}")

    output.mkdir(parents=True, exist_ok=True)
    records_by_path = {}
    all_records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records = list(load_jsonl(path))
        if limit is not None:
            records = records[:limit]
        records_by_path[path] = records
        all_records.extend(records)

    graph_index = {}
    graph_ids = {str(record["graph_id"]) for record in all_records}
    if graph_dir is not None:
        graph_index = load_graphs_from_directory(graph_dir, split, graph_ids)
    elif graph_path is not None:
        graph_index = load_graphs(graph_path, graph_ids)

    count = 0
    for path in paths:
        output_path = output / path.name
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records_by_path[path]:
                graph = graph_index.get(str(record["graph_id"]))
                json.dump(
                    build_retrieval_record(record, max_evidence, graph, max_hops, ranking_weights),
                    handle,
                    ensure_ascii=False,
                )
                handle.write("\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-evidence", type=positive_int_or_unlimited, default=15)
    parser.add_argument("--question-match-weight", type=int, default=DEFAULT_RANKING_WEIGHTS["question_match"])
    parser.add_argument("--source-priority-weight", type=int, default=DEFAULT_RANKING_WEIGHTS["source_priority"])
    parser.add_argument("--noise-penalty", type=int, default=DEFAULT_RANKING_WEIGHTS["noise_penalty"])
    parser.add_argument("--generic-relation-penalty", type=int, default=DEFAULT_RANKING_WEIGHTS["generic_relation_penalty"])
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--graph-path", type=Path, default=None)
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=None,
        help="Directory containing <split>/<graph_id>.json graph files for direct lookup.",
    )
    args = parser.parse_args()
    if args.max_hops < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--max-hops and --limit must be positive")
    count = export_dataset(
        args.data_dir,
        args.split,
        args.output,
        args.max_evidence,
        args.source_file,
        args.limit,
        args.graph_path or (None if args.graph_dir is not None else args.data_dir / "new_graphs.jsonl"),
        args.graph_dir,
        args.max_hops,
        {
            "question_match": args.question_match_weight,
            "source_priority": args.source_priority_weight,
            "noise_penalty": args.noise_penalty,
            "generic_relation_penalty": args.generic_relation_penalty,
        },
    )
    print(f"Wrote {count} KILT records to {args.output}")


if __name__ == "__main__":
    main()
