"""Build extractive multi-entity summaries in a KILT-shaped format."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Triple = Tuple[str, str, str]


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


def normalize_triples(edges: Sequence[Sequence[object]]) -> List[Triple]:
    triples = []
    for edge in edges:
        if len(edge) != 3:
            raise ValueError(f"Every edge must contain three values: {edge!r}")
        triples.append((str(edge[0]), str(edge[1]), str(edge[2])))
    return triples


def answer_terms(record: dict) -> set[str]:
    answer = record.get("answer")
    values = record.get("answer_entities", [])
    if not isinstance(values, list):
        values = [values]
    else:
        values = list(values)
    if answer is not None:
        values.append(answer)
    return {str(value).lower() for value in values if value is not None}


def reasoning_path_triples(record: dict) -> List[Triple]:
    """Read triples from either supported original reasoning-path field."""
    triples = []
    for field in ("reasoning_path", "original_reasoning_path"):
        value = record.get(field, [])
        if value:
            triples.extend(normalize_triples(value))
    return triples


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
    answer_terms_set: set[str],
    top_k: int,
    preferred_triples: set[Triple] | None = None,
) -> List[Tuple[str, List[Tuple[int, Triple]]]]:
    """Select direct, answer-relevant edges independently for each topic."""
    selected = []
    for topic in (str(entity) for entity in topic_entities):
        topic_lower = topic.lower()
        ranked = []
        for index, triple in enumerate(triples):
            head, relation, tail = triple
            endpoints = {head.lower(), tail.lower()}
            direct = int(topic_lower in endpoints)
            answer_hit = int(bool(endpoints & answer_terms_set))
            preferred = int(preferred_triples is not None and triple in preferred_triples)
            score = (direct, preferred, answer_hit, -len(relation), -index)
            if direct:
                ranked.append((score, index, triple))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected.append((topic, [(index, triple) for _, index, triple in ranked[:top_k]]))
    return selected


def triple_to_text(triple: Triple) -> str:
    head, relation, tail = triple
    relation_text = relation.replace("_", " ").replace(".", " ")
    return f"{head} {relation_text.strip()} {tail}."


def build_annotation_request(record: dict, evidence: Sequence[dict]) -> dict:
    return {
        "question": record["question"],
        "topic_entities": record.get("topic_entities", []),
        "candidate_triples": [item["triple"] for item in evidence],
        "evidence": [item["text"] for item in evidence],
        "instruction": "Write a concise multi-entity summary grounded only in the candidate triples.",
    }


def run_annotator(command: str, request: dict) -> str:
    result = subprocess.run(
        shlex.split(command),
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    summary = result.stdout.strip()
    if not summary:
        raise ValueError(f"Annotator produced no summary: {command}")
    return summary


def build_record(
    record: dict,
    top_k: int,
    max_evidence: int,
    full_graph: Sequence[Triple] | None = None,
    annotator_commands: Sequence[str] | None = None,
) -> dict:
    graph_id = str(record["graph_id"])
    original_triples = normalize_triples(record.get("edges", []))
    path_triples = reasoning_path_triples(record)
    triples = combine_candidates(full_graph or original_triples, original_triples, path_triples)
    topics = [str(entity) for entity in record.get("topic_entities", [])]
    selected = select_topk_by_entity(
        triples,
        topics,
        answer_terms(record),
        top_k,
        preferred_triples=set(original_triples),
    )

    evidence = []
    seen_triples = set()
    for topic, topic_triples in selected:
        for triple_index, triple in topic_triples:
            if triple in seen_triples:
                continue
            seen_triples.add(triple)
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
                }
            )
            if len(evidence) >= max_evidence:
                break
        if len(evidence) >= max_evidence:
            break

    extractive_summary = " ".join(item["text"] for item in evidence)
    if not extractive_summary:
        extractive_summary = "No supporting evidence was selected."

    annotation_request = build_annotation_request(record, evidence)
    outputs = []
    for command in annotator_commands or []:
        outputs.append(
            {
                "answer": run_annotator(command, annotation_request),
                "provenance": evidence,
                "meta": {"annotator": command},
            }
        )
    if not outputs:
        outputs.append(
            {
                "answer": extractive_summary,
                "provenance": evidence,
                "meta": {"annotator": "extractive"},
            }
        )

    return {
        "id": f"m3gqa-{graph_id}",
        "input": f"Summarize the relevant facts needed to answer: {record['question']}",
        "output": outputs,
        "meta": {
            "graph_id": record["graph_id"],
            "topic_entities": topics,
            "source_answer": record.get("answer"),
            "answer_entities": record.get("answer_entities", []),
            "top_k_per_entity": top_k,
            "max_evidence": max_evidence,
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
    top_k: int,
    max_evidence: int,
    source_file: str | None = None,
    limit: int | None = None,
    graph_path: Path | None = None,
    annotator_commands: Sequence[str] | None = None,
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
    if graph_path is not None:
        graph_ids = {str(record["graph_id"]) for record in all_records}
        graph_index = load_graphs(graph_path, graph_ids)

    count = 0
    for path in paths:
        output_path = output / path.name
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records_by_path[path]:
                graph = graph_index.get(str(record["graph_id"]))
                json.dump(
                    build_record(record, top_k, max_evidence, graph, annotator_commands),
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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-evidence", type=int, default=15)
    parser.add_argument("--source-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--graph-path", type=Path, default=None)
    parser.add_argument(
        "--annotator-command",
        action="append",
        default=[],
        help="Command that reads an annotation request as JSON from stdin and writes one summary to stdout. Repeat for multiple annotators.",
    )
    args = parser.parse_args()
    if args.top_k < 1 or args.max_evidence < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--top-k, --max-evidence, and --limit must be positive")
    count = export_dataset(
        args.data_dir,
        args.split,
        args.output,
        args.top_k,
        args.max_evidence,
        args.source_file,
        args.limit,
        args.graph_path or args.data_dir / "new_graphs.jsonl",
        args.annotator_command,
    )
    print(f"Wrote {count} KILT records to {args.output}")


if __name__ == "__main__":
    main()
