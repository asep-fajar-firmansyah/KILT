"""Export annotation records as a nested synthetic benchmark keyed by record and entity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from kilt_synmes.pipeline import load_jsonl


def select_annotator(record: dict, annotator: str | None) -> dict:
    outputs = record.get("output", [])
    if not outputs:
        raise ValueError(f"Record {record.get('id')} has no annotator output")
    if annotator is None:
        return outputs[0]
    for output in outputs:
        if output.get("meta", {}).get("annotator") == annotator:
            return output
    available = [output.get("meta", {}).get("annotator") for output in outputs]
    raise KeyError(f"Annotator {annotator!r} not in record {record.get('id')}; available: {available}")


def build_benchmark(
    records: Iterable[dict],
    annotator: str | None = None,
    include_question: bool = False,
) -> Dict[str, dict]:
    benchmark: Dict[str, dict] = {}
    for record in records:
        summary = select_annotator(record, annotator)
        entry: Dict[str, object] = {}
        if include_question:
            entry["question"] = record["input"]
        for group in summary.get("summary_by_entity", []):
            entry[group["topic_entity"]] = {"triples": group["triples"]}
        benchmark[record["id"]] = entry
    return benchmark


def export_benchmark(
    input_path: Path,
    output_path: Path,
    annotator: str | None = None,
    include_question: bool = False,
    indent: int = 2,
) -> int:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Annotation input does not exist: {input_path}. "
            "Run kilt_synmes.annotate first to produce multi-entity summaries."
        )
    paths = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    if not paths:
        raise FileNotFoundError(f"No annotation JSONL files found in {input_path}")

    if input_path.is_dir() or output_path.suffix != ".json":
        output_path.mkdir(parents=True, exist_ok=True)
        targets = [output_path / f"{path.stem}.json" for path in paths]
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        targets = [output_path]

    count = 0
    for path, target in zip(paths, targets):
        benchmark = build_benchmark(load_jsonl(path), annotator, include_question)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(benchmark, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        count += len(benchmark)
    return count


def render_tree(benchmark: Dict[str, dict], name: str = "synthetic_benchmark") -> str:
    lines = [name]
    record_ids = list(benchmark)
    for record_position, record_id in enumerate(record_ids):
        record_last = record_position == len(record_ids) - 1
        lines.append(f"{'└──' if record_last else '├──'}{record_id}")
        record_prefix = "   " if record_last else "│  "
        entities = [key for key in benchmark[record_id] if key != "question"]
        for entity_position, entity in enumerate(entities):
            entity_last = entity_position == len(entities) - 1
            lines.append(f"{record_prefix}{'└──' if entity_last else '├──'}\"{entity}\"")
            entity_prefix = record_prefix + ("   " if entity_last else "│  ")
            triples = benchmark[record_id][entity]["triples"]
            lines.append(f"{entity_prefix}└─triples")
            for triple_position, triple in enumerate(triples):
                triple_last = triple_position == len(triples) - 1
                marker = "└──" if triple_last else "├──"
                lines.append(f"{entity_prefix}  {marker}{' | '.join(triple)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Annotation JSONL file or directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--annotator",
        default=None,
        help="Annotator name to export, e.g. annotator-2 (default: the first annotator)",
    )
    parser.add_argument("--include-question", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--print-tree", action="store_true", help="Print the exported structure")
    args = parser.parse_args()

    count = export_benchmark(
        args.input, args.output, args.annotator, args.include_question, args.indent
    )
    if args.print_tree:
        paths = sorted(args.output.glob("*.json")) if args.output.is_dir() else [args.output]
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                print(render_tree(json.load(handle), path.stem))
    print(f"Wrote {count} benchmark records to {args.output}")


if __name__ == "__main__":
    main()
