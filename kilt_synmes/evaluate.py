"""Evaluate annotation evidence against M3GQA reference edges."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


DEFAULT_GOLD_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "M3GQA" / "test"
METRICS = ("precision", "recall", "f1", "full_support", "exact_match")
ANSWER_METRICS = ("answer_presence", "answer_coverage", "all_answers_present")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def triple_set(triples: list) -> set[tuple[str, str, str]]:
    result = set()
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3 or not all(
            isinstance(value, str) for value in triple
        ):
            raise ValueError("Each edge must contain three strings: head, relation, tail")
        result.add(tuple(normalize(value) for value in triple))
    return result


def edge_scores(selected: set, gold: set) -> dict:
    matched = len(selected & gold)
    precision = matched / len(selected) if selected else float(not gold)
    recall = matched / len(gold) if gold else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * matched / (len(selected) + len(gold)) if selected or gold else 1.0,
        "full_support": int(gold <= selected),
        "exact_match": int(selected == gold),
        "matched_edges": matched,
        "selected_edges": len(selected),
        "gold_edges": len(gold),
    }


def answer_scores(selected: set, answer: object) -> dict:
    values = answer if isinstance(answer, list) else [answer]
    answers = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("Gold answers must be strings, numbers, or lists of these")
        normalized = normalize(str(value))
        if normalized:
            answers.add(normalized)
    entities = {value for head, relation, tail in selected for value in (head, tail)}
    present = answers & entities
    return {
        "answer_presence": int(bool(present)) if answers else None,
        "answer_coverage": len(present) / len(answers) if answers else None,
        "all_answers_present": int(answers <= entities) if answers else None,
        "gold_answers": sorted(answers),
        "present_answers": sorted(present),
        "missing_answers": sorted(answers - entities),
        "answer_triples": sorted(
            triple for triple in selected if triple[0] in present or triple[2] in present
        ),
    }


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected an object")
                records.append(record)
            except ValueError as error:
                raise ValueError(f"Invalid JSON record in {path.name}, line {line_number}") from error
    return records


def evaluate_annotations(annotation_path: Path, gold_path: Path) -> dict:
    gold_by_question = {}
    gold_records = read_records(gold_path)
    for gold_index, record in enumerate(gold_records):
        question = normalize(record.get("question", ""))
        if not question:
            raise ValueError(f"Ground-truth record {gold_index} has no question")
        gold_by_question.setdefault(question, []).append((
            gold_index, triple_set(record["edges"]), record.get("answer")
        ))

    results = []
    unmatched = []
    seen = set()
    matched_gold_indices = set()
    annotation_records = read_records(annotation_path)
    for annotation_index, record in enumerate(annotation_records):
        record_id = str(record.get("id", ""))
        question = normalize(record.get("input", ""))
        if question and question in seen:
            raise ValueError(f"Duplicate annotation question at record {annotation_index}")
        seen.add(question)
        matches = gold_by_question.get(question, [])
        if len(matches) > 1:
            raise ValueError(f"Ambiguous ground-truth question at annotation record {annotation_index}")
        if not matches:
            unmatched.append(record_id)
            continue
        gold_index, gold, answer = matches[0]
        matched_gold_indices.add(gold_index)
        outputs = record.get("output", [])
        output = outputs[0] if outputs else {}
        selected = triple_set([
            triple
            for group in output.get("summary_by_entity", [])
            for triple in group.get("triples", [])
        ])
        results.append({
            "id": record_id,
            "annotation_index": annotation_index,
            **edge_scores(selected, gold),
            **answer_scores(selected, answer),
            "missing_triples": sorted(gold - selected),
            "extra_triples": sorted(selected - gold),
        })

    count = len(results)
    macro = {
        metric: sum(record[metric] for record in results) / count if count else None
        for metric in METRICS
    }
    answer_records = [record for record in results if record["gold_answers"]]
    for metric in ANSWER_METRICS:
        macro[metric] = (
            sum(record[metric] for record in answer_records) / len(answer_records)
            if answer_records else None
        )
    matched = sum(record["matched_edges"] for record in results)
    selected_count = sum(record["selected_edges"] for record in results)
    gold_count = sum(record["gold_edges"] for record in results)
    micro_precision = matched / selected_count if selected_count else float(not gold_count)
    micro_recall = matched / gold_count if gold_count else 1.0
    return {
        "dataset": annotation_path.name,
        "gold_file": gold_path.name,
        "scope": "First annotator output; all records in the selected dataset, ignoring search filters",
        "matching": "Question text only (graph IDs ignored); directed triples, NFC and whitespace normalization, deduplicated across entities",
        "warning": "Reference-edge agreement, not factual correctness. Candidates include source gold edges; this is not independent retrieval evaluation.",
        "answer_evaluation": "Gold answer presence in complete heads/tails only, with NFC and whitespace normalization. No substring, relation, alias, or case-insensitive matching. Presence is not QA accuracy or proof of sufficient reasoning; computed answers need not appear literally.",
        "answer_evaluated_records": len(answer_records),
        "evaluated_records": count,
        "annotation_records": len(annotation_records),
        "gold_records": len(gold_records),
        "unmatched_annotation_ids": unmatched,
        "missing_annotation_indices": [
            index for index in range(len(gold_records))
            if index not in matched_gold_indices
        ],
        "macro": macro,
        "micro": {
            "precision": micro_precision if count else None,
            "recall": micro_recall if count else None,
            "f1": (2 * matched / (selected_count + gold_count) if selected_count + gold_count else 1.0) if count else None,
        },
        "mean_selected_edges": selected_count / count if count else None,
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_annotations(args.annotations, args.gold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()