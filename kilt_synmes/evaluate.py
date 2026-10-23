"""Evaluate annotation evidence against M3GQA reference edges."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


DEFAULT_GOLD_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "M3GQA" / "test"
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


def read_record_at(path: Path, record_index: int) -> dict:
    if record_index < 0:
        raise ValueError("Annotation index must be nonnegative")
    with path.open(encoding="utf-8") as handle:
        current_index = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if current_index == record_index:
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Expected an object")
                    return record
                except ValueError as error:
                    raise ValueError(f"Invalid JSON record in {path.name}, line {line_number}") from error
            current_index += 1
    raise ValueError(f"Annotation index {record_index} is outside {path.name}")


def evaluate_annotation(annotation_path: Path, gold_path: Path, annotation_index: int) -> dict:
    gold_by_question = {}
    gold_records = read_records(gold_path)
    for gold_index, record in enumerate(gold_records):
        question = normalize(record.get("question", ""))
        if not question:
            raise ValueError(f"Ground-truth record {gold_index} has no question")
        gold_by_question.setdefault(question, []).append((
            gold_index, triple_set(record["edges"]), record.get("answer")
        ))

    record = read_record_at(annotation_path, annotation_index)
    record_id = str(record.get("id", ""))
    question = normalize(record.get("input", ""))
    matches = gold_by_question.get(question, [])
    if len(matches) > 1:
        raise ValueError(f"Ambiguous ground-truth question at annotation record {annotation_index}")

    result = None
    if matches:
        _, gold, answer = matches[0]
        outputs = record.get("output", [])
        output = outputs[0] if outputs else {}
        selected = triple_set([
            triple
            for group in output.get("summary_by_entity", [])
            for triple in group.get("triples", [])
        ])
        result = {
            "id": record_id,
            "annotation_index": annotation_index,
            **edge_scores(selected, gold),
            **answer_scores(selected, answer),
            "missing_triples": sorted(gold - selected),
            "extra_triples": sorted(selected - gold),
        }

    return {
        "dataset": annotation_path.name,
        "gold_file": gold_path.name,
        "annotation_index": annotation_index,
        "record": result,
        "scope": "First annotator output for the current record only",
        "matching": "Question text only (graph IDs ignored); directed triples, NFC and whitespace normalization, deduplicated across entities",
        "warning": "Reference-edge agreement, not factual correctness. Candidates include source gold edges; this is not independent retrieval evaluation.",
        "answer_evaluation": "Gold answer presence in complete heads/tails only, with NFC and whitespace normalization. No substring, relation, alias, or case-insensitive matching. Presence is not QA accuracy or proof of sufficient reasoning; computed answers need not appear literally.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int, help="Zero-based annotation record index")
    args = parser.parse_args()
    print(json.dumps(evaluate_annotation(args.annotations, args.gold, args.index), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()