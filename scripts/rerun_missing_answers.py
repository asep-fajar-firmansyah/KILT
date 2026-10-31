"""Find and rerun annotation records where not all gold answers are present."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kilt_synmes.evaluate import (  # noqa: E402
    evaluate_annotation,
    evaluate_record,
    index_gold_questions,
    read_records,
)
from kilt_synmes.viewer import rerun_record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument(
        "--model",
        help="Override the saved backend, e.g. ollama:qwen3, hf:org/model, or heuristic",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Only check and rerun this zero-based annotation record index",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List matching records without rerunning them"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Rerun without asking for confirmation"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print model and annotation-step progress"
    )
    args = parser.parse_args()
    if args.index is not None and args.index < 0:
        parser.error("--index must be nonnegative")

    records = read_records(args.annotations)
    gold_by_question = index_gold_questions(args.gold)
    targets = []
    for index, record in enumerate(records):
        if args.index is not None and index != args.index:
            continue
        result = evaluate_record(record, gold_by_question, index)
        if result and result["all_answers_present"] == 0:
            targets.append((index, result["id"], result["missing_answers"]))

    if not targets:
        print("No records with all_answers_present=0 were found.")
        return 0

    print(f"Found {len(targets)} record(s):")
    for index, record_id, missing_answers in targets:
        print(f"index={index} id={record_id} missing_answers={missing_answers}")

    if args.dry_run:
        return 0
    if not args.yes:
        try:
            confirmed = input("Rerun these records? [y/N] ").strip().lower() in {"y", "yes"}
        except EOFError:
            confirmed = False
        if not confirmed:
            print("No records rerun.")
            return 0

    failures = 0
    for index, record_id, _ in targets:
        try:
            progress_callback = None
            if args.verbose:
                print(
                    f"Starting rerun for index={index} id={record_id}...",
                    file=sys.stderr,
                    flush=True,
                )

                def progress_callback(event: dict) -> None:
                    if event.get("type") == "progress":
                        message = (
                            f"{event.get('annotator', 'annotator')} "
                            f"step {event.get('step')}/{event.get('total')} "
                            f"value={event.get('value')}"
                        )
                    else:
                        message = str(event.get("message", event))
                    print(message, file=sys.stderr, flush=True)

            version = rerun_record(
                args.annotations,
                args.retrieval,
                index,
                progress_callback=progress_callback,
                model_override=args.model,
                ollama_url=args.ollama_url,
            )
            result = evaluate_annotation(args.annotations, args.gold, index)["record"]
            print(
                f"Reran index={index} id={record_id}; version={version}; "
                f"all_answers_present={result['all_answers_present']}; "
                f"missing_answers={result['missing_answers']}"
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            failures += 1
            print(f"Failed index={index} id={record_id}: {error}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())