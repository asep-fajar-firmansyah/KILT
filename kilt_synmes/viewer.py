"""Serve a read-only viewer for annotation-pool entity summaries."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from kilt_synmes.evaluate import DEFAULT_GOLD_ROOT, evaluate_annotation, read_record_at
from kilt_synmes.tot import ToTConfig


POOL_FOLDERS = ("annotation-pools", "annotation-pools-2", "annotation-pools-3")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "predictions" / "synmes"
WRITE_LOCK = threading.RLock()


def dataset_paths(root: Path) -> dict[str, Path]:
    root = root.resolve()
    return {
        f"{folder}/{path.name}": path
        for folder in POOL_FOLDERS
        for path in sorted((root / folder).glob("*.jsonl"))
        if path.is_file() and path.resolve().is_relative_to(root)
    }


def dataset_model(path: Path) -> str | None:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for output in record.get("output", []):
                model = output.get("meta", {}).get("model")
                if model:
                    return str(model)
            return None
    return None


def project_record(record: dict) -> dict:
    return {
        "id": str(record.get("id", "")),
        "output": [
            {
                "model": output.get("meta", {}).get("model")
                or output.get("meta", {}).get("annotator", f"annotator-{index + 1}"),
                "summary_by_entity": [
                    {
                        "topic_entity": group.get("topic_entity", ""),
                        "answer": group.get("answer", ""),
                        "triples": group.get("triples", []),
                    }
                    for group in output.get("summary_by_entity", [])
                ],
            }
            for index, output in enumerate(record.get("output", []))
        ],
    }


def load_summaries(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(project_record(json.loads(line)))
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid annotation record in {path.name}, line {line_number}") from error
    return records


def read_history(path: Path, annotation_index: int) -> list[dict]:
    if not path.is_file():
        return []
    versions = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("index") == annotation_index:
                    versions.append(entry)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid record history in {path.name}, line {line_number}") from error
    return versions


def history_path_for(annotation_path: Path) -> Path:
    return annotation_path.parent.parent / "versions" / annotation_path.parent.name / annotation_path.name


def project_versions(annotation_path: Path, annotation_index: int) -> list[dict]:
    history_path = history_path_for(annotation_path)
    archived = read_history(history_path, annotation_index)
    versions = [
        {
            "version": entry["version"],
            "created_at": entry["created_at"],
            "record": project_record(entry["record"]),
        }
        for entry in archived
    ]
    current = read_record_at(annotation_path, annotation_index)
    versions.append({
        "version": len(versions) + 1,
        "created_at": "Current",
        "record": project_record(current),
    })
    return versions


def write_records_atomically(path: Path, records: list[dict]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def rerun_record(annotation_path: Path, retrieval_path: Path, annotation_index: int) -> int:
    from kilt_synmes.annotate import annotate_record, build_annotators

    with WRITE_LOCK:
        return _rerun_record_locked(annotation_path, retrieval_path, annotation_index, annotate_record, build_annotators)


def _rerun_record_locked(annotation_path: Path, retrieval_path: Path, annotation_index: int, annotate_record, build_annotators) -> int:
    records = []
    with annotation_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if annotation_index >= len(records):
        raise ValueError(f"Annotation index {annotation_index} is outside {annotation_path.name}")
    current = records[annotation_index]
    retrieval = read_record_at(retrieval_path, annotation_index)
    if str(current.get("id", "")) != str(retrieval.get("id", "")):
        raise ValueError("Annotation and retrieval records do not match at the selected index")
    outputs = current.get("output", [])
    if not outputs:
        raise ValueError("The selected record has no annotator output to rerun")

    selected_output = outputs[0]
    output_meta = selected_output.get("meta", {})
    model = output_meta.get("model")
    if not model:
        raise ValueError("The selected annotator model is not recorded in this dataset")
    record_meta = current.get("meta", {})
    weights = record_meta.get("objective_weights")
    if not isinstance(weights, dict):
        weights = ToTConfig().objective_weights
    config = ToTConfig(
        max_summary_len=record_meta.get("max_summary_len", 5),
        n_candidates_per_task=record_meta.get("n_candidates_per_task", 2),
        n_evals=record_meta.get("n_evals", 3),
        breadth_limit=record_meta.get("breadth_limit", 3),
        thought_temperature=record_meta.get("thought_temperature", 0.8),
        eval_temperature=record_meta.get("eval_temperature", 0.3),
        objective_weights=weights,
    )
    annotator = build_annotators(
        1, str(model), None, output_meta.get("seed", 42), "http://localhost:11434"
    )
    generated = annotate_record(
        retrieval,
        annotator,
        config,
        progress=False,
        per_entity_budget=record_meta.get("budget_scope", "per-entity") != "total",
    )
    updated = copy.deepcopy(current)
    updated.setdefault("output", [])[0] = generated["output"][0]

    history_path = history_path_for(annotation_path)
    history = read_history(history_path, annotation_index)
    version = len(history) + 1
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "index": annotation_index,
            "record_id": str(current.get("id", "")),
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "record": current,
        }, ensure_ascii=False) + "\n")
    records[annotation_index] = updated
    write_records_atomically(annotation_path, records)
    return version + 1


def make_handler(root: Path, gold_root: Path = DEFAULT_GOLD_ROOT) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status: int, value: object) -> None:
            self.respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            if request.path == "/":
                self.respond(200, Path(__file__).with_name("viewer.html").read_bytes(), "text/html; charset=utf-8")
                return
            paths = dataset_paths(root)
            route = request.path.rstrip("/")
            if route.endswith("/api/datasets"):
                self.send_json(200, [
                    {
                        "key": key,
                        "folder": key.split("/")[0],
                        "name": path.name,
                        "model": dataset_model(path),
                    }
                    for key, path in paths.items()
                ])
                return
            if route.endswith("/api/records"):
                key = parse_qs(request.query).get("dataset", [""])[0]
                if key not in paths:
                    self.send_json(404, {"error": "Dataset not found"})
                    return
                try:
                    records = load_summaries(paths[key])
                except (OSError, ValueError) as error:
                    self.send_json(422, {"error": str(error)})
                    return
                self.send_json(200, records)
                return
            if route.endswith("/api/versions"):
                query = parse_qs(request.query)
                key = query.get("dataset", [""])[0]
                if key not in paths:
                    self.send_json(404, {"error": "Dataset not found"})
                    return
                try:
                    annotation_index = int(query.get("index", [""])[0])
                    if annotation_index < 0:
                        raise ValueError
                    versions = project_versions(paths[key], annotation_index)
                except ValueError as error:
                    self.send_json(400, {"error": "A valid nonnegative annotation index is required"})
                    return
                except (OSError, KeyError, TypeError, AttributeError) as error:
                    self.send_json(422, {"error": f"Unable to load record versions: {error}"})
                    return
                self.send_json(200, versions)
                return
            if route.endswith("/api/evaluate"):
                query = parse_qs(request.query)
                key = query.get("dataset", [""])[0]
                if key not in paths:
                    self.send_json(404, {"error": "Dataset not found"})
                    return
                try:
                    annotation_index = int(query.get("index", [""])[0])
                    if annotation_index < 0:
                        raise ValueError
                except ValueError:
                    self.send_json(400, {"error": "A nonnegative annotation index is required"})
                    return
                gold_path = gold_root / paths[key].name
                if not gold_path.is_file() or not gold_path.resolve().is_relative_to(gold_root.resolve()):
                    self.send_json(404, {"error": "Matching ground-truth file not found"})
                    return
                try:
                    report = evaluate_annotation(paths[key], gold_path, annotation_index)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    self.send_json(422, {"error": f"Unable to evaluate dataset: {error}"})
                    return
                self.send_json(200, report)
                return
            self.send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            request = urlsplit(self.path)
            if not request.path.rstrip("/").endswith("/api/rerun"):
                self.send_json(404, {"error": "Not found"})
                return
            query = parse_qs(request.query)
            key = query.get("dataset", [""])[0]
            if key not in dataset_paths(root):
                self.send_json(404, {"error": "Dataset not found"})
                return
            paths = dataset_paths(root)
            annotation_path = paths[key]
            retrieval_path = root.resolve() / "retrieval" / annotation_path.name
            if not retrieval_path.is_file() or not retrieval_path.resolve().is_relative_to(root.resolve()):
                self.send_json(404, {"error": "Matching retrieval file not found"})
                return
            try:
                annotation_index = int(query.get("index", [""])[0])
                if annotation_index < 0:
                    raise ValueError
            except ValueError:
                self.send_json(400, {"error": "A nonnegative annotation index is required"})
                return
            try:
                latest_version = rerun_record(annotation_path, retrieval_path, annotation_index)
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                self.send_json(422, {"error": f"Unable to rerun selected record: {error}"})
                return
            self.send_json(200, {"version": latest_version})

    return ViewerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    with ThreadingHTTPServer((args.host, args.port), make_handler(args.data_dir, args.gold_dir)) as server:
        print(f"Entity summary viewer: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()