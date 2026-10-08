"""Serve a read-only viewer for annotation-pool entity summaries."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


POOL_FOLDERS = ("annotation-pools", "annotation-pools-2", "annotation-pools-3")
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "predictions" / "synmes"


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


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
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
            self.send_json(404, {"error": "Not found"})

    return ViewerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")
    with ThreadingHTTPServer((args.host, args.port), make_handler(args.data_dir)) as server:
        print(f"Entity summary viewer: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()