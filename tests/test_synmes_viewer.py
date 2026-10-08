import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from kilt_synmes.viewer import dataset_paths, load_summaries, make_handler, project_record


def annotation_record():
    return {
        "id": "record-1",
        "input": "Hidden question",
        "meta": {"search_trace": "hidden"},
        "output": [
            {
                "answer": "Hidden combined answer",
                "provenance": ["hidden"],
                "meta": {"annotator": "annotator-1", "model": "ollama:qwen3", "search_trace": "hidden"},
                "summary_by_entity": [{
                    "topic_entity": "Entity <A>",
                    "answer": "Entity A lives in Paris.",
                    "triples": [["Entity A", "lives_in", "Paris"]],
                    "candidate_indices": [5],
                }],
            },
            {"meta": {"annotator": "annotator-2", "model": "hf:example/model"}, "summary_by_entity": []},
        ],
    }


class TestSynMESViewer(unittest.TestCase):
    def test_projection_only_contains_entity_summaries(self):
        projected = project_record(annotation_record())
        self.assertEqual(set(projected), {"id", "output"})
        self.assertEqual(len(projected["output"]), 2)
        self.assertEqual(projected["output"][0]["model"], "ollama:qwen3")
        self.assertEqual(projected["output"][1]["model"], "hf:example/model")
        group = projected["output"][0]["summary_by_entity"][0]
        self.assertEqual(set(group), {"topic_entity", "answer", "triples"})
        self.assertEqual(group["triples"], [["Entity A", "lives_in", "Paris"]])
        self.assertNotIn("hidden", json.dumps(projected).lower())

    def test_load_handles_blank_lines_empty_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "records.jsonl"
            source.write_text("\n" + json.dumps(annotation_record()) + "\n\n", encoding="utf-8")
            self.assertEqual(len(load_summaries(source)), 1)
            source.write_text("", encoding="utf-8")
            self.assertEqual(load_summaries(source), [])
            source.write_text("\nnot json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_summaries(source)

    def test_http_limits_access_to_requested_pools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("annotation-pools", "annotation-pools-2", "annotation-pools-3", "retrieval"):
                (root / folder).mkdir()
                (root / folder / "sample.jsonl").write_text(json.dumps(annotation_record()) + "\n", encoding="utf-8")
            self.assertEqual(len(dataset_paths(root)), 3)
            (root / "annotation-pools" / "outside.jsonl").symlink_to(root.parent / "outside.jsonl")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(url + "/") as response:
                    page = response.read()
                    self.assertIn(b"Entity summaries", page)
                    self.assertIn(b'id="model"', page)
                    self.assertNotIn(b"Prediction folder", page)
                    self.assertNotIn(b'id="folder"', page)
                with urlopen(url + "/api/datasets") as response:
                    self.assertEqual(len(json.load(response)), 3)
                with urlopen(url + "/workspace/dataset/api/datasets") as response:
                    self.assertEqual(len(json.load(response)), 3)
                with urlopen(url + "/api/records?dataset=annotation-pools/sample.jsonl") as response:
                    self.assertEqual(json.load(response), [project_record(annotation_record())])
                with urlopen(url + "/workspace/dataset/api/records?dataset=annotation-pools/sample.jsonl") as response:
                    self.assertEqual(json.load(response), [project_record(annotation_record())])
                for path in ("/api/records?dataset=../sample.jsonl", "/api/records?dataset=retrieval/sample.jsonl", "/setup.py"):
                    with self.assertRaises(HTTPError) as error:
                        urlopen(url + path)
                    self.assertEqual(error.exception.code, 404)
                (root / "annotation-pools" / "sample.jsonl").write_text("invalid", encoding="utf-8")
                with self.assertRaises(HTTPError) as error:
                    urlopen(url + "/api/records?dataset=annotation-pools/sample.jsonl")
                self.assertEqual(error.exception.code, 422)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()