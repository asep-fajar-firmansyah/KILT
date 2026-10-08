import json
import tempfile
import unittest
from pathlib import Path

from kilt_synmes.benchmark import build_benchmark, export_benchmark, render_tree


def annotation_record() -> dict:
    return {
        "id": "m3gqa-1",
        "input": "Which entity practises law?",
        "output": [
            {
                "answer": "...",
                "summary_by_entity": [
                    {
                        "topic_entity": "Entity A",
                        "answer": "...",
                        "candidate_indices": [0],
                        "triples": [["Entity A", "people.person.profession", "Lawyer"]],
                    },
                    {
                        "topic_entity": "Entity B",
                        "answer": "...",
                        "candidate_indices": [2],
                        "triples": [["Entity B", "business.industry", "Law"]],
                    },
                ],
                "meta": {"annotator": "annotator-1"},
            },
            {
                "answer": "...",
                "summary_by_entity": [
                    {
                        "topic_entity": "Entity A",
                        "answer": "...",
                        "candidate_indices": [1],
                        "triples": [["Entity A", "people.person.place_of_birth", "Boston"]],
                    }
                ],
                "meta": {"annotator": "annotator-2"},
            },
        ],
        "meta": {"topic_entities": ["Entity A", "Entity B"]},
    }


class TestSynMESBenchmark(unittest.TestCase):
    def test_nests_records_by_entity_then_triples(self):
        benchmark = build_benchmark([annotation_record()])

        self.assertEqual(list(benchmark), ["m3gqa-1"])
        self.assertEqual(list(benchmark["m3gqa-1"]), ["Entity A", "Entity B"])
        self.assertEqual(
            benchmark["m3gqa-1"]["Entity A"]["triples"],
            [["Entity A", "people.person.profession", "Lawyer"]],
        )

    def test_selects_named_annotator_and_optional_question(self):
        benchmark = build_benchmark(
            [annotation_record()], annotator="annotator-2", include_question=True
        )

        self.assertEqual(benchmark["m3gqa-1"]["question"], "Which entity practises law?")
        self.assertEqual(list(benchmark["m3gqa-1"]), ["question", "Entity A"])

        with self.assertRaises(KeyError):
            build_benchmark([annotation_record()], annotator="annotator-9")

    def test_export_writes_json_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "annotation.jsonl"
            source.write_text(json.dumps(annotation_record()) + "\n", encoding="utf-8")
            target = Path(directory) / "synthetic_benchmark.json"

            count = export_benchmark(source, target)

            self.assertEqual(count, 1)
            self.assertIn("Entity B", json.loads(target.read_text(encoding="utf-8"))["m3gqa-1"])

    def test_render_tree_shows_entity_and_triple_levels(self):
        tree = render_tree(build_benchmark([annotation_record()]))

        self.assertIn("m3gqa-1", tree)
        self.assertIn('"Entity A"', tree)
        self.assertIn("triples", tree)


if __name__ == "__main__":
    unittest.main()
