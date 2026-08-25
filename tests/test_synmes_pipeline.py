import sys
import unittest

from kilt_synmes.pipeline import build_record, select_topk_by_entity


class TestSynMESPipeline(unittest.TestCase):
    def test_build_record_selects_top_k_per_entity_and_deduplicates(self):
        record = {
            "graph_id": 7,
            "question": "Which city connects the entities?",
            "answer": "City X",
            "answer_entities": ["City X"],
            "topic_entities": ["Entity A", "Entity B"],
            "edges": [
                ["Entity A", "related_to", "City X"],
                ["Entity B", "related_to", "City X"],
                ["Entity A", "has_type", "Person"],
            ],
        }

        full_graph = [
            ("Entity A", "related_to", "City X"),
            ("Entity A", "has_type", "Person"),
            ("Entity B", "related_to", "City X"),
            ("Entity B", "located_in", "City Y"),
        ]
        output = build_record(record, top_k=2, max_evidence=10, full_graph=full_graph)

        self.assertEqual(output["id"], "m3gqa-7")
        self.assertEqual(len(output["output"]), 1)
        self.assertEqual(len(output["output"][0]["provenance"]), 4)
        self.assertEqual(output["output"][0]["provenance"][0]["source_type"], "structured_graph")
        self.assertEqual(output["meta"]["top_k_per_entity"], 2)
        self.assertTrue(output["meta"]["retrieved_from_full_graph"])

    def test_build_record_uses_reasoning_path_candidates(self):
        record = {
            "graph_id": 8,
            "question": "How are the entities connected?",
            "topic_entities": ["Entity A"],
            "edges": [],
            "reasoning_path": [["Entity A", "connected_to", "Entity B"]],
        }

        output = build_record(record, top_k=1, max_evidence=1)

        self.assertEqual(
            output["output"][0]["provenance"][0]["triple"],
            ["Entity A", "connected_to", "Entity B"],
        )
        self.assertEqual(output["meta"]["candidate_sources"]["reasoning_path"], 1)

    def test_build_record_runs_each_annotator(self):
        record = {
            "graph_id": 9,
            "question": "Summarize the relation.",
            "topic_entities": ["Entity A"],
            "edges": [["Entity A", "related_to", "Entity B"]],
        }
        command = f'{sys.executable} -c "import sys; sys.stdin.read(); print(\'Annotated summary.\')"'

        output = build_record(
            record,
            top_k=1,
            max_evidence=1,
            annotator_commands=[command, command],
        )

        self.assertEqual(len(output["output"]), 2)
        self.assertEqual([item["answer"] for item in output["output"]], ["Annotated summary."] * 2)
        self.assertEqual(len(output["output"][0]["provenance"]), 1)

    def test_select_topk_preserves_topic_entity_occurrences(self):
        selected = select_topk_by_entity(
            [("Entity A", "related_to", "Entity B")],
            ["Entity A", "Entity A"],
            set(),
            top_k=1,
        )

        self.assertEqual([topic for topic, _ in selected], ["Entity A", "Entity A"])


if __name__ == "__main__":
    unittest.main()
