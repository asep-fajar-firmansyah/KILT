import json
import tempfile
import unittest
from pathlib import Path

from kilt_synmes.pipeline import (
    build_retrieval_record,
    load_graphs_from_directory,
    positive_int_or_unlimited,
    select_topk_by_entity,
)
from kilt_synmes.visualize import render_html


class TestSynMESPipeline(unittest.TestCase):
    def test_retrieval_selects_per_seed_evidence_and_deduplicates(self):
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
        output = build_retrieval_record(record, max_evidence=10, full_graph=full_graph)

        self.assertEqual(output["id"], "m3gqa-7")
        self.assertEqual(len(output["provenance"]), 4)
        self.assertEqual(output["provenance"][0]["source_type"], "structured_graph")
        self.assertEqual(output["meta"]["max_evidence_per_topic"], 10)
        self.assertTrue(output["meta"]["retrieved_from_full_graph"])

    def test_retrieval_uses_reasoning_path_candidates(self):
        record = {
            "graph_id": 8,
            "question": "How are the entities connected?",
            "topic_entities": ["Entity A"],
            "edges": [],
            "reasoning_path": [["Entity A", "connected_to", "Entity B"]],
        }

        output = build_retrieval_record(record, max_evidence=1)

        self.assertEqual(
            output["provenance"][0]["triple"],
            ["Entity A", "connected_to", "Entity B"],
        )
        self.assertEqual(output["meta"]["candidate_sources"]["reasoning_path"], 1)
        self.assertEqual(output["provenance"][0]["topic_entity"], "reasoning_path")

    def test_retrieval_preserves_all_source_edges_before_evidence_cap(self):
        record = {
            "graph_id": 15,
            "question": "Which Entity A fact matters?",
            "topic_entities": ["Entity A"],
            "edges": [
                ["Entity A", "related_to", "Entity B"],
                ["Entity B", "located_in", "Entity C"],
            ],
        }
        full_graph = [
            ("Entity A", "related_to", "Entity B"),
            ("Entity B", "located_in", "Entity C"),
            ("Entity A", "has_type", "Entity Type"),
        ]

        output = build_retrieval_record(record, max_evidence=1, full_graph=full_graph)

        self.assertEqual(
            output["candidate_triples"][:2],
            [["Entity A", "related_to", "Entity B"], ["Entity B", "located_in", "Entity C"]],
        )
        self.assertEqual(output["meta"]["source_evidence_count"], 2)

    def test_retrieval_budget_scales_with_topic_entity_count(self):
        record = {
            "graph_id": 16,
            "question": "Find topic facts.",
            "topic_entities": ["Entity A", "Entity B"],
            "edges": [],
        }
        full_graph = [
            ("Entity A", "related_to", "Entity C"),
            ("Entity A", "related_to", "Entity D"),
            ("Entity B", "related_to", "Entity E"),
        ]

        output = build_retrieval_record(record, max_evidence=1, full_graph=full_graph)

        self.assertEqual(output["meta"]["max_evidence_per_topic"], 1)
        self.assertEqual(output["meta"]["max_retrieved_evidence"], 2)
        self.assertEqual(output["meta"]["retrieved_evidence_count"], 2)

    def test_retrieval_expands_direct_edges_from_reasoning_path_entities(self):
        record = {
            "graph_id": 12,
            "question": "Which extension is relevant?",
            "topic_entities": ["Entity A"],
            "edges": [],
            "reasoning_path": [["Entity B", "leads_to", "Entity C"]],
        }
        full_graph = [
            ("Entity A", "related_to", "Entity X"),
            ("Entity B", "leads_to", "Entity C"),
            ("Entity C", "has_extension", "Entity D"),
        ]

        output = build_retrieval_record(record, max_evidence=3, full_graph=full_graph)

        self.assertIn(
            ["Entity C", "has_extension", "Entity D"],
            output["candidate_triples"],
        )

    def test_retrieval_allows_unlimited_budgets(self):
        record = {
            "graph_id": 13,
            "question": "Find Entity A facts.",
            "topic_entities": ["Entity A"],
            "edges": [],
        }
        full_graph = [
            ("Entity A", "related_to", "Entity B"),
            ("Entity A", "related_to", "Entity C"),
            ("Entity A", "related_to", "Entity D"),
        ]

        output = build_retrieval_record(record, max_evidence=None, full_graph=full_graph)

        self.assertEqual(len(output["candidate_triples"]), 3)
        self.assertIsNone(output["meta"]["max_evidence_per_topic"])
        self.assertIsNone(output["meta"]["max_retrieved_evidence"])

    def test_positive_int_or_unlimited(self):
        self.assertEqual(positive_int_or_unlimited("3"), 3)
        self.assertIsNone(positive_int_or_unlimited("unlimited"))

    def test_render_html_visualizes_candidate_triples(self):
        record = {
            "id": "m3gqa-14",
            "question": "Which entity is linked?",
            "candidate_triples": [["Entity A", "linked_to", "Entity B"]],
            "meta": {"topic_entities": ["Entity A"]},
        }

        document = render_html(record)

        self.assertIn("<svg", document)
        self.assertIn("Entity A", document)
        self.assertIn("linked_to", document)
        self.assertIn("pointerdown", document)

    def test_build_retrieval_record_stops_before_annotation(self):
        record = {
            "graph_id": 10,
            "question": "Which entity is connected?",
            "topic_entities": ["Entity A"],
            "edges": [["Entity A", "related_to", "Entity B"]],
        }

        output = build_retrieval_record(record, max_evidence=1)

        self.assertEqual(output["candidate_triples"], [["Entity A", "related_to", "Entity B"]])
        self.assertNotIn("output", output)
        self.assertNotIn("input", output)

    def test_retrieval_expands_question_relevant_path_between_seed_entities(self):
        record = {
            "graph_id": 11,
            "question": "Which bridge connects Entity A and Entity C?",
            "topic_entities": ["Entity A", "Entity C"],
            "edges": [],
        }
        full_graph = [
            ("Entity A", "bridge", "Entity B"),
            ("Entity B", "bridge", "Entity C"),
        ]

        output = build_retrieval_record(
            record,
            max_evidence=2,
            full_graph=full_graph,
            max_hops=2,
        )

        self.assertEqual(
            output["candidate_triples"],
            [["Entity A", "bridge", "Entity B"], ["Entity B", "bridge", "Entity C"]],
        )
        self.assertEqual(output["meta"]["max_hops"], 2)
        self.assertIn("bridge", output["meta"]["question_terms"])

    def test_load_graphs_from_directory_uses_graph_id_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "test" / "38.json"
            graph_path.parent.mkdir()
            graph_path.write_text(
                json.dumps(
                    {
                        "graph_id": 38,
                        "subgraph": [["Entity A", "related_to", "Entity B"]],
                    }
                ),
                encoding="utf-8",
            )

            graphs = load_graphs_from_directory(Path(temp_dir), "test", {"38"})

        self.assertEqual(graphs, {"38": [("Entity A", "related_to", "Entity B")]})

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
