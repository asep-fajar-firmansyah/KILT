import unittest

from kilt_synmes.annotate import annotate_record, build_annotators, mean_pairwise_jaccard
from kilt_synmes.tot import (
    HeuristicScorer,
    ToTConfig,
    aggregate_votes,
    extract_json_objects,
    semantic_role,
)


def retrieval_record() -> dict:
    triples = [
        ["Entity A", "people.person.profession", "Lawyer"],
        ["Entity A", "common.topic.subject_of", "Article"],
        ["Entity B", "business.business_operation.industry", "Law"],
        ["Entity B", "location.location.containedby", "United States"],
        ["Entity A", "people.person.place_of_birth", "Boston"],
    ]
    owners = ["Entity A", "Entity A", "Entity B", "Entity B", "Entity A"]
    return {
        "id": "m3gqa-1",
        "question": "Which entity practises law in the United States?",
        "candidate_triples": triples,
        "provenance": [
            {
                "source_id": f"m3gqa-1-edge-{index}",
                "source_type": "structured_graph",
                "triple_index": index,
                "topic_entity": owners[index],
                "triple": triple,
            }
            for index, triple in enumerate(triples)
        ],
        "meta": {"graph_id": 1, "topic_entities": ["Entity A", "Entity B"]},
    }


class ScriptedBackend:
    """Returns a fixed index per task and constant evaluation scores."""

    def __init__(self, indices):
        self.indices = indices
        self.calls = 0

    def chat(self, messages, temperature=0.0, max_new_tokens=512, n=1):
        prompt = messages[0]["content"]
        self.calls += 1
        if prompt.startswith("EVALUATE"):
            count = prompt.count("SUMMARY ")
            entries = ", ".join(
                f'{{"idx": {i}, "relatedness": 0.8, "informativeness": 0.7, "coverage": 0.6}}'
                for i in range(count)
            )
            return [f"[{entries}]"] * n
        for task, index in self.indices.items():
            if task.upper() in prompt:
                return [str(index)] * n
        return ["1"] * n


class TestSynMESAnnotation(unittest.TestCase):
    def setUp(self):
        self.record = retrieval_record()
        self.config = ToTConfig(max_summary_len=3, n_candidates_per_task=2, n_evals=1)

    def test_heuristic_annotators_produce_one_summary_each(self):
        annotators = build_annotators(3, "heuristic", None, seed=42, ollama_url="http://localhost:11434")

        output = annotate_record(self.record, annotators, self.config)

        self.assertEqual(len(output["output"]), 3)
        self.assertEqual(output["meta"]["annotators"], ["annotator-1", "annotator-2", "annotator-3"])
        self.assertEqual(output["meta"]["budget_scope"], "per-entity")
        for summary in output["output"]:
            per_entity = summary["meta"]["triples_per_seed_entity"]
            self.assertTrue(all(count <= 3 for count in per_entity.values()))
            self.assertEqual(len(summary["provenance"]), len(summary["meta"]["selected_triples"]))
            self.assertTrue(summary["answer"].strip())

    def test_per_entity_budget_caps_each_entity(self):
        config = ToTConfig(max_summary_len=2, n_candidates_per_task=2, n_evals=1)
        annotators = build_annotators(1, "heuristic", None, seed=42, ollama_url="http://localhost:11434")

        summary = annotate_record(self.record, annotators, config)["output"][0]

        per_entity = summary["meta"]["triples_per_seed_entity"]
        self.assertEqual(sorted(per_entity.values()), [2, 2])
        self.assertEqual(len(summary["meta"]["selected_candidate_indices"]), 4)

    def test_total_budget_scope_caps_whole_summary(self):
        annotators = build_annotators(1, "heuristic", None, seed=42, ollama_url="http://localhost:11434")

        output = annotate_record(self.record, annotators, self.config, per_entity_budget=False)

        self.assertEqual(output["meta"]["budget_scope"], "total")
        self.assertEqual(len(output["output"][0]["meta"]["selected_candidate_indices"]), 3)

    def test_summary_covers_every_topic_entity(self):
        annotators = build_annotators(1, "heuristic", None, seed=7, ollama_url="http://localhost:11434")

        summary = annotate_record(self.record, annotators, self.config)["output"][0]

        self.assertTrue(all(count > 0 for count in summary["meta"]["entity_coverage"].values()))

    def test_llm_backend_thoughts_drive_selection(self):
        backend = ScriptedBackend({"relatedness": 1, "informativeness": 3, "coverage": 4})
        annotators = [{"name": "annotator-1", "model": "scripted", "seed": 0, "backend": backend}]

        summary = annotate_record(self.record, annotators, self.config)["output"][0]

        self.assertIn(summary["meta"]["selected_candidate_indices"][0], {0, 2, 3})
        self.assertGreater(backend.calls, 0)

    def test_unparsable_backend_output_falls_back_to_heuristics(self):
        class GarbageBackend:
            def chat(self, messages, temperature=0.0, max_new_tokens=512, n=1):
                return ["I cannot decide."] * n

        annotators = [{"name": "annotator-1", "model": "garbage", "seed": 0, "backend": GarbageBackend()}]

        summary = annotate_record(self.record, annotators, self.config, per_entity_budget=False)["output"][0]

        self.assertEqual(len(summary["meta"]["selected_candidate_indices"]), 3)

    def test_annotator_agreement_uses_mean_pairwise_jaccard(self):
        self.assertEqual(mean_pairwise_jaccard([[1, 2], [1, 2]]), 1.0)
        self.assertAlmostEqual(mean_pairwise_jaccard([[1, 2], [2, 3]]), 0.3333, places=4)

    def test_aggregate_votes_parses_fenced_json(self):
        raw = '```json\n[{"idx": 0, "relatedness": 1.0, "informativeness": 1.0, "coverage": 1.0}]\n```'

        scores = aggregate_votes(1, [raw], {"relatedness": 0.4, "informativeness": 0.4, "coverage": 0.2})

        self.assertEqual(scores, [1.0])
        self.assertEqual(extract_json_objects("no json here"), [])
        self.assertIsNone(aggregate_votes(1, ["no json here"], {"relatedness": 1.0, "informativeness": 0.0, "coverage": 0.0}))

    def test_heuristic_scorer_rewards_novel_roles_and_entities(self):
        triples = [tuple(triple) for triple in self.record["candidate_triples"]]
        scorer = HeuristicScorer(triples, self.record["question"], ["Entity A", "Entity B"])

        self.assertEqual(semantic_role("location.location.containedby"), "location")
        self.assertGreater(scorer.coverage(2, [0]), scorer.coverage(1, [0]))


if __name__ == "__main__":
    unittest.main()
