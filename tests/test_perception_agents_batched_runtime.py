import copy
import itertools
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from perception_agents_batched_runtime import BatchedAgentRuntime, _attach_metrics, encode_batch, run_agents_batched
from perception_agents_core import run_agents


GOOD = [100, 100, 300, 300]
BAD = [300, 100, 100, 300]


class FakeRoles:
    def __init__(self, scripts):
        self.scripts = copy.deepcopy(scripts)
        self.positions = {path: {role: 0 for role in ("what", "where", "verify")} for path in scripts}
        self.calls = []

    def call(self, role, requests):
        self.calls.append((role, copy.deepcopy(requests)))
        results = []
        for request in requests:
            path = request["image_path"]
            index = self.positions[path][role]
            self.positions[path][role] += 1
            results.append(copy.deepcopy(self.scripts[path][role][index]))
        return results

    def single(self, path, mode, budget):
        return run_agents(
            lambda feedback: self.call("what", [{"image_path": path, "feedback": feedback}])[0],
            lambda category, feedback: self.call("where", [{"image_path": path, "category": category, "feedback": feedback}])[0],
            lambda candidate: self.call("verify", [{"image_path": path, "candidate": candidate}])[0],
            max_revisions=budget, mode=mode)

    def batch(self, mode, budget):
        return run_agents_batched(list(self.scripts),
                                  lambda requests: self.call("what", requests),
                                  lambda requests: self.call("where", requests),
                                  lambda requests: self.call("verify", requests),
                                  max_revisions=budget, mode=mode)


def script(categories=("fire", "flood", "fire"), verdicts=("B", "C", "A"), boxes=None):
    return {"what": [{"category": category, "score": .2 + .2 * index, "raw": str(category)}
                     for index, category in enumerate(categories)],
            "where": boxes or [GOOD, [400, 400, 600, 600], [600, 600, 800, 800]],
            "verify": [{"verdict": verdict, "score": .8 - .1 * index, "raw": verdict}
                       for index, verdict in enumerate(verdicts)]}


class BatchedControllerTests(unittest.TestCase):
    def assert_matches_serial(self, scripts, mode="full", budget=2):
        batch = FakeRoles(scripts)
        actual = batch.batch(mode, budget)
        serial = FakeRoles(scripts)
        expected = [serial.single(path, mode, budget) for path in scripts]
        self.assertEqual(actual, expected)
        # Request contents, including previous hypotheses and feedback, also match.
        for path in scripts:
            actual_calls = [(role, row) for role, rows in batch.calls for row in rows if row["image_path"] == path]
            expected_calls = [(role, row) for role, rows in serial.calls for row in rows if row["image_path"] == path]
            self.assertEqual(actual_calls, expected_calls)
        return batch, actual

    def test_mixed_agents_are_real_batches_and_keep_all_traces(self):
        scripts = {
            "accept.png": script(verdicts=("A", "D", "D")),
            "relocalize.png": script(verdicts=("B", "A", "D")),
            "reclassify.png": script(verdicts=("C", "A", "D")),
            "false_negative.png": script(categories=(None, "flood", "fire"), verdicts=("C", "A", "D")),
            "negative.png": script(categories=(None, None, None), verdicts=("D", "D", "D")),
            "exhausted.png": script(verdicts=("B", "B", "B")),
            "bad_accept.png": script(verdicts=("A", "D", "D"), boxes=[BAD] * 3),
            "empty_relocalize.png": script(categories=(None, None, None), verdicts=("B", "B", "B")),
        }
        batch, results = self.assert_matches_serial(scripts)
        self.assertEqual(len(batch.calls[0][1]), len(scripts))
        self.assertTrue(any(role == "where" and len(requests) > 1 for role, requests in batch.calls))
        self.assertTrue(any(role == "verify" and len(requests) > 1 for role, requests in batch.calls))
        self.assertEqual(results[3]["category"], "flood")
        self.assertTrue(results[5]["abstained"])
        self.assertEqual(results[5]["trace"]["revisions"], 2)

    def test_all_verdict_paths_modes_and_revision_budgets_match_core(self):
        # Covers all 125 three-verdict trajectories, with absent/present starts.
        scripts = {}
        for index, verdicts in enumerate(itertools.product(("A", "B", "C", "D", "UNKNOWN"), repeat=3)):
            scripts[str(index)] = script(categories=(None if index % 2 else "fire", "flood", None), verdicts=verdicts,
                                         boxes=[BAD if index % 3 == 0 else GOOD, GOOD, GOOD])
        for mode, budget in itertools.product(("full", "no_verify", "verify_only"), (0, 1, 2)):
            with self.subTest(mode=mode, budget=budget):
                self.assert_matches_serial(scripts, mode, budget)

    def test_full_verdict_names_and_empty_categories(self):
        self.assert_matches_serial({
            "one": script(categories=("", "fire", "fire"), verdicts=("reclassify", "relocalize", "accept")),
            "two": script(categories=("no_event", None, None), verdicts=("no_event", "accept", "accept")),
            "three": script(categories=(None, None, None), verdicts=("accept", "accept", "accept")),
        })

    def test_output_order_is_input_order_despite_different_stop_times(self):
        scripts = {str(i): script(verdicts=("B", "C", "A") if i % 2 else ("D", "D", "D")) for i in range(11)}
        shuffled = list(scripts)
        random.Random(43).shuffle(shuffled)
        self.assert_matches_serial({path: scripts[path] for path in shuffled})

    def test_empty_and_invalid_requests(self):
        fail = lambda _: self.fail("empty batch invoked callback")
        self.assertEqual(run_agents_batched([], fail, fail, fail), [])
        for settings in ({"mode": "bad"}, {"max_revisions": -1}):
            with self.assertRaises(ValueError):
                run_agents_batched([], fail, fail, fail, **settings)
        with self.assertRaisesRegex(ValueError, "what returned"):
            run_agents_batched(["a"], lambda _: [], fail, fail)

    def test_metrics_keep_expected_schema_and_truth_free_requests(self):
        _, results = self.assert_matches_serial({"image.png": script()})
        result = _attach_metrics(results[0], 1200, 4)
        self.assertEqual(result["agent_calls"], {"what": 2, "where": 3, "verify": 3})
        self.assertEqual(result["num_calls"], 8)
        self.assertEqual(result["revisions"], 2)
        self.assertEqual(result["inference_batch_size"], 4)
        self.assertAlmostEqual(result["verify_score"], .6)
        self.assertEqual(json.loads(result["raw_output"]), result["trace"])
        self.assertIn("amortized", result["timing_scope"])

    def test_callback_mutation_does_not_leak_between_hypotheses(self):
        scripts = {"one": script(), "two": script()}
        expected = FakeRoles(scripts).batch("full", 2)
        fake = FakeRoles(scripts)
        def callback(role, requests):
            values = fake.call(role, requests)
            for row in requests:
                row.clear()
            return values
        actual = run_agents_batched(list(scripts), lambda rows: callback("what", rows),
                                   lambda rows: callback("where", rows), lambda rows: callback("verify", rows))
        self.assertEqual(actual, expected)


class EncodingTests(unittest.TestCase):
    def test_where_integer_logit_limit_is_not_moved_as_a_tensor(self):
        class FakeTensor:
            def to(self, device):
                self.device = device
                return self
        class FakeTokenizer:
            padding_side = 'right'
        class FakeProcessor:
            tokenizer = FakeTokenizer()
            def apply_chat_template(self, messages, **kwargs):
                return {'input_ids': FakeTensor()}
        runtime = BatchedAgentRuntime.__new__(BatchedAgentRuntime)
        runtime.processor = FakeProcessor()
        runtime.config = {'input': {'min_pixels': 65536, 'max_pixels': 995328}, 'agents': {'marker_width': 4}}
        runtime.category_text = 'fire, flood'
        runtime.device = 'cpu'
        inputs = runtime._inputs([{'image_path': 'image.png', 'category': 'fire'}], 'where')
        self.assertEqual(inputs['logits_to_keep'], 1)
        self.assertEqual(inputs['input_ids'].device, 'cpu')

    def test_where_keeps_category_and_feedback_per_row_and_left_padding(self):
        class FakeTokenizer:
            padding_side = "right"
        class FakeProcessor:
            tokenizer = FakeTokenizer()
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return {"input_ids": "fake", "image_grid_thw": "one image per row"}
        processor = FakeProcessor()
        config = {"input": {"min_pixels": 65536, "max_pixels": 995328}, "agents": {"marker_width": 4}}
        requests = [{"image_path": "one.png", "category": "fire", "feedback": None},
                    {"image_path": "two.png", "category": "flood", "feedback": {
                        "verdict": "reclassify", "candidate": {"category": "fire", "bbox_1000": GOOD,
                                                                   "gt_label": "SECRET"}}}]
        encoded = encode_batch(processor, config, requests, "where", "fire, flood")
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertFalse(processor.kwargs["add_generation_prompt"])
        self.assertTrue(processor.kwargs["processor_kwargs"]["padding"])
        self.assertEqual(encoded["logits_to_keep"], 1)
        self.assertEqual([messages[-1]["content"] for messages in processor.messages], ["fire<vis>", "flood<vis>"])
        self.assertEqual([messages[1]["content"][0]["image"] for messages in processor.messages], ["one.png", "two.png"])
        self.assertNotIn("SECRET", json.dumps(processor.messages))


if __name__ == "__main__":
    unittest.main()
