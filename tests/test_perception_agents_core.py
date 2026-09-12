import copy
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from perception_agents_core import bbox_iou, make_candidates, make_messages, oracle_verdict, run_agents, valid_bbox


class CoreTests(unittest.TestCase):
    def test_geometry_and_oracle(self):
        good = [100, 100, 300, 300]
        self.assertTrue(valid_bbox(good))
        for bad in ([300, 100, 100, 300], [0, 0, 0, 10], [0, 0, 1001, 10], [0, 0, float("nan"), 10]):
            self.assertFalse(valid_bbox(bad))
        self.assertEqual(bbox_iou(good, good), 1)
        self.assertAlmostEqual(bbox_iou([0, 0, 200, 200], [100, 0, 300, 200]), 1 / 3)
        self.assertEqual(oracle_verdict(False, None, None, "fire", good), "no_event")
        self.assertEqual(oracle_verdict(True, "fire", good, None, None), "reclassify")
        self.assertEqual(oracle_verdict(True, "fire", good, "flood", good), "reclassify")
        self.assertEqual(oracle_verdict(True, "fire", good, "fire", None), "relocalize")
        self.assertEqual(oracle_verdict(True, "fire", good, "fire", good), "accept")

    def test_seeded_candidates_cover_targets_without_mutation(self):
        for gt in ([100, 100, 300, 300], [0, 0, 1000, 1000], [999, 999, 1000, 1000]):
            original = copy.deepcopy(gt)
            candidates = make_candidates(True, "fire", gt, ["fire", "flood"], random.Random(43))
            self.assertEqual(candidates, make_candidates(True, "fire", gt, ["fire", "flood"], random.Random(43)))
            self.assertEqual(gt, original)
            verdicts = [oracle_verdict(True, "fire", gt, item["category"], item["bbox_1000"]) for item in candidates]
            self.assertEqual({key: verdicts.count(key) for key in set(verdicts)}, {"accept": 4, "relocalize": 4, "reclassify": 4})
            self.assertTrue(any(item["category"] == "flood" for item in candidates))
            self.assertTrue(any(item["category"] is None for item in candidates))
            self.assertTrue(any(item["bbox_1000"] == [0, 0, 1000, 1000] for item in candidates))
            self.assertTrue(all(item["bbox_1000"] is None or valid_bbox(item["bbox_1000"]) for item in candidates))
        negative = make_candidates(False, None, None, ["fire", "flood"], random.Random(43))
        self.assertTrue(all(oracle_verdict(False, None, None, **{"candidate_category": c["category"], "candidate_box": c["bbox_1000"]}) == "no_event" for c in negative))

    def test_messages_have_one_image_and_no_annotation_feedback(self):
        image = object()
        feedback = {"candidate": {"category": "fire", "bbox_1000": [10, 20, 30, 40], "gt_label": "SECRET_A"}, "verdict": "relocalize", "gt_bbox": "SECRET_B"}
        messages = make_messages(image, "where", "fire: fire", {"category": "fire", "gt_label": "SECRET_C"}, feedback)
        self.assertIs(messages[1]["content"][0]["image"], image)
        text = messages[1]["content"][1]["text"]
        self.assertNotIn("SECRET", text)
        self.assertNotIn("gt_", text)
        self.assertEqual(len(messages[1]["content"]), 2)
        self.assertEqual(make_messages(image, "verify", "fire", answer="A")[-1], {"role": "assistant", "content": "A"})

    def test_false_negative_can_recover(self):
        what_feedback = []
        proposals = iter([{"category": None, "score": 0.1}, {"category": "fire", "score": 0.8}])
        checks = iter([{"verdict": "C", "score": 0.9}, {"verdict": "A", "score": 0.75}])
        def what(feedback):
            what_feedback.append(feedback)
            return next(proposals)
        result = run_agents(what, lambda category, feedback: [100, 100, 200, 200], lambda candidate: next(checks))
        self.assertEqual(result["category"], "fire")
        self.assertAlmostEqual(result["presence_score"], 0.6)
        self.assertTrue(result["verified"])
        self.assertEqual(result["trace"]["revisions"], 1)
        self.assertEqual([call["role"] for call in result["trace"]["calls"]], ["what", "verify", "what", "where", "verify"])
        self.assertIsNone(what_feedback[0])
        self.assertEqual(what_feedback[1]["verdict"], "reclassify")
        self.assertNotIn("gt", json.dumps(result))

    def test_relocalization_reruns_where_only(self):
        boxes = iter([[100, 100, 200, 200], [600, 600, 800, 800]])
        checks = iter([{"verdict": "relocalize", "score": 0.9}, {"verdict": "accept", "score": 1}])
        feedbacks = []
        def where(category, feedback):
            feedbacks.append(feedback)
            return next(boxes)
        result = run_agents(lambda feedback: {"category": "fire", "score": 0.8}, where, lambda candidate: next(checks))
        self.assertEqual(result["bbox_1000"], [600, 600, 800, 800])
        self.assertEqual(sum(c["role"] == "what" for c in result["trace"]["calls"]), 1)
        self.assertEqual(feedbacks[1]["candidate"]["bbox_1000"], [100, 100, 200, 200])
        self.assertEqual(result["trace"]["initial_bbox_1000"], [100, 100, 200, 200])

    def test_category_feedback_recomputes_box(self):
        proposals = iter([{"category": "fire", "score": 0.9}, {"category": "flood", "score": 0.7}])
        checks = iter([{"verdict": "C", "score": 0.9}, {"verdict": "A", "score": 1}])
        categories = []
        def where(category, feedback):
            categories.append(category)
            return [100, 100, 200, 200]
        result = run_agents(lambda feedback: next(proposals), where, lambda candidate: next(checks))
        self.assertEqual(categories, ["fire", "flood"])
        self.assertEqual(result["category"], "flood")

    def test_budget_exhaustion_abstains(self):
        result = run_agents(lambda feedback: {"category": "fire", "score": 0.9}, lambda category, feedback: [0, 0, 100, 100], lambda candidate: {"verdict": "B", "score": 0.9}, max_revisions=2)
        self.assertTrue(result["valid"])
        self.assertTrue(result["abstained"])
        self.assertFalse(result["verified"])
        self.assertIsNone(result["category"])
        self.assertEqual(result["trace"]["revisions"], 2)
        self.assertEqual(len(result["trace"]["rounds"]), 3)

    def test_invalid_box_and_verdict_never_accepted(self):
        for box, verdict in (([30, 0, 10, 20], "A"), ([0, 0, 10, 20], "UNKNOWN")):
            result = run_agents(lambda feedback: {"category": "fire", "score": 1}, lambda category, feedback: box, lambda candidate: {"verdict": verdict, "score": 1})
            self.assertTrue(result["abstained"])
            self.assertFalse(result["verified"])

    def test_verifier_can_reject_false_positive(self):
        result = run_agents(lambda feedback: {"category": "fire", "score": 1}, lambda category, feedback: [0, 0, 10, 20], lambda candidate: {"verdict": "D", "score": 0.9})
        self.assertIsNone(result["category"])
        self.assertTrue(result["verified"])
        self.assertFalse(result["abstained"])
        self.assertEqual(result["presence_score"], 0)

    def test_ablation_modes(self):
        what = lambda feedback: {"category": "fire", "score": 0.8}
        where = lambda category, feedback: [0, 0, 10, 20]
        def must_not_verify(candidate):
            self.fail("no_verify invoked verifier")
        result = run_agents(what, where, must_not_verify, mode="no_verify")
        self.assertEqual(result["presence_score"], 0.8)
        self.assertFalse(result["verified"])
        result = run_agents(what, where, lambda candidate: {"verdict": "B", "score": 1}, mode="verify_only")
        self.assertEqual(result["trace"]["revisions"], 0)
        self.assertTrue(result["abstained"])


if __name__ == "__main__":
    unittest.main()
