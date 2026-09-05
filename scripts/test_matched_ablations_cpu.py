#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
import unittest
from run_matched_ablations import (check_budget, check_epoch, check_result,
                                  fingerprint, stage_names, state)


class MatchedTests(unittest.TestCase):
    def setUp(self):
        self.config = {"experiment": "test", "data": {"protocols": ["session_disjoint"]},
                       "train": {"epochs": 12, "learning_rate": 1e-4,
                                 "vision_learning_rate": 1e-6, "seeds": [43]}}

    def payload(self):
        return dict(experiment="test", protocol="session_disjoint", seed=43,
                    checkpoint_epoch=12, rows=[{"record_uid": "a"}],
                    metrics={"table4": dict(ap50=.1, c_f1=.2, g_map50=.1,
                                           n_fpr=.1, p_recall=.8)})

    def test_budget(self):
        check_budget(self.config)
        self.config["train"]["epochs"] = 8
        with self.assertRaises(ValueError):
            check_budget(self.config)

    def test_epoch_one_is_not_complete(self):
        with self.assertRaises(ValueError):
            check_epoch(1)
        check_epoch(12)

    def test_stage_dependencies(self):
        self.assertEqual(stage_names(["no_roi"]), ["full_ms", "full_cls", "no_roi"])
        self.assertEqual(stage_names(["single_scale"]), ["full_ms", "full_cls",
                                                         "single_scale_ms", "single_scale_cls"])

    def test_uid_and_metrics(self):
        payload = self.payload()
        check_result(payload, self.config, {"a"})
        payload["rows"].append({"record_uid": "a"})
        with self.assertRaises(ValueError):
            check_result(payload, self.config, {"a"})
        payload = self.payload()
        del payload["metrics"]["table4"]["p_recall"]
        with self.assertRaises(ValueError):
            check_result(payload, self.config, {"a"})

    def test_states_and_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ckpt, result, receipt = (root / "run/last.pt", root / "result.json",
                                     root / "run/matched_run_config.json")
            self.assertEqual(state(self.config, ckpt, result, receipt, {"a"}), "new")
            ckpt.parent.mkdir()
            ckpt.touch()
            with self.assertRaises(ValueError):
                state(self.config, ckpt, result, receipt, {"a"}, lambda _: 1)
            with self.assertRaises(ValueError):
                state(self.config, ckpt, result, receipt, {"a"}, lambda _: 12)
            receipt.write_text(json.dumps({"config_sha256": fingerprint(self.config)}))
            self.assertEqual(state(self.config, ckpt, result, receipt, {"a"},
                                   lambda _: 12), "trained")
            result.write_text(json.dumps(self.payload()))
            self.assertEqual(state(self.config, ckpt, result, receipt, {"a"},
                                   lambda _: 12), "done")
            self.config["input"] = {"max_pixels": 640}
            with self.assertRaises(ValueError):
                state(self.config, ckpt, result, receipt, {"a"}, lambda _: 12)

    def test_receipt_only_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipt = root / "matched_run_config.json"
            receipt.write_text(json.dumps({"config_sha256": fingerprint(self.config)}))
            self.assertEqual(state(self.config, root / "last.pt", None, receipt, {"a"}), "new")
            receipt.write_text(json.dumps({"config_sha256": "different"}))
            with self.assertRaises(ValueError):
                state(self.config, root / "last.pt", None, receipt, {"a"})

    def test_interrupted_before_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tensorboard").mkdir()
            with self.assertRaises(ValueError):
                state(self.config, root / "last.pt", None, root / "receipt.json", {"a"})


if __name__ == "__main__":
    unittest.main()
