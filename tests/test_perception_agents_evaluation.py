"""Calibration and trace invariants without Torch, model downloads or datasets."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "test_perception_agents.py"
SPEC = importlib.util.spec_from_file_location("perception_agents_evaluation_under_test", SCRIPT)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "output": "saved", "device": "cuda", "protocol": "session_disjoint", "seed": 43,
            "model": {"path": "model"}, "spatial": {"num_heads": 8},
            "input": {"max_pixels": 995328}, "prompt": {"system": "trained prompt"},
            "agents": {"max_revisions": 2}, "data": {"labels": "labels.txt"},
            "validation": {"max_n_fpr": 0.1},
            "test": {"batch_size": 1, "max_new_tokens": 64},
            "train": {"learning_rate": 0.00002},
        }
        self.labels = ["event"]
        self.definitions = {"event": "training ontology definition"}
        self.source_hashes = {"core": "original", "runtime": "original"}

    def signature(self, config=None, mode="full", labels=None, definitions=None, sources=None):
        return evaluation.inference_signature(
            self.config if config is None else config, mode,
            self.labels if labels is None else labels,
            self.definitions if definitions is None else definitions,
            self.source_hashes if sources is None else sources,
        )

    def test_saved_config_owns_inference_and_training_limits_do_not_leak(self):
        saved = {**self.config, "max_train_samples": 4, "max_val_samples": 4}
        requested = {
            "output": "new_output", "device": "cpu", "test": {"batch_size": 2, "max_new_tokens": 999},
            "agents": {"max_revisions": 0}, "prompt": {"system": "untrained prompt"},
            "input": {"max_pixels": 100},
        }
        result = evaluation.evaluation_configuration(saved, requested)
        for key in ("agents", "prompt", "input", "train"):
            self.assertEqual(result[key], saved[key])
        self.assertEqual(result["test"], {"batch_size": 2, "max_new_tokens": 64})
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["output"], "new_output")
        self.assertNotIn("max_val_samples", result)
        self.assertEqual(result["training_sample_limits"]["max_val_samples"], 4)
        self.assertEqual(saved["test"]["batch_size"], 1)
        requested["max_val_samples"] = 2
        self.assertEqual(evaluation.evaluation_configuration(saved, requested)["max_val_samples"], 2)

    def test_signature_changes_with_inference_semantics(self):
        expected = self.signature()
        changes = {
            "agents": {"max_revisions": 1}, "prompt": {"system": "changed"},
            "input": {"max_pixels": 200}, "test": {"max_new_tokens": 32},
            "model": {"path": "other"}, "spatial": {"num_heads": 4},
            "data": {"labels": "other.txt"}, "validation": {"max_n_fpr": 0.2},
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[key] = value
                self.assertNotEqual(self.signature(config), expected)
        for kwargs in (
            {"mode": "no_verify"}, {"mode": "verify_only"},
            {"labels": ["other"]}, {"definitions": {"event": "changed"}},
            {"sources": {"core": "changed", "runtime": "original"}},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertNotEqual(self.signature(**kwargs), expected)

    def test_signature_allows_device_batch_size_and_split_limits(self):
        config = copy.deepcopy(self.config)
        config["device"] = "cpu"
        config["test"]["batch_size"] = 4
        config["max_val_samples"] = 4
        config["max_test_samples"] = 4
        self.assertEqual(self.signature(config), self.signature())

    def calibration(self, checkpoint):
        return {
            "schema_version": 1, "split": "val", "mode": "full",
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": "sha",
            "inference_signature": self.signature(), "limited_run": False, "threshold": 0.5,
        }

    def test_only_matching_validation_calibration_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            calibration = self.calibration(checkpoint)
            validate = lambda item: evaluation.validate_calibration(
                item, self.config, checkpoint, "sha", self.signature(), "full")
            self.assertEqual(validate(calibration), 0.5)
            for key, value in (
                ("schema_version", 0), ("split", "test"), ("mode", "no_verify"),
                ("checkpoint", str(checkpoint / "other")),
                ("checkpoint_sha256", "changed adapter or spatial weights"),
                ("inference_signature", "changed prompts"),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate({**calibration, key: value})

    def test_tiny_validation_cannot_calibrate_full_test(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            calibration = {**self.calibration(checkpoint), "limited_run": True}
            with self.assertRaises(ValueError):
                evaluation.validate_calibration(calibration, self.config, checkpoint,
                                                "sha", self.signature(), "full")
            config = {**self.config, "max_test_samples": 4}
            self.assertEqual(evaluation.validate_calibration(
                calibration, config, checkpoint, "sha", self.signature(), "full"), 0.5)

    def test_threshold_rejects_invalid_values_and_accepts_reject_all_sentinel(self):
        import math
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            calibration = self.calibration(checkpoint)
            for threshold in (True, False, None, "0.5", -0.1, 1.01, float("nan"), float("inf")):
                with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                    evaluation.validate_calibration({**calibration, "threshold": threshold},
                                                    self.config, checkpoint, "sha", self.signature(), "full")
            sentinel = math.nextafter(1.0, math.inf)
            self.assertEqual(evaluation.validate_calibration(
                {**calibration, "threshold": sentinel}, self.config, checkpoint, "sha", self.signature(), "full"), sentinel)

    def test_trace_counts_support_controller_and_runtime_outputs(self):
        predictions = [
            {"trace": {"calls": [{"role": "what"}, {"role": "where"}, {"role": "verify"}],
                       "revisions": 1, "abstained": True}},
            {"agent_calls": {"what": 2, "where": 1, "verify": 2},
             "revisions": 0, "abstained": False, "verified": True},
        ]
        metrics = evaluation.agent_metrics(predictions)
        self.assertEqual(metrics["agent_mean_calls"], {"what": 1.5, "where": 1.0, "verify": 1.5})
        self.assertEqual(metrics["agent_call_rates"], {"what": 1.0, "where": 1.0, "verify": 1.0})
        self.assertEqual(metrics["revision_rate"], 0.5)
        self.assertEqual(metrics["mean_revisions"], 0.5)
        self.assertEqual(metrics["abstention_rate"], 0.5)
        self.assertEqual(metrics["verified_rate"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            samples = [SimpleNamespace(record_uid="image-a"), SimpleNamespace(record_uid="image-b")]
            evaluation.write_traces(path, samples, predictions)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["record_uid"] for row in rows], ["image-a", "image-b"])
            self.assertEqual(rows[0]["trace"], predictions[0]["trace"])
            self.assertEqual(rows[1]["agent_calls"], predictions[1]["agent_calls"])

    def test_file_fingerprint_tracks_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_schema.json"
            path.write_text('{"schema_version": 1}', encoding="utf-8")
            before = evaluation.file_sha256(path)
            path.write_text('{"schema_version": 2}', encoding="utf-8")
            self.assertNotEqual(evaluation.file_sha256(path), before)


if __name__ == "__main__":
    unittest.main()
