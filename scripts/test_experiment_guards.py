"""Guard regression tests. No model imports, training, GPU or network calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import run_matched_ablations as matched
import run_paper_shifts as shifts
from clear_uav.experiment_guard import (artifact_identity, assert_unlocked,
                                       stage_lock, verify_artifact)


class LockTests(unittest.TestCase):
    def test_same_stage_excluded_different_stage_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one.lock", Path(directory) / "two.lock"
            with stage_lock(first, "one"):
                with self.assertRaises(BlockingIOError):
                    with stage_lock(first, "one"):
                        self.fail("duplicate started")
                with stage_lock(second, "two"):
                    self.assertTrue(first.exists() and second.exists())
            self.assertFalse(first.exists() or second.exists())

    def test_other_process_cannot_start_same_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.lock"
            source = str(matched.ROOT / "src")
            code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                    "from clear_uav.experiment_guard import stage_lock; "
                    "guard=stage_lock(Path(sys.argv[2]),'one'); guard.__enter__()")
            with stage_lock(path, "one"):
                result = subprocess.run([sys.executable, "-c", code, source, str(path)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("BlockingIOError", result.stderr)

    def test_release_after_exception_and_no_stale_lock_stealing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.lock"
            with self.assertRaises(KeyboardInterrupt):
                with stage_lock(path, "one"):
                    raise KeyboardInterrupt()
            self.assertFalse(path.exists())
            path.write_text(json.dumps({"host": "other-host", "pid": 9999999}))
            with self.assertRaisesRegex(BlockingIOError, "other-host"):
                assert_unlocked(path, "one")
            self.assertTrue(path.exists())

    def test_artifact_identity_portable_between_roots_and_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            a.mkdir(); b.mkdir()
            original, copied = a / "last.pt", b / "last.pt"
            original.write_bytes(b"checkpoint")
            copied.write_bytes(b"checkpoint")
            identity = artifact_identity(original, a)
            verify_artifact(copied, identity, b)
            copied.write_bytes(b"CHECKPOINT")  # same length is insufficient
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_artifact(copied, identity, b)


class MatchedGuardTests(unittest.TestCase):
    def test_main_owns_lock_through_child_and_result_sealing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "results/table4/qwen_ground_cls_t4/session_disjoint/seed43_test.json"
            reference.parent.mkdir(parents=True)
            reference.write_text(json.dumps({"rows": [{"record_uid": "a"}]}))
            config = {"experiment": "test", "data": {"protocols": ["session_disjoint"]},
                      "train": {"epochs": 12, "learning_rate": 1e-4,
                                "vision_learning_rate": 1e-6, "seeds": [43]},
                      "output": {"checkpoint": "run/last.pt", "test_results": "result.json"}}
            loader = types.ModuleType("clear_uav.experiment_config")
            loader.load_yaml_with_base = lambda _: config
            def child(*args, **kwargs):
                self.assertTrue(matched.lock_path("full_ms").exists())
                with self.assertRaises(BlockingIOError):
                    with stage_lock(matched.lock_path("full_ms"), "full_ms"):
                        self.fail("duplicate child")
                (root / "run/last.pt").write_bytes(b"checkpoint")
            with patch.object(matched, "ROOT", root), patch.object(sys, "argv", ["runner", "--only", "full_ms"]), \
                    patch.dict(sys.modules, {"clear_uav.experiment_config": loader}), \
                    patch.object(matched, "run_with_progress", side_effect=child), \
                    patch.object(matched, "load_epoch", return_value=12):
                matched.main()
                self.assertFalse(matched.lock_path("full_ms").exists())
                saved = json.loads((root / "run/matched_run_config.json").read_text())
                self.assertIn("checkpoint_identity", saved)
                self.assertEqual(saved["checkpoint_origin"], "trained_by_this_run")

    def test_sealed_checkpoint_or_result_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, result, receipt = root / "last.pt", root / "result.json", root / "receipt.json"
            config = {"experiment": "sample"}
            payload = dict(experiment="sample", protocol="session_disjoint", seed=43,
                           checkpoint_epoch=12, rows=[{"record_uid": "a"}],
                           metrics={"table4": dict(ap50=.1, c_f1=.1, g_map50=.1, n_fpr=.1, p_recall=.1)})
            checkpoint.write_bytes(b"model")
            result.write_text(json.dumps(payload))
            receipt.write_text(json.dumps({"config_sha256": matched.fingerprint(config)}))
            with patch.object(matched, "ROOT", root):
                matched.seal_artifacts(checkpoint, result, receipt, "trained_by_this_run")
                self.assertEqual(matched.state(config, checkpoint, result, receipt, {"a"}, lambda _: 12), "done")
                checkpoint.write_bytes(b"other")
                with self.assertRaisesRegex(ValueError, "differs"):
                    matched.state(config, checkpoint, result, receipt, {"a"}, lambda _: 12)
                checkpoint.write_bytes(b"model")
                payload["metrics"]["table4"]["c_f1"] = .2
                result.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "differs"):
                    matched.state(config, checkpoint, result, receipt, {"a"}, lambda _: 12)

    def test_initialization_refuses_still_locked_ms(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(matched, "ROOT", Path(directory)):
            with stage_lock(matched.lock_path("full_ms"), "full_ms"):
                with self.assertRaises(BlockingIOError):
                    matched.check_initialization("no_roi", lambda _: self.fail("must stop before load"), {"a"})


class ShiftGuardTests(unittest.TestCase):
    def test_main_holds_stage_lock_then_skips_sealed_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(shifts, "ROOT", Path(directory)):
            root = Path(directory)
            artifact, marker = root / "result.json", root / "unseen_site_ground_cls_test_results.json"
            step = ([sys.executable, "scripts/test_qwen_ground_cls.py"], self.config(), artifact, marker)
            def child(*args, **kwargs):
                self.assertTrue(shifts.lock_path(marker).exists())
                with self.assertRaises(BlockingIOError):
                    with stage_lock(shifts.lock_path(marker), marker.stem):
                        self.fail("duplicate child")
                artifact.write_text(json.dumps(self.payload()))
            with patch.object(shifts, "plan", return_value=[step]), \
                    patch.object(shifts, "dependency_names", return_value=[]), \
                    patch.object(sys, "argv", ["runner"]), \
                    patch.object(shifts, "run_with_progress", side_effect=child) as run:
                shifts.main()
                self.assertEqual(run.call_count, 1)
                self.assertFalse(shifts.lock_path(marker).exists())
                shifts.main()
                self.assertEqual(run.call_count, 1)

    def config(self):
        return {"experiment": "paper_shift_ground_cls", "data": {"protocols": ["unseen_site"]},
                "train": {"seeds": [43], "epochs": 12}, "output": {}}

    def payload(self):
        return {"experiment": "paper_shift_ground_cls", "protocol": "unseen_site", "seed": 43,
                "checkpoint_epoch": 12, "rows": [{"record_uid": "a"}],
                "metrics": {"table4": {"threshold": .5}}}

    def test_skip_binds_artifact_identity_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(shifts, "ROOT", Path(directory)):
            root = Path(directory)
            artifact, marker = root / "result.json", root / "unseen_site_ground_cls_test_results.json"
            dependency = root / "last.pt"
            dependency.write_bytes(b"checkpoint")
            artifact.write_text(json.dumps(self.payload()))
            receipt = {"config_sha256": matched.fingerprint(self.config()), "stage": marker.stem,
                       "protocol": "unseen_site", "seed": 43,
                       "artifact_identity": artifact_identity(artifact, root),
                       "dependencies": {"ms": artifact_identity(dependency, root)}}
            marker.write_text(json.dumps(receipt))
            self.assertTrue(shifts.completed(self.config(), artifact, marker))
            dependency.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "differs"):
                shifts.completed(self.config(), artifact, marker)
            dependency.write_bytes(b"checkpoint")
            artifact.write_text(json.dumps(self.payload()) + " ")
            with self.assertRaisesRegex(ValueError, "differs"):
                shifts.completed(self.config(), artifact, marker)

    def test_legacy_completion_receipt_is_not_silently_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, marker = Path(directory) / "result.json", Path(directory) / "unseen_site_ground_cls_test_results.json"
            artifact.write_text(json.dumps(self.payload()))
            marker.write_text(json.dumps({"config_sha256": matched.fingerprint(self.config()), "artifact": str(artifact)}))
            with self.assertRaisesRegex(ValueError, "Legacy/unbound"):
                shifts.completed(self.config(), artifact, marker)

    def test_result_identity_and_incomplete_epoch_rejected(self):
        marker = Path("unseen_site_ground_cls_test_results.json")
        for field, value in (("experiment", "wrong"), ("protocol", "forward_temporal"),
                             ("seed", 44), ("checkpoint_epoch", 4), ("rows", [])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory) / "result.json"
                payload = self.payload(); payload[field] = value
                artifact.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    shifts.validate_artifact(self.config(), artifact, marker)

    def test_result_without_explicit_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "result.json"
            payload = self.payload()
            del payload["seed"]
            artifact.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "experiment/protocol/seed"):
                shifts.validate_artifact(self.config(), artifact,
                                         Path("unseen_site_ground_cls_test_results.json"))

    def test_qwen_receipt_binds_adapter_and_cached_predictions(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(shifts, "ROOT", Path(directory)):
            root = Path(directory)
            adapter = root / "qwen/best"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            (adapter.parent / "training_data.json").write_text("{}")
            cache = root / "val.json"; cache.write_text("[]")
            config = self.config()
            config["output"] = {"adapter": "qwen/best", "validation_predictions": "val.json"}
            marker = root / "unseen_site_qwen_calibration.json"
            original = shifts.supporting_artifacts(config, marker)
            self.assertIn("adapter/adapter_model.safetensors", original)
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(original, shifts.supporting_artifacts(config, marker))


if __name__ == "__main__":
    unittest.main()
