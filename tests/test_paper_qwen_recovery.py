"""Offline recovery safety tests: stdlib only; no model import or child job.

Run from any directory:
    python -B path/to/code/tests/test_paper_qwen_recovery.py
"""
from contextlib import ExitStack, contextmanager, redirect_stdout
import builtins
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


CODE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE / "src"))
import recover_paper_qwen as recovery
import run_paper_shifts as runner


@contextmanager
def forbid_execution():
    """Fail if recovery crosses into heavy imports or process execution."""
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (name.split(".")[0] in {"torch", "transformers", "peft", "numpy", "tensorboard"}
                or name == "clear_uav.table4"
                or name == "clear_uav" and "table4" in (fromlist or ())):
            raise AssertionError(f"Heavy/model import forbidden in offline tests: {name}")
        return original_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded_import), \
            mock.patch("subprocess.Popen", side_effect=AssertionError("No child job allowed")), \
            mock.patch.object(runner, "run_with_progress", side_effect=AssertionError("No training/inference allowed")) as launch:
        yield launch
        launch.assert_not_called()


class EquivalentTests(unittest.TestCase):
    def test_matching_nested_structure_and_float_tolerance(self):
        recovery.equivalent({"a": [True, {"score": 0.5 + 1e-11}], "b": None},
                            {"a": [True, {"score": 0.5}], "b": None}, "result")
        recovery.equivalent([1, 2], (1.0, 2.0), "sequence")

    def test_boolean_is_not_number_in_either_direction(self):
        for actual, expected in ((True, 1), (1, True), (False, 0), (0.0, False)):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                recovery.equivalent(actual, expected, "strict_bool")

    def test_nonfinite_numbers_never_match(self):
        for actual, expected in ((float("nan"), float("nan")), (float("nan"), 0.5),
                                 (0.5, float("nan")), (float("inf"), float("inf")),
                                 (float("-inf"), float("-inf"))):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                recovery.equivalent({"v": [actual]}, {"v": [expected]}, "finite")

    def test_list_shape_order_and_scalar_type_are_checked(self):
        for actual, expected in (([1], [1, 2]), ([2, 1], [1, 2]), ("12", [1, 2]),
                                 ({"0": 1}, [1]), (["1"], [1]), (1, "1")):
            with self.subTest(actual=actual, expected=expected), self.assertRaises(ValueError):
                recovery.equivalent(actual, expected, "shape")

    def test_dict_missing_extra_and_wrong_container_are_rejected(self):
        for actual in ({}, {"a": 1, "b": 2}, [("a", 1)]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                recovery.equivalent(actual, {"a": 1}, "fields")


class UniqueRowsTests(unittest.TestCase):
    def test_maps_unique_rows_without_altering_rows(self):
        rows = [{"record_uid": "a", "score": 0.1}, {"record_uid": "b"}]
        result = recovery.unique_rows(rows, "cache")
        self.assertEqual(list(result), ["a", "b"])
        self.assertIs(result["a"], rows[0])

    def test_duplicate_uid_rejected_even_if_rows_equal(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            recovery.unique_rows([{"record_uid": "a"}, {"record_uid": "a"}], "cache")

    def test_missing_empty_nonstring_and_nonobject_rows_rejected(self):
        for row in ({}, {"record_uid": None}, {"record_uid": ""}, {"record_uid": 1}, None, "a"):
            with self.subTest(row=row), self.assertRaises(ValueError):
                recovery.unique_rows([row], "cache")

    def test_empty_or_nonlist_collection_rejected(self):
        for rows in ([], {}, None, ({"record_uid": "a"},)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                recovery.unique_rows(rows, "cache")


class ExpectedUpdatesTests(unittest.TestCase):
    def setUp(self):
        self.train = {"batch_size": 4, "gradient_accumulation": 2, "epochs": 3,
                      "samples_per_epoch": "positive_records"}

    def test_16112_positive_records_produce_6042_optimizer_updates(self):
        samples = [SimpleNamespace(presence=True)] * 16112 + [SimpleNamespace(presence=False)] * 2500
        self.assertEqual(recovery.expected_updates(self.train, samples), 6042)

    def test_default_budget_counts_only_positive_records(self):
        train = dict(self.train)
        del train["samples_per_epoch"]
        samples = [SimpleNamespace(presence=1)] * 9 + [SimpleNamespace(presence=0)] * 50
        self.assertEqual(recovery.expected_updates(train, samples), 6)

    def test_explicit_budget_uses_two_roundings_and_epochs(self):
        self.train["samples_per_epoch"] = 1003
        self.assertEqual(recovery.expected_updates(self.train, []), 378)

    def test_none_budget_uses_all_records(self):
        self.train["samples_per_epoch"] = None
        samples = [SimpleNamespace(presence=False)] * 9
        self.assertEqual(recovery.expected_updates(self.train, samples), 6)

    def test_invalid_sample_budgets_are_rejected(self):
        for budget in (True, False, 0, -1, 1.5, "unknown"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                recovery.expected_updates({**self.train, "samples_per_epoch": budget}, [])

    def test_empty_positive_population_is_rejected(self):
        with self.assertRaises(ValueError):
            recovery.expected_updates(self.train, [SimpleNamespace(presence=0)])

    def test_invalid_training_dimensions_are_rejected(self):
        for key in ("batch_size", "gradient_accumulation", "epochs"):
            for value in (True, False, 0, -1, 1.5):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    recovery.expected_updates({**self.train, "samples_per_epoch": 8, key: value}, [])


class RecoveryOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="qwen_recovery_test_")))
        self.protocol = "unseen_site"
        self.config = {"experiment": "paper_shift_qwen", "data": {"protocols": [self.protocol]},
                       "output": {"root": "outputs/qwen/unseen_site/seed43"}}
        self.calibration = self.root / "outputs/calibration.json"
        self.result = self.root / "results/test.json"
        self.source = self.root / "inputs/frozen_test.csv"
        self.adapter = self.root / "outputs/adapter.safetensors"
        self.cache = self.root / "outputs/predictions.json"
        for path, text in ((self.calibration, json.dumps({"threshold": 0.5})),
                           (self.result, json.dumps({"experiment": "paper_shift_qwen", "protocol": self.protocol,
                                                    "seed": 43, "rows": [{"record_uid": "test_1"}],
                                                    "metrics": {"threshold": 0.5}})),
                           (self.source, "record_uid\ntest_1\n"), (self.adapter, "fake adapter bytes"),
                           (self.cache, '[{"record_uid":"test_1"}]')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.markers = [self.root / f"receipts/{self.protocol}_qwen_{suffix}.json"
                        for suffix in ("calibration", "results")]
        self.catalog = {marker.stem: (["NEVER_EXECUTE", "train" if i == 0 else "test"],
                                      copy.deepcopy(self.config), artifact, marker)
                        for i, (artifact, marker) in enumerate(zip((self.calibration, self.result), self.markers))}
        self.evidence = {"checks": ["fake deterministic verifier"], "records": {"val": 1, "test": 1}}
        self.verifier = mock.Mock(side_effect=self.verify_with_locks)
        self.active_locks = []
        self.lock_history = []
        self.stack.enter_context(mock.patch.object(runner, "ROOT", self.root))
        self.stack.enter_context(mock.patch.object(runner, "stage_lock", side_effect=self.fake_lock))
        self.support_mock = self.stack.enter_context(mock.patch.object(runner, "supporting_artifacts", side_effect=self.support))
        self.source_mock = self.stack.enter_context(mock.patch.object(recovery, "source_files", return_value=[self.source]))
        self.launch = self.stack.enter_context(forbid_execution())
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.initial = {p: p.read_bytes() for p in (self.calibration, self.result, self.source, self.adapter, self.cache)}

    @contextmanager
    def fake_lock(self, path, label):
        self.assertNotIn(label, self.active_locks)
        self.active_locks.append(label)
        self.lock_history.append(("enter", label))
        try:
            yield
        finally:
            self.active_locks.remove(label)
            self.lock_history.append(("exit", label))

    def support(self, config, marker):
        return {"adapter": runner.artifact_identity(self.adapter, self.root),
                "full_frame_predictions": runner.artifact_identity(self.cache, self.root)}

    def verify_with_locks(self, config, protocol, root):
        self.assertEqual(self.active_locks, [p.stem for p in self.markers])
        self.assertEqual(config, self.config)
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(root, self.root)
        return copy.deepcopy(self.evidence)

    def recover(self):
        recovery.recover_existing_qwen(self.protocol, self.catalog, runner, verifier=self.verifier)

    def assert_no_receipts(self):
        self.assertTrue(all(not marker.exists() for marker in self.markers))
        self.assertEqual(self.active_locks, [])
        self.launch.assert_not_called()

    def assert_inputs_untouched(self):
        self.assertEqual({p: p.read_bytes() for p in self.initial}, self.initial)

    def test_invalid_verifier_writes_neither_receipt(self):
        self.verifier.side_effect = ValueError("Invalid frozen split/metric evidence")
        with self.assertRaisesRegex(ValueError, "Invalid frozen"):
            self.recover()
        self.assert_no_receipts()
        self.assert_inputs_untouched()

    def test_missing_test_output_rejected_before_verification(self):
        self.result.unlink()
        with self.assertRaisesRegex(ValueError, "calibration AND test"):
            self.recover()
        self.verifier.assert_not_called()
        self.assert_no_receipts()

    def test_missing_calibration_rejected_before_verification(self):
        self.calibration.unlink()
        with self.assertRaisesRegex(ValueError, "calibration AND test"):
            self.recover()
        self.verifier.assert_not_called()
        self.assert_no_receipts()

    def test_bad_existing_calibration_receipt_preserved(self):
        self.assert_bad_existing_receipt_preserved(0)

    def test_bad_existing_test_receipt_preserved(self):
        self.assert_bad_existing_receipt_preserved(1)

    def assert_bad_existing_receipt_preserved(self, index):
        marker = self.markers[index]
        marker.parent.mkdir(parents=True, exist_ok=True)
        original = b'{"config_sha256":"unrelated-existing-receipt"}\n'
        marker.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Changed config"):
            self.recover()
        self.assertEqual(marker.read_bytes(), original)
        self.assertFalse(self.markers[1 - index].exists())
        self.verifier.assert_not_called()
        self.assert_inputs_untouched()

    def test_duplicate_result_rows_rejected_before_verifier(self):
        payload = json.loads(self.result.read_text())
        payload["rows"] *= 2
        self.result.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.recover()
        self.verifier.assert_not_called()
        self.assert_no_receipts()

    def test_mismatched_stage_configs_rejected(self):
        self.catalog[self.markers[1].stem][1]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            self.recover()
        self.verifier.assert_not_called()
        self.assert_no_receipts()

    def test_source_adapter_cache_and_output_hash_changes_rejected(self):
        for path in (self.source, self.adapter, self.cache, self.calibration, self.result):
            with self.subTest(path=path.name):
                original = path.read_bytes()

                def mutate(*args):
                    self.verify_with_locks(*args)
                    path.write_bytes(original + b"\nchanged during verification")
                    return copy.deepcopy(self.evidence)

                self.verifier.side_effect = mutate
                try:
                    with self.assertRaisesRegex(ValueError, "changed during recovery"):
                        self.recover()
                    self.assert_no_receipts()
                finally:
                    path.write_bytes(original)

    def test_good_recovery_binds_both_receipts_and_correct_dependency(self):
        self.recover()
        self.verifier.assert_called_once()
        calibration, result = [json.loads(marker.read_text()) for marker in self.markers]
        self.assertEqual(calibration["dependencies"], {})
        self.assertEqual(result["dependencies"], {self.markers[0].stem: calibration["artifact_identity"]})
        for receipt, marker, artifact in zip((calibration, result), self.markers, (self.calibration, self.result)):
            self.assertEqual(receipt["stage"], marker.stem)
            self.assertEqual(receipt["protocol"], self.protocol)
            self.assertEqual(receipt["seed"], 43)
            self.assertEqual(receipt["config_sha256"], runner.fingerprint(self.config))
            self.assertEqual(receipt["resolved_config"], self.config)
            self.assertEqual(receipt["artifact_identity"], runner.artifact_identity(artifact, self.root))
            self.assertEqual(receipt["supporting_artifacts"], self.support(self.config, marker))
            adoption = receipt["recovery"]
            self.assertTrue(adoption["no_training_or_inference"])
            self.assertEqual(adoption["kind"], "validated_existing_outputs")
            self.assertEqual(adoption["evidence"], self.evidence)
            self.assertEqual(adoption["source_identities"],
                             {self.source.relative_to(self.root).as_posix(): runner.artifact_identity(self.source, self.root)})
            self.assertTrue(runner.completed(self.config, artifact, marker))
        self.assertEqual(self.active_locks, [])
        self.assertEqual(len(self.lock_history), 4)
        self.assert_inputs_untouched()

    def test_second_recovery_is_byte_for_byte_idempotent_without_reverification(self):
        self.recover()
        original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.markers}
        self.verifier.reset_mock()
        self.source_mock.reset_mock()
        self.recover()
        self.verifier.assert_not_called()
        self.source_mock.assert_not_called()
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.markers}, original)
        self.assert_inputs_untouched()

    def test_one_valid_existing_receipt_is_preserved_while_other_is_recovered(self):
        self.recover()
        original = self.markers[0].read_bytes()
        self.markers[1].unlink()
        self.verifier.reset_mock()
        self.recover()
        self.verifier.assert_called_once()
        self.assertEqual(self.markers[0].read_bytes(), original)
        recovered = json.loads(self.markers[1].read_text())
        self.assertEqual(recovered["dependencies"],
                         {self.markers[0].stem: runner.artifact_identity(self.calibration, self.root)})
        self.assert_inputs_untouched()


class RecoveryCliTests(unittest.TestCase):
    def call_cli(self, argv, recover_mock):
        with mock.patch.object(sys, "argv", ["run_paper_shifts.py", *argv]), \
                mock.patch.object(runner, "plan", return_value=[]), \
                mock.patch.object(recovery, "recover_existing_qwen", recover_mock), \
                mock.patch.object(runner, "completed", side_effect=AssertionError("Normal runner must not execute")), \
                forbid_execution():
            runner.main()

    def test_qwen_recovery_returns_without_normal_runner_or_training(self):
        recover_mock = mock.Mock()
        self.call_cli(["--protocol", "unseen_site", "unseen_site", "--only", "qwen", "--recover-existing"], recover_mock)
        recover_mock.assert_called_once_with("unseen_site", {}, runner)

    def test_recovery_requires_only_qwen(self):
        for groups in ([], ["ground"], ["tiling"], ["qwen", "ground"], ["qwen", "tiling"]):
            with self.subTest(groups=groups):
                recover_mock = mock.Mock()
                argv = ["--recover-existing"] + (["--only", *groups] if groups else [])
                with self.assertRaisesRegex(ValueError, "requires --only qwen"):
                    self.call_cli(argv, recover_mock)
                recover_mock.assert_not_called()

    def test_verifier_failure_cannot_fall_through_to_training(self):
        recover_mock = mock.Mock(side_effect=ValueError("Recovery refused"))
        with self.assertRaisesRegex(ValueError, "Recovery refused"):
            self.call_cli(["--protocol", "forward_temporal", "--only", "qwen", "--recover-existing"], recover_mock)
        recover_mock.assert_called_once_with("forward_temporal", {}, runner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
