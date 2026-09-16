"""CPU checks of the shipped protocol configs and orchestration; no model is loaded."""
import copy
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import yaml

import perception_no_spatial_protocols as protocol
import run_perception_no_spatial as runner


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="perception-protocol-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        shutil.copytree(protocol.ROOT / "configs/yaml", self.root / "configs/yaml")
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("train_perception_qwen.py", "train_perception_continue.py", "perception_extension_runtime.py"):
            (scripts / name).write_text("# Mock source receipt\n", encoding="utf-8")
        for name in protocol.PROTOCOLS:
            folder = self.root / "um7" / name
            folder.mkdir(parents=True)
            for part, suffix, date in (("train", "a", "2025-10-31 10:00:00"),
                                       ("val", "b", "2025-11-01 10:00:00"),
                                       ("test_inputs", "c", "2026-01-01 10:00:00")):
                row = {"record_uid": suffix, "content_group_id": f"g_{suffix}",
                       "site_id": f"site_{suffix}" if name == "unseen_site" else "shared_site",
                       "session_id": "shared_session" if name == "unseen_site" else f"session_{suffix}",
                       "detected_at": date}
                with (folder / f"{part}.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
        self.addCleanup(patch.stopall)
        patch.object(protocol, "ROOT", self.root).start()
        self.events = []
        self.baseline = types.ModuleType("train_perception_qwen")
        self.baseline.build_model = self.base_builder
        self.original_builder = self.baseline.build_model
        self.baseline.train = self.fake_train
        continuation = types.ModuleType("train_perception_continue")
        continuation.build_model = self.continue_builder
        continuation.checkpoint_fingerprint = lambda path: "mock-checkpoint-fingerprint"
        self.evaluator = types.ModuleType("test_perception_continue")
        self.evaluator.evaluate = Mock()
        self.table4 = types.ModuleType("clear_uav.table4")
        self.table4.training_data_state = lambda config, name: {"protocol": name, "state": "fixed"}
        self.table4.read_discovery_samples = Mock(return_value=[types.SimpleNamespace(presence=True),
                                                               types.SimpleNamespace(presence=False)])
        self.table4.no_event_split = Mock(return_value={"train": [1], "val": [2], "test": [3]})
        package = types.ModuleType("clear_uav")
        package.__path__ = []
        patch.dict(sys.modules, {"train_perception_qwen": self.baseline,
                                 "train_perception_continue": continuation,
                                 "test_perception_continue": self.evaluator,
                                 "clear_uav": package, "clear_uav.table4": self.table4}).start()

    def base_builder(self, config):
        self.events.append((config["protocol"], "base_factory"))

    def continue_builder(self, config):
        self.events.append((config["protocol"], "continuation_factory"))

    def fake_train(self, config):
        self.baseline.build_model(config)
        self.events.append((config["protocol"], "train", config["train"]["epochs"]))
        output = protocol.output_path(config)
        (output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (output / "history.json").write_text(json.dumps([{"epoch": i} for i in range(1, 4)]))
        best = output / "best"
        best.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors", "box_head.pt"):
            (best / name).write_bytes(b"mock checkpoint")

    def complete(self, name):
        with redirect_stdout(io.StringIO()):
            protocol.train_stage(name, "base")
            protocol.train_stage(name, "continue")

    def test_real_configs_change_only_protocol_and_remove_audit_pointer(self):
        for name in protocol.PROTOCOLS:
            for stage in protocol.STAGES:
                with self.subTest(protocol=name, stage=stage):
                    actual = protocol.load_config(name, stage)
                    expected = protocol.read_config(protocol.reference_path(stage))
                    expected["protocol"] = name
                    expected.pop("matched_spatial_config", None)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual["train"]["epochs"], 3)
                    self.assertNotIn("spatial", actual)
                    self.assertNotIn("spatial_lora", actual)
                    self.assertEqual(actual["data"]["labels"], "./configs/core18_complete.txt")
            base = protocol.load_config(name, "base")
            continuation = protocol.load_config(name, "continue")
            self.assertNotIn("initial_checkpoint", base)
            self.assertEqual((self.root / continuation["initial_checkpoint"].format(**continuation)).resolve(),
                             (protocol.output_path(base) / "best").resolve())

    def test_changed_hyperparameter_or_session_checkpoint_is_rejected(self):
        path = protocol.config_path("forward_temporal", "continue")
        original = protocol.read_config(path)
        for changed in (original | {"initial_checkpoint": "outputs/perception_qwen/session_disjoint/seed43/best"},
                        original | {"train": original["train"] | {"batch_size": 4}},
                        original | {"spatial": {"enabled": True}}):
            path.write_text(yaml.safe_dump(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Settings differ"):
                protocol.load_config("forward_temporal", "continue")

    def test_actual_all_orchestration_orders_base_continue_calibrated_test(self):
        for name in protocol.PROTOCOLS:
            with self.subTest(protocol=name), patch.object(sys, "argv", ["run", "--protocol", name]), \
                    patch.object(runner.os, "chdir"), patch.object(runner, "check_protocol", return_value={}) as check, \
                    patch.object(runner.subprocess, "run") as launch, redirect_stdout(io.StringIO()):
                runner.main()
                check.assert_called_once_with(name)
                self.assertEqual([call.args[0][1:] for call in launch.call_args_list], [
                    ["scripts/train_perception_no_spatial.py", "--protocol", name, "--stage", "base"],
                    ["scripts/train_perception_no_spatial.py", "--protocol", name, "--stage", "continue"],
                    ["scripts/test_perception_no_spatial.py", "--protocol", name, "--split", "all"]])
                self.assertTrue(all(call.kwargs == {"check": True} for call in launch.call_args_list))

    def test_test_mode_recalibrates_and_dry_run_never_loads_data_or_starts_jobs(self):
        for name in protocol.PROTOCOLS:
            self.assertEqual(runner.commands(name, "test")[0][-2:], ["--split", "all"])
            with patch.object(sys, "argv", ["run", "--protocol", name, "--dry-run"]), \
                    patch.object(runner.os, "chdir"), \
                    patch.object(runner, "check_protocol", side_effect=AssertionError("data read")), \
                    patch.object(runner.subprocess, "run", side_effect=AssertionError("job launch")), \
                    redirect_stdout(io.StringIO()):
                runner.main()

    def test_both_protocols_use_original_factories_for_three_plus_three_epochs(self):
        for name in protocol.PROTOCOLS:
            with self.subTest(protocol=name):
                self.complete(name)
                self.assertEqual([event[1:] for event in self.events if event[0] == name],
                                 [("base_factory",), ("train", 3), ("continuation_factory",), ("train", 3)])
                self.assertIs(self.baseline.build_model, self.original_builder)
                for stage in protocol.STAGES:
                    self.assertTrue(protocol.training_complete(protocol.load_config(name, stage)))
        self.assertNotEqual(protocol.output_path(protocol.load_config("unseen_site", "base")),
                            protocol.output_path(protocol.load_config("forward_temporal", "base")))

    def test_completed_stages_are_reused_without_retraining(self):
        self.complete("unseen_site")
        events = copy.deepcopy(self.events)
        self.complete("unseen_site")
        self.assertEqual(self.events, events)

    def test_continuation_cannot_start_without_own_completed_base(self):
        with self.assertRaisesRegex(ValueError, "base training first"):
            protocol.train_stage("unseen_site", "continue")
        self.assertEqual(self.events, [])
        self.assertFalse(protocol.output_path(protocol.load_config("unseen_site", "continue")).exists())

    def test_evaluation_delegates_unchanged_config_and_requested_split(self):
        for name in protocol.PROTOCOLS:
            self.complete(name)
            expected = protocol.load_config(name, "continue")
            for split in ("val", "test", "all"):
                protocol.evaluate_protocol(name, split)
                self.evaluator.evaluate.assert_called_with(expected, split)
            self.assertEqual(expected["validation"], {"fallback_threshold": 0.5, "max_n_fpr": 0.1})

    def test_split_contract_respects_protocol_specific_isolation(self):
        for name in protocol.PROTOCOLS:
            contract = protocol.split_contract(protocol.load_config(name, "base"))
            self.assertEqual(contract["disjoint_keys"][-1], "site_id" if name == "unseen_site" else "session_id")
            # The public fixtures have no test_labels_private.csv by design.
            self.assertFalse((self.root / "um7" / name / "test_labels_private.csv").exists())
        path = self.root / "um7/unseen_site/test_inputs.csv"
        path.write_text(path.read_text().replace("site_c", "site_a"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "site_id overlaps"):
            protocol.split_contract(protocol.load_config("unseen_site", "base"))
        path = self.root / "um7/forward_temporal/test_inputs.csv"
        path.write_text(path.read_text().replace("2026-01-01", "2025-10-01"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dates are not ordered"):
            protocol.split_contract(protocol.load_config("forward_temporal", "base"))

    def test_check_reads_only_train_val_and_passes_current_protocol_to_negatives(self):
        for name in protocol.PROTOCOLS:
            self.table4.read_discovery_samples.reset_mock()
            receipt = protocol.check_protocol(name)
            self.assertEqual(receipt["epochs"], {"base": 3, "continue": 3})
            self.assertEqual([call.args[1:] for call in self.table4.read_discovery_samples.call_args_list],
                             [(name, "train"), (name, "val")])
            self.table4.no_event_split.assert_called_with(protocol.load_config(name, "base"), name)

    def test_incomplete_checkpoint_or_changed_split_cannot_be_reused(self):
        self.complete("unseen_site")
        config = protocol.load_config("unseen_site", "continue")
        history = protocol.output_path(config) / "history.json"
        history.write_text('[{"epoch": 1}]')
        with self.assertRaisesRegex(RuntimeError, "unfinished epochs"):
            protocol.training_complete(config)
        history.write_text(json.dumps([{"epoch": i} for i in range(1, 4)]))
        manifest = self.root / "um7/unseen_site/train.csv"
        manifest.write_text(manifest.read_text().replace("2025-10-31", "2025-10-30"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Split manifests changed"):
            protocol.training_complete(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
