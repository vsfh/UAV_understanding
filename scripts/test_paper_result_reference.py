"""CPU-only regression tests for independently sourced paper-result audits."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from audit_table4_results import compute
from paper_result_reference import build_expected_rows, negative_filenames
from update_paper_results import audit, check_targets, load_result


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.labels = ["event"] + [f"unused{i}" for i in range(17)]
        self.write("configs/core18_complete.txt", "\n".join(self.labels))
        self.write("configs/ontology.yaml", yaml.safe_dump({"events": [{"name": n} for n in [*self.labels, "outside"]]}))
        settings = {"root": "um7/no_event", "group_gap_seconds": 60,
                    "split_ratios": {"train": .7, "val": .1, "test": .2}, "seed": 43}
        config = {"data": {"root": "um7", "labels": "configs/core18_complete.txt",
                           "ontology": "configs/ontology.yaml", "bbox_annotations": "um7/boxes.json",
                           "no_event": settings}}
        self.write("configs/yaml/table4_qwen3vl_t4.yaml", yaml.safe_dump(config))
        self.settings = settings
        self.names = ["photo - 2025-01-01T000000.000.png", "photo - 2025-01-01T000001.000.png",
                      "photo - 2025-01-01T000300.000.png", "photo - 2025-01-01T000600.000.png"]
        self.write("inventory.json", json.dumps({"filenames": self.names,
                    "filenames_sha256": hashlib.sha256("\n".join(sorted(self.names)).encode()).hexdigest()}))
        coco = {"images": [{"id": i, "file_name": f"{uid}.jpg", "width": 10, "height": 20}
                           for i, uid in enumerate(("a", "b", "out"))],
                "annotations": [{"image_id": i, "bbox": [1, 2, 3, 4]} for i in range(3)]}
        self.write("um7/boxes.json", json.dumps(coco))
        for protocol in ("session_disjoint", "unseen_site", "forward_temporal"):
            self.csv(f"um7/{protocol}/test_inputs.csv", [
                {"record_uid": uid, "context_path": f"data/{uid}.jpg", "content_group_id": f"group-{uid}"}
                for uid in ("a", "b", "out")])
            self.csv(f"um7/{protocol}/test_labels_private.csv", [
                {"record_uid": uid, "source_class": label}
                for uid, label in (("a", "event"), ("b", "event"), ("out", "outside"))])

    def write(self, name, text):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def csv(self, name, rows):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def reference(self, protocol="unseen_site"):
        return build_expected_rows(self.root, protocol, "inventory.json")

    def rows(self, protocol="unseen_site"):
        rows = copy.deepcopy(self.reference(protocol)["rows"])
        for row in rows:
            row["prediction"] = {"presence_score": .8, "category": "event",
                                 "bbox_1000": [100., 100., 400., 300.], "valid": True, "latency_ms": 1.}
        return rows

    def result(self, rows=None, protocol="unseen_site", system="qwen", **overrides):
        rows = self.rows(protocol) if rows is None else rows
        experiment = ({"qwen": "table4_qwen3vl_t4", "tiling": "table4_qwen3vl_adaptive_tiling"}[system]
                      if protocol == "session_disjoint" else "paper_shift_"+system)
        payload = {"experiment": experiment, "protocol": protocol, "seed": 43,
                   "rows": rows, "metrics": compute(rows, self.labels, .5)}
        payload.update(overrides)
        model = {"qwen": "qwen3vl_t4", "tiling": "qwen3vl_adaptive_tiling", "ground_cls": "ground_cls"}[system]
        directory = "table4" if protocol == "session_disjoint" else "table4_shifts"
        return self.write(f"results/{directory}/{model}/{protocol}/seed43_test.json", json.dumps(payload))

    def test_source_population_and_bbox_are_independent(self):
        ref = self.reference()
        self.assertEqual(ref["provenance"]["records"], 3)
        self.assertEqual(ref["provenance"]["positive_records"], 2)
        self.assertEqual(ref["provenance"]["negative_records"], 1)
        self.assertEqual(ref["provenance"]["excluded_outside_label_subset"], 1)
        self.assertEqual(ref["rows"][0]["target"]["bbox_1000"], [100., 100., 400., 300.])
        self.result()
        report = audit(self.root, "inventory.json")
        self.assertEqual(report["domains"]["unseen_site"]["qwen"]["records"], 3)

    def test_first_domain_result_missing_uid_rejected(self):
        self.result(self.rows()[:-1])
        with self.assertRaisesRegex(ValueError, "frozen source reference"):
            audit(self.root, "inventory.json")

    def test_common_omission_across_domain_systems_rejected(self):
        rows = self.rows()[:-1]
        self.result(rows)
        self.result(rows, system="tiling")
        with self.assertRaisesRegex(ValueError, "frozen source reference"):
            audit(self.root, "inventory.json")

    def test_wrong_experiment_rejected(self):
        self.result(experiment="another_experiment")
        with self.assertRaisesRegex(ValueError, "Wrong experiment"):
            audit(self.root, "inventory.json")

    def test_duplicate_prediction_uid_rejected(self):
        rows = self.rows()
        self.result([*rows, rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
            audit(self.root, "inventory.json")

    def test_polluted_negative_target_rejected(self):
        for field, value in (("category", "event"), ("bbox_1000", [0, 0, 10, 10])):
            rows = self.rows()
            negative = next(r for r in rows if not r["target"]["presence"])
            negative["target"][field] = value
            path = self.result(rows)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Negative target"):
                load_result(path, "unseen_site", expected_experiment="paper_shift_qwen")

    def test_group_and_positive_box_changes_rejected(self):
        reference = self.reference()["rows"]
        rows = self.rows()
        rows[0]["group_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Group"):
            check_targets(rows, reference)
        rows = self.rows()
        rows[0]["target"]["bbox_1000"][0] += 1
        with self.assertRaisesRegex(ValueError, "Target box"):
            check_targets(rows, reference)

    def test_missing_shift_seed_rejected_legacy_is_explicit(self):
        path = self.result()
        data = json.loads(path.read_text())
        del data["seed"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Wrong seed"):
            audit(self.root, "inventory.json")
        with self.assertRaisesRegex(ValueError, "Wrong seed"):
            load_result(path, "unseen_site", expected_experiment="paper_shift_qwen", allow_legacy_missing_seed=True)

    def test_legacy_session_missing_seed_flagged_wrong_seed_rejected(self):
        path = self.result(protocol="session_disjoint")
        data = json.loads(path.read_text())
        del data["seed"]
        path.write_text(json.dumps(data))
        report = audit(self.root, "inventory.json")
        self.assertTrue(report["domains"]["session_disjoint"]["qwen"]["missing_seed_metadata"])
        data["seed"] = 44
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Wrong seed"):
            audit(self.root, "inventory.json")

    def test_ground_epoch_rejected(self):
        self.result(system="ground_cls", checkpoint_epoch=11)
        with self.assertRaisesRegex(ValueError, "Incomplete checkpoint"):
            audit(self.root, "inventory.json")

    def test_duplicate_source_uid_rejected(self):
        path = self.root/"um7/unseen_site/test_inputs.csv"
        path.write_text(path.read_text()+"a,data/a.jpg,group-a\n")
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            self.reference()

    def test_missing_private_label_rejected(self):
        self.csv("um7/unseen_site/test_labels_private.csv", [{"record_uid": "a", "source_class": "event"}])
        with self.assertRaisesRegex(ValueError, "identical IDs"):
            self.reference()

    def test_inventory_hash_and_duplicates_rejected(self):
        self.write("inventory.json", json.dumps({"filenames": self.names, "filenames_sha256": "wrong"}))
        with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
            negative_filenames(self.root, self.settings, "inventory.json")
        self.write("inventory.json", json.dumps([*self.names, self.names[0]]))
        with self.assertRaisesRegex(ValueError, "Duplicate negative"):
            negative_filenames(self.root, self.settings, "inventory.json")

    def test_real_directory_matches_offline_inventory(self):
        directory = self.root/"um7/no_event"
        directory.mkdir()
        for name in self.names:
            (directory/name).touch()
        actual = build_expected_rows(self.root, "unseen_site")
        self.assertEqual(actual["rows"], self.reference()["rows"])

    def test_reference_module_has_no_model_imports(self):
        tree = ast.parse(Path(__file__).with_name("paper_result_reference.py").read_text())
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
        self.assertFalse(imports & {"torch", "transformers", "peft", "clear_uav"})


if __name__ == "__main__":
    unittest.main()
