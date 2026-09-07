import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_table4_results import compute
from update_paper_results import check_targets, load_result, summarize, matched_table, VARIANTS
import run_paper_shifts as shifts


class PredictionTests(unittest.TestCase):
    def rows(self):
        return [dict(record_uid="a", group_id="one", target=dict(presence=True, category="event", bbox_1000=[0, 0, 10, 10]),
                     prediction=dict(presence_score=.8, category="event", bbox_1000=[0, 0, 10, 10], valid=True, latency_ms=1)),
                dict(record_uid="b", group_id="two", target=dict(presence=False, category=None, bbox_1000=None),
                     prediction=dict(presence_score=.9, category="event", bbox_1000=[0, 0, 10, 10], valid=True, latency_ms=1))]

    def test_positive_ap_recomputed_not_copied(self):
        rows = self.rows()
        data = dict(protocol="session_disjoint", seed=43, checkpoint_epoch=12, rows=rows,
                    metrics=compute(rows, ["event"], .5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(data))
            payload, actual = load_result(path, "session_disjoint", 12)
            result = summarize(path, payload, actual, ["event"])
            self.assertEqual(result["metrics"]["g_map50"], .5)
            self.assertEqual(result["positive_only"]["g_map50"], 1.)

    def test_changed_target_rejected(self):
        rows = self.rows()
        for field, value in (("category", "different"), ("bbox_1000", [0, 0, 11, 10])):
            changed = copy.deepcopy(rows)
            changed[0]["target"][field] = value
            with self.assertRaises(ValueError):
                check_targets(changed, rows)
        with self.assertRaises(ValueError):
            check_targets(rows[:1], rows)

    def test_partial_epoch_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(dict(protocol="session_disjoint", checkpoint_epoch=4, rows=self.rows())))
            with self.assertRaises(ValueError):
                load_result(path, "session_disjoint", 12)
            path.write_text(json.dumps(dict(protocol="session_disjoint", checkpoint_epoch=12, rows=[self.rows()[0]]*2)))
            with self.assertRaises(ValueError):
                load_result(path, "session_disjoint", 12)

    def test_stale_metrics_rejected(self):
        rows = self.rows()
        data = dict(metrics=compute(rows, ["event"], .5))
        data["metrics"]["g_map50"] = 1.
        with self.assertRaisesRegex(ValueError, "g_map50"):
            summarize(Path("unused"), data, rows, ["event"])

    def test_missing_rows_remain_pending(self):
        table = matched_table({"matched": {name: None for name in VARIANTS}})
        self.assertEqual(table.count(r"\pending"), 35)


class ShiftTests(unittest.TestCase):
    def config(self, path):
        protocol = "unseen_site" if "unseen_site" in str(path) else "forward_temporal"
        return {"data": {"protocols": [protocol]},
                "output": {k: str(path) + k for k in ("checkpoint", "test_results", "calibration", "results")}}

    def test_independent_queue(self):
        with patch.object(shifts, "load_yaml_with_base", side_effect=self.config):
            ground = shifts.plan(("unseen_site",), ("ground",))
            qwen = shifts.plan(("forward_temporal",), ("qwen",))
            self.assertEqual(len(ground), 3)
            self.assertEqual(len(qwen), 2)
            self.assertTrue(all("unseen_site_ground" in item[3].stem for item in ground))
            self.assertTrue(all("forward_temporal_qwen" in item[3].stem for item in qwen))
            self.assertEqual(len(shifts.plan()), 14)

    def test_single_stage_and_dedup(self):
        with patch.object(shifts, "load_yaml_with_base", side_effect=self.config):
            self.assertEqual(len(shifts.plan(("unseen_site",), ("ground_ms",))), 1)
            self.assertEqual(len(shifts.plan(("unseen_site",), ("ground", "ground_ms"))), 3)

    def test_dependencies(self):
        self.assertEqual(shifts.dependency_names(Path("unseen_site_tiling_results.json")),
                         ["unseen_site_qwen_results", "unseen_site_tiling_calibration"])
        self.assertEqual(shifts.dependency_names(Path("forward_temporal_ground_cls_checkpoint.json")),
                         ["forward_temporal_ground_ms_checkpoint"])

    def test_missing_dependency_fails_without_starting_jobs(self):
        marker = Path("unseen_site_ground_cls_checkpoint.json")
        catalog = {"unseen_site_ground_ms_checkpoint": ([], {}, Path("not-there"), Path("not-there-receipt"))}
        with self.assertRaisesRegex(ValueError, "No dependency is started"):
            shifts.require_dependencies(marker, catalog)
        shifts.require_dependencies(marker, catalog, scheduled=catalog)


if __name__ == "__main__":
    unittest.main()
