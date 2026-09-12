"""CPU-only geometry/leakage contracts; no model weights or training required."""
import importlib.util
import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_perception_zoom.py"
spec = importlib.util.spec_from_file_location("perception_zoom_contract", SCRIPT)
zoom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zoom)


class ZoomContracts(unittest.TestCase):
    def test_crop_clamps_edges_and_preserves_minimum_side(self):
        bounds = zoom.crop_bounds([0.99, 0.98, 1.02, 1.05], (1920, 1080))
        self.assertTrue(0 <= bounds[0] < bounds[2] <= 1920)
        self.assertTrue(0 <= bounds[1] < bounds[3] <= 1080)
        self.assertGreaterEqual(bounds[2] - bounds[0], 96)
        self.assertGreaterEqual(bounds[3] - bounds[1], 96)

    def test_tiny_image_bounds(self):
        self.assertEqual(zoom.crop_bounds([0.2, 0.2, 0.3, 0.3], (32, 24)), (0, 0, 32, 24))

    def test_negative_and_invalid_proposals_stop(self):
        for box in [None, [], [0, 0, 0, 0], [0.8, 0.8, 0.2, 0.2],
                    [float("nan"), 0, 1, 1], [-2, -2, -1, -1], [0, 0, math.inf, 1]]:
            self.assertIsNone(zoom.crop_bounds(box, (1920, 1080)))
        self.assertFalse(zoom.should_review(0.99, False))
        self.assertFalse(zoom.should_review(float("nan"), True))

    def test_negative_prediction_cannot_request_target_derived_crop(self):
        config = {"zoom": {"crop_expansion": 1.8, "min_crop_side": 96, "min_crop_fraction": 0.12}}
        row = {"prediction": {"valid": True, "category": None, "bbox_1000": [100, 100, 500, 500]},
               "image_size": [1920, 1080]}
        self.assertIsNone(zoom.proposal_bounds(row, config))
        row["prediction"]["category"] = "some_event"
        self.assertIsNotNone(zoom.proposal_bounds(row, config))
        row["prediction"]["valid"] = False
        self.assertIsNone(zoom.proposal_bounds(row, config))

    def test_source_fingerprint_detects_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box_head.pt"
            path.write_bytes(b"old head")
            before = zoom.checkpoint_digest(directory)
            path.write_bytes(b"new head")
            self.assertNotEqual(before, zoom.checkpoint_digest(directory))

    def test_calibration_binds_checkpoint_and_blocks_tiny_to_full_test(self):
        config = {"protocol": "session_disjoint", "seed": 43, "zoom": {"proposal_sha256": "proposal"}, "data": {}}
        checkpoint = Path("best").resolve()
        calibration = {"checkpoint": str(checkpoint), "checkpoint_sha256": "model", "protocol": "session_disjoint",
                       "seed": 43, "proposal_sha256": "proposal", "limited_run": True}
        zoom.validate_calibration(calibration, config, checkpoint, "model", 2)
        with self.assertRaisesRegex(ValueError, "tiny-run"):
            zoom.validate_calibration(calibration, config, checkpoint, "model", None)
        with self.assertRaisesRegex(ValueError, "content"):
            zoom.validate_calibration(calibration, config, checkpoint, "different-model", 2)
        calibration["limited_run"] = False
        zoom.validate_calibration(calibration, config, checkpoint, "model", None)

    def test_evaluation_limits_are_explicit_and_do_not_inherit_training_smoke(self):
        saved = {"device": "cuda", "test": {"batch_size": 1}, "output": "saved", "model": {"path": "trained-model"},
                 "limits": {"train": 2, "val": 2, "test": 2}}
        runtime = {"device": "cpu", "test": {"batch_size": 3}, "output": "run", "seed": 43, "protocol": "p",
                   "model": {"path": "wrong-model"}}
        with patch.object(zoom, "read_config", side_effect=lambda path: copy.deepcopy(saved)):
            actual = zoom.evaluation_config(runtime)
            self.assertEqual(actual["limits"], {})
            self.assertEqual(actual["training_limits"], {"train": 2, "val": 2, "test": 2})
            self.assertEqual(actual["model"]["path"], "trained-model")
            runtime["limits"] = {"test": 4}
            self.assertEqual(zoom.evaluation_config(runtime)["limits"], {"test": 4})

    def test_final_output_fpr_retains_real_output_requirements(self):
        samples = [SimpleNamespace(presence=False) for _ in range(3)]
        predictions = [
            {"category": None, "bbox_1000": None, "presence_score": 0.9, "valid": True},
            {"category": "event", "bbox_1000": [100, 100, 100, 500], "presence_score": 0.9, "valid": True},
            {"category": "event", "bbox_1000": [100, 100, 500, 500], "presence_score": 0.9, "valid": True}]
        self.assertAlmostEqual(zoom.final_output_fpr(samples, predictions, 0.5), 1 / 3)

    def test_coordinate_roundtrip(self):
        bounds, size = (100, 80, 1500, 900), (1920, 1080)
        original = [0.02, 0.03, 0.98, 0.96]  # deliberately extends outside crop
        local = zoom.original_to_crop(original, bounds, size)
        restored = zoom.crop_to_original(local, bounds, size, clamp=False)
        for a, b in zip(original, restored):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(zoom.crop_to_original([-9, -9, 9, 9], bounds, size), [0, 0, 1, 1])

    def test_gate_metadata_and_inference_boundary_do_not_read_ground_truth(self):
        class ImageOnly:
            image_path = Path("synthetic.png")
            record_uid = "row"
            @property
            def label(self):
                raise AssertionError("Inference read target category")
            @property
            def bbox_1000(self):
                raise AssertionError("Inference read target ROI")
            @property
            def presence(self):
                raise AssertionError("Inference read target presence")
        safe = zoom.inference_sample(ImageOnly())
        self.assertEqual(vars(safe), {"record_uid": "row", "image_path": Path("synthetic.png")})
        prediction = {"presence_score": 0.3, "bbox_1000": [100, 100, 300, 300]}
        values = zoom.gate_metadata(prediction, (10, 10, 30, 30), (100, 100), 0.1)
        self.assertEqual(len(values), 8)
        self.assertEqual(values[-1], 0.1)
        self.assertFalse(zoom.should_review(0.49, True))
        self.assertTrue(zoom.should_review(0.5, True))


if __name__ == "__main__":
    unittest.main()
