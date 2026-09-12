"""CPU-only action-policy invariants; does not load a VLM or image data."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from perception_actions_core import (ActionPolicy, aligned_iou, apply_action,
    grpo_loss, group_advantages, imitation_loss, rollout, trajectory_logits,
    trajectory_reward, valid_boxes)
from perception_actions_core import update_rollout_buffer
from perception_actions_evaluation import (CALIBRATION_VERSION, calibration_preflight,
    checkpoint_fingerprint, validate_calibration, valid_final_event)

SETTINGS = {"max_steps": 6, "center_step": 0.25, "log_scale_step": 0.35,
            "decay": 0.75, "minimum_size": 1e-4, "oracle_stop_margin": 1e-4}


class ConstantPolicy(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, features, boxes, step, max_steps):
        logits = boxes.new_full((len(boxes), 9), -100)
        logits[:, self.action] = 100
        return logits


class BiasPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(9))

    def forward(self, features, boxes, step, max_steps):
        return self.bias[None, :].expand(len(boxes), -1)


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "best"
        self.checkpoint.mkdir()
        self.weights = self.checkpoint / "action_policy.pt"
        self.weights.write_bytes(b"first")
        (self.checkpoint / "processor_config.json").write_text('{"size": 64}')
        self.calibration_path = self.root / "calibration.json"
        self.calibration = {"schema_version": CALIBRATION_VERSION,
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_fingerprint": checkpoint_fingerprint(self.checkpoint),
            "threshold": .5, "limited_run": False}

    def _save(self, calibration=None):
        self.calibration_path.write_text(json.dumps(calibration or self.calibration), encoding="utf-8")

    def test_same_path_same_size_weight_replacement_invalidates_calibration(self):
        self._save()
        original = calibration_preflight("test", self.calibration_path, self.checkpoint)
        stat = self.weights.stat()
        self.weights.write_bytes(b"other")
        os.utime(self.weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(original, checkpoint_fingerprint(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "contents changed"):
            calibration_preflight("test", self.calibration_path, self.checkpoint)

    def test_processor_changes_are_included_in_checkpoint_fingerprint(self):
        original = checkpoint_fingerprint(self.checkpoint)
        (self.checkpoint / "processor_config.json").write_text('{"size": 32}')
        self.assertNotEqual(original, checkpoint_fingerprint(self.checkpoint))

    def test_legacy_calibration_cannot_be_silently_reused(self):
        for missing in ["schema_version", "checkpoint_fingerprint", "limited_run"]:
            with self.subTest(missing=missing):
                old = self.calibration.copy()
                old.pop(missing)
                self._save(old)
                with self.assertRaisesRegex(ValueError, "Legacy/incomplete"):
                    calibration_preflight("test", self.calibration_path, self.checkpoint)

    def test_preflight_rejects_tiny_calibration_before_full_test(self):
        self.calibration["limited_run"] = True
        self._save()
        with self.assertRaisesRegex(ValueError, "tiny-run calibration"):
            calibration_preflight("test", self.calibration_path, self.checkpoint)
        calibration_preflight("test", self.calibration_path, self.checkpoint, max_test_samples=2)

    def test_all_split_limits_are_checked_before_checkpoint_reading(self):
        with self.assertRaisesRegex(ValueError, "tiny validation followed by full test"):
            calibration_preflight("all", self.calibration_path, self.root / "does-not-exist", max_val_samples=2)
        # Both full/full and tiny/tiny generate fresh validation calibration.
        calibration_preflight("all", self.calibration_path, self.checkpoint)
        calibration_preflight("all", self.calibration_path, self.checkpoint, max_val_samples=2, max_test_samples=2)

    def test_final_event_requires_finite_ordered_valid_box(self):
        event = {"category": "event", "bbox_1000": [10, 20, 30, 40], "presence_score": .8, "valid": True}
        self.assertTrue(valid_final_event(event, .5, ["event"]))
        bad_boxes = [None, [10, 20, 30], [10, 20, 10, 40], [30, 20, 10, 40],
                     [10, 40, 30, 20], [float("nan"), 20, 30, 40], [10, 20, float("inf"), 40],
                     [-1, 20, 30, 40], [10, 20, 1001, 40]]
        for box in bad_boxes:
            with self.subTest(box=box):
                self.assertFalse(valid_final_event(event | {"bbox_1000": box}, .5, ["event"]))
        for change in [{"presence_score": float("nan")}, {"presence_score": float("inf")},
                       {"presence_score": .2}, {"valid": False}, {"category": None},
                       {"category": "unsupported"}, {"category": "no_event"}]:
            with self.subTest(change=change):
                self.assertFalse(valid_final_event(event | change, .5, ["event"]))


class ActionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.features = torch.randn(3, 12)
        self.boxes = torch.tensor([[.2, .2, .6, .6], [.001, .001, .002, .004], [.0, .0, 1., 1.]])

    def test_every_action_has_valid_ordered_bounds(self):
        boxes = self.boxes
        for step in range(30):
            for action in range(9):
                result = apply_action(boxes, torch.full((len(boxes),), action), step % 6, SETTINGS)
                self.assertTrue(torch.isfinite(result).all())
                self.assertTrue(((result >= 0) & (result <= 1)).all())
                self.assertTrue((result[:, 2:] > result[:, :2]).all())
            boxes = result

    def test_illegal_action_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_action(self.boxes, torch.tensor([0, 9, 1]), 0, SETTINGS)

    def test_stop_is_terminal_and_box_is_unchanged(self):
        result = rollout(ConstantPolicy(0), self.features, self.boxes, SETTINGS, greedy=True)
        torch.testing.assert_close(result["boxes"], self.boxes)
        self.assertTrue(result["mask"][:, 0].all())
        self.assertFalse(result["mask"][:, 1:].any())
        torch.testing.assert_close(result["log_probs"][:, 1:], torch.zeros(3, 5))

    def test_disabled_rows_never_act(self):
        enabled = torch.tensor([True, False, True])
        result = rollout(ConstantPolicy(2), self.features, self.boxes, SETTINGS, greedy=True, enabled=enabled)
        self.assertFalse(result["mask"][1].any())
        torch.testing.assert_close(result["boxes"][1], self.boxes[1])

    def test_coarse_to_fine_reduces_step_distance(self):
        box = self.boxes[:1]
        action = torch.tensor([2])
        first = (apply_action(box, action, 0, SETTINGS) - box).abs().sum()
        last = (apply_action(box, action, 5, SETTINGS) - box).abs().sum()
        self.assertGreater(first, last)

    def test_multistep_log_probability_matches_policy(self):
        policy = ActionPolicy(12, 16)
        trajectory = rollout(policy, self.features, self.boxes, SETTINGS)
        logits = trajectory_logits(policy, self.features, trajectory, SETTINGS)
        selected = logits.log_softmax(-1).gather(-1, trajectory["actions"][..., None]).squeeze(-1)
        torch.testing.assert_close(trajectory["log_probs"], selected * trajectory["mask"])
        torch.testing.assert_close(trajectory["log_probs"].sum(1), (selected * trajectory["mask"]).sum(1))

    def test_group_advantages_do_not_mix_images(self):
        advantages = group_advantages(torch.tensor([1., 2., 10., 10.]), 2)
        self.assertLess(float(advantages[0]), 0)
        self.assertGreater(float(advantages[1]), 0)
        torch.testing.assert_close(advantages[2:], torch.zeros(2))

    def test_grpo_gradient_reaches_only_current_policy(self):
        policy = ActionPolicy(12, 16)
        old, reference = copy.deepcopy(policy).requires_grad_(False), copy.deepcopy(policy).requires_grad_(False)
        features = self.features.repeat_interleave(4, 0)
        boxes = self.boxes.repeat_interleave(4, 0)
        with torch.no_grad():
            trajectory = rollout(old, features, boxes, SETTINGS)
        advantages = torch.tensor([-1., -.3, .3, 1.]).repeat(3)
        loss, parts = grpo_loss(policy, reference, features, trajectory, advantages, SETTINGS, {"clip": .2, "beta": .03})
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in old.parameters()))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        self.assertAlmostEqual(parts["ratio"], 1.0, places=5)
        self.assertAlmostEqual(parts["kl"], 0.0, places=5)

    def test_reward_prefers_improvement_and_masks_ineligible(self):
        initial = torch.tensor([[.0, .0, .4, .4], [.0, .0, .4, .4]])
        target = torch.tensor([[.2, .2, .6, .6], [.2, .2, .6, .6]])
        trajectory = {"actions": torch.tensor([[2, 4], [2, 4]]), "mask": torch.ones(2, 2, dtype=torch.bool)}
        config = {"final_iou": 1., "iou_gain": 1., "step_cost": 0.}
        bad = trajectory_reward(initial, initial, target, torch.tensor([True, False]), trajectory, config)
        good = trajectory_reward(initial, target, target, torch.tensor([True, False]), trajectory, config)
        self.assertGreater(float(good[0]), float(bad[0]))
        self.assertEqual(float(good[1]), 0)

    def test_imitation_trains_policy_and_excludes_negative_rows(self):
        policy = ActionPolicy(12, 16)
        features = self.features.clone().requires_grad_(True)
        target = self.boxes.clone()
        loss = imitation_loss(policy, features, self.boxes, target, torch.tensor([True, False, True]), SETTINGS)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in policy.parameters() if p.grad is not None), 0)
        torch.testing.assert_close(features.grad[1], torch.zeros(12))

    def _frozen_stop_buffer(self, old_policy):
        features, boxes = self.features[:1], self.boxes[:1]
        with torch.no_grad():
            # Uniform behavior can legally emit STOP. Greedy collection makes
            # this numerical clipping test deterministic, not an RL recipe.
            trajectory = rollout(old_policy, features, boxes, SETTINGS, greedy=True)
        return {"features": features, "initial": boxes, "target": boxes,
                "eligible": torch.ones(1, dtype=torch.bool), "grouped_features": features,
                "trajectory": trajectory, "advantages": torch.ones(1), "rewards": torch.ones(1)}

    def test_reused_rollout_changes_ratio_and_activates_clipping(self):
        policy = BiasPolicy()
        reference = copy.deepcopy(policy).requires_grad_(False)
        item = self._frozen_stop_buffer(reference)
        original_old_probs = item["trajectory"]["log_probs"].clone()
        optimizer = torch.optim.SGD(policy.parameters(), lr=3.0)
        scheduler = CountingScheduler()
        config = {"clip": .2, "beta": 0., "updates_per_rollout": 2, "imitation_anchor": 0., "max_grad_norm": 100.}
        history = update_rollout_buffer(policy, reference, [item], SETTINGS, config, optimizer, scheduler)
        self.assertEqual(len(history), 2)
        self.assertEqual(scheduler.steps, 2)
        self.assertAlmostEqual(history[0]["ratio"], 1., places=5)
        self.assertGreater(history[1]["ratio"], 1.2)
        self.assertAlmostEqual(history[1]["clip_fraction"], 1., places=5)
        self.assertAlmostEqual(history[1]["loss"], -1.2, places=5)
        torch.testing.assert_close(item["trajectory"]["log_probs"], original_old_probs)
        torch.testing.assert_close(reference.bias, torch.zeros(9))
        # The second optimizer step is clipped; it cannot further grow STOP's
        # probability when KL and the imitation anchor are disabled.
        expected_bias = torch.full((9,), -3. / 9)
        expected_bias[0] = 3. * 8. / 9
        torch.testing.assert_close(policy.bias, expected_bias)

    def test_short_buffer_normalization_and_optimizer_budget(self):
        first, second = BiasPolicy(), BiasPolicy()
        reference = copy.deepcopy(first).requires_grad_(False)
        item = self._frozen_stop_buffer(reference)
        config = {"clip": .2, "beta": 0., "updates_per_rollout": 4, "imitation_anchor": 0., "max_grad_norm": 100.}
        one_scheduler, two_scheduler = CountingScheduler(), CountingScheduler()
        one = update_rollout_buffer(first, reference, [item], SETTINGS, config,
            torch.optim.SGD(first.parameters(), lr=.1), one_scheduler, max_updates=1)
        two = update_rollout_buffer(second, reference, [item, item], SETTINGS, config,
            torch.optim.SGD(second.parameters(), lr=.1), two_scheduler, max_updates=1)
        self.assertEqual(len(one), 1)
        self.assertEqual(len(two), 1)
        self.assertEqual(one_scheduler.steps, 1)
        self.assertEqual(two_scheduler.steps, 1)
        torch.testing.assert_close(first.bias, second.bias)


if __name__ == "__main__":
    unittest.main()
