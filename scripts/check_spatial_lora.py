"""CPU-only geometry, gradient, hook and checkpoint checks; no Qwen weights."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from transformers.utils import ModelOutput

from perception_spatial_lora import SpatialLowRankResidual, VisualSpatialAdapter
from perception_spatial_head import (
    SpatialInteraction, checkpoint_fingerprint, evaluation_signature,
    load_spatial_heads, save_spatial_heads,
)


@dataclass
class VisualOutput(ModelOutput):
    last_hidden_state: torch.Tensor = None
    pooler_output: torch.Tensor = None
    deepstack_features: list = None


class TinyVisual(nn.Module):
    def forward(self, hidden_states, grid_thw=None):
        self.original = VisualOutput(hidden_states, hidden_states * 2,
                                     [hidden_states * 3, hidden_states * 4])
        return self.original


def tiny_heads(with_adapter=True):
    model = nn.Module()
    model.box_head = nn.Linear(8, 4)
    model.spatial_head = SpatialInteraction(8, 4, 2, dropout=0)
    model.spatial_adapter = (VisualSpatialAdapter(8, 1, 2, rank=4, dropout=0)
                             if with_adapter else None)
    return model


class SpatialLoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_zero_identity_and_gradient_startup(self):
        adapter = SpatialLowRankResidual(8, rank=4, dropout=0)
        features = torch.randn(12, 8, requires_grad=True)
        weights = torch.randn_like(features)
        output = adapter(features, [(1, 2, 3), (1, 3, 2)])
        torch.testing.assert_close(output, features, rtol=0, atol=0)
        (output * weights).sum().backward()
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0)
        for layer in (adapter.down, adapter.position, adapter.local):
            self.assertEqual(layer.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(features.grad, weights)
        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.up.weight.normal_(std=0.1)
        (adapter(features, [(2, 2, 3)]) * weights).sum().backward()
        for layer in (adapter.down, adapter.position, adapter.local, adapter.up):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_rectangular_images_and_frames_are_independent(self):
        adapter = SpatialLowRankResidual(8, rank=4, dropout=0).eval()
        with torch.no_grad():
            adapter.up.weight.normal_(std=0.2)
        first, second = torch.randn(12, 8), torch.randn(6, 8)
        joined = adapter(torch.cat((first, second)), [(2, 2, 3), (1, 3, 2)])
        independent = torch.cat((adapter(first, [(2, 2, 3)]),
                                 adapter(second, [(1, 3, 2)])))
        torch.testing.assert_close(joined, independent)
        frames = torch.cat([adapter(frame, [(1, 2, 3)]) for frame in first.split(6)])
        torch.testing.assert_close(joined[:12], frames)

    def test_coordinates_are_xy_with_x_fastest(self):
        adapter = SpatialLowRankResidual(2, rank=2, alpha=2, dropout=0)
        with torch.no_grad():
            adapter.down.weight.zero_()
            adapter.local.weight.zero_()
            adapter.position.weight.copy_(torch.eye(2))
            adapter.up.weight.copy_(torch.eye(2))
        xy = torch.tensor([[-2/3, -0.5], [0, -0.5], [2/3, -0.5],
                           [-2/3, 0.5], [0, 0.5], [2/3, 0.5]])
        output = adapter(torch.zeros(12, 2), [(2, 2, 3)])
        torch.testing.assert_close(output, torch.nn.functional.silu(xy).repeat(2, 1))

    def test_hook_copies_modeloutput_and_handles_both_grid_styles(self):
        adapter = VisualSpatialAdapter(8, 2, merge_size=2, rank=4, dropout=0)
        with torch.no_grad():
            for branch in adapter.branches:
                branch.up.weight.normal_(std=0.1)
        visual = TinyVisual()
        handle = visual.register_forward_hook(adapter.inject, with_kwargs=True)
        features = torch.randn(12, 8)
        grids = torch.tensor([[1, 4, 6], [1, 6, 4]])
        result = visual(features, grid_thw=grids)
        original = visual.original
        self.assertIsInstance(result, VisualOutput)
        self.assertIsNot(result, original)
        self.assertIs(result.last_hidden_state, original.last_hidden_state)
        self.assertIsNot(result.deepstack_features, original.deepstack_features)
        for index, stream in enumerate([result.pooler_output, *result.deepstack_features]):
            raw = features * (index + 2)
            expected = adapter.branches[index](raw, [(1, 2, 3), (1, 3, 2)])
            torch.testing.assert_close(stream, expected)
            self.assertFalse(torch.equal(stream, raw))
        torch.testing.assert_close(original.pooler_output, features * 2)
        torch.testing.assert_close(original.deepstack_features[0], features * 3)
        self.assertIs(result['pooler_output'], result.pooler_output)
        positional = visual(features, grids)
        torch.testing.assert_close(positional.pooler_output, result.pooler_output)
        handle.remove()

    def test_checkpoint_round_trip_and_fingerprint_include_adapter(self):
        with TemporaryDirectory() as directory:
            source, target = tiny_heads(), tiny_heads()
            with torch.no_grad():
                source.spatial_adapter.branches[0].up.weight.normal_()
            save_spatial_heads(source, directory)
            load_spatial_heads(target, directory, require_spatial=True)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
            before = checkpoint_fingerprint(directory)
            with torch.no_grad():
                source.spatial_adapter.branches[0].up.weight.add_(1)
            torch.save(source.spatial_adapter.state_dict(), Path(directory) / 'spatial_adapter.pt')
            self.assertNotEqual(checkpoint_fingerprint(directory), before)

    def test_legacy_spatial_and_box_only_checkpoints(self):
        with TemporaryDirectory() as directory:
            source, target = tiny_heads(False), tiny_heads(False)
            save_spatial_heads(source, directory)
            self.assertFalse((Path(directory) / 'spatial_adapter.pt').exists())
            load_spatial_heads(target, directory, require_spatial=True)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
        with TemporaryDirectory() as directory:
            torch.save(source.box_head.state_dict(), Path(directory) / 'box_head.pt')
            load_spatial_heads(target, directory, require_spatial=False)
            with self.assertRaises(FileNotFoundError):
                load_spatial_heads(target, directory, require_spatial=True)

    def test_legacy_evaluation_signature_is_unchanged(self):
        config = {'protocol': 'session_disjoint', 'seed': 43, 'model': {'path': 'base'},
                  'spatial': {'interaction_dim': 256}, 'unrelated': 'ignored'}
        keys = ('protocol', 'seed', 'model', 'spatial', 'input', 'prompt', 'data', 'validation')
        old_signature = hashlib.sha256(json.dumps(
            {key: config.get(key) for key in keys}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(evaluation_signature(config), old_signature)
        config['spatial_lora'] = {'enabled': True, 'rank': 32}
        self.assertNotEqual(evaluation_signature(config), old_signature)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
