"""Torch-only spatial geometry, padding, gradients, and checkpoint tests."""
import sys
import tempfile
import unittest
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from perception_spatial_head import (SpatialInteraction, spatial_memory, save_spatial_heads, load_spatial_heads,
    checkpoint_fingerprint, evaluation_configuration, evaluation_signature, validate_calibration, final_output_fpr)


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.hidden = torch.randn(2, 12, 16, requires_grad=True)
        self.ids = torch.tensor([[0, 0, 7, 7, 7, 7, 2, 3, 4, 8, 9, 10],
                                 [0, 1, 2, 7, 7, 7, 7, 7, 7, 4, 8, 9]])
        self.mask = self.ids.ne(0).long()
        self.grid = torch.tensor([[1, 4, 4], [1, 4, 6]])

    def memory(self, maximum=2048):
        return spatial_memory(self.hidden, self.ids, self.mask, 7, self.grid, 2, maximum)

    def test_mixed_grid_geometry_padding_and_gradients(self):
        memory, coords, padding = self.memory()
        self.assertEqual(tuple(memory.shape), (2, 6, 16))
        self.assertEqual(padding.tolist(), [[False] * 4 + [True] * 2, [False] * 6])
        torch.testing.assert_close(memory[0, :4], self.hidden[0, 2:6])
        torch.testing.assert_close(memory[1], self.hidden[1, 3:9])
        torch.testing.assert_close(coords[0, :4, :2], torch.tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]]))
        torch.testing.assert_close(coords[1, :3, 0], torch.tensor([-2/3, 0., 2/3]))
        self.assertTrue(torch.equal(coords[:, :, 2], torch.zeros(2, 6)))
        memory.sum().backward()
        self.assertEqual(float(self.hidden.grad[0, 2:6].sum()), 64.)
        self.assertEqual(float(self.hidden.grad[0, :2].sum()), 0.)

    def test_padding_never_contributes(self):
        head = SpatialInteraction(16, 8, 2, dropout=0., residual_gate_init=.5).eval()
        memory, coords, padding = self.memory()
        query = torch.randn(2, 16)
        expected = head(query, memory, coords, padding)
        corrupted_memory, corrupted_coords = memory.detach().clone(), coords.clone()
        corrupted_memory[padding] = 10000
        corrupted_coords[padding] = -10000
        torch.testing.assert_close(head(query, corrupted_memory, corrupted_coords, padding), expected)

    def test_gradients_reach_query_image_projection_and_image_states(self):
        head = SpatialInteraction(16, 8, 2, dropout=0., residual_gate_init=.5)
        memory, coords, padding = self.memory()
        query = torch.randn(2, 16, requires_grad=True)
        head(query, memory, coords, padding).square().sum().backward()
        for tensor in [query, self.hidden, head.query_proj.weight, head.image_proj.weight, head.position[0].weight]:
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_zero_gate_preserves_baseline_and_starts_learning(self):
        head = SpatialInteraction(16, 8, 2, dropout=0.)
        query = torch.randn(2, 16)
        actual = head(query, *self.memory())
        torch.testing.assert_close(actual, query, rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertGreater(head.residual_gate.grad.abs().item(), 0)

    def test_subsampling_retains_original_coordinates(self):
        full_memory, full_coords, _ = self.memory()
        memory, coords, padding = self.memory(maximum=3)
        selected = torch.linspace(0, 5, 3).round().long()
        torch.testing.assert_close(memory[1], full_memory[1, selected])
        torch.testing.assert_close(coords[1], full_coords[1, selected])
        self.assertFalse(padding.any())

    def test_wrong_grid_and_hidden_mask_are_rejected(self):
        wrong = self.grid.clone()
        wrong[0] = torch.tensor([1, 2, 2])
        with self.assertRaises(ValueError):
            spatial_memory(self.hidden, self.ids, self.mask, 7, wrong)
        wrong_mask = self.mask.clone()
        wrong_mask[0, 2] = 0
        with self.assertRaises(ValueError):
            spatial_memory(self.hidden, self.ids, wrong_mask, 7, self.grid)

    def test_checkpoint_roundtrip_and_baseline_warmstart(self):
        def model():
            value = nn.Module()
            value.box_head = nn.Linear(16, 4)
            value.spatial_head = SpatialInteraction(16, 8, 2, dropout=0., residual_gate_init=.5)
            return value
        source, restored = model(), model()
        with tempfile.TemporaryDirectory() as directory:
            save_spatial_heads(source, directory)
            load_spatial_heads(restored, directory, require_spatial=True)
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            Path(directory, "spatial_head.pt").unlink()
            restored = model()
            before = {k: v.clone() for k, v in restored.spatial_head.state_dict().items()}
            load_spatial_heads(restored, directory)
            for name, value in before.items():
                torch.testing.assert_close(value, restored.spatial_head.state_dict()[name], rtol=0, atol=0)
            with self.assertRaises(FileNotFoundError):
                load_spatial_heads(restored, directory, require_spatial=True)

    def test_eval_limits_do_not_leak_from_training(self):
        saved = {'max_train_samples': 4, 'max_val_samples': 2, 'max_test_samples': 2,
                 'device': 'cuda', 'test': {'batch_size': 2}, 'output': 'old'}
        runtime = {'device': 'cpu', 'test': {'batch_size': 1}, 'output': 'new'}
        actual = evaluation_configuration(saved, runtime)
        self.assertNotIn('max_val_samples', actual)
        self.assertNotIn('max_test_samples', actual)
        self.assertEqual(actual['training_sample_limits']['max_val_samples'], 2)
        self.assertEqual(actual['output'], 'new')
        self.assertEqual(saved['max_val_samples'], 2)
        explicit = evaluation_configuration(saved, runtime | {'max_val_samples': 3, 'max_test_samples': 5})
        self.assertEqual((explicit['max_val_samples'], explicit['max_test_samples']), (3, 5))

    def test_checkpoint_bound_calibration_rejects_changed_head_and_tiny_full_test(self):
        config = {'protocol': 'session_disjoint', 'seed': 43, 'data': {}, 'prompt': {}, 'max_test_samples': 2}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / 'box_head.pt').write_bytes(b'box')
            (checkpoint / 'spatial_head.pt').write_bytes(b'spatial')
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'adapter')
            fingerprint = checkpoint_fingerprint(checkpoint)
            calibration = {'split': 'val', 'protocol': config['protocol'], 'seed': 43,
                           'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': fingerprint,
                           'evaluation_signature': evaluation_signature(config), 'threshold': .5, 'limited_run': True}
            validate_calibration(calibration, config, checkpoint, fingerprint)
            full = config | {'max_test_samples': None}
            with self.assertRaisesRegex(ValueError, 'tiny-run'):
                validate_calibration(calibration, full, checkpoint, fingerprint)
            with self.assertRaises(ValueError):
                validate_calibration(calibration, config | {'prompt': {'user': 'changed'}}, checkpoint, fingerprint)
            (checkpoint / 'spatial_head.pt').write_bytes(b'updated-spatial')
            self.assertNotEqual(checkpoint_fingerprint(checkpoint), fingerprint)
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                validate_calibration(calibration, config, checkpoint, checkpoint_fingerprint(checkpoint))
            # Baseline threshold selection supports an all-reject sentinel.
            validate_calibration(calibration | {'threshold': math.nextafter(1., math.inf)}, config, checkpoint, fingerprint)

    def test_final_output_fpr_requires_supported_valid_class_box_and_score(self):
        good = {'category': 'event', 'bbox_1000': [10., 20., 800., 900.], 'presence_score': .9, 'valid': True}
        predictions = [good, good | {'category': None}, good | {'bbox_1000': [800, 20, 10, 900]},
                       good | {'category': 'unsupported'}, good | {'presence_score': .1}, good]
        samples = [SimpleNamespace(presence=False) for _ in range(5)] + [SimpleNamespace(presence=True)]
        self.assertEqual(final_output_fpr(samples, predictions, .5, ['event']), .2)
        self.assertIsNone(final_output_fpr([SimpleNamespace(presence=True)], [good], .5, ['event']))

    def test_baseline_predict_replays_through_model_forward(self):
        # Import the actual baseline with lightweight data/tokenizer dependencies.
        # The wrapper must be called; calling vlm directly would bypass spatial.
        source = Path(__file__).resolve().parents[1] / 'scripts' / 'test_perception_qwen.py'
        if not source.is_file():
            source = Path(__file__).resolve().parents[3] / 'perception_qwen_implementation' / 'scripts' / 'test_perception_qwen.py'
        if not source.is_file():
            self.skipTest('Baseline source is not available in this checkout')
        class Tokenizer:
            eos_token_id, pad_token_id = 9, 0
            def encode(self, text, **kwargs): return [3, 8]
            def batch_decode(self, tokens, **kwargs): return ['event'] * len(tokens)
            def decode(self, tokens, **kwargs): return 'event<vis>'
        class Collator:
            def __init__(self, *args, **kwargs): pass
            def __call__(self, samples):
                return ({'input_ids': torch.tensor([[1, 7, 2]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long),
                         'mm_token_type_ids': torch.tensor([[0, 1, 0]])}, None, None)
        class FakeVLM:
            def generate(self, **kwargs):
                return SimpleNamespace(sequences=torch.tensor([[1, 7, 2, 3, 8, 9]]), scores=[torch.zeros(1, 12)])
            def __call__(self, *args, **kwargs): raise AssertionError('Spatial forward was bypassed')
        class FakeModel:
            vis_id = 8
            vlm = FakeVLM()
            def eval(self): return self
            def __call__(self, inputs):
                self.replay = inputs
                return None, torch.tensor([[.1, .2, .8, .9]])
        baseline = ModuleType('train_perception_qwen')
        for name in ('build_model', 'output_path', 'read_config'):
            setattr(baseline, name, lambda *args: None)
        baseline.VIS, baseline.DEFAULT_CONFIG, baseline.Collator = '<vis>', '', Collator
        constraints = ModuleType('clear_uav.generation_constraints')
        constraints.label_prefix_allowed_tokens = lambda *args, **kwargs: None
        table4 = ModuleType('clear_uav.table4')
        for name in ('definitions_from_config', 'read_discovery_samples', 'save_results', 'select_threshold', 'table4_metrics'):
            setattr(table4, name, lambda *args, **kwargs: None)
        table4.event_probability = lambda *args: .9
        progress = ModuleType('tqdm')
        progress.tqdm = lambda sequence, **kwargs: sequence
        modules = {'train_perception_qwen': baseline, 'clear_uav': ModuleType('clear_uav'),
                   'clear_uav.generation_constraints': constraints, 'clear_uav.table4': table4, 'tqdm': progress}
        with mock.patch.dict(sys.modules, modules):
            spec = importlib.util.spec_from_file_location('_spatial_predict_contract', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = FakeModel()
            predictions = module.predict(model, SimpleNamespace(tokenizer=Tokenizer()), [object()],
                                         {'device': 'cpu', 'test': {'batch_size': 1}}, ['event'])
        self.assertEqual(model.replay['input_ids'].tolist(), [[1, 7, 2, 3, 8, 9]])
        self.assertEqual(model.replay['mm_token_type_ids'].tolist(), [[0, 1, 0, 0, 0, 0]])
        self.assertTrue(predictions[0]['valid'])
        self.assertEqual(predictions[0]['category'], 'event')
        self.assertAlmostEqual(predictions[0]['bbox_1000'][0], 100., places=4)


if __name__ == "__main__":
    unittest.main()
