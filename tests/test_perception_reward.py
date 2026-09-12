"""CPU tensor and cache-boundary tests. No Transformers, dataset, or 8B weights."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))
import perception_reward_model as rm


class RewardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_roi_sampling_coordinates_and_gradients(self):
        # Full-image 2x2 bins on a 4x4 ramp sample at original pixel (0.5,0.5), etc.
        feature = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).requires_grad_()
        full_box = torch.tensor([[0., 0., 1., 1.]], requires_grad=True)
        result = rm.region_grid(feature, full_box, size=2)
        torch.testing.assert_close(result[0, 0], torch.tensor([[2.5, 4.5], [10.5, 12.5]]))
        result.sum().backward()
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertGreater(full_box.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(full_box.grad).all())

    def test_visual_model_uses_class_image_and_box_parameters(self):
        model = rm.VisualQualityModel(3, 2, hidden_dim=16, roi_size=2)
        features = torch.randn(2, 3, 5, 5, requires_grad=True)
        boxes = torch.tensor([[.1, .2, .6, .7], [.2, .1, .9, .8]], requires_grad=True)
        model(features, boxes, torch.tensor([0, 1])).sum().backward()
        for value in (features, boxes, model.class_embedding.weight, model.net[1].weight):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_candidates_are_valid_deterministic_and_class_conditioned(self):
        gt = torch.tensor([.2, .3, .6, .8])
        args = (gt, 1, 3, 12)
        a = rm.candidate_targets(*args, torch.Generator().manual_seed(9))
        b = rm.candidate_targets(*args, torch.Generator().manual_seed(9))
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y)
        boxes, classes, quality = a
        self.assertTrue(rm.validate_box(boxes).all())
        self.assertEqual(quality[0].item(), 1.)
        self.assertEqual(quality[1].item(), 0.)
        self.assertEqual(classes[0].item(), 1)
        self.assertNotEqual(classes[1].item(), 1)
        torch.testing.assert_close(quality, rm.aligned_iou(boxes, gt) * (classes == 1))
        _, _, negative_quality = rm.candidate_targets(None, 0, 3, 12, torch.Generator().manual_seed(2))
        self.assertEqual(negative_quality.sum().item(), 0.)
        with self.assertRaises(ValueError):
            rm.candidate_targets([.5, .1, .2, .6], 0, 3, 4, torch.Generator())

    def test_pairwise_loss_prefers_correct_order_and_handles_ties(self):
        targets = torch.tensor([[1., .4, 0.]])
        good = torch.tensor([[2., 0., -2.]], requires_grad=True)
        self.assertLess(rm.pairwise_loss(good, targets).item(), rm.pairwise_loss(-good, targets).item())
        rm.pairwise_loss(good, targets).backward()
        self.assertLess(good.grad[0, 0].item(), 0.)
        self.assertGreater(good.grad[0, 2].item(), 0.)
        tied = torch.ones(1, 3, requires_grad=True)
        loss = rm.pairwise_loss(tied, torch.zeros_like(tied))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(tied.grad.sum().item(), 0.)

    def fixture(self, directory):
        root = Path(directory)
        base, adapter, data, cache = (root / name for name in ('base', 'adapter', 'data', 'cache'))
        for path in (base, adapter, data / 'session_disjoint', cache / 'train', cache / 'val'):
            path.mkdir(parents=True)
        for name, content in {'config.json': '{}', 'model.safetensors': 'fake-base-weights',
                              'model.safetensors.index.json': '{"weight_map":{}}'}.items():
            (base / name).write_text(content)
        for name, content in {'adapter_config.json': '{}', 'adapter_model.safetensors': 'fake-adapter',
                              'box_head.pt': 'fake-head', 'preprocessor_config.json': '{"size":4}',
                              'tokenizer.json': '{}', 'chat_template.jinja': '{{messages}}'}.items():
            (adapter / name).write_text(content)
        for name, content in {'labels.txt': 'a\nb\n', 'definitions.json': '{}', 'boxes.json': '{}'}.items():
            (data / name).write_text(content)
        for split in ('train', 'val'):
            (data / 'session_disjoint' / f'{split}.csv').write_text(f'record_uid,session_id\n{split}_0,{split}_session\n')
        config = {'protocol': 'session_disjoint', 'seed': 43, 'input': {'max_pixels': 16}, 'prompt': {'user': 'classify'},
                  'model': {'path': str(base), 'initial_checkpoint': str(adapter)},
                  'data': {'root': str(data), 'labels': str(data / 'labels.txt'),
                           'definitions': str(data / 'definitions.json'), 'bbox_annotations': str(data / 'boxes.json')},
                  'reward_model': {'cache': str(cache), 'checkpoint': str(root / 'reward.pt'),
                                   'feature_channels': 2, 'grid_size': 4, 'train': {'epochs': 1}}}
        spec = rm.cache_spec(config)
        manifest = {'spec': spec, 'splits': {}}
        for split in ('train', 'val'):
            uid = f'{split}_0'
            dest = cache / split / f'{uid}.pt'
            image_path = data / f'{split}_image.fake'
            image_path.write_text(f'{split}_image_bytes')
            image_sha = rm.file_sha(image_path)
            torch.save({'features': torch.randn(2, 4, 4).half(), 'image_sha256': image_sha,
                        'spec_hash': rm.spec_hash(spec)}, dest)
            manifest['splits'][split] = [{'record_uid': uid, 'group_id': f'{split}_group',
                'session_id': f'{split}_session', 'image_path': str(image_path),
                'image_sha256': image_sha, 'feature_sha256': rm.file_sha(dest)}]
        (cache / 'manifest.json').write_text(json.dumps(manifest))
        architecture = {'channels': 2, 'num_classes': 2, 'hidden_dim': 16, 'roi_size': 2}
        model = rm.VisualQualityModel(**architecture)
        torch.save({'labels': ['a', 'b'], 'feature_spec_hash': rm.spec_hash(spec),
                    'training_metadata': rm.reward_training_metadata(config, manifest),
                    'architecture': architecture, 'state_dict': model.state_dict()}, config['reward_model']['checkpoint'])
        return config, manifest

    def test_model_processor_and_weight_stat_changes_invalidate_spec(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = self.fixture(directory)
            base = Path(config['model']['path'])
            with mock.patch.object(rm, 'file_sha', wraps=rm.file_sha) as spy:
                rm.base_fingerprint(base)
                self.assertFalse(any(str(call.args[0]).endswith('.safetensors') for call in spy.call_args_list))
            (base / 'model.safetensors').write_text('changed fake base weight length')
            self.assertNotEqual(rm.cache_spec(config), manifest['spec'])
            before = rm.cache_spec(config)
            (Path(config['model']['initial_checkpoint']) / 'preprocessor_config.json').write_text('{"size":8}')
            self.assertNotEqual(rm.cache_spec(config), before)

    def test_feature_checksums_metadata_shape_and_finiteness(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = self.fixture(directory)
            cache, spec = Path(config['reward_model']['cache']), manifest['spec']
            row = manifest['splits']['train'][0]
            self.assertEqual(tuple(rm.load_cached_feature(cache, 'train', row, spec).shape), (2, 4, 4))
            path = cache / 'train' / 'train_0.pt'
            state = torch.load(path, weights_only=True)
            for key, value in [('spec_hash', 'wrong'), ('image_sha256', 'wrong'),
                               ('features', torch.zeros(2, 3, 4)), ('features', torch.full((2, 4, 4), float('nan')))]:
                changed = state | {key: value}
                torch.save(changed, path)
                with self.assertRaises(ValueError):
                    rm.load_cached_feature(cache, 'train', row, spec)
                row['feature_sha256'] = rm.file_sha(path)
                with self.assertRaises(ValueError):
                    rm.load_cached_feature(cache, 'train', row, spec)

    def test_manifest_rejects_uid_image_group_and_session_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = self.fixture(directory)
            rm.validate_manifest(manifest)
            for field in ('record_uid', 'image_sha256', 'group_id', 'session_id'):
                changed = copy.deepcopy(manifest)
                changed['splits']['val'][0][field] = changed['splits']['train'][0][field]
                with self.assertRaises(ValueError):
                    rm.validate_manifest(changed)
            missing = copy.deepcopy(manifest)
            missing['splits']['train'][0]['session_id'] = None
            with self.assertRaises(ValueError):
                rm.validate_manifest(missing)

    def test_training_metadata_binds_uids_preferences_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = self.fixture(directory)
            initial = rm.reward_training_metadata(config, manifest)
            changed = copy.deepcopy(manifest)
            changed['splits']['train'][0]['record_uid'] = 'another_train_uid'
            self.assertNotEqual(rm.reward_training_metadata(config, changed), initial)
            Path(config['data']['bbox_annotations']).write_text('{"updated":true}')
            self.assertNotEqual(rm.reward_training_metadata(config, manifest), initial)
            preference = Path(directory) / 'preferences.jsonl'
            preference.write_text('{"pair":1}\n')
            config['reward_model']['preference_pairs'] = str(preference)
            first = rm.reward_training_metadata(config, manifest)
            preference.write_text('{"pair":2}\n')
            self.assertNotEqual(rm.reward_training_metadata(config, manifest), first)

    def test_provider_frozen_scores_no_gt_and_no_rehashing_large_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(directory)
            provider = rm.CachedVisualReward(config, ['a', 'b'], 'cpu')
            self.assertTrue(all(not p.requires_grad for p in provider.model.parameters()))
            boxes = torch.tensor([[.1, .2, .8, .9], [.5, .5, .2, .8]])
            # score has only UID, candidate boxes, and generated category. It may
            # hash the small feature tensor, but must never reopen model weights.
            with mock.patch.object(rm, 'cache_spec', side_effect=AssertionError('Repeated model fingerprint')):
                scores = provider.score('train_0', boxes, 'a')
                self.assertEqual(scores.shape, (2,))
                self.assertTrue(0 < scores[0] < 1)
                self.assertEqual(scores[1].item(), 0.)
                self.assertFalse(scores.requires_grad)
                self.assertEqual(provider.score('val_0', boxes, 'no_event').sum().item(), 0.)
            with self.assertRaises(ValueError):
                provider.score('test_0', boxes, 'a')
            Path(config['data']['bbox_annotations']).write_text('{"updated":true}')
            with self.assertRaises(ValueError):
                rm.CachedVisualReward(config, ['a', 'b'], 'cpu')

    def test_explicit_heldout_features_do_not_enter_training_or_rl_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = self.fixture(directory)
            before = rm.reward_training_metadata(config, manifest)
            cache = Path(config['reward_model']['cache'])
            (cache / 'test').mkdir()
            path = cache / 'test' / 'test_0.pt'
            image_path = Path(directory) / 'test_image.fake'
            image_path.write_text('test image bytes')
            image_sha = rm.file_sha(image_path)
            torch.save({'features': torch.randn(2, 4, 4).half(), 'image_sha256': image_sha,
                        'spec_hash': rm.spec_hash(manifest['spec'])}, path)
            row = {'record_uid': 'test_0', 'group_id': 'test_group', 'session_id': 'test_session',
                   'image_path': str(image_path), 'image_sha256': image_sha, 'feature_sha256': rm.file_sha(path)}
            manifest['splits']['test'] = [row]
            rm.validate_manifest(manifest)
            self.assertEqual(rm.reward_training_metadata(config, manifest), before)
            self.assertEqual(tuple(rm.load_cached_feature(cache, 'test', row, manifest['spec']).shape), (2, 4, 4))
            (cache / 'manifest.json').write_text(json.dumps(manifest))
            provider = rm.CachedVisualReward(config, ['a', 'b'], 'cpu')
            with self.assertRaises(ValueError):
                provider.score('test_0', [[.1, .1, .8, .8]], 'a')

    def test_frozen_binding_accepts_rm_epoch_override_and_rejects_changed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = self.fixture(directory)
            checkpoint = torch.load(config['reward_model']['checkpoint'], weights_only=True)
            self.assertEqual(checkpoint['training_metadata']['training_settings']['epochs'], 1)
            config['reward_model']['train']['epochs'] = 5
            config['reward_model']['train']['learning_rate'] = .01
            rm.validate_reward_binding(checkpoint['training_metadata'], config, manifest)
            provider = rm.CachedVisualReward(config, ['a', 'b'], 'cpu')
            self.assertEqual(provider.score('train_0', [[.1, .2, .8, .9]], 'a').shape, (1,))
            changed = copy.deepcopy(manifest)
            changed['splits']['train'][0]['record_uid'] = 'different_training_record'
            with self.assertRaises(ValueError):
                rm.validate_reward_binding(checkpoint['training_metadata'], config, changed)
            preference = Path(directory) / 'new_preferences.jsonl'
            preference.write_text('{"pair":1}\n')
            config['reward_model']['preference_pairs'] = str(preference)
            with self.assertRaises(ValueError):
                rm.validate_reward_binding(checkpoint['training_metadata'], config, manifest)
            self.assertEqual(checkpoint['training_metadata']['training_settings']['epochs'], 1)

    def test_quality_evaluator_accepts_rm_epoch_override_using_saved_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest = self.fixture(directory)
            config['reward_model']['train']['epochs'] = 5
            config['device'], config['output'] = 'cpu', directory
            source = Path(__file__).resolve().parents[1] / 'scripts' / 'test_perception_reward.py'
            train_module = ModuleType('train_perception_reward')
            train_module.read_config = lambda path: config
            train_module.cached_rows = lambda runtime, split: (Path(config['reward_model']['cache']), manifest, [object()])
            observed_settings = []
            def fake_batch(rows, cache, split, labels, settings, seed, device):
                observed_settings.append(settings)
                return (torch.randn(4, 2, 4, 4), torch.tensor([[.1, .1, .7, .7]]).expand(4, -1),
                        torch.zeros(4, dtype=torch.long), torch.tensor([[1., .5, .2, 0.]]))
            train_module.batch_for_rows = fake_batch
            table4 = ModuleType('clear_uav.table4')
            table4.labels_from_config = lambda runtime: ['a', 'b']
            modules = {'train_perception_reward': train_module, 'clear_uav': ModuleType('clear_uav'), 'clear_uav.table4': table4}
            with mock.patch.dict(sys.modules, modules):
                spec = importlib.util.spec_from_file_location('_frozen_quality_contract', source)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                module.evaluate_quality(config, 'val')
            self.assertEqual(observed_settings[0]['epochs'], 1)
            result = json.loads((Path(directory) / 'rm' / 'val_quality.json').read_text())
            self.assertEqual(result['reward_training_settings']['epochs'], 1)
            self.assertTrue(result['rmse'] >= 0)


if __name__ == '__main__':
    unittest.main()
