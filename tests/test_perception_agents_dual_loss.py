"""CPU tests for the real batched objective, with a tiny deterministic causal VLM.

No Hugging Face model, GPU or image file is loaded. Run in the existing remote
torch/torchvision environment. The local bundled Python skips these tests when
torch is absent; a skip is not GPU or gradient verification.
"""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import perception_agents_dual_common as common
from perception_agents_data import example_set

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torchvision.ops import generalized_box_iou_loss
except ImportError:
    torch = None


VIS_ID = 21


if torch is not None:
    class TinyVLM(nn.Module):
        """Causal cumulative context makes left-padding and answer shift observable."""
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 7)
            self.context = nn.Linear(7, 7)
            self.lm_head = nn.Linear(7, 32)
            self.requested_windows = []

        def forward(self, input_ids, attention_mask, labels=None, logits_to_keep=0, use_cache=False):
            self.requested_windows.append(logits_to_keep)
            mask = attention_mask.unsqueeze(-1)
            embeddings = self.embedding(input_ids) * mask
            hidden = torch.tanh(self.context(embeddings.cumsum(1) / mask.cumsum(1).clamp_min(1)))
            selected = hidden[:, -logits_to_keep:, :] if logits_to_keep else hidden
            logits = self.lm_head(selected)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].reshape(-1), ignore_index=-100)
            return SimpleNamespace(logits=logits, loss=loss, hidden_states=hidden)


    class TinySpatial(nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = TinyVLM()
            self.spatial_head = nn.Linear(7, 7)
            self.box_head = nn.Linear(7, 4)
            self.seen_positions = []

        def forward(self, inputs):
            ids = inputs['input_ids']
            positions = ((ids == VIS_ID) * torch.arange(ids.shape[1])).amax(1)
            self.seen_positions.append(positions.tolist())
            output = self.vlm(**inputs, use_cache=False)
            hidden = output.hidden_states
            query = hidden[torch.arange(len(ids)), positions]
            mask = inputs['attention_mask'].unsqueeze(-1)
            memory = (hidden * mask).sum(1) / mask.sum(1)
            state = query + self.spatial_head(memory)
            raw = self.box_head(state).sigmoid()
            boxes = torch.cat((torch.minimum(raw[:, :2], raw[:, 2:]),
                               torch.maximum(raw[:, :2], raw[:, 2:])), dim=-1)
            return output.loss, boxes


def fake_encode(processor, config, batch):
    """Variable prompt/answer lengths, prompt <vis>, assistant <vis>, left padding."""
    rows, targets = [], []
    for sample, example in zip(batch['samples'], batch['examples']):
        index = int(sample.record_uid.rsplit('_', 1)[1])
        prompt = [1, VIS_ID, 4, 5] + [6] * (index % 4)
        if batch['role'] == 'verify':
            answer = [10 + ord(example['answer']) - ord('A'), 2]
        else:
            category = [8] if index % 2 == 0 else [8, 9, 10]
            answer = category + ([VIS_ID] if sample.presence else []) + [2]
        rows.append(prompt + answer)
        targets.append([-100] * len(prompt) + answer)
    width = max(map(len, rows))
    return {
        'input_ids': torch.tensor([[0] * (width - len(row)) + row for row in rows]),
        'attention_mask': torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows]),
        'labels': torch.tensor([[-100] * (width - len(row)) + row for row in targets]),
    }


@unittest.skipIf(torch is None, "requires the remote CPU torch/torchvision environment")
class DualLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4300)
        self.samples = [SimpleNamespace(record_uid=f'row_{index}', label='fire' if index % 2 == 0 else None,
                                       presence=index % 2 == 0, image_path=Path(f'image_{index}.png'),
                                       bbox_1000=[100 + 10 * index, 150, 700, 800] if index % 2 == 0 else None)
                        for index in range(5)]
        self.cached = {sample.record_uid: {'category': 'flood', 'bbox_1000': [50, 50, 200, 200]}
                       for sample in self.samples}
        self.labels = ['fire', 'flood']
        self.config = {'device': 'cpu', 'loss': {'language': .7, 'l1': .4, 'giou': 1.2},
                       'agents': {'candidate_seed': 4300, 'accept_iou': .5, 'verifier_candidates_per_image': 4},
                       'train': {'batch_size': 4}, 'validation': {'batch_size': 3}}

    def assert_gradients_equal(self, batched, serial):
        for (name, parameter), (other_name, other) in zip(batched.named_parameters(), serial.named_parameters()):
            self.assertEqual(name, other_name)
            if parameter.grad is None or other.grad is None:
                self.assertIs(parameter.grad, other.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=3e-5, atol=3e-6, msg=name)

    def reference_role_loss(self, model, sample, example):
        """Original batch1 recipe: full logits, model CE mean, scalar L1/GIoU."""
        batch = {'role': example['role'], 'samples': [sample], 'examples': [example]}
        inputs = fake_encode(None, self.config, batch)
        if example['role'] == 'where':
            language, boxes = model(inputs)
            targets = torch.tensor([sample.bbox_1000], dtype=torch.float32) / 1000
            value = self.config['loss']['language'] * language
            value = value + self.config['loss']['l1'] * F.l1_loss(boxes, targets)
            return value + self.config['loss']['giou'] * generalized_box_iou_loss(boxes, targets, reduction='mean')
        return self.config['loss']['language'] * model.vlm(**inputs, use_cache=False).loss

    def test_answer_window_preserves_shifted_loss_and_all_logits_gradients(self):
        full = torch.randn(4, 13, 32, requires_grad=True)
        labels = torch.full((4, 13), -100)
        for row, positions in enumerate(([8, 9, 10, 11, 12], [11, 12], [7, 9, 12], [12])):
            labels[row, positions] = torch.arange(len(positions)) + 3
        weights = torch.tensor([1/3, 1/2, 1/12, 1/8])
        window = common.answer_window(labels)
        self.assertEqual(window, 7)
        cropped_loss = (common.per_example_language_loss(full[:, -window:], labels) * weights).sum()
        expected = sum(F.cross_entropy(full[row, :-1], labels[row, 1:], ignore_index=-100) * weights[row]
                       for row in range(4))
        torch.testing.assert_close(cropped_loss, expected)
        gradient = torch.autograd.grad(cropped_loss, full, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(expected, full)[0]
        torch.testing.assert_close(gradient, reference_gradient)
        self.assertEqual(torch.count_nonzero(gradient[:, :6]).item(), 0)
        self.assertGreater(torch.count_nonzero(gradient[2, 6]).item(), 0)
        self.assertEqual(torch.count_nonzero(gradient[:, -1]).item(), 0)

    def test_answer_window_rejects_missing_supervision_or_missing_predecessor(self):
        for labels in (torch.full((2, 6), -100), torch.tensor([[1, -100, -100], [-100, 2, 3]]),
                       torch.tensor([[-100, 2, 3], [-100, -100, -100]])):
            with self.assertRaises(ValueError):
                common.answer_window(labels)

    def test_training_and_validation_full_objective_and_gradients_equal_serial(self):
        for validation in (False, True):
            with self.subTest(validation=validation):
                batched = TinySpatial()
                serial = copy.deepcopy(batched)
                criterion = common.RoleBatchLoss(batched, None, self.config)
                draws = [7, 11, 14, 19, 22]
                with mock.patch.object(common, 'encode_batch', side_effect=fake_encode):
                    parts = list(common.role_batches(self.samples, self.cached, self.labels, self.config,
                                                    epoch=1, validation=validation, draw_indices=draws))
                    value = sum(criterion(batch) for batch in parts) / len(self.samples)
                reference = 0
                for sample, draw in zip(self.samples, draws):
                    examples = example_set(sample, self.cached[sample.record_uid], self.labels, self.config['agents'],
                                           epoch=1, validation=validation, draw_index=draw)
                    self.assertAlmostEqual(sum(example['weight'] for example in examples), 1)
                    self.assertEqual(sum(example['role'] == 'where' for example in examples), int(sample.presence))
                    self.assertEqual(sum(example['role'] == 'verify' for example in examples), 4 if validation else 1)
                    for example in examples:
                        reference = reference + self.reference_role_loss(serial, sample, example) * example['weight'] / len(self.samples)
                torch.testing.assert_close(value, reference, rtol=2e-6, atol=1e-6)
                value.backward()
                reference.backward()
                self.assert_gradients_equal(batched, serial)
                self.assertGreater(batched.box_head.weight.grad.abs().sum().item(), 0)
                self.assertGreater(batched.spatial_head.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(window > 0 for window in batched.vlm.requested_windows))
                self.assertTrue(all(window == 0 for window in serial.vlm.requested_windows))
                self.assertEqual(len(batched.vlm._forward_hooks), 0)

    def test_where_uses_last_vis_and_full_hidden_sequence_with_cropped_logits(self):
        model = TinySpatial()
        examples = [next(example for example in example_set(sample, self.cached[sample.record_uid], self.labels,
                                                             self.config['agents']) if example['role'] == 'where')
                    for sample in self.samples if sample.presence]
        batch = {'role': 'where', 'samples': [sample for sample in self.samples if sample.presence], 'examples': examples}
        encoded = fake_encode(None, self.config, batch)
        expected = ((encoded['input_ids'] == VIS_ID) * torch.arange(encoded['input_ids'].shape[1])).amax(1)
        with mock.patch.object(common, 'encode_batch', side_effect=fake_encode):
            loss = common.RoleBatchLoss(model, None, self.config)(batch)
        self.assertEqual(model.seen_positions, [expected.tolist()])
        self.assertTrue((expected > 1).all().item())
        self.assertLess(model.vlm.requested_windows[0], encoded['input_ids'].shape[1])
        loss.backward()
        self.assertGreater(model.vlm.embedding.weight.grad[VIS_ID].abs().sum().item(), 0)
        self.assertEqual(len(model.vlm._forward_hooks), 0)

    def test_where_hook_removed_when_forward_fails(self):
        model = TinySpatial()
        sample = self.samples[0]
        example = next(e for e in example_set(sample, self.cached[sample.record_uid], self.labels,
                                              self.config['agents']) if e['role'] == 'where')
        batch = {'role': 'where', 'samples': [sample], 'examples': [example]}
        with mock.patch.object(common, 'encode_batch', side_effect=fake_encode), \
                mock.patch.object(model, 'forward', side_effect=RuntimeError('test interruption')):
            with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                common.RoleBatchLoss(model, None, self.config)(batch)
        self.assertEqual(len(model.vlm._forward_hooks), 0)

    def test_what_and_verify_do_not_update_box_or_spatial_heads(self):
        for role in ('what', 'verify'):
            with self.subTest(role=role):
                model = TinySpatial()
                batches = list(common.role_batches(self.samples, self.cached, self.labels, self.config))
                with mock.patch.object(common, 'encode_batch', side_effect=fake_encode):
                    value = sum(common.RoleBatchLoss(model, None, self.config)(batch)
                                for batch in batches if batch['role'] == role)
                value.backward()
                self.assertTrue(all(parameter.grad is None for parameter in model.box_head.parameters()))
                self.assertTrue(all(parameter.grad is None for parameter in model.spatial_head.parameters()))
                self.assertGreater(model.vlm.lm_head.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
