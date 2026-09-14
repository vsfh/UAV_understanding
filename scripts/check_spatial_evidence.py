"""CPU-only frozen-readout checks. No model weights, training or subprocesses."""
import copy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from perception_spatial_head import SpatialInteraction
import run_spatial_evidence as evidence
import summarize_spatial_evidence as summary


def decode(head, query):
    raw = head(query).sigmoid()
    return torch.cat((torch.minimum(raw[:, :2], raw[:, 2:]),
                      torch.maximum(raw[:, :2], raw[:, 2:])), -1) * 1000


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.model = nn.Module()
        self.model.spatial_head = SpatialInteraction(8, 4, 2, dropout=0, residual_gate_init=0.7)
        self.model.box_head = nn.Linear(8, 4)
        self.model.eval()
        self.capture = evidence.PairedSpatialReadout(self.model, seed=43)
        self.args = (torch.randn(2, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 3),
                     torch.tensor([[False, False, False, True, True], [False] * 5]))

    def tearDown(self):
        self.capture.handle.remove()

    def run_head(self, args=None, uids=('a', 'b')):
        self.capture.uids = list(uids)
        with torch.no_grad():
            full = self.model.spatial_head(*(args or self.args))
        return full, {name: torch.tensor(boxes) for name, boxes in self.capture.boxes.items()}

    def test_full_and_no_spatial_match_direct_decoding(self):
        full, boxes = self.run_head()
        torch.testing.assert_close(boxes['full'], decode(self.model.box_head, full))
        torch.testing.assert_close(boxes['no_spatial'], decode(self.model.box_head, self.args[0]))

    def test_padding_invariance_and_valid_only_global_mean(self):
        _, baseline = self.run_head()
        query, memory, coords, padding = [item.clone() for item in self.args]
        memory[padding], coords[padding] = 1e6, -1e6
        original = self.model.spatial_head.forward
        with patch.object(self.model.spatial_head, 'forward', wraps=original) as spy:
            _, altered = self.run_head((query, memory, coords, padding))
        for name in evidence.CONDITIONS:
            torch.testing.assert_close(baseline[name], altered[name])
        global_memory = spy.call_args_list[-1].args[1]
        for row in range(2):
            expected = memory[row, ~padding[row]].mean(0).expand_as(memory[row])
            torch.testing.assert_close(global_memory[row], expected)

    def test_uid_permutation_determinism_and_batch_order(self):
        original = self.model.spatial_head.forward
        with patch.object(self.model.spatial_head, 'forward', wraps=original) as spy:
            _, first = self.run_head()
            shuffled_first = spy.call_args_list[1].args[2].clone()
        torch.rand(100)  # Global RNG consumption must not change UID-local shuffle.
        with patch.object(self.model.spatial_head, 'forward', wraps=original) as spy:
            _, second = self.run_head(tuple(item.flip(0) for item in self.args), ('b', 'a'))
            shuffled_second = spy.call_args_list[1].args[2]
        torch.testing.assert_close(shuffled_first, shuffled_second.flip(0), rtol=0, atol=0)
        for name in evidence.CONDITIONS:
            torch.testing.assert_close(first[name], second[name].flip(0))
        self.assertEqual(evidence.seeded_order(['b', 'a', 'c'], 43),
                         evidence.seeded_order(['c', 'b', 'a'], 43))

    def test_joint_memory_coordinate_permutation_preserves_full(self):
        _, baseline = self.run_head()
        query, memory, coords, padding = self.args
        order = torch.tensor([3, 1, 4, 0, 2])
        _, permuted = self.run_head((query, memory[:, order], coords[:, order], padding[:, order]))
        torch.testing.assert_close(baseline['full'], permuted['full'])

    def test_negative_target_keeps_predicted_event_and_new_box(self):
        row = {'record_uid': 'negative', 'target': {'presence': False, 'category': 'no_event'},
               'prediction': {'valid': True, 'category': 'event', 'bbox_1000': [1, 2, 3, 4],
                              'raw_output': 'event<vis>', 'presence_score': 0.8}}
        original = copy.deepcopy(row)
        boxes = {name: [[10, 20, 30, 40]] for name in evidence.CONDITIONS}
        result = evidence.paired_record(row, boxes, 0)
        self.assertTrue(result['replayed'])
        for prediction in result['predictions'].values():
            self.assertEqual(prediction['category'], 'event')
            self.assertEqual(prediction['bbox_1000'], boxes['full'][0])
        self.assertEqual(row, original)
        self.assertEqual(result['cached_prediction'], original['prediction'])

    def test_cached_completion_preserves_vis_and_stops_at_first_eos(self):
        tokenizer = SimpleNamespace(eos_token_id=2)
        with patch.object(tokenizer, 'encode', create=True, return_value=[7, 81, 2, 0, 2]) as encode:
            self.assertEqual(evidence.cached_completion(tokenizer, 'event<vis><eos><pad>'), [7, 81, 2])
            encode.assert_called_once_with('event<vis><eos><pad>', add_special_tokens=False)
        with patch.object(tokenizer, 'encode', create=True, return_value=[7, 81]):
            self.assertEqual(evidence.cached_completion(tokenizer, 'event<vis>'), [7, 81])

    def test_progress_repairs_only_partial_final_line(self):
        record = {'record_uid': 'a', 'predictions': dict.fromkeys(evidence.CONDITIONS, {})}
        valid = json.dumps(record).encode()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.jsonl'
            self.assertEqual(evidence.read_progress(path), set())
            path.write_bytes(valid + b'\n{"record_uid":')
            self.assertEqual(evidence.read_progress(path), {'a'})
            self.assertEqual(path.read_bytes(), valid + b'\n')
            path.write_bytes(valid)
            evidence.read_progress(path)
            self.assertEqual(path.read_bytes(), valid + b'\n')
            for invalid in (valid + b'\n' + valid + b'\n', b'broken\n' + valid,
                            valid + b'\nbroken\n', b'{"record_uid":"a","predictions":{}}\n'):
                path.write_bytes(invalid)
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    evidence.read_progress(path)
                self.assertEqual(path.read_bytes(), invalid)

    def test_parent_timeout_is_bounded_and_still_summarizes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'paired_predictions.jsonl').write_text('{}\n')
            spec = {'output': directory, 'max_hours': 20}
            responses = [subprocess.TimeoutExpired('mock-worker', 1), SimpleNamespace(returncode=0)]
            with patch.object(evidence, 'read_yaml', return_value=spec), \
                    patch.object(evidence.sys, 'argv', ['run_spatial_evidence.py']), \
                    patch.object(evidence.subprocess, 'run', side_effect=responses) as run:
                evidence.main()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args_list[0].kwargs['timeout'], 12 * 3600 - 120)
            self.assertLessEqual(run.call_args_list[1].kwargs['timeout'], 12 * 3600)
            status = json.loads((output / 'status.json').read_text())
            self.assertEqual(status['worker'], 'deadline_stopped')
            self.assertFalse(status['training'])

    def test_actual_summary_complete_partial_and_frozen_fields(self):
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source', Path(directory) / 'output'
            source.mkdir()
            output.mkdir()
            trained = {'protocol': 'session_disjoint', 'seed': 43}
            (source / 'config.yaml').write_text(json.dumps(trained))
            positive = {'record_uid': 'p', 'group_id': 'group_p',
                        'target': {'presence': True, 'category': 'event', 'bbox_1000': [100, 100, 600, 600]},
                        'prediction': {'valid': True, 'category': 'event', 'bbox_1000': [100, 100, 600, 600],
                                       'presence_score': .9, 'latency_ms': 1}}
            negative = {'record_uid': 'n', 'group_id': 'group_n',
                        'target': {'presence': False, 'category': None, 'bbox_1000': None},
                        'prediction': {'valid': True, 'category': None, 'bbox_1000': None,
                                       'presence_score': .1, 'latency_ms': 1}}
            cached = {**trained, 'rows': [positive, negative], 'metrics': {'threshold': .5}}
            (source / 'test_results.json').write_text(json.dumps(cached))
            (output / 'plan.json').write_text(json.dumps({'record_uids': ['n', 'p'], 'total_records': 2}))
            config = Path(directory) / 'spec.yaml'
            config.write_text(json.dumps({'source_dir': str(source), 'output': str(output), 'split': 'test'}))
            paired = [evidence.paired_record(positive), evidence.paired_record(negative)]
            paired[0]['predictions']['no_spatial']['bbox_1000'] = [700, 700, 800, 800]
            path = output / 'paired_predictions.jsonl'
            def check(records):
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
                with patch.object(summary.sys, 'argv', ['summary', '--config', str(config)]), \
                        patch.object(summary, 'labels_from_config', return_value=['event']):
                    summary.main()
                return json.loads((output / 'summary.json').read_text())
            full = check(paired)
            self.assertEqual(full['state'], 'COMPLETE')
            self.assertEqual(full['conditions']['full']['j50'], 1)
            self.assertEqual(full['conditions']['no_spatial']['j50'], 0)
            self.assertNotIn('median_ms', full['conditions']['full']['benchmark'])
            self.assertEqual(check(paired[:1])['state'], 'PARTIAL')
            self.assertEqual(check(paired[1:])['state'], 'PARTIAL')  # Negatives-only partial output.
            paired[0]['predictions']['no_spatial']['presence_score'] = .5
            with self.assertRaisesRegex(ValueError, 'more than ROI'):
                check(paired)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
