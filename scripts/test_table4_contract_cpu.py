#!/usr/bin/env python3
"""Check shared metric semantics without importing GPU/model dependencies."""
import ast
import json
import math
import statistics
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from audit_table4_results import compute, normalize

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT/'src/clear_uav/table4.py').read_text(encoding='utf-8'))
names = {'box_iou','average_precision','presence_ap','localization_ap50','macro_f1','presence_at_threshold','table4_metrics','save_results'}
pure = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
scope = dict(statistics=statistics, math=math, Counter=Counter, Path=Path, json=json)
exec(compile(pure, '<table4 pure metric functions>', 'exec'), scope)

class ContractTests(unittest.TestCase):
    def test_misaligned_rejected(self):
        with self.assertRaises(ValueError):
            scope['table4_metrics']([object()], [], [], .5, True)

    def test_candidate_before_gating_and_negative_false_alarm(self):
        samples = [SimpleNamespace(presence=True, label='event', bbox_1000=[0,0,100,100]),
                   SimpleNamespace(presence=False, label=None, bbox_1000=None)]
        pred = [dict(presence_score=.2, category='event', bbox_1000=[0,0,100,100], valid=True, latency_ms=2),
                dict(presence_score=.9, category='event', bbox_1000=[0,0,100,100], valid=True, latency_ms=2)]
        result = scope['table4_metrics'](samples, pred, ['event'], .5, True)
        self.assertEqual(result['c_f1'], 1)
        self.assertEqual(result['n_fpr'], 1)
        self.assertEqual(result['p_recall'], 0)
        self.assertEqual(result['g_map50'], .5)
        self.assertEqual(result['timing_scopes'], {'legacy_unspecified': 2})

    def test_duplicate_save_rejected(self):
        sample = SimpleNamespace(record_uid='same')
        with self.assertRaises(ValueError):
            scope['save_results'](Path('must_not_be_created.json'), {}, '', [sample,sample], [{},{}], {})

if __name__ == '__main__':
    unittest.main()
