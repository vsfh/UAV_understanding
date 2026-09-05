#!/usr/bin/env python3
"""Repair only aggregate Qwen-Ground AP; preserve rows and a dated backup."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from audit_table4_results import compute, normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    data = json.loads(args.result.read_text())
    rows = [normalize(r) for r in data['rows']]
    assert len(rows) == len({r['record_uid'] for r in rows}) == 5338
    metrics = data['metrics']
    recomputed = compute(rows, data['labels'], metrics['table4']['threshold'])
    print(json.dumps(recomputed, indent=2))
    if args.write:
        backup = args.result.with_suffix('.json.before_uniform_ap_20260905')
        if not backup.exists():
            shutil.copy2(args.result, backup)
        digest = hashlib.sha256(json.dumps(data['rows'], sort_keys=True).encode()).hexdigest()
        metrics.setdefault('g_map50_legacy_class_scores', metrics['g_map50'])
        metrics['g_map50'] = recomputed['g_map50']
        metrics['table4'].update(recomputed)
        data['metric_provenance'] = dict(ap_ranking='presence_score_single_candidate',
            source='saved_per_record_predictions', raw_rows_modified=False,
            raw_rows_sha256=digest, previous_metrics_backup=str(backup),
            timing='legacy_batch8_transfer_and_forward_amortized_not_end_to_end')
        temporary = args.result.with_suffix('.json.uniform_ap.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        assert hashlib.sha256(json.dumps(json.loads(temporary.read_text())['rows'], sort_keys=True).encode()).hexdigest() == digest
        temporary.replace(args.result)
