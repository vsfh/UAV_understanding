#!/usr/bin/env python3
"""CPU-only Table IV audit. Writes a separate receipt, never predictions.

AP uses the repository's non-interpolated ranked precision definition, one
candidate per frame and presence-score ranking for every system.
"""
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def normalize(row):
    if 'target' in row:
        return row
    return dict(record_uid=row['record_uid'], group_id=row.get('group_id'),
                target=dict(presence=row['target_presence'], bbox_1000=row['target_bbox_1000'],
                            category=row['target_category']),
                prediction=dict(presence_score=row['presence_score'],
                                bbox_1000=row.get('candidate_bbox_1000', row['prediction_bbox_1000']),
                                category=row.get('candidate_category', row['prediction_category']),
                                valid=True, latency_ms=row['latency_ms']))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0.0


def ap(items, positives):
    if not positives:
        return None
    hits = 0
    total = 0.0
    for rank, (_, hit) in enumerate(sorted(items, key=lambda x: x[0], reverse=True), 1):
        if hit:
            hits += 1
            total += hits/rank
    return total/positives


def compute(rows, labels, threshold, classification=True):
    positives = sum(r['target']['presence'] for r in rows)
    negatives = len(rows)-positives
    pairs = [(r['prediction']['presence_score'], r['target']['presence']) for r in rows]
    tp = sum(y and s >= threshold for s, y in pairs)
    fp = sum(not y and s >= threshold for s, y in pairs)
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/positives if positives else 0
    aps, f1s = [], []
    for label in labels:
        support = sum(r['target']['presence'] and r['target']['category'] == label for r in rows)
        selected = [r for r in rows if r['prediction']['category'] == label and r['prediction']['bbox_1000'] is not None]
        aps.append(ap([(r['prediction']['presence_score'], r['target']['presence'] and r['target']['category'] == label and iou(r['target']['bbox_1000'], r['prediction']['bbox_1000']) >= .5) for r in selected], support))
        hits = sum(r['target']['presence'] and r['target']['category'] == label and r['prediction']['category'] == label for r in rows)
        predicted = sum(r['target']['presence'] and r['prediction']['category'] == label for r in rows)
        f1s.append(2*hits/(support+predicted) if support+predicted else 0)
    return dict(p_ap=ap(pairs, positives), n_fpr=fp/negatives if negatives else None,
                p_precision=precision, p_recall=recall,
                p_f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                ap50=ap([(r['prediction']['presence_score'], r['target']['presence'] and iou(r['target']['bbox_1000'], r['prediction']['bbox_1000']) >= .5) for r in rows if r['prediction']['bbox_1000'] is not None], positives),
                c_f1=statistics.mean(f1s) if classification else None,
                g_map50=statistics.mean(x for x in aps if x is not None) if classification else None,
                valid_rate=statistics.mean(r['prediction']['valid'] for r in rows),
                median_ms=statistics.median(r['prediction']['latency_ms'] for r in rows),
                positive_records=positives, negative_records=negatives, threshold=threshold,
                false_positives=fp, true_positives=tp)


def audit(root):
    files = sorted((root/'results'/'table4').glob('*/session_disjoint/seed43_test.json'))
    if not files:
        files = sorted((root/'table4').glob('*/session_disjoint/seed43_test.json'))
    if not files:
        raise ValueError('No Table IV results found')
    reports, baseline = [], None
    for path in files:
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = [normalize(r) for r in data['rows']]
        ids = [r['record_uid'] for r in rows]
        assert len(ids) == len(set(ids)) == 5338, (path, 'duplicate/missing records')
        labels = sorted({r['target']['category'] for r in rows if r['target']['presence']})
        assert len(labels) == 18, (path, 'ontology mismatch')
        if baseline is None:
            baseline = {r['record_uid']: r for r in rows}
        assert set(ids) == set(baseline), (path, 'UID set mismatch')
        out_of_bounds = []
        for row in rows:
            ref = baseline[row['record_uid']]
            assert row['group_id'] == ref['group_id'], 'group changed'
            a, b, pred = row['target'], ref['target'], row['prediction']
            assert (a['presence'], a['category']) == (b['presence'], b['category']), 'target changed'
            if a['presence']:
                assert all(abs(x-y) < .001 for x,y in zip(a['bbox_1000'],b['bbox_1000'])), 'target box changed'
            assert math.isfinite(pred['presence_score']) and 0 <= pred['presence_score'] <= 1
            assert math.isfinite(pred['latency_ms']) and pred['latency_ms'] >= 0
            assert pred['category'] is None or pred['category'] in labels
            box = pred['bbox_1000']
            if box is not None:
                assert len(box) == 4 and all(math.isfinite(x) for x in box), (path, row['record_uid'], box)
                if any(x < -.01 or x > 1000.01 for x in box):
                    out_of_bounds.append(row['record_uid'])
                assert box[0] < box[2] and box[1] < box[3], (path, row['record_uid'], 'invalid box')
        stored = data['metrics'].get('table4', data['metrics'])
        metrics = compute(rows, labels, stored['threshold'], stored['c_f1'] is not None)
        assert (metrics['positive_records'], metrics['negative_records']) == (4778, 560)
        deltas = {k: metrics[k]-stored[k] for k in metrics if k in stored and isinstance(metrics[k], (int,float)) and stored[k] is not None and abs(metrics[k]-stored[k]) > 1e-8}
        cal = root/'outputs'/'table4'/path.parents[1].name/'session_disjoint'/'seed43'/'calibration.json'
        declared_calibration = data.get('metric_provenance', {}).get('calibration')
        if declared_calibration:
            cal = root/declared_calibration
        calibration = None
        if cal.exists():
            calibration = json.loads(cal.read_text())
            assert math.isclose(calibration['threshold'], stored['threshold'], abs_tol=1e-12), (path, cal, 'threshold provenance mismatch')
        batches = Counter()
        for row in rows:
            pred = row['prediction']
            calls = pred.get('raw_output')
            if isinstance(calls, list):
                batches.update(str(x['prediction'].get('inference_batch_size', 'unknown')) for x in calls)
            else:
                batches[str(pred.get('inference_batch_size', 'unknown'))] += 1
        group_counts = {}
        for group in sorted({r['group_id'] for r in rows if not r['target']['presence']}):
            subset = [r for r in rows if not r['target']['presence'] and r['group_id'] == group]
            group_counts[group] = dict(n=len(subset), fp=sum(r['prediction']['presence_score'] >= stored['threshold'] for r in subset))
        reports.append(dict(system=path.parents[1].name, source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            metrics=metrics, stored_metric_deltas=deltas, calibration=calibration,
                            out_of_bounds_boxes=len(out_of_bounds), out_of_bounds_examples=out_of_bounds[:5],
                            per_call_batch_sizes=dict(batches), negative_groups=group_counts,
                            routes=dict(Counter(r['prediction'].get('route', 'single') for r in rows))))
    cache = root/'outputs/table4/qwen3vl_t4/session_disjoint/seed43/validation_predictions.json'
    cache_check = None
    if cache.exists():
        val = json.loads(cache.read_text())
        val_ids = [r['record_uid'] for r in val]
        assert len(val_ids) == len(set(val_ids)) == 2728
        assert not set(val_ids).intersection(baseline), 'validation/test overlap'
        cache_check = dict(records=len(val_ids), unique=True, test_overlap=0)
    return dict(protocol='session_disjoint', seed=43, uid_set_sha256=hashlib.sha256('\n'.join(sorted(baseline)).encode()).hexdigest(),
                validation_cache=cache_check, results=reports,
                note='Timing values are legacy recorded costs, not matched batch-one end-to-end latency. Metric changes are reported separately; inputs are not edited.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for r in report['results']:
        print(r['system'], json.dumps(r['metrics']), 'deltas=', r['stored_metric_deltas'])
