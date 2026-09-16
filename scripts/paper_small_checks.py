"""CPU-only diagnostics and shared metrics for the paper's small checks."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

DEFAULT = 'configs/yaml/paper_small_checks.yaml'


def read_config(path=DEFAULT):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def load_rows(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return data['rows']


def paired_rows(config):
    rows = {name: load_rows(path) for name, path in config['predictions'].items()}
    canonical = rows['direct']
    ids = [r['record_uid'] for r in canonical]
    assert len(ids) == len(set(ids)), 'Repeated record UID'
    for name, values in rows.items():
        mapping = {r['record_uid']: r for r in values}
        assert len(mapping) == len(values) and set(mapping) == set(ids), name + ': mismatched records'
        rows[name] = [mapping[uid] for uid in ids]
        assert all(a['target'] == b['target'] for a, b in zip(canonical, rows[name])), name + ': mismatched targets'
    return rows


def iou(a, b):
    if a is None or b is None:
        return 0.0
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return overlap / max(area_a + area_b - overlap, 1e-12)


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return [center-half, center+half]


def multi_ap(rows, k='all', category=None):
    """Non-interpolated AP; one reference per image, duplicates are FP.

    Top-k is selected per IMAGE before class filtering. Stable ties follow
    source record order and then saved proposal order, matching the paper.
    """
    positives = sum(r['target']['presence'] and (category is None or r['target']['category'] == category) for r in rows)
    if not positives:
        return None
    ranked = []
    for index, row in enumerate(rows):
        candidates = sorted(row['candidates'], key=lambda c: -c['score'])
        candidates = candidates if k == 'all' else candidates[:int(k)]
        ranked.extend((c['score'], index, c) for c in candidates if category is None or c.get('category') == category)
    ranked.sort(key=lambda x: -x[0])
    used, tp, area = set(), 0, 0.0
    for rank, (_, index, candidate) in enumerate(ranked, 1):
        target = rows[index]['target']
        correct = (index not in used and target['presence'] and
                   (category is None or target['category'] == category) and
                   iou(target['bbox_1000'], candidate['bbox_1000']) >= .5)
        if correct:
            used.add(index)
            tp += 1
            area += tp / rank
    return area / positives


def candidate_summary(rows, ks=(1, 10, 'all'), classified=False):
    labels = sorted({r['target']['category'] for r in rows if r['target']['presence']})
    result = {}
    for k in ks:
        aps = {label: multi_ap(rows, k, label) for label in labels} if classified else {}
        result[str(k)] = {'ap50': multi_ap(rows, k), 'g_map50': float(np.mean(list(aps.values()))) if aps else None,
                          'per_class_g_ap50': aps}
    return result


def single_candidates(rows):
    return [{'record_uid': r['record_uid'], 'target': r['target'], 'candidates':
             [] if r['prediction']['bbox_1000'] is None else
             [{'bbox_1000': r['prediction']['bbox_1000'], 'score': r['prediction']['presence_score'],
               'category': r['prediction']['category']}]} for r in rows]


def diagnostics(config):
    paired = paired_rows(config)
    output = Path(config['output'])
    labels = sorted({r['target']['category'] for r in paired['direct'] if r['target']['presence']})
    result, table = {}, []
    for name, rows in paired.items():
        positive = [r for r in rows if r['target']['presence']]
        errors = Counter()
        per_class = {}
        for label in labels:
            subset = [r for r in positive if r['target']['category'] == label]
            counts = Counter()
            for r in subset:
                c = r['prediction']['category'] == label
                b = iou(r['target']['bbox_1000'], r['prediction']['bbox_1000']) >= .5
                counts[f'class_{int(c)}_roi_{int(b)}'] += 1
            errors.update(counts)
            per_class[label] = {'n': len(subset), 'j50': counts['class_1_roi_1']/len(subset),
                                'class_recall': (counts['class_1_roi_1']+counts['class_1_roi_0'])/len(subset),
                                'errors': dict(counts)}
        result[name] = {'n_positive': len(positive), 'per_class': per_class, 'errors': dict(errors),
                        'macro_j50': float(np.mean([v['j50'] for v in per_class.values()])),
                        'other16_macro_j50': float(np.mean([v['j50'] for k,v in per_class.items()
                                                         if k not in ['green_algae_duckweed','crop_lodging']]))}
    for label in labels:
        a, b = (result[n]['per_class'][label] for n in ['direct','perception'])
        table.append({'class': label, 'N': a['n'], 'direct_J50': 100*a['j50'],
                      'perception_J50': 100*b['j50'], 'delta_pp': 100*(b['j50']-a['j50'])})
    result['perception_wins'] = sum(r['delta_pp'] > 0 for r in table)
    groups = defaultdict(list)
    for r in paired['perception']:
        if not r['target']['presence']:
            groups[r['group_id']].append(r['prediction']['presence_score'] >= config['alert_threshold'])
    result['negative_groups'] = [{'group': key, 'n': len(v), 'fp': sum(v), 'n_fpr': sum(v)/len(v),
                                  'iid_frame_wilson95': wilson(sum(v),len(v))} for key,v in sorted(groups.items())]
    n = sum(len(v) for v in groups.values()); fp = sum(sum(v) for v in groups.values())
    result['negative_total'] = {'n':n, 'fp':fp, 'n_fpr':fp/n, 'iid_frame_wilson95':wilson(fp,n),
                                'note':'Frame-IID interval, not cross-group generalization uncertainty.'}
    # Actual validation values, not a probability-calibration claim.
    val = load_rows(config['validation']['perception'])
    result['validation_score'] = {}
    for subset, values in [('all', val), ('positive',[r for r in val if r['target']['presence']]),
                           ('negative',[r for r in val if not r['target']['presence']])]:
        scores = [r['prediction']['presence_score'] for r in values]
        result['validation_score'][subset] = {'n':len(scores), 'quantiles':np.quantile(scores,[0,.05,.5,.95,1]).tolist()}
    result['sources_sha256'] = {k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in config['predictions'].items()}
    save_json(output/'diagnostics.json', result)
    with (output/'per_class.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(json.dumps({k:result[k]['macro_j50'] for k in paired}, indent=2))


def perturb(config):
    paired = {k:[r for r in v if r['target']['presence']] for k,v in paired_rows(config).items()}
    refs = np.array([r['target']['bbox_1000'] for r in paired['direct']], float)
    center = (refs[:,:2]+refs[:,2:])/2; size = refs[:,2:]-refs[:,:2]
    rng = np.random.default_rng(config['roi_perturbation']['seed'])
    result = []
    for fraction in config['roi_perturbation']['fractions']:
        trials = []
        for _ in tqdm(range(config['roi_perturbation']['repeats']), desc=f'ROI jitter {fraction:.0%}'):
            c = center + rng.uniform(-fraction,fraction,size.shape)*size
            s = size * rng.uniform(1-fraction,1+fraction,size.shape)
            boxes = np.clip(np.concatenate([c-s/2,c+s/2],axis=1),0,1000)
            trial = {'reference_mean_iou':float(np.mean([iou(a,b) for a,b in zip(refs,boxes)]))}
            for name, rows in paired.items():
                trial[name] = sum(r['prediction']['category']==r['target']['category'] and
                                  iou(b,r['prediction']['bbox_1000'])>=.5 for r,b in zip(rows,boxes))/len(rows)
            trial['delta_pp'] = 100*(trial['perception']-trial['direct'])
            trials.append(trial)
        result.append({'fraction':fraction, 'trials':trials,
                       'summary':{key:{'mean':float(np.mean([t[key] for t in trials])),
                                       'min':min(t[key] for t in trials), 'max':max(t[key] for t in trials)} for key in trials[0]}})
    save_json(Path(config['output'])/'roi_perturbation.json',
              {'note':'Synthetic reference sensitivity, NOT inter-annotator agreement or human ROI utility. Same perturbations for both systems.',
               'settings':config['roi_perturbation'], 'results':result})


def shifts(config):
    result = {}
    for protocol in ['session_disjoint','unseen_site','forward_temporal']:
        direct = (f'results/table4/qwen3vl_t4/{protocol}/seed43_test.json' if protocol=='session_disjoint' else
                  f'results/table4_shifts/qwen3vl_t4/{protocol}/seed43_test.json')
        perception = f'outputs/perception_qwen/{protocol}/seed43/test_results.json'
        if not Path(direct).exists() or not Path(perception).exists():
            result[protocol] = {'status':'missing existing results', 'paths':[direct,perception]}; continue
        pair = paired_rows({'predictions':{'direct':direct,'perception':perception}})
        result[protocol] = {'status':'UID and targets matched; training-schedule comparability requires separate review',
                            'models':{name:candidate_summary(single_candidates(rows),[1],True)['1'] |
                                      {'j50':sum(r['prediction']['category']==r['target']['category'] and
                                                iou(r['target']['bbox_1000'],r['prediction']['bbox_1000'])>=.5
                                                for r in rows if r['target']['presence']) /
                                               sum(r['target']['presence'] for r in rows)} for name,rows in pair.items()},
                            'note':'Perception BASE, not continued checkpoint. Site/time isolation is positive-only.'}
    save_json(Path(config['output'])/'existing_shifts.json',result)


def summary(config):
    root=Path(config['output'])
    files=['diagnostics.json','roi_perturbation.json','existing_shifts.json','efficiency_direct.json',
           'efficiency_perception.json','dfine_metrics.json','dfine_dinov2_metrics.json','grounding_dino_metrics.json']
    save_json(root/'summary.json',{name:json.loads((root/name).read_text()) if (root/name).exists() else
                                  {'status':'not run'} for name in files})
    print('Results:',root/'summary.json')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['analysis','perturb','shifts','summary'])
    parser.add_argument('--config',default=DEFAULT)
    args=parser.parse_args(); config=read_config(args.config)
    {'analysis':diagnostics,'perturb':perturb,'shifts':shifts,'summary':summary}[args.stage](config)
