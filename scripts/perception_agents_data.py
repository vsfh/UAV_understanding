"""GT is used only here to construct training targets, never by the agent runtime."""
from __future__ import annotations

import hashlib
import random

from perception_agents_core import VIS, VERDICT_CODES, make_candidates, oracle_verdict


def example_set(sample, proposal, labels, settings, epoch=0, validation=False, draw_index=0):
    seed = int(hashlib.sha256(sample.record_uid.encode()).hexdigest()[:12],16) + settings['candidate_seed']
    rng = random.Random(seed)
    candidates = make_candidates(sample.presence,sample.label,sample.bbox_1000,labels,rng)
    oracle = lambda c:oracle_verdict(sample.presence,sample.label,sample.bbox_1000,
                                   c['category'],c['bbox_1000'],settings['accept_iou'])
    # Always include the frozen model's actual proposal, then cover available verdicts.
    chosen = [dict(category=proposal['category'],bbox_1000=proposal['bbox_1000'])]
    buckets = {}
    for candidate in candidates:
        buckets.setdefault(oracle(candidate),[]).append(candidate)
    keys = list(buckets)
    rng.shuffle(keys)
    for key in keys:
        chosen.append(rng.choice(buckets[key]))
    count = settings['verifier_candidates_per_image']
    chosen = chosen[:count]
    while len(chosen)<count:
        chosen.append(rng.choice(candidates))
    selected = chosen if validation else [chosen[(epoch+seed+draw_index) % len(chosen)]]
    answer = sample.label+VIS if sample.presence else 'no_event'
    what_feedback = None
    if epoch%2 and oracle(chosen[0]) == 'reclassify':
        what_feedback = {'verdict':'reclassify','candidate':chosen[0]}
    result = [{'role':'what','answer':answer,'candidate':None,'feedback':what_feedback,'weight':1.0}]
    if sample.presence:
        # Alternate ordinary localization and full-image corrective localization.
        bad_boxes = [c for c in candidates if oracle(c)=='relocalize']
        if oracle(chosen[0])=='relocalize':
            bad_boxes.insert(0,chosen[0])
        feedback = {'verdict':'relocalize','candidate':bad_boxes[0]} if epoch%2 and bad_boxes else None
        if what_feedback is not None:
            feedback=what_feedback
        result.append({'role':'where','answer':answer,'candidate':{'category':sample.label,'bbox_1000':None},
                       'feedback':feedback,'weight':1.0})
    for candidate in selected:
        result.append({'role':'verify','answer':VERDICT_CODES[oracle(candidate)],'candidate':candidate,
                       'feedback':None,'weight':1.0/len(selected)})
    # Each role has equal total weight per image, irrespective of verifier candidate count.
    total = sum(e['weight'] for e in result)
    for e in result:
        e['weight'] /= total
    return result
