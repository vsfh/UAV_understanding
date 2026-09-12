"""Evaluate the RM-trained action policy, or the frozen ROI reward quality.

Policy testing needs only the policy checkpoint and images. The RM is a
training-only verifier; it is not called with test GT or used at inference.
Quality evaluation creates synthetic candidates from evaluation annotations
strictly to measure the frozen verifier, never to select a policy prediction.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from perception_reward_model import VisualQualityModel, resolved, spec_hash, validate_reward_binding
from train_perception_reward import read_config, cached_rows, batch_for_rows


def evaluate_quality(config, split='val', limit=None):
    from clear_uav.table4 import labels_from_config
    cache, manifest, rows = cached_rows(config, split)
    if limit is not None:
        if limit < 1:
            raise ValueError('Sample limit must be positive')
        rows = rows[:limit]
    labels = labels_from_config(config)
    checkpoint = resolved(config['reward_model']['checkpoint'], config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['labels'] != labels or saved['feature_spec_hash'] != spec_hash(manifest['spec']):
        raise ValueError('Reward checkpoint and feature cache differ')
    validate_reward_binding(saved.get('training_metadata'), config, manifest)
    model = VisualQualityModel(**saved['architecture']).to(config['device'])
    model.load_state_dict(saved['state_dict']); model.eval()
    # Keep candidate sampling/ranking settings tied to this frozen RM run; a
    # later YAML's optimizer settings are neither a provenance error nor a new
    # training run. The actual training settings remain in the checkpoint.
    settings = saved['training_metadata']['training_settings']
    batch_size = settings.get('batch_images', 4)
    sums = {'squared_error': 0., 'absolute_error': 0., 'pair_correct': 0., 'pair_count': 0., 'candidates': 0}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            f, b, c, target = batch_for_rows(rows[start:start + batch_size], cache, split,
                labels, settings, 19000000 + start, config['device'])
            scores = model(f, b, c).sigmoid().view_as(target)
            delta = target[:, :, None] - target[:, None, :]
            selected = delta > settings.get('ranking_margin', .05)
            ranked = scores[:, :, None] - scores[:, None, :]
            sums['pair_correct'] += ((ranked > 0) & selected).sum().item()
            sums['pair_count'] += selected.sum().item()
            sums['squared_error'] += (scores - target).square().sum().item()
            sums['absolute_error'] += (scores - target).abs().sum().item()
            sums['candidates'] += target.numel()
    result = {'split': split, 'checkpoint': str(checkpoint.resolve()), 'images': len(rows),
        'limited_run': limit is not None, 'feature_spec_hash': saved['feature_spec_hash'],
        'reward_training_settings': saved['training_metadata']['training_settings'],
        'candidate_seed': 19000000, 'candidate_distribution': 'GT-jitter, random, wrong-class, full-frame',
        'rmse': (sums['squared_error'] / sums['candidates']) ** .5,
        'mae': sums['absolute_error'] / sums['candidates'],
        'pairwise_accuracy': sums['pair_correct'] / sums['pair_count'] if sums['pair_count'] else None,
        'pair_count': int(sums['pair_count']),
        'target': 'class-correct contextual-box IoU (not an independent semantic quality judgment)'}
    output = Path(config['output']) / 'rm' / f'{split}_quality.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/yaml/perception_reward.yaml')
    parser.add_argument('--mode', choices=['policy', 'quality'], default='policy')
    parser.add_argument('--split', choices=['val', 'test', 'all'], default='all')
    parser.add_argument('--stage', choices=['sft', 'grpo'], default='grpo')
    parser.add_argument('--checkpoint', help='Action policy checkpoint (policy mode only)')
    parser.add_argument('--output')
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--max-test-samples', type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.output:
        config['output'] = args.output
        config['reward_model']['cache'] = str(Path(args.output) / 'feature_cache')
        config['reward_model']['checkpoint'] = str(Path(args.output) / 'rm' / 'best.pt')
    if args.mode == 'quality':
        if args.checkpoint:
            parser.error('quality uses reward_model.checkpoint from the YAML')
        for split in (['val', 'test'] if args.split == 'all' else [args.split]):
            evaluate_quality(config, split, args.max_val_samples if split == 'val' else args.max_test_samples)
    else:
        from test_perception_actions import evaluate
        evaluate(config, args.split, args.stage, args.checkpoint, args.max_val_samples, args.max_test_samples)


if __name__ == '__main__':
    main()
