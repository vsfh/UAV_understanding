"""Prepare frozen visual features, train a visual reward, then action-only GRPO.

Default reward targets are GT class agreement times contextual-box IoU.
Optional train-only human preference pairs add a semantic ranking objective.
No claim of reproducing Perceval or learning unannotated evidence preferences.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from clear_uav.table4 import read_discovery_samples, labels_from_config
from train_perception_qwen import Collator
from perception_extension_runtime import build_baseline as build_model
from perception_reward_model import (VisualQualityModel, candidate_targets, pairwise_loss,
    resolved, feature_spec, cache_spec, spec_hash, validate_box, file_sha,
    validate_manifest, load_cached_feature, reward_training_metadata)

DEFAULT = 'configs/yaml/perception_reward.yaml'


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    config['output'] = str(resolved(config['output'], config))
    config['model']['initial_checkpoint'] = str(resolved(config['model']['initial_checkpoint'], config))
    return config


def selected_samples(config, split, limit=None):
    rows = read_discovery_samples(config, config['protocol'], split)
    if limit is not None:
        if limit <= 0:
            raise ValueError('Sample limits must be positive')
        # Must match actions.train samples[:N] for bounded end-to-end smoke runs.
        rows = rows[:limit]
    if not rows:
        raise ValueError(f'No {split} records')
    return rows


def session_metadata(config, split):
    """DiscoverySample.group_id is content_group_id, NOT the CSV session_id."""
    name = 'test_inputs.csv' if split == 'test' else f'{split}.csv'
    path = Path(config['data']['root']) / config['protocol'] / name
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {row['record_uid']: row.get('session_id') or None for row in rows}


def record_session(sample, sessions, protocol):
    # Added no_event images do not occur in the positive CSV. Their neggrp ID
    # is the timestamp-gap temporal cluster defined by table4.no_event_groups.
    session = sessions.get(sample.record_uid)
    if session is None and sample.record_uid.startswith('neg_'):
        session = sample.group_id
    if protocol == 'session_disjoint' and not session:
        raise ValueError(f'Missing CSV session_id / negative temporal group: {sample.record_uid}')
    return session


def prepare(config, max_train=None, max_val=None, splits=('train', 'val'), max_test=None):
    if not splits or any(split not in ('train', 'val', 'test') for split in splits):
        raise ValueError('Reward feature preparation supports train, val, or explicit held-out test')
    cache = resolved(config['reward_model']['cache'], config)
    spec = cache_spec(config)
    manifest_path = cache / 'manifest.json'
    manifest = {'spec': spec, 'splits': {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['spec'] != spec:
            raise ValueError('Existing reward cache has different provenance; use a new cache directory')
    cache.mkdir(parents=True, exist_ok=True)
    model, processor = build_model(config, config['model']['initial_checkpoint'])
    model.eval().requires_grad_(False)
    settings = feature_spec(config)
    projection = torch.randn(model.vlm.config.text_config.hidden_size, settings['channels'],
        generator=torch.Generator().manual_seed(settings['projection_seed']))
    projection = (projection / projection.shape[0] ** .5).to(config['device'])
    captured = {}

    def capture(module, args, output):
        hidden = output[0]
        selected = hidden[0, captured['mask'][0]].float() @ projection
        t, h, w = captured['grid']
        merge = model.vlm.config.vision_config.spatial_merge_size
        h, w = h // merge, w // merge
        if t != 1 or len(selected) != h * w:
            raise ValueError('Reward extractor requires one static image and matching spatial tokens')
        grid = selected.T.reshape(1, settings['channels'], h, w)
        captured['features'] = F.adaptive_avg_pool2d(grid, settings['grid_size'])[0].to(torch.float16).cpu()

    hook = model.vlm.get_base_model().model.language_model.register_forward_hook(capture)
    collator = Collator(processor, config, training=False)
    try:
        limits = {'train': max_train, 'val': max_val, 'test': max_test}
        for split in splits:
            limit = limits[split]
            rows = selected_samples(config, split, limit)
            sessions = session_metadata(config, split)
            (cache / split).mkdir(exist_ok=True)
            records = []
            existing_records = {r['record_uid']: r for r in manifest['splits'].get(split, [])}
            for sample in tqdm(rows, desc=f'prepare reward {split}'):
                dest = cache / split / f'{sample.record_uid}.pt'
                image_hash = file_sha(sample.image_path)
                if dest.exists():
                    old = torch.load(dest, map_location='cpu', weights_only=True)
                    if old['image_sha256'] != image_hash or old['spec_hash'] != spec_hash(spec):
                        raise ValueError(f'Stale feature: {sample.record_uid}')
                    if sample.record_uid in existing_records:
                        load_cached_feature(cache, split, existing_records[sample.record_uid], spec)
                else:
                    inputs = {k: v.to(config['device']) for k, v in collator([sample])[0].items()}
                    captured['mask'] = inputs['input_ids'] == model.vlm.config.image_token_id
                    captured['grid'] = inputs['image_grid_thw'][0].tolist()
                    with torch.inference_mode(), torch.autocast(torch.device(config['device']).type, dtype=torch.bfloat16):
                        model.vlm(**inputs, use_cache=False, logits_to_keep=1)
                    torch.save({'features': captured.pop('features'), 'image_sha256': image_hash,
                                'spec_hash': spec_hash(spec)}, dest)
                # The feature manifest contains no labels or GT boxes.
                row = {'record_uid': sample.record_uid, 'group_id': sample.group_id,
                       'session_id': record_session(sample, sessions, config['protocol']),
                       'image_path': str(sample.image_path.resolve()),
                       'image_sha256': image_hash, 'feature_sha256': file_sha(dest)}
                load_cached_feature(cache, split, row, spec)
                records.append(row)
            manifest['splits'][split] = records
            validate_manifest(manifest)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    finally:
        hook.remove()
    print(f'Prepared {list(splits)} frozen image features: {cache}')


def cached_rows(config, split):
    cache = resolved(config['reward_model']['cache'], config)
    manifest = json.loads((cache / 'manifest.json').read_text())
    if manifest['spec'] != cache_spec(config):
        raise ValueError('Stale reward cache configuration')
    validate_manifest(manifest)
    wanted = {r['record_uid']: r for r in manifest['splits'][split]}
    rows = [r for r in selected_samples(config, split) if r.record_uid in wanted]
    if len(rows) != len(wanted):
        raise ValueError('Feature cache and current dataset IDs differ')
    sessions = session_metadata(config, split)
    for sample in rows:
        row = wanted[sample.record_uid]
        if row['image_sha256'] != file_sha(sample.image_path):
            raise ValueError(f'Current dataset image changed: {sample.record_uid}')
        if row.get('group_id') != sample.group_id or row.get('session_id') != record_session(sample, sessions, config['protocol']):
            raise ValueError(f'Current dataset grouping changed: {sample.record_uid}')
        load_cached_feature(cache, split, row, manifest['spec'])
    return cache, manifest, rows


def batch_for_rows(rows, cache, split, labels, settings, seed, device, manifest=None):
    images, boxes, classes, quality = [], [], [], []
    if manifest is None:
        manifest = json.loads((Path(cache) / 'manifest.json').read_text())
        validate_manifest(manifest)
    by_id = {r['record_uid']: r for r in manifest['splits'][split]}
    count = settings.get('candidates_per_image', 8)
    for i, sample in enumerate(rows):
        feature = load_cached_feature(cache, split, by_id[sample.record_uid], manifest['spec'])
        target = torch.tensor(sample.bbox_1000).float() / 1000 if sample.presence else None
        cls = labels.index(sample.label) if sample.presence else 0
        b, c, q = candidate_targets(target, cls, len(labels), count, torch.Generator().manual_seed(seed + i))
        if not validate_box(b).all():
            raise ValueError('Invalid candidate box')
        images.append(feature.unsqueeze(0).expand(count, -1, -1, -1))
        boxes.append(b); classes.append(c); quality.append(q)
    return (torch.cat(images).float().to(device), torch.cat(boxes).to(device),
            torch.cat(classes).to(device), torch.stack(quality).to(device))


def preferences(config, allowed_ids, labels):
    path = config['reward_model'].get('preference_pairs')
    if not path:
        return []
    records = [json.loads(line) for line in resolved(path, config).read_text().splitlines() if line.strip()]
    for row in records:
        if row['record_uid'] not in allowed_ids or row['category'] not in labels:
            raise ValueError('Preference pair is outside training IDs/ontology')
        if not validate_box(torch.tensor([row['better_box'], row['worse_box']])).all():
            raise ValueError('Preference boxes must be valid normalized xyxy')
    return records


def fit_reward(config, epochs=None):
    torch.manual_seed(config['seed'])
    rm = config['reward_model']
    settings = rm['train']
    out = Path(config['output']) / 'rm'
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite reward run: {out}')
    cache, manifest, train_rows = cached_rows(config, 'train')
    _, _, val_rows = cached_rows(config, 'val')
    metadata = reward_training_metadata(config, manifest)
    labels = labels_from_config(config)
    architecture = {'channels': feature_spec(config)['channels'], 'num_classes': len(labels),
                    'hidden_dim': rm.get('hidden_dim', 256), 'roi_size': rm.get('roi_size', 4)}
    model = VisualQualityModel(**architecture).to(config['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.get('learning_rate', 1e-3), weight_decay=.01)
    pairs = preferences(config, {r.record_uid for r in train_rows}, labels)
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    best, history = float('inf'), []
    bs = settings.get('batch_images', 4)
    epoch_count = settings.get('epochs', 5) if epochs is None else epochs
    if epoch_count < 1 or bs < 1:
        raise ValueError('Reward epochs and batch_images must be positive')
    for epoch in range(epoch_count):
        random.Random(config['seed'] + epoch).shuffle(train_rows)
        values = {}
        for split, rows in (('train', train_rows), ('val', val_rows)):
            model.train(split == 'train')
            sums, n = {'loss': 0., 'mse': 0., 'pairwise': 0.}, 0
            for start in tqdm(range(0, len(rows), bs), desc=f'visual reward {epoch + 1} {split}'):
                group = rows[start:start + bs]
                seed = config['seed'] + start + (epoch * 100000 if split == 'train' else 9000000)
                f, b, c, targets = batch_for_rows(group, cache, split, labels, settings, seed, config['device'], manifest)
                with torch.set_grad_enabled(split == 'train'):
                    logits = model(f, b, c).reshape_as(targets)
                    rank = pairwise_loss(logits, targets, margin=settings.get('ranking_margin', .05))
                    loss = F.binary_cross_entropy_with_logits(logits, targets) + settings.get('ranking_weight', .2) * rank
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite reward loss in {split}')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                sums['loss'] += loss.item() * len(group)
                sums['mse'] += F.mse_loss(logits.detach().sigmoid(), targets).item() * len(group)
                sums['pairwise'] += rank.item() * len(group)
                n += len(group)
            values[split] = {k: v / n for k, v in sums.items()}
            if split == 'train' and pairs:
                model.train()
                by_id = {r['record_uid']: r for r in manifest['splits']['train']}
                for pair in pairs:
                    feature = load_cached_feature(cache, 'train', by_id[pair['record_uid']], manifest['spec']).float().to(config['device'])
                    b = torch.tensor([pair['better_box'], pair['worse_box']], device=config['device'])
                    cls = torch.full((2,), labels.index(pair['category']), device=config['device'], dtype=torch.long)
                    logits = model(feature[None].expand(2, -1, -1, -1), b, cls)
                    loss = settings.get('human_preference_weight', .2) * F.softplus(-(logits[0] - logits[1]))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite human preference loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
        history.append({'epoch': epoch + 1, **values})
        if values['val']['loss'] < best:
            best = values['val']['loss']
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'architecture': architecture, 'labels': labels, 'training_metadata': metadata,
                        'feature_spec_hash': spec_hash(manifest['spec']), 'epoch': epoch + 1,
                        'target': 'class-conditioned contextual ROI IoU; optional train-only preferences'}, out / 'best.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'Reward model saved: {out / "best.pt"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT)
    parser.add_argument('--stage', choices=['prepare', 'rm', 'sft', 'grpo', 'train'], default='train')
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-val-samples', type=int)
    parser.add_argument('--max-test-samples', type=int)
    parser.add_argument('--prepare-split', choices=['train_val', 'test'], default='train_val')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--rm-epochs', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--checkpoint', help='Action warmstart; use the same SFT checkpoint for rules/RM GRPO')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.rm_epochs is not None or (args.stage == 'rm' and args.epochs is not None):
        config['reward_model']['train']['epochs'] = args.rm_epochs or args.epochs
    if args.output:
        config['output'] = args.output
        config['reward_model']['cache'] = str(Path(args.output) / 'feature_cache')
        config['reward_model']['checkpoint'] = str(Path(args.output) / 'rm' / 'best.pt')
    for key in ('max_train_samples', 'max_val_samples'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for stage in ('sft', 'grpo'):
        for key in ('epochs', 'steps'):
            if getattr(args, key) is not None:
                config[stage][key] = getattr(args, key)
    if args.checkpoint and args.stage == 'train':
        parser.error('Use --stage grpo --checkpoint for a shared SFT start; train creates its own SFT stage')
    if args.stage == 'prepare':
        prepare(config, args.max_train_samples, args.max_val_samples,
                splits=('test',) if args.prepare_split == 'test' else ('train', 'val'),
                max_test=args.max_test_samples)
    elif args.stage == 'rm':
        fit_reward(config, args.rm_epochs or args.epochs)
    else:
        from train_perception_actions import train
        if args.stage == 'train':
            prepare(config, args.max_train_samples, args.max_val_samples)
            fit_reward(config, args.rm_epochs)
            train(config, 'sft')
            train(config, 'grpo')
        else:
            train(config, args.stage, args.checkpoint)


if __name__ == '__main__':
    main()
