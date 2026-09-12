"""Train-only visual region-quality reward; independent of the policy's features.

Frozen baseline image-token grids are cached before reward/policy training.
This is a supervised visual quality model, not a pretrained hallucination PRM.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TARGET_SCHEMA = 'class_conditioned_contextual_iou_v1_optional_train_pairs'


def resolved(value, config):
    return Path(str(value).format(protocol=config['protocol'], seed=config['seed']))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def feature_spec(config):
    rm = config['reward_model']
    return {'channels': int(rm.get('feature_channels', 64)),
            'grid_size': int(rm.get('grid_size', 16)),
            'projection_seed': int(rm.get('projection_seed', 1729))}


def checkpoint_fingerprint(path):
    path = Path(path)
    patterns = ('*.json', '*.jinja', '*.txt', '*.model', '*.tiktoken',
                'adapter*.safetensors', 'adapter*.bin', 'box_head.pt')
    paths = {p for pattern in patterns for p in path.glob(pattern) if p.is_file()}
    files = {p.name: file_sha(p) for p in sorted(paths)}
    if not {'adapter_config.json', 'box_head.pt'} <= files.keys():
        raise ValueError(f'Expected a Perception checkpoint: {path}')
    if not any(p.name.startswith('adapter') and p.suffix in ('.bin', '.safetensors') for p in paths):
        raise ValueError(f'No adapter weight file in checkpoint: {path}')
    return files


def base_fingerprint(path):
    """Practical local identity: hash configs; stat 8B weight shards, not their data.

    This detects replaced/updated weights without rereading tens of GB. It is
    an operational cache identity, not a cryptographic hash of model weights.
    """
    path = Path(path).resolve()
    weights = sorted(set(path.glob('*.safetensors')) | set(path.glob('pytorch_model*.bin')))
    if not weights or not (path / 'config.json').is_file():
        raise ValueError(f'Expected a local base model with weight files: {path}')
    return {'path': str(path),
            'weight_stat': {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in weights},
            'config_hashes': {p.name: file_sha(p) for p in sorted(path.glob('*.json'))}}


def cache_spec(config):
    initial = resolved(config['model']['initial_checkpoint'], config)
    return {'schema': 2, 'protocol': config['protocol'], 'seed': config['seed'],
            'checkpoint': str(initial.resolve()), 'checkpoint_hashes': checkpoint_fingerprint(initial),
            'base_model': base_fingerprint(config['model']['path']),
            'input': config['input'], 'prompt': config['prompt'], 'features': feature_spec(config),
            'labels_sha256': file_sha(config['data']['labels']),
            'definitions_sha256': file_sha(config['data']['definitions'])}


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def validate_manifest(manifest):
    """group_id is content_group_id; session_id is recorded separately."""
    rows = manifest['splits']
    for split in ('train', 'val', 'test'):
        ids = [r['record_uid'] for r in rows.get(split, [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate reward cache UID inside {split}')
        for row in rows.get(split, []):
            uid = row['record_uid']
            if not isinstance(uid, str) or not uid or '/' in uid or '\\' in uid or uid in ('.', '..'):
                raise ValueError('Invalid reward cache UID')
            if not row.get('image_sha256') or not row.get('feature_sha256') or not row.get('image_path'):
                raise ValueError('Cache manifest requires image path and image/feature checksums; prepare a schema-2 cache')
    fields = ['record_uid', 'image_sha256', 'group_id']
    if manifest['spec']['protocol'] == 'session_disjoint':
        fields.append('session_id')
    for field in fields:
        for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            a = {r[field] for r in rows.get(left, []) if r.get(field)}
            b = {r[field] for r in rows.get(right, []) if r.get(field)}
            if a & b:
                raise ValueError(f'{left}/{right} reward cache leakage in {field}')
    if manifest['spec']['protocol'] == 'session_disjoint':
        if any(not r.get('session_id') for split in ('train', 'val', 'test') for r in rows.get(split, [])):
            raise ValueError('session_disjoint cache requires an explicit session_id for every record')


def load_cached_feature(cache, split, row, spec):
    """Validate each small tensor file against its immutable manifest entry."""
    path = Path(cache) / split / f"{row['record_uid']}.pt"
    if file_sha(path) != row['feature_sha256']:
        raise ValueError(f"Feature file checksum changed: {row['record_uid']}")
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('spec_hash') != spec_hash(spec) or state.get('image_sha256') != row['image_sha256']:
        raise ValueError(f"Feature provenance mismatch: {row['record_uid']}")
    feature = state.get('features')
    settings = spec['features']
    shape = (settings['channels'], settings['grid_size'], settings['grid_size'])
    if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != shape:
        raise ValueError(f"Wrong cached feature shape: {row['record_uid']}, expected {shape}")
    if not feature.is_floating_point() or not torch.isfinite(feature).all():
        raise ValueError(f"Nonfinite/nonfloating cached feature: {row['record_uid']}")
    return feature


def reward_training_metadata(config, manifest):
    """Bind RM selection and targets without ever providing GT to score()."""
    validate_manifest(manifest)
    splits = {split: sorted(manifest['splits'].get(split, []), key=lambda r: r['record_uid'])
              for split in ('train', 'val')}
    preference = config['reward_model'].get('preference_pairs')
    data_root = Path(config['data']['root']) / config['protocol']
    source_files = {'train_csv': data_root / 'train.csv', 'val_csv': data_root / 'val.csv',
                    'bbox_annotations': Path(config['data']['bbox_annotations'])}
    return {'target_schema': TARGET_SCHEMA,
            'train_uids_sha256': spec_hash([r['record_uid'] for r in splits['train']]),
            'val_uids_sha256': spec_hash([r['record_uid'] for r in splits['val']]),
            'cache_records_sha256': spec_hash(splits),
            'supervision_sources_sha256': {name: file_sha(path) for name, path in source_files.items()},
            'preference_sha256': file_sha(resolved(preference, config)) if preference else None,
            'training_settings': config['reward_model'].get('train', {})}


def validate_reward_binding(saved_metadata, config, manifest):
    """Check frozen RM provenance while retaining its actual training settings.

    A later quality/GRPO invocation does not retrain the RM: its YAML epochs,
    learning rate, and other RM optimizer options need not equal the settings
    that produced the checkpoint. Data, candidates' target schema, preference
    source, and cache identity must still match exactly.
    """
    if not isinstance(saved_metadata, dict):
        raise ValueError('Reward checkpoint has no saved training provenance')
    if not isinstance(saved_metadata.get('training_settings'), dict):
        raise ValueError('Reward checkpoint has no actual RM training settings')
    expected = reward_training_metadata(config, manifest)
    actual_binding = {key: value for key, value in saved_metadata.items() if key != 'training_settings'}
    expected_binding = {key: value for key, value in expected.items() if key != 'training_settings'}
    if actual_binding != expected_binding:
        raise ValueError('Reward checkpoint training UID/target/preference/cache binding mismatch')


def validate_box(boxes):
    return (torch.isfinite(boxes).all(-1) & (boxes >= 0).all(-1) & (boxes <= 1).all(-1)
            & (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1]))


def aligned_iou(boxes, target):
    lo = torch.maximum(boxes[..., :2], target[..., :2])
    hi = torch.minimum(boxes[..., 2:], target[..., 2:])
    inter = (hi - lo).clamp_min(0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(-1)
    return inter / (area + target_area - inter).clamp_min(1e-8)


def region_grid(features, boxes, size=4):
    """Differentiable sampling at ROI bin centres in normalized full-image space."""
    if features.ndim != 4 or boxes.shape != (features.shape[0], 4):
        raise ValueError('Expected [B,C,H,W] features and [B,4] boxes')
    offsets = (torch.arange(size, device=boxes.device, dtype=boxes.dtype) + .5) / size
    x = boxes[:, 0, None] + offsets * (boxes[:, 2] - boxes[:, 0])[:, None]
    y = boxes[:, 1, None] + offsets * (boxes[:, 3] - boxes[:, 1])[:, None]
    grid = torch.stack((x[:, None, :].expand(-1, size, -1),
                        y[:, :, None].expand(-1, -1, size)), -1)
    return F.grid_sample(features.float(), grid.float() * 2 - 1,
                         align_corners=False, padding_mode='border')


class VisualQualityModel(nn.Module):
    def __init__(self, channels, num_classes, hidden_dim=256, roi_size=4):
        super().__init__()
        self.roi_size = roi_size
        self.class_embedding = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(nn.LayerNorm(channels * (roi_size * roi_size + 1) + 8 + 32),
            nn.Linear(channels * (roi_size * roi_size + 1) + 8 + 32, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, features, boxes, category_ids):
        local = region_grid(features, boxes, self.roi_size).flatten(1)
        global_ = features.float().mean((-2, -1))
        geometry = torch.cat((boxes, (boxes[:, :2] + boxes[:, 2:]) / 2,
                              boxes[:, 2:] - boxes[:, :2]), -1)
        return self.net(torch.cat((local, global_, geometry.float(), self.class_embedding(category_ids)), -1)).squeeze(-1)


def candidate_targets(target_box, category_id, num_classes, count, generator):
    """Training/evaluation candidates and soft GT quality targets; never used in inference."""
    if count < 4 or num_classes < 2:
        raise ValueError('At least four candidates and two classes are required')
    raw = torch.rand(count, 4, generator=generator)
    low, high = torch.minimum(raw[:, :2], raw[:, 2:]), torch.maximum(raw[:, :2], raw[:, 2:])
    boxes = torch.cat((low, torch.maximum(high, low + 1e-4).clamp_max(1)), -1)
    classes = torch.randint(num_classes, (count,), generator=generator)
    quality = torch.zeros(count)
    if target_box is not None:
        gt = torch.as_tensor(target_box, dtype=torch.float32)
        if gt.shape != (4,) or not validate_box(gt) or not 0 <= category_id < num_classes:
            raise ValueError('GT candidate target must be a valid normalized xyxy box and category')
        boxes[0] = gt
        boxes[1] = gt
        classes[0] = category_id
        classes[1] = (category_id + 1) % num_classes
        boxes[2] = torch.tensor([0., 0., 1., 1.])
        classes[2:] = category_id
        for i in range(3, count):
            center = (gt[:2] + gt[2:]) / 2
            wh = (gt[2:] - gt[:2]).clamp_min(.01)
            center = center + torch.randn(2, generator=generator) * wh * .65
            wh = wh * (torch.randn(2, generator=generator) * .5).exp()
            lo, hi = (center - wh / 2).clamp(0, .999), (center + wh / 2).clamp(.001, 1)
            boxes[i] = torch.cat((lo, torch.maximum(hi, lo + .001).clamp_max(1)))
        quality = aligned_iou(boxes, gt) * (classes == category_id)
    return boxes, classes, quality


def pairwise_loss(logits, targets, margin=.05):
    delta = targets[:, :, None] - targets[:, None, :]
    selected = delta > margin
    differences = logits[:, :, None] - logits[:, None, :]
    if not selected.any():
        return logits.sum() * 0
    return F.softplus(-differences[selected]).mean()


class CachedVisualReward:
    """Frozen reward provider for action-only GRPO; score never reads sample targets."""
    def __init__(self, config, labels, device):
        self.device = torch.device(device)
        self.labels = list(labels)
        rm = config['reward_model']
        self.cache = resolved(rm['cache'], config)
        self.manifest = json.loads((self.cache / 'manifest.json').read_text())
        validate_manifest(self.manifest)
        self.spec = self.manifest['spec']
        if self.spec != cache_spec(config):
            raise ValueError('Reward feature cache provenance does not match this run')
        state = torch.load(resolved(rm['checkpoint'], config), map_location='cpu', weights_only=True)
        if state['labels'] != self.labels or state['feature_spec_hash'] != spec_hash(self.spec):
            raise ValueError('Reward checkpoint/cache/ontology mismatch')
        validate_reward_binding(state.get('training_metadata'), config, self.manifest)
        self.model = VisualQualityModel(**state['architecture']).to(self.device)
        self.model.load_state_dict(state['state_dict'], strict=True)
        self.model.eval().requires_grad_(False)
        self.rows = {}
        self._verified_image_stats = {}
        for split in ('train', 'val'):
            for row in self.manifest['splits'].get(split, []):
                if row['record_uid'] in self.rows:
                    raise ValueError('Train/val reward cache ID overlap')
                self.rows[row['record_uid']] = (split, row)

    @torch.no_grad()
    def score(self, record_uid, boxes, category):
        boxes = torch.as_tensor(boxes, device=self.device, dtype=torch.float32).reshape(-1, 4)
        if record_uid not in self.rows:
            raise ValueError(f'Reward cache lacks training/validation UID: {record_uid}')
        if category not in self.labels:
            return torch.zeros(len(boxes), device=self.device)
        split, row = self.rows[record_uid]
        image_path = Path(row['image_path'])
        stat = image_path.stat()
        image_stat = (stat.st_size, stat.st_mtime_ns)
        if self._verified_image_stats.get(record_uid) != image_stat:
            if file_sha(image_path) != row['image_sha256']:
                raise ValueError(f'Current reward image changed: {record_uid}')
            self._verified_image_stats[record_uid] = image_stat
        feature = load_cached_feature(self.cache, split, row, self.spec).float().to(self.device)
        classes = torch.full((len(boxes),), self.labels.index(category), device=self.device, dtype=torch.long)
        scores = self.model(feature.unsqueeze(0).expand(len(boxes), -1, -1, -1), boxes.clamp(0, 1), classes).sigmoid()
        return torch.where(validate_box(boxes), scores, torch.zeros_like(scores))
