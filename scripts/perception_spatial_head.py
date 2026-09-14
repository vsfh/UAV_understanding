"""Spatial rereading of final-layer image tokens; no raw-ViT/multiscale branch.

Kept torch-only so geometry, masking, gradients, and checkpoint tests never need
to load Qwen or the 8B checkpoint.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def spatial_memory(hidden, input_ids, attention_mask, image_token_id,
                   image_grid_thw, spatial_merge_size=2, max_image_tokens=2048):
    """Pack one image per sample into padded memory and normalized (x,y,t).

    Qwen's flattened merged image grid has x varying fastest. The original
    sample collator supplies exactly one image per example. Fail explicitly if
    a caller provides multiple images or a different token/grid layout.
    """
    batch, length, _ = hidden.shape
    if input_ids.shape != (batch, length) or attention_mask.shape != input_ids.shape:
        raise ValueError("Image memory requires matching hidden, IDs and attention mask")
    if image_grid_thw is None or tuple(image_grid_thw.shape) != (batch, 3):
        raise ValueError("Spatial perception expects exactly one image_grid_thw row per sample")
    if spatial_merge_size < 1 or max_image_tokens < 1:
        raise ValueError("spatial_merge_size and max_image_tokens must be positive")
    memories, coordinates = [], []
    for row in range(batch):
        positions = ((input_ids[row] == image_token_id) & attention_mask[row].bool()).nonzero().flatten()
        t, h, w = [int(v) for v in image_grid_thw[row].tolist()]
        if t < 1 or h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(f"Invalid merged image grid {(t, h, w)}")
        h, w = h // spatial_merge_size, w // spatial_merge_size
        if h < 1 or w < 1 or len(positions) != t * h * w:
            raise ValueError(f"Sample {row}: {len(positions)} image tokens disagree with merged grid {(t, h, w)}")
        axis = lambda size: (torch.arange(size, device=hidden.device, dtype=torch.float32) + 0.5) / size * 2 - 1
        tt, yy, xx = torch.meshgrid(axis(t), axis(h), axis(w), indexing="ij")
        xyz = torch.stack((xx, yy, tt), dim=-1).reshape(-1, 3)
        if len(positions) > max_image_tokens:
            selected = torch.linspace(0, len(positions) - 1, max_image_tokens, device=hidden.device).round().long()
            positions, xyz = positions[selected], xyz[selected]
        memories.append(hidden[row, positions])
        coordinates.append(xyz)
    maximum = max(len(memory) for memory in memories)
    memory = torch.stack([F.pad(item, (0, 0, 0, maximum - len(item))) for item in memories])
    coords = torch.stack([F.pad(item, (0, 0, 0, maximum - len(item))) for item in coordinates])
    lengths = torch.tensor([len(item) for item in memories], device=hidden.device)
    padding_mask = torch.arange(maximum, device=hidden.device)[None] >= lengths[:, None]
    return memory, coords, padding_mask


class SpatialInteraction(nn.Module):
    """The <vis> state rereads image states before the existing ROI MLP.

    A zero-initialized scalar residual gate preserves a baseline warm start.
    On the first step the gate learns; subsequent steps train the attention
    projections. All calculations in this head use FP32, like the ROI MLP.
    """
    def __init__(self, hidden_dim, interaction_dim=256, num_heads=8,
                 dropout=0.1, residual_gate_init=0.0):
        super().__init__()
        if interaction_dim % num_heads:
            raise ValueError("interaction_dim must be divisible by num_heads")
        self.query_proj = nn.Linear(hidden_dim, interaction_dim)
        self.image_proj = nn.Linear(hidden_dim, interaction_dim)
        self.position = nn.Sequential(nn.Linear(3, interaction_dim), nn.GELU(), nn.Linear(interaction_dim, interaction_dim))
        self.query_norm = nn.LayerNorm(interaction_dim)
        self.memory_norm = nn.LayerNorm(interaction_dim)
        self.attention = nn.MultiheadAttention(interaction_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(interaction_dim, hidden_dim)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def forward(self, query, memory, coordinates, padding_mask):
        if padding_mask.all(dim=1).any():
            raise ValueError("Each image must contribute at least one unmasked token")
        query = query.float()
        q = self.query_norm(self.query_proj(query)).unsqueeze(1)
        m = self.memory_norm(self.image_proj(memory.float()) + self.position(coordinates.float()))
        attended, _ = self.attention(q, m, m, key_padding_mask=padding_mask, need_weights=False)
        return query + torch.tanh(self.residual_gate) * self.output_proj(attended[:, 0])


def save_spatial_heads(model, checkpoint):
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(model.box_head.state_dict(), checkpoint / "box_head.pt")
    torch.save(model.spatial_head.state_dict(), checkpoint / "spatial_head.pt")
    if getattr(model, "spatial_adapter", None) is not None:
        torch.save(model.spatial_adapter.state_dict(), checkpoint / "spatial_adapter.pt")


def load_spatial_heads(model, checkpoint, require_spatial=False):
    """Baseline warm starts contain only box_head.pt; spatial reloads need both."""
    checkpoint = Path(checkpoint)
    model.box_head.load_state_dict(torch.load(checkpoint / "box_head.pt", map_location="cpu", weights_only=True))
    if (checkpoint / "spatial_head.pt").is_file():
        model.spatial_head.load_state_dict(torch.load(checkpoint / "spatial_head.pt", map_location="cpu", weights_only=True))
    elif require_spatial:
        raise FileNotFoundError(f"Missing spatial head in extension checkpoint: {checkpoint}")
    if getattr(model, "spatial_adapter", None) is not None:
        adapter_path = checkpoint / "spatial_adapter.pt"
        if adapter_path.is_file() or require_spatial:
            model.spatial_adapter.load_state_dict(torch.load(
                adapter_path, map_location="cpu", weights_only=True))


def checkpoint_fingerprint(checkpoint):
    """Bind calibration to every adapter, processor, and spatial/box head file."""
    checkpoint = Path(checkpoint)
    for name in ('box_head.pt', 'spatial_head.pt'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f'Missing spatial checkpoint file: {checkpoint / name}')
    digest = hashlib.sha256()
    for path in sorted(p for p in checkpoint.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(checkpoint)).encode())
        digest.update(b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        digest.update(b'\0')
    return digest.hexdigest()


def evaluation_configuration(saved, runtime):
    """Evaluation limits belong to the current invocation, not its SFT run."""
    result = copy.deepcopy(saved)
    result['training_sample_limits'] = {key: saved.get(key) for key in
                                       ('max_train_samples', 'max_val_samples', 'max_test_samples')}
    for key in ('max_train_samples', 'max_val_samples', 'max_test_samples'):
        result.pop(key, None)
    for key in ('max_val_samples', 'max_test_samples'):
        if runtime.get(key) is not None:
            result[key] = runtime[key]
    result['device'], result['test'], result['output'] = runtime['device'], copy.deepcopy(runtime['test']), runtime['output']
    return result


def evaluation_signature(config):
    values = {key: config.get(key) for key in ('protocol', 'seed', 'model', 'spatial', 'input', 'prompt', 'data', 'validation')}
    # Preserve existing signatures for old, non-adapter spatial checkpoints.
    if 'spatial_lora' in config:
        values['spatial_lora'] = config['spatial_lora']
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate_calibration(calibration, config, checkpoint, fingerprint):
    if calibration.get('split') != 'val':
        raise ValueError('Calibration must come from validation; rerun --split val')
    # Checkpoint location is provenance, not identity: allow moving the repository.
    if not fingerprint or calibration.get('checkpoint_sha256') != fingerprint:
        raise ValueError('Calibration checkpoint content differs; use the calibrated weights or rerun --split val')
    if calibration.get('protocol') != config['protocol'] or calibration.get('seed') != config['seed']:
        raise ValueError('Calibration protocol/seed differs; rerun --split val')
    if calibration.get('evaluation_signature') != evaluation_signature(config):
        raise ValueError('Calibration model/input/prompt/data/validation configuration differs; rerun --split val')
    if calibration.get('limited_run') and config.get('max_test_samples') is None and not config['data'].get('max_samples'):
        raise ValueError('Refusing full test with tiny-run calibration; run full --split val first')
    threshold = calibration.get('threshold')
    # Existing select_threshold may use nextafter(1,+inf) to reject all events.
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1., math.inf):
        raise ValueError('Invalid calibrated threshold')


def final_output_fpr(samples, predictions, threshold, labels):
    """Negative FPR requiring a valid emitted class AND valid ROI AND score gate."""
    if len(samples) != len(predictions):
        raise ValueError('Sample/prediction counts differ')
    supported = set(labels)
    emitted, negatives = 0, 0
    for sample, prediction in zip(samples, predictions):
        if sample.presence:
            continue
        negatives += 1
        box, score = prediction.get('bbox_1000'), prediction.get('presence_score', float('nan'))
        valid_box = (box is not None and len(box) == 4 and all(math.isfinite(float(x)) for x in box)
                     and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)
        emitted += bool(prediction.get('valid') and prediction.get('category') in supported and valid_box
                        and math.isfinite(float(score)) and score >= threshold)
    return emitted / negatives if negatives else None
