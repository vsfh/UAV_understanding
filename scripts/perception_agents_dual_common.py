"""Portable prepared samples and answer-only batched loss for shared agent SFT."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def cache_path(config):
    return Path(str(config['proposal_cache']).format(protocol=config['protocol'], seed=config['seed']))


def semantic_config(config):
    values = {key: copy.deepcopy(config[key]) for key in ('protocol', 'seed', 'data', 'spatial', 'input', 'prompt')}
    for key in ('root', 'ontology', 'labels', 'definitions', 'bbox_annotations'):
        values['data'].pop(key, None)
    values['data'].get('no_event', {}).pop('root', None)
    values['model'] = {'head_dim': config['model']['head_dim']}
    return values


def relocated_fingerprint(samples, current_root, original_root):
    """Reproduce the old hash with its original mount prefix; all GT and stat fields stay checked."""
    rows = []
    for sample in samples:
        stat = sample.image_path.stat()
        old_image = Path(original_root) / sample.image_path.resolve().relative_to(current_root)
        rows.append([sample.record_uid, sample.group_id, old_image.as_posix(), stat.st_size,
                     stat.st_mtime_ns, sample.presence, sample.label, sample.bbox_1000])
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def export_cache(config):
    """Once on the source host: freeze existing rows, never predict or select anew."""
    from train_perception_agents import read_splits, prepare_binding
    folder = cache_path(config)
    saved = json.loads((folder / 'manifest.json').read_text())
    source = copy.deepcopy(saved['source_config'])
    source['initial_checkpoint'] = config['initial_checkpoint']
    source['agents'] = {'bootstrap_max_samples': None}
    print('Checking existing proposal manifest and fixed train/val records (no inference)...', flush=True)
    splits = read_splits(source)
    actual = prepare_binding(source, splits)
    root = Path(source['data']['root']).resolve()
    if actual != saved:
        relative_checkpoint = Path(str(source['initial_checkpoint']).format(protocol=source['protocol'], seed=source['seed']))
        if relative_checkpoint.is_absolute():
            raise ValueError('For source cache export use the repository-relative initial_checkpoint path')
        original_repo = Path(saved['source_checkpoint'])
        for part in relative_checkpoint.parts:
            original_repo = original_repo.parent
        original_root = original_repo / source['data']['root']
        actual['source_checkpoint'] = saved['source_checkpoint']
        actual['splits'] = {s: relocated_fingerprint(v, root, original_root) for s, v in splits.items()}
        if actual != saved:
            differences = [k for k in saved if actual.get(k) != saved[k]]
            raise ValueError(f'Prepared cache differs beyond its mount prefix: {differences}')
        print('Original mount prefix differs; checkpoint, labels, GT and file stat fingerprints match.', flush=True)
    rows = {}
    for split, samples in splits.items():
        proposals = json.loads((folder / f'{split}.json').read_text())
        if {sample.record_uid for sample in samples} != set(proposals):
            raise ValueError(f'Incomplete prepared {split} proposals')
        rows[split] = []
        for sample in samples:
            rows[split].append({
                'record_uid': sample.record_uid, 'label': sample.label, 'presence': sample.presence,
                'image_path': sample.image_path.resolve().relative_to(root).as_posix(),
                'image_bytes': sample.image_path.stat().st_size,
                'bbox_1000': sample.bbox_1000, 'group_id': sample.group_id,
                'negative_subtype': sample.negative_subtype,
            })
    result = {'version': 1, 'semantic_config': semantic_config(source),
              'source_manifest_sha256': file_hash(folder / 'manifest.json'),
              'proposal_sha256': {s: file_hash(folder / f'{s}.json') for s in rows},
              'samples': rows}
    path = folder / 'portable_samples.json'
    text = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise FileExistsError(f'Different portable sample index already exists: {path}')
    path.write_text(text, encoding='utf-8')
    print(json.dumps({'portable_cache': str(path), 'samples': {s: len(v) for s, v in rows.items()}}, indent=2))


def load_cached_splits(config):
    """Read the exact exported rows. No discovery resampling and no inference."""
    from clear_uav.table4 import DiscoverySample
    from perception_spatial_head import checkpoint_fingerprint
    folder = cache_path(config)
    portable = json.loads((folder / 'portable_samples.json').read_text(encoding='utf-8'))
    manifest = json.loads((folder / 'manifest.json').read_text())
    if portable['version'] != 1 or portable['source_manifest_sha256'] != file_hash(folder / 'manifest.json'):
        raise ValueError('Portable index and original prepared manifest differ')
    if portable['semantic_config'] != semantic_config(config):
        raise ValueError('Prepared dataset, model geometry, input or prompts differ from requested config')
    checkpoint = Path(str(config['initial_checkpoint']).format(protocol=config['protocol'], seed=config['seed']))
    if checkpoint_fingerprint(checkpoint) != manifest['source_sha256']:
        raise ValueError('Spatial warm-start weights differ from the prepared candidates')
    for key, expected in manifest['category_files'].items():
        if file_hash(config['data'][key]) != expected:
            raise ValueError(f'Prepared category file changed: {key}')
    root = Path(config['data']['root'])
    splits, cached = {}, {}
    for split, rows in portable['samples'].items():
        if file_hash(folder / f'{split}.json') != portable['proposal_sha256'][split]:
            raise ValueError(f'Prepared {split} predictions changed')
        cached[split] = json.loads((folder / f'{split}.json').read_text())
        if len({r['record_uid'] for r in rows}) != len(rows) or {r['record_uid'] for r in rows} != set(cached[split]):
            raise ValueError(f'Prepared {split} record IDs differ')
        samples = []
        for row in rows:
            image = root / row['image_path']
            if image.stat().st_size != row['image_bytes']:
                raise ValueError(f'Prepared image size changed: {image}')
            samples.append(DiscoverySample(
                record_uid=row['record_uid'], label=row['label'], presence=row['presence'],
                image_path=image, evidence_path=None,
                bbox_1000=tuple(row['bbox_1000']) if row['bbox_1000'] is not None else None,
                group_id=row['group_id'], negative_subtype=row['negative_subtype']))
        limit = config.get(f'max_{split}_samples')
        splits[split] = samples[:limit] if limit is not None else samples
    return splits, cached, manifest


def build_fast_model(config, trainable=False, checkpoint=None):
    """Use local weights with explicitly selected attention and checkpointing."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from train_perception_spatial import SpatialPerceptionQwen
    from perception_spatial_head import load_spatial_heads
    from clear_uav.modeling import enable_offline_mode, require_local_model
    from perception_agents_core import VIS
    enable_offline_mode()
    source = Path(str(checkpoint or config['initial_checkpoint']).format(protocol=config['protocol'], seed=config['seed']))
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        require_local_model(config['model']['path']), dtype=torch.bfloat16,
        local_files_only=True, disable_mmap=config['model'].get('disable_mmap', False),
        attn_implementation=config['model'].get('attn_implementation', 'sdpa'))
    base.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
    vlm = PeftModel.from_pretrained(base, source, is_trainable=trainable)
    if trainable:
        vlm.enable_input_require_grads()
        if config['train'].get('gradient_checkpointing', False):
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            vlm.gradient_checkpointing_disable()
    model = SpatialPerceptionQwen(vlm, processor.tokenizer.convert_tokens_to_ids(VIS), config)
    load_spatial_heads(model, source, require_spatial=True)
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return model.to(config['device']), processor


def role_batches(samples, cached_split, labels, config, epoch=0, validation=False, draw_indices=None):
    from perception_agents_data import example_set
    groups = {role: [] for role in ('what', 'where', 'verify')}
    if draw_indices is None:
        draw_indices = range(len(samples))
    for sample, draw_index in zip(samples, draw_indices):
        for example in example_set(sample, cached_split[sample.record_uid], labels, config['agents'],
                                   epoch, validation=validation, draw_index=draw_index):
            groups[example['role']].append((sample, example))
    size = config['validation'].get('batch_size', config['train']['batch_size']) if validation else config['train']['batch_size']
    for role, rows in groups.items():
        for start in range(0, len(rows), size):
            chunk = rows[start:start + size]
            yield {'role': role, 'samples': [r[0] for r in chunk], 'examples': [r[1] for r in chunk]}


def encode_batch(processor, config, batch):
    from perception_agents_runtime import marked_image
    from perception_agents_core import make_messages
    from clear_uav.modeling import assistant_only_labels
    from clear_uav.table4 import category_block
    # One immutable category list per processor/run, avoiding repeated filesystem reads.
    if not hasattr(processor, '_agents_category_text'):
        processor._agents_category_text = category_block(config)
    images, messages = [], []
    for sample, example in zip(batch['samples'], batch['examples']):
        if batch['role'] == 'verify':
            image = marked_image(sample.image_path, example['candidate'], config['agents']['marker_width'])
            images.append(image)
        else:
            image = str(sample.image_path)
        messages.append(make_messages(image, batch['role'], processor._agents_category_text, candidate=example['candidate'],
                                      feedback=example['feedback'], answer=example['answer']))
    try:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors='pt', processor_kwargs={'padding': True, 'size': {
                'longest_edge': config['input']['max_pixels'], 'shortest_edge': config['input']['min_pixels']}})
    finally:
        for image in images:
            image.close()
    inputs['labels'] = assistant_only_labels(inputs['input_ids'], inputs['attention_mask'], processor.tokenizer)
    return inputs


def answer_window(labels):
    """The preceding token must be kept for causal prediction of the first answer token."""
    import torch
    valid = labels.ne(-100)
    if not valid.any(dim=1).all():
        raise ValueError('Every role must contain a supervised answer')
    first = torch.where(valid)[1].min().item()
    if first == 0:
        raise ValueError('An answer needs a preceding prompt token')
    return labels.shape[1] - first + 1


def per_example_language_loss(logits, labels):
    import torch.nn.functional as F
    targets = labels[:, -(logits.shape[1] - 1):]
    shifted = logits[:, :-1, :].float()
    token_loss = F.cross_entropy(shifted.reshape(-1, shifted.shape[-1]), targets.reshape(-1),
                                 ignore_index=-100, reduction='none').view_as(targets)
    return token_loss.sum(1) / targets.ne(-100).sum(1)


class RoleBatchLoss:
    """Sum of per-image, equal-role weighted losses, with only answer logits materialized."""
    def __init__(self, model, processor, config):
        self.model, self.processor, self.config = model, processor, config

    def __call__(self, batch):
        import torch
        import torch.nn.functional as F
        from torchvision.ops import generalized_box_iou_loss
        config = self.config
        device = torch.device(config['device'])
        inputs = {k: v.to(device) for k, v in encode_batch(self.processor, config, batch).items()}
        labels = inputs.pop('labels')
        inputs['logits_to_keep'] = answer_window(labels)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            if batch['role'] == 'where':
                captured = []
                hook = self.model.vlm.register_forward_hook(lambda module, args, output: captured.append(output.logits))
                try:
                    _, boxes = self.model(inputs)
                    logits = captured[0]
                finally:
                    hook.remove()
            else:
                logits = self.model.vlm(**inputs, use_cache=False).logits
        losses = config['loss']['language'] * per_example_language_loss(logits, labels)
        if batch['role'] == 'where':
            targets = torch.tensor([s.bbox_1000 for s in batch['samples']], device=device, dtype=torch.float32) / 1000
            losses = losses + config['loss']['l1'] * F.l1_loss(boxes.float(), targets, reduction='none').mean(1)
            losses = losses + config['loss']['giou'] * generalized_box_iou_loss(boxes.float(), targets, reduction='none')
        weights = torch.tensor([e['weight'] for e in batch['examples']], device=device, dtype=torch.float32)
        return (losses * weights).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-cache', action='store_true', required=True)
    parser.add_argument('--config', default='configs/yaml/perception_agents_dual_pro6000.yaml')
    args = parser.parse_args()
    import yaml
    export_cache(yaml.safe_load(Path(args.config).read_text()))
