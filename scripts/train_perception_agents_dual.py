"""Two-GPU shared What/Where/Verify SFT, using already completed proposal caches.

Launch with torchrun. Each rank owns a complete model. Role forwards can use
different heads and batch sizes, so gradients are summed explicitly once per
optimizer update instead of wrapping partial forwards in DistributedDataParallel.
The objective is the same global per-image mean, including a short final batch.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from datetime import timedelta


DEFAULT_CONFIG = 'configs/yaml/perception_agents_dual_pro6000.yaml'


def optimizer_groups(indices, global_batch_size):
    """Keep every sampled occurrence and its original epoch draw number."""
    if global_batch_size < 1:
        raise ValueError('Global batch size must be positive')
    for start in range(0, len(indices), global_batch_size):
        yield list(enumerate(indices[start:start + global_batch_size], start))


def shard_group(group, rank, world_size, local_batch_size):
    """Shard each global microbatch without dropping or duplicating examples."""
    if not 0 <= rank < world_size or local_batch_size < 1:
        raise ValueError('Invalid rank, world size, or local batch size')
    width = world_size * local_batch_size
    return [item for offset, item in enumerate(group)
            if (offset % width) // local_batch_size == rank]


def training_plan(sample_count, epochs, local_batch_size, accumulation,
                  world_size, steps=None):
    if min(sample_count, epochs, local_batch_size, accumulation, world_size) < 1:
        raise ValueError('Sampler size, epochs, batch size, accumulation, and world size must be positive')
    global_batch_size = local_batch_size * accumulation * world_size
    per_epoch = math.ceil(sample_count / global_batch_size)
    updates = per_epoch * epochs
    if steps is not None:
        if steps < 1:
            raise ValueError('steps must be positive')
        updates = min(updates, steps)
    return {'sampled_images_per_epoch': sample_count, 'epochs': epochs,
            'world_size': world_size, 'batch_size_per_gpu': local_batch_size,
            'gradient_accumulation': accumulation, 'global_batch_size': global_batch_size,
            'updates_per_epoch': per_epoch, 'optimizer_updates': updates}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def parameter_buckets(parameters, bucket_bytes):
    """Bound temporary FP32 communication storage, preserving parameter order."""
    current, size = [], 0
    for parameter in parameters:
        amount = parameter.numel() * 4
        if current and size + amount > bucket_bytes:
            yield current
            current, size = [], 0
        current.append(parameter)
        size += amount
    if current:
        yield current


def synchronize_initial_state(model, distributed):
    # Frozen base weights come from the same checkpoint; synchronize all trainable
    # tensors and buffers explicitly before any role-specific optimization.
    for parameter in model.parameters():
        if parameter.requires_grad:
            distributed.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        distributed.broadcast(buffer, src=0)


def sum_gradients(parameters, distributed, bucket_bytes):
    """SUM already normalized local gradients; preserve globally unused params.

    Every rank enters the same collectives even if its last group is empty or it
    has no positive example. This is essential for the conditional Where head.
    """
    import torch
    used = torch.tensor([p.grad is not None for p in parameters],
                        dtype=torch.int32, device=parameters[0].device)
    distributed.all_reduce(used, op=distributed.ReduceOp.MAX)
    used = used.cpu().tolist()
    offset = 0
    for bucket in parameter_buckets(parameters, bucket_bytes):
        flat = torch.zeros(sum(p.numel() for p in bucket), dtype=torch.float32,
                           device=bucket[0].device)
        position = 0
        for parameter in bucket:
            if parameter.grad is not None:
                flat[position:position + parameter.numel()].copy_(parameter.grad.reshape(-1))
            position += parameter.numel()
        distributed.all_reduce(flat, op=distributed.ReduceOp.SUM)
        position = 0
        for parameter in bucket:
            if used[offset]:
                gradient = flat[position:position + parameter.numel()].view_as(parameter)
                if parameter.grad is None:
                    parameter.grad = gradient.to(parameter.dtype).clone()
                else:
                    parameter.grad.copy_(gradient)
            else:
                parameter.grad = None
            position += parameter.numel()
            offset += 1


def peak_memory(distributed, device):
    import torch
    memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                           torch.cuda.max_memory_reserved(device) / 2**30], device=device)
    distributed.all_reduce(memory, op=distributed.ReduceOp.MAX)
    return {'peak_allocated_gib': memory[0].item(), 'peak_reserved_gib': memory[1].item()}


def validate(model, criterion, samples, cached, labels, config, rank, world_size,
             distributed, device):
    import torch
    from tqdm import tqdm
    from perception_agents_dual_common import role_batches
    model.eval()
    local_samples = samples[rank::world_size]
    batch_size = config.get('validation', {}).get('batch_size', config['train']['batch_size'])
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in tqdm(range(0, len(local_samples), batch_size),
                          desc='agents dual validation (rank 0 shard)', disable=rank != 0):
            chunk = local_samples[start:start + batch_size]
            for batch in role_batches(chunk, cached, labels, config, validation=True):
                totals[0].add_(criterion(batch).detach().double())
                totals[2] += len(batch['examples'])
            totals[1] += len(chunk)
    distributed.all_reduce(totals, op=distributed.ReduceOp.SUM)
    if totals[1].item() != len(samples):
        raise RuntimeError('Distributed validation did not cover each image exactly once')
    value = (totals[0] / totals[1]).item()
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite validation loss')
    elapsed = time.perf_counter() - started
    return {'val_loss': value, 'val_images': len(samples), 'val_seconds': elapsed,
            'val_images_per_second': len(samples) / elapsed,
            'val_role_examples': int(totals[2].item())}


def save_best(model, processor, config, output, epoch, validation):
    from perception_agents_runtime import SCHEMA_VERSION
    from perception_spatial_head import save_spatial_heads
    checkpoint = output / 'best'
    model.vlm.save_pretrained(checkpoint, save_embedding_layers=False)
    processor.save_pretrained(checkpoint)
    save_spatial_heads(model, checkpoint)
    write_json(checkpoint / 'agent_schema.json', {
        'schema_version': SCHEMA_VERSION, 'roles': ['what', 'where', 'verify'],
        'verdicts': {'A': 'accept', 'B': 'relocalize', 'C': 'reclassify', 'D': 'no_event'},
        'agents': config['agents'], 'epoch': epoch, 'val_loss': validation,
        'limited_run': config['run']['limited_run']})


def train(config, benchmark=False):
    import torch
    import torch.distributed as distributed
    import yaml
    from transformers import set_seed, get_cosine_schedule_with_warmup
    from tqdm import tqdm
    # Existing entry point adds the repository package to sys.path.
    import train_perception_qwen
    from clear_uav.table4 import definitions_from_config, discovery_sampler
    from perception_agents_runtime import output_path, SCHEMA_VERSION
    from perception_agents_dual_common import (
        build_fast_model, load_cached_splits, RoleBatchLoss, role_batches)

    rank, world_size = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size < 2:
        raise ValueError('Use torchrun --standalone --nproc_per_node=2 for dual training')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    config['device'] = str(device)
    distributed.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    output = output_path(config)
    try:
        if (output / 'best').exists() or (output / 'history.json').exists():
            raise FileExistsError(f'Training output already exists: {output}')
        settings = config['train']
        # The helper validates completed caches and maps the saved dataset to this
        # machine. No prepare, candidate generation, or data selection is run here.
        splits, cached, manifest = load_cached_splits(config)
        labels, _ = definitions_from_config(config)
        sample_count = len(discovery_sampler(splits['train'], settings, config['seed']))
        plan = training_plan(sample_count, settings['epochs'], settings['batch_size'],
                             settings['gradient_accumulation'], world_size, settings.get('steps'))
        if not splits['val']:
            raise ValueError('Validation split must be nonempty')
        set_seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        model, processor = build_fast_model(config, trainable=True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        synchronize_initial_state(model, distributed)
        # Identical starting parameters, independent per-rank dropout streams.
        set_seed(config['seed'] + rank)
        optimizer = torch.optim.AdamW([
            {'params': [p for p in model.vlm.parameters() if p.requires_grad],
             'lr': settings['learning_rate']},
            {'params': [p for module in (model.box_head, model.spatial_head)
                        for p in module.parameters() if p.requires_grad],
             'lr': settings['head_learning_rate']}],
            weight_decay=settings['weight_decay'], fused=True)
        updates = plan['optimizer_updates']
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(updates * settings['warmup_ratio']), updates)
        criterion = RoleBatchLoss(model, processor, config)
        config['run'] = {
            'schema_version': SCHEMA_VERSION, 'roles': ['what', 'where', 'verify'],
            'training': 'synchronous data parallel shared Spatial LoRA SFT',
            'gradient_reduction': 'FP32 SUM of per-image global-mean gradients once per update',
            'source_sha256': manifest['source_sha256'],
            'proposal_cache': str(config['proposal_cache']),
            'limited_run': bool(benchmark or settings.get('steps') is not None
                                or config.get('max_train_samples') or config.get('max_val_samples')),
            'benchmark': benchmark, **plan}
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            print(json.dumps({'stage': 'train', 'output': str(output), **config['run']},
                             ensure_ascii=False), flush=True)
        distributed.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        history, best, completed = [], float('inf'), 0
        training_seconds, total_images, total_roles = 0.0, 0, 0
        validation_seconds = 0.0
        started = time.perf_counter()
        bucket_bytes = int(config.get('distributed', {}).get('gradient_bucket_mb', 64) * 2**20)
        log_every = max(1, int(settings.get('log_every', 10)))
        benchmark_warmup = int(settings.get('benchmark_warmup_steps', 2))
        measured_seconds, measured_images, measured_roles = 0.0, 0, 0
        for epoch in range(settings['epochs']):
            # Broadcast one weighted draw stream. Repeated sampler draws remain
            # repeated; rank sharding itself never introduces extra repetitions.
            payload = [list(discovery_sampler(splits['train'], settings, config['seed'] + epoch))
                       if rank == 0 else None]
            distributed.broadcast_object_list(payload, src=0)
            indices = payload[0]
            epoch_images, epoch_loss, epoch_roles, epoch_seconds = 0, 0.0, 0, 0.0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            groups = list(optimizer_groups(indices, plan['global_batch_size']))[:updates - completed]
            for group in tqdm(groups, desc=f'agents dual train {epoch + 1}', disable=rank != 0):
                step_started = time.perf_counter()
                local = shard_group(group, rank, world_size, settings['batch_size'])
                local_totals = torch.zeros(2, dtype=torch.float64, device=device)
                for start in range(0, len(local), settings['batch_size']):
                    chunk = local[start:start + settings['batch_size']]
                    samples = [splits['train'][index] for _, index in chunk]
                    draws = [draw for draw, _ in chunk]
                    for batch in role_batches(samples, cached['train'], labels, config,
                                              epoch=epoch, draw_indices=draws):
                        loss = criterion(batch)
                        # Divide by real global group size, including the final
                        # short group, then SUM gradients across the two ranks.
                        (loss / len(group)).backward()
                        local_totals[0].add_(loss.detach().double())
                        local_totals[1] += len(batch['examples'])
                distributed.all_reduce(local_totals, op=distributed.ReduceOp.SUM)
                group_loss, group_roles = local_totals.cpu().tolist()
                if not math.isfinite(group_loss):
                    raise FloatingPointError('Nonfinite role loss on at least one rank')
                sum_gradients(parameters, distributed, bucket_bytes)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, settings['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - step_started
                completed += 1
                epoch_images += len(group)
                epoch_roles += int(group_roles)
                epoch_loss += group_loss
                epoch_seconds += step_seconds
                total_images += len(group)
                total_roles += int(group_roles)
                training_seconds += step_seconds
                if completed > benchmark_warmup:
                    measured_images += len(group)
                    measured_roles += int(group_roles)
                    measured_seconds += step_seconds
                if completed % log_every == 0 or completed == updates or group is groups[-1]:
                    memory = peak_memory(distributed, device)
                    progress = {
                        'stage': 'train', 'epoch': epoch + 1, 'updates': completed,
                        'total_updates': updates, 'epoch_images': epoch_images,
                        'sampled_images_per_epoch': len(indices), 'train_loss': epoch_loss / epoch_images,
                        'last_step_loss': group_loss / len(group), 'gradient_norm': gradient_norm.item(),
                        'images_per_second': epoch_images / epoch_seconds,
                        'role_examples_per_second': epoch_roles / epoch_seconds,
                        'last_step_seconds': step_seconds,
                        'training_eta_seconds_excludes_validation':
                            (updates - completed) * training_seconds / completed,
                        'elapsed_seconds': time.perf_counter() - started,
                        'limited_run': config['run']['limited_run'], **memory}
                    if rank == 0:
                        write_json(output / 'progress.json', progress)
                        print(json.dumps(progress), flush=True)
            if rank == 0:
                write_json(output / 'progress.json', {
                    'stage': 'validation', 'epoch': epoch + 1, 'updates': completed,
                    'train_loss': epoch_loss / epoch_images, 'val_images': len(splits['val']),
                    'elapsed_seconds': time.perf_counter() - started,
                    'limited_run': config['run']['limited_run']})
            validation = validate(model, criterion, splits['val'], cached['val'], labels,
                                  config, rank, world_size, distributed, device)
            validation_seconds += validation['val_seconds']
            entry = {
                'epoch': epoch + 1, 'updates': completed, 'train_loss': epoch_loss / epoch_images,
                'train_images': epoch_images, 'train_seconds': epoch_seconds,
                'train_images_per_second': epoch_images / epoch_seconds,
                'train_role_examples_per_second': epoch_roles / epoch_seconds,
                'spatial_gate': model.spatial_head.residual_gate.detach().item(),
                'limited_run': config['run']['limited_run'], **validation,
                **peak_memory(distributed, device)}
            history.append(entry)
            if rank == 0:
                if validation['val_loss'] < best and not benchmark:
                    best = validation['val_loss']
                    save_best(model, processor, config, output, epoch + 1, best)
                write_json(output / 'history.json', history)
                print(json.dumps(entry), flush=True)
            distributed.barrier()
            if completed >= updates:
                break
        result = {
            'stage': 'completed', 'updates': completed, 'train_images': total_images,
            'train_seconds': training_seconds, 'val_seconds': validation_seconds,
            'elapsed_seconds': time.perf_counter() - started,
            'train_images_per_second': total_images / training_seconds,
            'train_role_examples_per_second': total_roles / training_seconds,
            'measured_images_per_second': measured_images / measured_seconds if measured_seconds else None,
            'measured_role_examples_per_second': measured_roles / measured_seconds if measured_seconds else None,
            'excluded_warmup_steps': min(completed, benchmark_warmup),
            'limited_run': config['run']['limited_run'], 'best_saved': not benchmark,
            **peak_memory(distributed, device)}
        if rank == 0:
            write_json(output / 'progress.json', result)
            if benchmark:
                write_json(output / 'benchmark.json', {**result, 'plan': plan,
                    'estimate_note': 'Measured throughput excludes warmup steps and validation; full run also includes validation and evaluation.'})
            print(json.dumps(result), flush=True)
    finally:
        distributed.destroy_process_group()


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--stage', choices=('plan', 'train', 'benchmark'), default='plan')
    parser.add_argument('--output')
    parser.add_argument('--proposal-cache')
    parser.add_argument('--require-full-run', action='store_true')
    for name in ('epochs', 'steps', 'max-train-samples', 'max-val-samples',
                 'batch-size', 'gradient-accumulation'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    config = copy.deepcopy(yaml.safe_load(Path(args.config).read_text(encoding='utf-8')))
    for name in ('output', 'proposal_cache', 'max_train_samples', 'max_val_samples'):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    for name in ('epochs', 'steps', 'batch_size', 'gradient_accumulation', 'gradient_checkpointing'):
        if getattr(args, name) is not None:
            config['train'][name] = getattr(args, name)
    if args.stage == 'benchmark':
        if args.steps is None:
            config['train']['steps'] = 20
        if args.max_val_samples is None:
            config['max_val_samples'] = 16
    limited = bool(config['train'].get('steps') is not None
                   or config.get('max_train_samples') or config.get('max_val_samples'))
    if args.require_full_run and limited:
        parser.error('all requires a full-run YAML without steps or sample limits; use smoke for a short run')
    if limited and args.output is None:
        suffix = 'benchmark' if args.stage == 'benchmark' else 'limited'
        config['output'] = config['output'].rstrip('/') + '_' + suffix + '_' + time.strftime('%Y%m%d_%H%M%S')
    for name in ('epochs', 'batch_size', 'gradient_accumulation'):
        if config['train'][name] < 1:
            parser.error(f'{name} must be positive')
    for value in (config['train'].get('steps'), config.get('max_train_samples'), config.get('max_val_samples')):
        if value is not None and value < 1:
            parser.error('steps and sample limits must be positive')
    if not config.get('proposal_cache'):
        parser.error('proposal_cache must point to the completed train/val proposal directory')
    if args.stage == 'plan':
        print(json.dumps({
            'config': config,
            'stages': ['load completed train/val proposal cache',
                       'synchronous two-GPU What/Where/Verify SFT',
                       'distributed val calibration and test'],
            'global_batch_size': config['train']['batch_size'] * config['train']['gradient_accumulation'] * 2,
            'candidate_prediction': 'skipped; this entry point has no prepare stage',
            'execution': 'Plan only: no dataset scan, model loading, or GPU allocation.'},
            ensure_ascii=False, indent=2))
    else:
        train(config, benchmark=args.stage == 'benchmark')


if __name__ == '__main__':
    main()
