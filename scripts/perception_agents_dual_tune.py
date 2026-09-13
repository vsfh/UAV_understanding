"""Measure candidate global-batch-8 configurations, then optionally train and test."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/yaml/perception_agents_dual_pro6000.yaml')
    parser.add_argument('--output')
    parser.add_argument('--proposal-cache')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--memory-gib', type=float, default=76)
    parser.add_argument('--mode', choices=('full', 'all', 'no_verify', 'verify_only'), default='full')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.steps <= 2:
        parser.error('Use at least 3 benchmark steps to leave measurements after warmup')
    config = yaml.safe_load(Path(args.config).read_text())
    if config['train'].get('steps') is not None or any(config.get(k) is not None for k in ('max_train_samples', 'max_val_samples', 'max_test_samples')):
        parser.error('tune/fast requires a full-run YAML without steps or sample limits')
    if args.steps <= config['train'].get('benchmark_warmup_steps', 2):
        parser.error('--steps must exceed train.benchmark_warmup_steps')
    for name in ('output', 'proposal_cache'):
        if getattr(args, name):
            config[name] = getattr(args, name)
    output = Path(config['output'].format(protocol=config['protocol'], seed=config['seed']))
    if args.run and ((output / 'best').exists() or (output / 'history.json').exists()):
        raise FileExistsError(f'Choose a new --output before tuning for a new training run: {output}')
    folder = output.parent / (output.name + '_tuning_' + time.strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True)
    launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    results = []
    for batch, accumulation, checkpointing in ((4, 1, False), (2, 2, False), (4, 1, True), (2, 2, True)):
        name = f'b{batch}_gc{int(checkpointing)}'
        candidate = copy.deepcopy(config)
        candidate['train'].update(batch_size=batch, gradient_accumulation=accumulation,
                                  gradient_checkpointing=checkpointing)
        candidate['output'] = str(folder / name)
        config_path = folder / (name + '.yaml')
        config_path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        log_path = folder / (name + '.log')
        command = launcher + ['scripts/train_perception_agents_dual.py', '--stage', 'benchmark',
            '--config', str(config_path), '--output', candidate['output'], '--steps', str(args.steps),
            '--max-val-samples', '8']
        print(json.dumps({'candidate': name, 'command': command, 'log': str(log_path)}), flush=True)
        with log_path.open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = {'name': name, 'batch_size': batch, 'gradient_accumulation': accumulation,
               'gradient_checkpointing': checkpointing, 'returncode': result.returncode}
        if result.returncode:
            if 'out of memory' not in log_path.read_text(errors='replace').lower():
                raise RuntimeError(f'Benchmark failed for a reason other than OOM; inspect {log_path}')
            row['eligible'] = False
            row['reason'] = 'out of memory'
        else:
            measured = json.loads((folder / name / 'benchmark.json').read_text())
            row.update(measured)
            row['eligible'] = (measured['peak_reserved_gib'] <= args.memory_gib
                               and measured['measured_images_per_second'] is not None)
        results.append(row)
        (folder / 'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    eligible = [row for row in results if row['eligible']]
    if not eligible:
        raise RuntimeError(f'No measured candidate fits {args.memory_gib} GiB; see {folder}/results.json')
    best = max(eligible, key=lambda row: row['measured_images_per_second'])
    config['train'].update({key: best[key] for key in ('batch_size', 'gradient_accumulation', 'gradient_checkpointing')})
    selected = folder / 'selected.yaml'
    selected.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({'selected_config': str(selected), 'winner': best, 'note': 'Fastest measured candidate; input and global batch unchanged.'}), flush=True)
    if args.run:
        subprocess.run(launcher + ['scripts/train_perception_agents_dual.py', '--stage', 'train',
                       '--config', str(selected)], check=True)
        subprocess.run(launcher + ['scripts/test_perception_agents_dual.py', '--split', 'all',
                       '--mode', args.mode, '--config', str(selected)], check=True)


if __name__ == '__main__':
    main()
