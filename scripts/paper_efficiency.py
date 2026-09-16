"""Short same-hardware profile of the saved Direct/Perception checkpoints."""
import argparse
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
from paper_small_checks import DEFAULT, read_config, save_json


def main(args):
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from clear_uav import table4
    config=read_config(args.config)
    device=torch.device('cuda')
    settings=config['profile']
    if args.model=='perception':
        from train_perception_qwen import build_model
        import test_perception_qwen as evaluation
        recipe=read_config(config['perception_config']); recipe['device']='cuda'
        model,processor=build_model(recipe,Path(config['perception_checkpoint']))
        labels=table4.labels_from_config(recipe)
        evaluation.tqdm=functools.partial(tqdm,disable=True)
        run=lambda batch:evaluation.predict(model,processor,batch,recipe,labels)
        generation_model=model.vlm
    else:
        recipe=read_config(config['direct_config'])
        model,processor=table4.load_qwen_adapter(recipe,config['protocol'],config['seed'],device)
        table4.tqdm=functools.partial(tqdm,disable=True)
        run=lambda batch:table4.predict_qwen_discovery(model,processor,batch,recipe,device,'profile')
        generation_model=model
    model.eval()
    samples=table4.read_discovery_samples(recipe,config['protocol'],'val')
    # Same deterministic validation cohort for both models; never select on speed.
    samples=sorted(samples,key=lambda s:s.record_uid)
    random.Random(20260916).shuffle(samples)
    samples=samples[:settings['samples']]
    generation_times=[]; replay_times=[]; pixels=[]
    original_generate=generation_model.generate

    def timed_generate(*a,**kw):
        grid=kw.get('image_grid_thw')
        if grid is not None:
            patch_size=generation_model.config.vision_config.patch_size
            pixels.extend((grid.prod(-1)*patch_size**2).cpu().tolist())
        torch.cuda.synchronize(); started=time.perf_counter()
        out=original_generate(*a,**kw)
        torch.cuda.synchronize(); generation_times.append((time.perf_counter()-started)*1000)
        return out

    generation_model.generate=timed_generate
    if args.model=='perception':
        original_forward=model.forward

        def timed_replay(*a,**kw):
            torch.cuda.synchronize(); started=time.perf_counter()
            out=original_forward(*a,**kw)
            torch.cuda.synchronize(); replay_times.append((time.perf_counter()-started)*1000)
            return out

        model.forward=timed_replay
    results=[]
    for size in settings['batch_sizes']:
        recipe.setdefault('test',{})['batch_size']=size
        for _ in range(settings['warmup_batches']): run(samples[:size])
        generation_times.clear(); replay_times.clear(); pixels.clear()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        batch_ms=[]
        for start in tqdm(range(0,len(samples),size),desc=f'{args.model} batch={size}',unit='batch'):
            torch.cuda.synchronize(); started=time.perf_counter()
            run(samples[start:start+size])
            torch.cuda.synchronize(); batch_ms.append((time.perf_counter()-started)*1000)
        results.append({'batch_size':size,'n_images':len(samples),
            'batch_latency_ms_p50':float(np.median(batch_ms)), 'batch_latency_ms_p95':float(np.quantile(batch_ms,.95)),
            'throughput_images_s':len(samples)/(sum(batch_ms)/1000),
            'generation_ms_per_batch_p50':float(np.median(generation_times)),
            'replay_ms_per_batch_p50':float(np.median(replay_times)) if replay_times else 0,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,
            'processed_image_pixels_min_median_max':[min(pixels),float(np.median(pixels)),max(pixels)] if pixels else None})
    save_json(Path(config['output'])/f'efficiency_{args.model}.json',{
        'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'parameter_dtype':str(next(model.parameters()).dtype),
        'scope':'Image loading/preprocessing, generation, replay if applicable, and parsing. Model loading excluded. CUDA synchronized.',
        'caveat':'Throughput is sequential batch-loop throughput, not an optimized serving benchmark. Shared GPU load can bias timings.',
        'records':[s.record_uid for s in samples], 'input_settings':recipe['input'], 'results':results})
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['direct','perception'],required=True)
    parser.add_argument('--config',default=DEFAULT)
    main(parser.parse_args())
