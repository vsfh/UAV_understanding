"""Generate selected weak classes from existing descriptions and trained LoRAs.

Reuses uav_synthesis_pipeline.crops/outpaint/export_annotations unchanged. No
description generation, downloads, fine-tuning, or modification of real splits.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

DEFAULT_CONFIG='configs/yaml/uav_synthesis_weak.yaml'


def read_records(path):
    """Retain complete JSONL records; an interrupted last write may be incomplete."""
    raw=Path(path).read_bytes()
    lines=raw.splitlines()
    records=[]
    ignored_tail=False
    for index,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError):
            if index!=len(lines)-1 or raw.endswith(b'\n'):
                raise
            ignored_tail=True
    return records,ignored_tail,hashlib.sha256(raw).hexdigest()


def safe_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or path==root.resolve():
        raise ValueError(f'Output path escapes synthetic root: {relative}')
    return path


def validate_record(record,label,root):
    if record['event_class']!=label or not record['description'].strip() or record.get('is_synthetic') is not True:
        raise ValueError(f'Invalid prompt record in {label}')
    for field in ('crop_relative','original_relative'):
        safe_path(root,record[field])
    safe_path(root,'metadata/'+record['synthetic_id']+'.json')
    width,height=record['target_original_size']
    x1,y1,x2,y2=record['bbox_xyxy']
    if not (width>0 and height>0 and 0<=x1<x2<=width and 0<=y1<y2<=height):
        raise ValueError('Invalid synthetic placement box')
    if any(size<=0 for size in record['target_crop_size']):
        raise ValueError('Invalid crop size')


def select_records(args):
    groups=[]
    sources={}
    for label in args.classes:
        path=args.work_dir/'prompts'/f'{label}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'No existing descriptions for {label}: {path}')
        rows,tail,digest=read_records(path)
        rows=rows[:args.per_class]
        if not rows:
            raise ValueError(f'No complete descriptions for {label}')
        for row in rows:
            validate_record(row,label,args.synthetic_root)
        groups.append(rows)
        sources[label]={'source':str(path),'selected':len(rows),'ignored_incomplete_tail':tail,'sha256':digest}
    records=[r for batch in itertools.zip_longest(*groups) for r in batch if r is not None]
    for key in ('synthetic_id','crop_relative','original_relative'):
        if len({r[key] for r in records})!=len(records):
            raise ValueError(f'Duplicate {key} in selected descriptions')
    return records,sources


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def binding(args):
    return {'classes':args.classes,'per_class':args.per_class,'work_dir':str(args.work_dir),
            'synthetic_root':str(args.synthetic_root),'pipeline_config':str(args.config),
            'crop_inference_steps':args.crop_inference_steps,'inpaint_inference_steps':args.inpaint_inference_steps,
            'save_conditions':args.save_conditions,'hf_cache':str(args.hf_cache),
            'pipeline_sha256':sha256(args.project_root/'scripts/uav_synthesis_pipeline.py'),
            'crop_lora_sha256':sha256(args.work_dir/'crop_lora/pytorch_lora_weights.safetensors'),
            'inpaint_lora_sha256':sha256(args.work_dir/'inpaint_lora/pytorch_lora_weights.safetensors')}


def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def prepare(args):
    manifest=args.selection_dir/'selection.json'
    if manifest.exists():
        return load_selection(args)
    # Dedicated output root prevents a filtered export from replacing an earlier dataset.
    if args.synthetic_root.exists() and any(args.synthetic_root.iterdir()):
        raise FileExistsError('A new selection needs an empty synthetic root; use a new --synthetic-root')
    records,sources=select_records(args)
    atomic_json(manifest,{'schema_version':1,'binding':binding(args),'sources':sources,'records':records})
    return records


def load_selection(args):
    selected=json.loads((args.selection_dir/'selection.json').read_text(encoding='utf-8'))
    if selected['binding']!=binding(args):
        raise ValueError('Selection settings or LoRA/pipeline changed; use a new selection/output directory')
    return selected['records']


def image_ok(path,size):
    if not path.exists():
        return False
    from PIL import Image,UnidentifiedImageError
    try:
        with Image.open(path) as image:
            same=image.size==tuple(size)
            image.verify()
        return same
    except (OSError,ValueError,UnidentifiedImageError,SyntaxError):
        return False


def record_state(args,record,repair=False):
    """Check only this selection's generated files; repair interrupted outputs."""
    import uav_synthesis_pipeline as pipeline
    crop=safe_path(args.synthetic_root,record['crop_relative'])
    original=safe_path(args.synthetic_root,record['original_relative'])
    metadata=safe_path(args.synthetic_root,'metadata/'+record['synthetic_id']+'.json')
    crop_ok=image_ok(crop,record['target_crop_size'])
    original_ok=image_ok(original,record['target_original_size'])
    annotation=None
    if metadata.exists():
        try:
            annotation=json.loads(metadata.read_text(encoding='utf-8'))
        except (json.JSONDecodeError,UnicodeDecodeError):
            pass
    x1,y1,x2,y2=record['bbox_xyxy']
    metadata_ok=annotation is not None and all(annotation.get(k)==record[k] for k in
        ('synthetic_id','event_class','crop_relative','original_relative','bbox_xyxy','target_original_size'))
    metadata_ok=metadata_ok and annotation.get('bbox_xywh')==[x1,y1,x2-x1,y2-y1]
    if repair:
        if not crop_ok:
            for path in (crop,original,metadata):
                path.unlink(missing_ok=True)
            original_ok=metadata_ok=False
        else:
            # A crop may have been saved just before a description JSON write was interrupted.
            description=safe_path(args.synthetic_root,'description/'+str(Path(record['crop_relative']).with_suffix('.json')))
            atomic_json(description,{'image_path':record['crop_relative'],'commercial_event':record['event_class'],
                'description':record['description'],'is_synthetic':True,'generator':pipeline.KLEIN_MODEL,
                'lora':str(args.work_dir/'crop_lora'),'template_uid':record['template_uid'],'seed':record['seed']})
            if not original_ok:
                original.unlink(missing_ok=True)
            if not metadata_ok:
                metadata.unlink(missing_ok=True)
    return {'crop':crop_ok,'original':original_ok,'complete':crop_ok and original_ok and metadata_ok}


def upstream(args,records):
    import uav_synthesis_pipeline as pipeline
    # Worker-local iterator injection keeps the original diffusion implementation.
    pipeline.prompt_records=lambda unused:iter(records)
    return pipeline


def export(args,records):
    pipeline=upstream(args,records)
    pipeline.export_annotations(args)


def worker(args,kind):
    records=load_selection(args)[args.offset:args.offset+args.count]
    pipeline=upstream(args,records)
    if kind=='_outpaint':
        # Parent exports the cumulative completed set, not just this chunk.
        pipeline.export_annotations=lambda unused:None
        pipeline.outpaint(args)
    else:
        pipeline.crops(args)


@contextmanager
def selection_lock(args):
    import fcntl
    args.selection_dir.mkdir(parents=True,exist_ok=True)
    with (args.selection_dir/'.run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def generate(args):
    with selection_lock(args):
        records=prepare(args)
        states={r['synthetic_id']:record_state(args,r,repair=True) for r in records}
        for offset in range(0,len(records),args.chunk_size):
            chunk=records[offset:offset+args.chunk_size]
            if all(states[r['synthetic_id']]['complete'] for r in chunk):
                continue
            for kind,needed in [('_crops',any(not states[r['synthetic_id']]['crop'] for r in chunk)),
                                ('_outpaint',True)]:
                if not needed:
                    continue
                command=[sys.executable,str(Path(__file__).resolve()),'--config',str(args.runner_config),
                         '--stage',kind,'--gpu',args.gpu,'--selection-dir',str(args.selection_dir),
                         '--synthetic-root',str(args.synthetic_root),'--per-class',str(args.per_class),
                         '--chunk-size',str(args.chunk_size),'--crop-inference-steps',str(args.crop_inference_steps),
                         '--inpaint-inference-steps',str(args.inpaint_inference_steps),'--offset',str(offset),
                         '--count',str(len(chunk)),'--classes',*args.classes]
                subprocess.run(command,check=True)
            for record in chunk:
                states[record['synthetic_id']]=record_state(args,record,repair=True)
                if not states[record['synthetic_id']]['complete']:
                    raise RuntimeError(f'Incomplete generated triple: {record["synthetic_id"]}')
            complete=[r for r in records if states[r['synthetic_id']]['complete']]
            export(args,complete)
            print(json.dumps({'completed':len(complete),'total':len(records),
                              'by_class':dict(Counter(r['event_class'] for r in complete))}),flush=True)
        export(args,[r for r in records if states[r['synthetic_id']]['complete']])


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(DEFAULT_CONFIG))
    parser.add_argument('--stage',choices=['plan','prepare','all','status','export','_crops','_outpaint'],default='plan')
    parser.add_argument('--gpu')
    parser.add_argument('--classes',nargs='+')
    for name in ('per-class','chunk-size','crop-inference-steps','inpaint-inference-steps'):
        parser.add_argument('--'+name,type=int)
    for name in ('selection-dir','synthetic-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--count',type=int,default=0)
    cli=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    runner_config=(root/cli.config).resolve()
    config=yaml.safe_load(runner_config.read_text(encoding='utf-8'))
    values=dict(config)
    for key,value in vars(cli).items():
        if value is not None and key!='config':values[key]=value
    args=SimpleNamespace(**values)
    args.project_root=root
    args.runner_config=runner_config
    args.config=(root/config['pipeline_config']).resolve()
    for key in ('work_dir','selection_dir','synthetic_root','hf_cache'):
        setattr(args,key,(root/Path(getattr(args,key))).resolve())
    args.crop_model_dir=args.hf_cache/'flux2-klein-base-4b'
    args.inpaint_model_dir=args.hf_cache/'sdxl-inpainting-0.1'
    args.max_images=None
    args.gpu=str(args.gpu)
    if min(args.per_class,args.chunk_size,args.crop_inference_steps,args.inpaint_inference_steps)<1:
        parser.error('Counts and inference steps must be positive')
    if len(args.classes)!=len(set(args.classes)):
        parser.error('Classes must not repeat')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    os.environ['HF_HOME']=str(args.hf_cache/'huggingface')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import uav_synthesis_pipeline as pipeline
    pipeline.configuration(args)
    if args.stage.startswith('_'):
        worker(args,args.stage)
    elif args.stage=='all':
        generate(args)
    elif args.stage=='prepare':
        with selection_lock(args):
            records=prepare(args)
        print(json.dumps({'selected':dict(Counter(r['event_class'] for r in records)),'total':len(records)}))
    else:
        selected=args.selection_dir/'selection.json'
        records=load_selection(args) if selected.exists() else select_records(args)[0]
        if args.stage=='plan':
            print(json.dumps({'selected':dict(Counter(r['event_class'] for r in records)),
                'total':len(records),'chunk_size':args.chunk_size,'synthetic_root':str(args.synthetic_root),
                'crop_lora_exists':(args.work_dir/'crop_lora/pytorch_lora_weights.safetensors').is_file(),
                'inpaint_lora_exists':(args.work_dir/'inpaint_lora/pytorch_lora_weights.safetensors').is_file(),
                'stages':['existing descriptions','crop chunk','original + bbox chunk','cumulative export'],
                'generates_descriptions':False,'starts_training':False},indent=2))
        else:
            if not selected.exists():
                raise FileNotFoundError('No frozen selection yet; run prepare or all')
            states=[record_state(args,r) for r in records]
            complete=[r for r,s in zip(records,states) if s['complete']]
            if args.stage=='export':
                with selection_lock(args):
                    export(args,complete)
            print(json.dumps({'selected':len(records),'crops':sum(s['crop'] for s in states),
                'originals':sum(s['original'] for s in states),'complete':len(complete),
                'complete_by_class':dict(Counter(r['event_class'] for r in complete))},indent=2))


if __name__=='__main__':
    main()
