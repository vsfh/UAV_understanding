"""Prepare real Spatial proposals, then supervised What/Where/Verify role training.

No RL or model downloads. A single shared LoRA and existing spatial/box heads
are trained; each role is forwarded separately to bound activation memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


DEFAULT_CONFIG = 'configs/yaml/perception_agents.yaml'


def fingerprint_rows(samples):
    rows=[]
    for s in samples:
        stat=s.image_path.stat()
        rows.append([s.record_uid,s.group_id,str(s.image_path.resolve()),stat.st_size,stat.st_mtime_ns,
                     s.presence,s.label,s.bbox_1000])
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def prepare_binding(config,splits):
    from perception_spatial_head import checkpoint_fingerprint
    checkpoint=Path(str(config['initial_checkpoint']).format(protocol=config['protocol'],seed=config['seed']))
    source={k:config[k] for k in ('protocol','seed','data','model','spatial','input','prompt')}
    return {'source_config':source,'source_checkpoint':str(checkpoint.resolve()),
            'source_sha256':checkpoint_fingerprint(checkpoint),
            'category_files':{k:hashlib.sha256(Path(config['data'][k]).read_bytes()).hexdigest() for k in ('labels','definitions')},
            'splits':{k:fingerprint_rows(v) for k,v in splits.items()}}


def read_splits(config):
    from clear_uav.table4 import read_discovery_samples
    from train_perception_spatial import subset
    full={s:read_discovery_samples(config,config['protocol'],s) for s in ('train','val')}
    for field in ('record_uid','group_id','image_path'):
        sets=[{str(getattr(s,field)) for s in rows if getattr(s,field) is not None} for rows in full.values()]
        if sets[0]&sets[1]:
            raise ValueError(f'Train/val overlap in {field}')
    return {s:subset(rows,config.get(f'max_{s}_samples') or config['agents'].get('bootstrap_max_samples'))
            for s,rows in full.items()}


def prepare(config):
    from perception_agents_runtime import build_model,output_path
    from clear_uav.table4 import definitions_from_config
    from test_perception_qwen import predict
    output=output_path(config)/'proposals'
    if output.exists():
        raise FileExistsError(f'Proposal cache exists: {output}; use a new --output or run train')
    splits=read_splits(config)
    binding=prepare_binding(config,splits)
    model,processor=build_model(config)
    labels,_=definitions_from_config(config)
    output.mkdir(parents=True)
    for split,samples in splits.items():
        # Use the already-trained Spatial model's original prompt, not untrained agent roles.
        predictions=predict(model,processor,samples,config,labels)
        rows={s.record_uid:{'category':p['category'],'bbox_1000':p['bbox_1000']} for s,p in zip(samples,predictions)}
        (output/f'{split}.json').write_text(json.dumps(rows),encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')


def role_loss(model,processor,config,sample,example):
    import torch
    import torch.nn.functional as F
    from torchvision.ops import generalized_box_iou_loss
    from perception_agents_runtime import encode_role
    inputs=encode_role(processor,config,sample.image_path,example['role'],candidate=example['candidate'],
                       feedback=example['feedback'],answer=example['answer'],supervised=True)
    device=torch.device(config['device'])
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        if example['role']=='where':
            language,boxes=model(inputs)
        else:
            language=model.vlm(**inputs,use_cache=False).loss
    total=config['loss']['language']*language
    if example['role']=='where':
        target=torch.tensor([sample.bbox_1000],device=device,dtype=torch.float32)/1000
        total=total+config['loss']['l1']*F.l1_loss(boxes.float(),target)
        total=total+config['loss']['giou']*generalized_box_iou_loss(boxes.float(),target,reduction='mean')
    return total


def train(config):
    import torch
    import yaml
    from transformers import set_seed,get_cosine_schedule_with_warmup
    from tqdm import tqdm
    from clear_uav.table4 import definitions_from_config,discovery_sampler
    from perception_agents_runtime import build_model,output_path,SCHEMA_VERSION
    from perception_agents_data import example_set
    from perception_spatial_head import save_spatial_heads
    set_seed(config['seed'])
    settings=config['train']
    if settings['batch_size']!=1:
        raise ValueError('Role-serial training currently uses batch_size=1; use gradient_accumulation')
    output=output_path(config)
    if (output/'best').exists() or (output/'history.json').exists():
        raise FileExistsError(f'Training output already exists: {output}')
    splits=read_splits(config)
    proposals=output/'proposals'
    saved=json.loads((proposals/'manifest.json').read_text(encoding='utf-8'))
    if saved!=prepare_binding(config,splits):
        raise ValueError('Proposal cache data/checkpoint/config differs; prepare a new output')
    cached={s:json.loads((proposals/f'{s}.json').read_text(encoding='utf-8')) for s in splits}
    for split,samples in splits.items():
        if {s.record_uid for s in samples}!=set(cached[split]):
            raise ValueError('Proposal cache record IDs differ')
    labels,_=definitions_from_config(config)
    model,processor=build_model(config,is_trainable=True)
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.vlm.parameters() if p.requires_grad],'lr':settings['learning_rate']},
        {'params':list(model.box_head.parameters())+list(model.spatial_head.parameters()),'lr':settings['head_learning_rate']}],
        weight_decay=settings['weight_decay'])
    sampler=discovery_sampler(splits['train'],settings,config['seed'])
    if not len(sampler) or not splits['val']:
        raise ValueError('Training sampler and validation split must be nonempty')
    accumulation=settings['gradient_accumulation']
    updates=math.ceil(len(sampler)/accumulation)*settings['epochs']
    if settings.get('steps') is not None:
        updates=min(updates,settings['steps'])
    scheduler=get_cosine_schedule_with_warmup(optimizer,int(updates*settings['warmup_ratio']),updates)
    config['run']={'schema_version':SCHEMA_VERSION,'roles':['what','where','verify'],
                   'training':'shared Spatial LoRA SFT; bbox loss only on Where',
                   'source_sha256':saved['source_sha256'],'limited_run':bool(config.get('max_train_samples') or config.get('max_val_samples') or config['agents'].get('bootstrap_max_samples'))}
    output.mkdir(parents=True,exist_ok=True)
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    history=[]
    best=float('inf')
    completed=0
    for epoch in range(settings['epochs']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total=0.
        seen=0
        indices=list(discovery_sampler(splits['train'],settings,config['seed']+epoch))
        # A bounded smoke run still ends with a correctly normalized accumulation group.
        indices=indices[:max(0,(updates-completed)*accumulation)]
        for step,index in enumerate(tqdm(indices,desc=f'agents train {epoch+1}')):
            sample=splits['train'][index]
            group_size=min(accumulation,len(indices)-(step//accumulation)*accumulation)
            examples=example_set(sample,cached['train'][sample.record_uid],labels,config['agents'],epoch,draw_index=step)
            image_loss=0.
            for example in examples:
                loss=role_loss(model,processor,config,sample,example)*example['weight']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite {example["role"]} loss')
                (loss/group_size).backward()
                image_loss+=loss.detach().item()
            total+=image_loss
            seen+=1
            if (step+1)%accumulation==0 or step+1==len(indices):
                torch.nn.utils.clip_grad_norm_(model.parameters(),settings['max_grad_norm'])
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                completed+=1
        model.eval()
        validation=0.
        with torch.inference_mode():
            for sample in tqdm(splits['val'],desc='agents validation loss'):
                examples=example_set(sample,cached['val'][sample.record_uid],labels,config['agents'],0,validation=True)
                validation+=sum(role_loss(model,processor,config,sample,e).item()*e['weight'] for e in examples)
        validation/=len(splits['val'])
        if not math.isfinite(validation):
            raise FloatingPointError('Nonfinite validation loss')
        entry={'epoch':epoch+1,'updates':completed,'train_loss':total/seen,'val_loss':validation,
               'spatial_gate':model.spatial_head.residual_gate.detach().item()}
        history.append(entry)
        if validation<best:
            best=validation
            checkpoint=output/'best'
            model.vlm.save_pretrained(checkpoint,save_embedding_layers=False)
            processor.save_pretrained(checkpoint)
            save_spatial_heads(model,checkpoint)
            (checkpoint/'agent_schema.json').write_text(json.dumps({'schema_version':SCHEMA_VERSION,
                'roles':['what','where','verify'],'verdicts':{'A':'accept','B':'relocalize','C':'reclassify','D':'no_event'},
                'agents':config['agents'],'epoch':epoch+1,'val_loss':validation},indent=2),encoding='utf-8')
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True)
        if completed>=updates:
            break


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--stage',choices=['plan','prepare','train'],default='plan')
    for name in ('output','device'):
        parser.add_argument('--'+name)
    for name in ('max-train-samples','max-val-samples','epochs','steps'):
        parser.add_argument('--'+name,type=int)
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    for name in ('output','device','max_train_samples','max_val_samples'):
        if getattr(args,name) is not None:
            config[name]=getattr(args,name)
    for name in ('epochs','steps'):
        if getattr(args,name) is not None:
            config['train'][name]=getattr(args,name)
    for value in [config['train']['epochs'],config['train']['gradient_accumulation'],config['agents']['verifier_candidates_per_image']]:
        if value<1:
            parser.error('Epochs, accumulation, and candidate count must be positive')
    if config['train'].get('steps') is not None and config['train']['steps']<1:
        parser.error('--steps must be positive')
    if args.stage=='plan':
        print(json.dumps({'config':config,'stages':['prepare frozen Spatial proposals (train/val only)',
             'train shared What/Where/Verify SFT','calibrate on val, evaluate on test'],
             'inference':'full image -> What -> Where -> Verify -> up to two targeted revisions; one shared model',
             'remote_execution':'This command prints a plan only.'},ensure_ascii=False,indent=2))
    else:
        # Populate clear_uav imports through the existing repository entry point.
        import train_perception_qwen
        {'prepare':prepare,'train':train}[args.stage](config)


if __name__=='__main__':
    main()
