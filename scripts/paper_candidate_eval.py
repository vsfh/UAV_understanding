"""Existing detector weights -> cached multi-proposals -> top-k reference AP.

No training. D-FINE is class-agnostic. Grounding DINO class IDs come from
canonical prompt spans. D-FINE+DINOv2 classifies only the top 10 proposals.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from paper_small_checks import DEFAULT, read_config, save_json, load_rows, candidate_summary


def cache_rows(path):
    if not path.exists():
        return None, []
    with path.open(encoding='utf-8') as handle:
        meta=json.loads(next(handle))['_meta']
        return meta, [json.loads(line) for line in handle]


def main(args):
    import torch
    from PIL import Image
    from tqdm import tqdm
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from clear_uav.table4 import (read_discovery_samples, load_dfine, load_grounding_dino,
                                 definition_prompts, labels_from_config, load_dinov2_classifier, crop_from_box)
    config=read_config(args.config)
    name=args.model
    model_config=read_config(config['dinov2_config' if name=='dfine_dinov2' else name+'_config'])
    protocol,seed=config['protocol'],config['seed']
    samples=read_discovery_samples(model_config,protocol,'test')
    # Bind to the exact cohort underlying the paper, including all negatives.
    source=load_rows(config['predictions']['direct'])
    expected={r['record_uid']:r for r in source}
    assert len(samples)==len(expected) and {s.record_uid for s in samples}==set(expected)
    by_uid={s.record_uid:s for s in samples}; samples=[by_uid[r['record_uid']] for r in source]
    for sample in samples:
        target=expected[sample.record_uid]['target']
        assert list(sample.bbox_1000 or [])==list(target['bbox_1000'] or []) and sample.label==target['category']
    if args.limit:
        samples=samples[:args.limit]
    output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
    suffix=f'_smoke{args.limit}' if args.limit else ''
    cache=output/f'{name}{suffix}_candidates.jsonl'
    fingerprint=hashlib.sha256(json.dumps({'model_config':model_config,'name':name,
        'uids':[s.record_uid for s in samples], 'floor':config['grounding_minimum_score'],
        'classifier_k':config['classifier_top_k']},sort_keys=True).encode()).hexdigest()
    old,rows=cache_rows(cache)
    if old is not None:
        assert old['fingerprint']==fingerprint, 'Cache settings differ; choose a new output folder.'
        assert [r['record_uid'] for r in rows]==[s.record_uid for s in samples[:len(rows)]], 'Incomplete/out-of-order cache'
    meta={'fingerprint':fingerprint,'model':name,'n_expected':len(samples),'nms':'none',
          'candidate_scope':('top-10 D-FINE boxes, classified with DINOv2' if name=='dfine_dinov2' else
                             'all returned D-FINE postprocessor candidates, score floor 0' if name=='dfine' else
                             'all queries above configured score floor across original prompt chunks'),
          'score_floor':config['grounding_minimum_score'] if name=='grounding_dino' else 0,
          'ap':'non-interpolated, one-to-one reference matching; stable source/proposal-order score ties',
          'score':'detector score (also for class-specific AP)', 'category_mapping':
          'max sigmoid token score over each canonical label+definition span' if name=='grounding_dino' else None}
    if old is None:
        cache.write_text(json.dumps({'_meta':meta})+'\n',encoding='utf-8')
    device=torch.device('cuda')
    if len(rows)<len(samples):
        if name=='dfine':
            path=Path(model_config['output']['model'].format(protocol=protocol,seed=seed))
            model,processor=load_dfine(path,device,True)
        elif name=='grounding_dino':
            model,processor=load_grounding_dino(model_config,device)
            labels,prompts=definition_prompts(model_config)
            chunks=[]; chunk_size=model_config['inference']['prompt_chunk_size']
            for start in range(0,len(prompts),chunk_size):
                parts=[p.rstrip('. ') for p in prompts[start:start+chunk_size]]
                text='. '.join(parts)+'.'; spans=[]; offset=0
                for part in parts:
                    spans.append((offset,offset+len(part))); offset+=len(part)+2
                offsets=processor.tokenizer(text,return_offsets_mapping=True)['offset_mapping']
                positions=[[i for i,(a,b) in enumerate(offsets) if b>a and b>lo and a<hi] for lo,hi in spans]
                assert all(positions), 'Empty class span'
                chunks.append((text,positions,labels[start:start+chunk_size]))
        else:
            model,processor=load_dinov2_classifier(model_config,protocol,seed,device)
            labels=labels_from_config(model_config)
            _,proposals=cache_rows(output/f'dfine{suffix}_candidates.jsonl')
            assert len(proposals)==len(samples), 'Run D-FINE candidates first'
            proposal_map={r['record_uid']:r for r in proposals}
        with cache.open('a',encoding='utf-8') as handle, torch.inference_mode():
            for sample in tqdm(samples[len(rows):],desc=name+' proposals',unit='image',initial=len(rows),total=len(samples)):
                with Image.open(sample.image_path) as original:
                    image=original.convert('RGB')
                candidates=[]
                if name=='dfine':
                    inputs=processor(images=image,return_tensors='pt').to(device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        outputs=model(**inputs)
                    prediction=processor.post_process_object_detection(outputs,target_sizes=[(image.height,image.width)],threshold=0.)[0]
                    for score,box in zip(prediction['scores'].float().cpu().tolist(),prediction['boxes'].float().cpu().tolist()):
                        candidates.append({'score':score,'bbox_1000':[v*1000/(image.width if i%2==0 else image.height) for i,v in enumerate(box)],'category':None})
                elif name=='grounding_dino':
                    for text,positions,chunk_labels in chunks:
                        inputs=processor(images=image,text=text,return_tensors='pt').to(device)
                        assert inputs['input_ids'].shape[-1] == len(processor.tokenizer(text)['input_ids']), 'Prompt truncation'
                        outputs=model(**inputs)
                        probabilities=outputs.logits[0].float().sigmoid()
                        scores=probabilities.max(-1).values
                        class_scores=torch.stack([probabilities[:,idx].max(-1).values for idx in positions],-1)
                        class_indices=class_scores.argmax(-1)
                        box=outputs.pred_boxes[0].float()
                        xyxy=torch.cat((box[:,:2]-box[:,2:]/2,box[:,:2]+box[:,2:]/2),-1)*1000
                        keep=scores>config['grounding_minimum_score']
                        for score,b,index in zip(scores[keep].cpu().tolist(),xyxy[keep].cpu().tolist(),class_indices[keep].cpu().tolist()):
                            candidates.append({'score':score,'bbox_1000':b,'category':chunk_labels[index]})
                else:
                    candidates=sorted(proposal_map[sample.record_uid]['candidates'],key=lambda c:-c['score'])[:config['classifier_top_k']]
                    size=config['classifier_batch_size']
                    for start in range(0,len(candidates),size):
                        chunk=candidates[start:start+size]
                        valid=[]; crops=[]
                        for c in chunk:
                            b=c['bbox_1000']
                            # Invalid/outside crops remain unclassified; never fabricate an image.
                            if min(1000,b[2])<=max(0,b[0]) or min(1000,b[3])<=max(0,b[1]):
                                c['category']=None; continue
                            crop=crop_from_box(image,b,model_config['cascade']['context_margin'])
                            if not crop.width or not crop.height:
                                c['category']=None; continue
                            valid.append(c); crops.append(crop)
                        if crops:
                            pixels=processor(images=crops,return_tensors='pt')['pixel_values'].to(device)
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                indexes=model(pixels).argmax(-1).cpu().tolist()
                            for c,index in zip(valid,indexes): c['category']=labels[index]
                candidates.sort(key=lambda c:-c['score'])
                row={'record_uid':sample.record_uid,'target':expected[sample.record_uid]['target'],'candidates':candidates}
                handle.write(json.dumps(row)+'\n'); handle.flush(); rows.append(row)
        del model
        torch.cuda.empty_cache()
    ks=[1,config['classifier_top_k']] if name=='dfine_dinov2' else config['candidate_k']
    result=candidate_summary(rows,ks,classified=name!='dfine')
    original_name='dfine_t1' if name=='dfine' else 'grounding_dino_t1' if name=='grounding_dino' else 'dfine_dinov2_t5'
    original=load_rows(f'results/table4/{original_name}/{protocol}/seed{seed}_test.json')
    old_by_uid={r['record_uid']:r for r in original}
    diffs=[]
    for row in rows:
        prediction=old_by_uid[row['record_uid']]['prediction']; candidates=row['candidates']
        if candidates and prediction['bbox_1000'] is not None:
            diffs.append(max(abs(a-b) for a,b in zip(candidates[0]['bbox_1000'],prediction['bbox_1000'])))
    save_json(output/f'{name}{suffix}_metrics.json',{'settings':meta,'n_images':len(rows),
        'n_candidates':sum(len(r['candidates']) for r in rows), 'metrics':result,
        'old_top1_max_box_coordinate_difference':max(diffs,default=None),
        'note':'New inference, fixed original weights. G-DINO class mapping is an added evaluation convention, not a trained classifier.'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['dfine','dfine_dinov2','grounding_dino'],required=True)
    parser.add_argument('--config',default=DEFAULT)
    parser.add_argument('--limit',type=int,default=0,help='Smoke run only, saved under separate filenames')
    main(parser.parse_args())
