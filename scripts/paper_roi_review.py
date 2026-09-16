"""Prepare blinded independent ROI annotation and score returned JSON files."""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from paper_small_checks import DEFAULT, read_config, save_json, paired_rows, iou


def prepare(config):
    from PIL import Image
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from clear_uav.table4 import read_discovery_samples
    pair=paired_rows(config)
    recipe=read_config(config['perception_config'])
    samples=read_discovery_samples(recipe,config['protocol'],'test')
    paths={s.record_uid:s.image_path for s in samples}
    positives=[r for r in pair['direct'] if r['target']['presence']]
    models={name:{r['record_uid']:r for r in rows} for name,rows in pair.items()}
    root=Path(config['output'])/'roi_review'
    if (root/'private_key.json').exists():
        print('Review package already exists; preserving assignments and reviewer files:',root); return
    (root/'images').mkdir(parents=True,exist_ok=True)
    rng=random.Random(config['roi_review']['seed'])
    classes=defaultdict(list)
    for row in positives: classes[row['target']['category']].append(row)
    selected=[]
    for label,rows in sorted(classes.items()):
        rows.sort(key=lambda r:(r['target']['bbox_1000'][2]-r['target']['bbox_1000'][0])*
                              (r['target']['bbox_1000'][3]-r['target']['bbox_1000'][1]))
        # Twenty/class, five from each ROI-area quartile. No model-error selection.
        for index,stratum in enumerate(np.array_split(np.array(rows,dtype=object),4)):
            pool=list(stratum); rng.shuffle(pool)
            n=min(len(pool),config['roi_review']['per_class']//4)
            for row in pool[:n]: selected.append((row,index,len(pool),n))
    rng.shuffle(selected)
    draw=[]; utility=[]; key=[]
    for index,(row,stratum,population,n) in enumerate(tqdm(selected,desc='prepare blind ROI review',unit='image')):
        uid=row['record_uid']; code=f'R{index+1:04d}'
        image_file=code+'.jpg'
        with Image.open(paths[uid]) as image:
            # Native-resolution export. Browser fit + optional native-size scrolling.
            image.convert('RGB').save(root/'images'/image_file,quality=95)
            width,height=image.size
        task={'id':code,'image':'images/'+image_file,'label':row['target']['category'], 'width':width,'height':height}
        draw.append(task)
        names=['direct','perception']; rng.shuffle(names)
        candidates=[{'id':f'C{i+1}','box':models[name][uid]['prediction']['bbox_1000']} for i,name in enumerate(names)]
        utility.append(task|{'candidates':candidates})
        key.append({'id':code,'record_uid':uid,'label':task['label'],'reference':row['target']['bbox_1000'],
                    'group_id':row.get('group_id'),'area_quartile':stratum,'stratum_population':population,
                    'stratum_sample':n,'weight':population/n,
                    'candidate_models':{f'C{i+1}':name for i,name in enumerate(names)}})
    manifest_hash=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    save_json(root/'private_key.json',{'manifest_hash':manifest_hash,'sampling':config['roi_review'],'records':key})
    template=Path(__file__).with_name('paper_roi_review.html').read_text(encoding='utf-8')
    for reviewer in config['roi_review']['annotators']:
        for mode,tasks in [('draw',draw),('utility',utility)]:
            items=list(tasks); random.Random(config['roi_review']['seed']+ord(reviewer[0])).shuffle(items)
            payload={'reviewer':reviewer,'mode':mode,'manifest_hash':manifest_hash,'tasks':items}
            html=template.replace('__REVIEW_DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
            (root/f'{reviewer}_{mode}.html').write_text(html,encoding='utf-8')
    print('Prepared',len(draw),'images at',root)
    print('Give annotators only their HTML files + images/. Do not share private_key.json.')


def score(config,args):
    root=Path(config['output'])/'roi_review'
    key=json.loads((root/'private_key.json').read_text())
    draws=sorted([json.loads(Path(path).read_text()) for path in args.draw],key=lambda d:d['reviewer'])
    assert len(draws)==2 and draws[0]['reviewer']!=draws[1]['reviewer'], 'Two independent reviewers required'
    assert all(d['manifest_hash']==key['manifest_hash'] and d['mode']=='draw' for d in draws)
    ids={r['id'] for r in key['records']}
    assert all(set(d['responses'])<=ids for d in draws), 'Unknown review IDs'
    result={'status':'partial','expected':len(ids),'completed_pair':0,'per_record':[],
            'note':'Blank/uncertain records are not silently counted as agreement. Stratified unweighted and population-weighted summaries are distinct.'}
    for row in key['records']:
        answers=[d['responses'].get(row['id'],{}) for d in draws]
        boxes=[a.get('box') for a in answers]
        if not all(b and len(b)==4 and 0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000 for b in boxes): continue
        result['per_record'].append({'id':row['id'],'label':row['label'],'weight':row['weight'],
            'reviewer_iou':iou(*boxes),'A_reference_iou':iou(boxes[0],row['reference']),
            'B_reference_iou':iou(boxes[1],row['reference'])})
    records=result['per_record']; result['completed_pair']=len(records)
    if len(records)==len(ids): result['status']='complete'
    if records:
        values=np.array([r['reviewer_iou'] for r in records]); weights=np.array([r['weight'] for r in records])
        result.update({'unweighted_mean_iou':float(values.mean()),'median_iou':float(np.median(values)),
                       'q25_q75':np.quantile(values,[.25,.75]).tolist(),'fraction_iou_ge_05':float((values>=.5).mean()),
                       'population_weighted_mean_iou':float(np.average(values,weights=weights))})
    if args.utility:
        utilities=[json.loads(Path(path).read_text()) for path in args.utility]
        assert len({d['reviewer'] for d in utilities})==len(utilities)
        model_ratings=defaultdict(list); agreements=[]
        for d in utilities:
            assert d['manifest_hash']==key['manifest_hash'] and d['mode']=='utility'
            assert set(d['responses'])<=ids, 'Unknown utility review IDs'
        for row in key['records']:
            for candidate,model in row['candidate_models'].items():
                ratings=[d['responses'].get(row['id'],{}).get(candidate) for d in utilities]
                for rating in ratings:
                    if rating not in [None,'']:
                        assert rating in ['yes','no','uncertain']
                        model_ratings[model].append(rating)
                if len(ratings)==2 and all(ratings): agreements.append(ratings[0]==ratings[1])
        result['utility']={model:{r:ratings.count(r) for r in ['yes','no','uncertain']} for model,ratings in model_ratings.items()}
        result['utility_rating_agreement']=float(np.mean(agreements)) if agreements else None
        result['utility_note']='Counts are reviewer ratings, not independent image samples. Uncertain is separate. No adjudicated result is implied.'
    save_json(root/'agreement_results.json',result)
    save_json(root/'adjudication_needed.json',[r for r in records if r['reviewer_iou']<.5])
    print(json.dumps({k:v for k,v in result.items() if k!='per_record'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','score'])
    parser.add_argument('--config',default=DEFAULT)
    parser.add_argument('--draw',nargs=2,help='Independent A and B drawing exports')
    parser.add_argument('--utility',nargs=2,help='Independent A and B utility exports')
    args=parser.parse_args(); config=read_config(args.config)
    if args.stage=='prepare': prepare(config)
    else:
        if not args.draw: parser.error('score needs --draw A_draw.json B_draw.json')
        score(config,args)
