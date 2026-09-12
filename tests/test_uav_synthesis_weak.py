import copy
import importlib.util
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import uav_synthesis_weak as weak


class WeakTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.args=types.SimpleNamespace(work_dir=self.root/'work',synthetic_root=self.root/'synthetic',
            classes=['a','b'],per_class=2,selection_dir=self.root/'selection',config=self.root/'pipeline.yaml',
            project_root=self.root,hf_cache=self.root/'hf_cache',crop_inference_steps=50,
            inpaint_inference_steps=30,save_conditions=False)
        (self.args.work_dir/'prompts').mkdir(parents=True)
        self.rows={}
        for label in self.args.classes:
            self.rows[label]=[{'synthetic_id':f'syn_{label}_{i}','event_class':label,
                'description':label+' scene','crop_relative':f'period/{label}/cropped/{i}.png',
                'original_relative':f'period/{label}/original/{i}.png','target_crop_size':[16,16],
                'target_original_size':[32,24],'bbox_xyxy':[2,4,12,14],'is_synthetic':True,
                'template_uid':'train_only','seed':i} for i in range(3)]
            (self.args.work_dir/'prompts'/f'{label}.jsonl').write_text(
                ''.join(json.dumps(r)+'\n' for r in self.rows[label]),encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_per_class_limit_and_round_robin(self):
        rows,sources=weak.select_records(self.args)
        self.assertEqual([r['event_class'] for r in rows],['a','b','a','b'])
        self.assertEqual([r['synthetic_id'] for r in rows],['syn_a_0','syn_b_0','syn_a_1','syn_b_1'])
        self.assertEqual(sources['a']['selected'],2)
        self.assertEqual(rows[0],self.rows['a'][0])

    def test_incomplete_tail_is_ignored_without_modification(self):
        p=self.args.work_dir/'prompts/a.jsonl'
        original=p.read_bytes()+b'{"description":"broken\xe4\xb8'
        p.write_bytes(original)
        rows,tail,_=weak.read_records(p)
        self.assertTrue(tail)
        self.assertEqual(len(rows),3)
        self.assertEqual(p.read_bytes(),original)

    def test_internal_corruption_is_not_skipped(self):
        p=self.args.work_dir/'prompts/a.jsonl'
        p.write_text('broken\n'+json.dumps(self.rows['a'][0])+'\n')
        with self.assertRaises(json.JSONDecodeError):weak.read_records(p)

    def test_missing_class_and_path_escape_rejected(self):
        self.args.classes=['absent']
        with self.assertRaises(FileNotFoundError):weak.select_records(self.args)
        with self.assertRaises(ValueError):weak.safe_path(self.args.synthetic_root,'../real.jpg')

    def test_invalid_bbox_and_duplicate_ids_rejected(self):
        row=copy.deepcopy(self.rows['a'][0]);row['bbox_xyxy']=[0,0,33,24]
        with self.assertRaises(ValueError):weak.validate_record(row,'a',self.args.synthetic_root)
        p=self.args.work_dir/'prompts/a.jsonl'
        p.write_text((json.dumps(self.rows['a'][0])+'\n')*2)
        with self.assertRaises(ValueError):weak.select_records(self.args)

    def test_repair_incomplete_image_and_description(self):
        from PIL import Image
        from unittest.mock import patch
        module=types.ModuleType('uav_synthesis_pipeline');module.KLEIN_MODEL='test-model'
        r=self.rows['a'][0]
        crop=weak.safe_path(self.args.synthetic_root,r['crop_relative'])
        original=weak.safe_path(self.args.synthetic_root,r['original_relative'])
        metadata=self.args.synthetic_root/'metadata'/f'{r["synthetic_id"]}.json'
        crop.parent.mkdir(parents=True);original.parent.mkdir(parents=True)
        Image.new('RGB',(16,16),'red').save(crop)
        original.write_bytes(b'broken png')
        weak.atomic_json(metadata,dict(r,bbox_xywh=[2,4,10,10]))
        with patch.dict(sys.modules,{'uav_synthesis_pipeline':module}):
            state=weak.record_state(self.args,r,repair=True)
            self.assertTrue(state['crop']);self.assertFalse(state['complete'])
            self.assertFalse(original.exists())
            desc=self.args.synthetic_root/'description'/Path(r['crop_relative']).with_suffix('.json')
            self.assertEqual(json.loads(desc.read_text())['description'],r['description'])
            Image.new('RGB',(32,24),'blue').save(original)
            self.assertTrue(weak.record_state(self.args,r)['complete'])
            crop.write_bytes(b'partial')
            state=weak.record_state(self.args,r,repair=True)
            self.assertFalse(state['complete'])
            self.assertFalse(original.exists());self.assertFalse(metadata.exists())

    def test_upstream_receives_only_selected_records(self):
        from unittest.mock import patch
        module=types.ModuleType('uav_synthesis_pipeline')
        records,_=weak.select_records(self.args)
        with patch.dict(sys.modules,{'uav_synthesis_pipeline':module}):
            upstream=weak.upstream(self.args,records)
            self.assertEqual(list(upstream.prompt_records(None)),records)
            self.assertEqual(list(upstream.prompt_records(None)),records)


if __name__=='__main__':unittest.main()
