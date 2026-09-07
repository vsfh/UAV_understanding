"""Exercise the result writer without importing or loading vision models."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ast.parse((ROOT/'src/clear_uav/table4.py').read_text(encoding='utf-8'))
writer=next(n for n in SOURCE.body if isinstance(n,ast.FunctionDef) and n.name=='save_results')
namespace={'Path':Path,'json':json,'hashlib':hashlib,'box_iou':lambda a,b:1.}
exec(compile(ast.Module(body=[writer],type_ignores=[]),'result-writer','exec'),namespace)


class MetadataTests(unittest.TestCase):
    def test_every_internal_call_passes_current_seed(self):
        calls=[n for n in ast.walk(SOURCE) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='save_results']
        self.assertEqual(len(calls),8)
        for call in calls:
            value=next(k.value for k in call.keywords if k.arg=='seed')
            self.assertIsInstance(value,ast.Name)
            self.assertEqual(value.id,'seed')

    def test_seed_and_config_identity_are_saved_without_changing_scores(self):
        config={'experiment':'paper_shift_qwen','train':{'seeds':[43,44]}}
        sample=SimpleNamespace(record_uid='x',group_id='g',negative_subtype=None,presence=True,bbox_1000=[0,0,1,1],label='event')
        prediction={'bbox_1000':[0,0,1,1],'presence_score':.8}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            namespace['save_results'](path,config,'unseen_site',[sample],[prediction],{'c_f1':.5},seed=44)
            result=json.loads(path.read_text())
            self.assertEqual(result['seed'],44)
            self.assertEqual(result['experiment'],'paper_shift_qwen')
            self.assertEqual(result['metrics'],{'c_f1':.5})
            self.assertEqual(result['rows'][0]['prediction'],prediction)
            self.assertEqual(len(result['config_sha256']),64)
            with self.assertRaisesRegex(ValueError,'seed'):
                namespace['save_results'](path,config,'unseen_site',[],[],{},seed=45)


if __name__=='__main__': unittest.main()
