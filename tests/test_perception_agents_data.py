import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from perception_agents_data import example_set


class DataTests(unittest.TestCase):
    def setUp(self):
        self.settings={'candidate_seed':4300,'verifier_candidates_per_image':4,'accept_iou':.5}
        self.labels=['water','garbage']
        self.sample=SimpleNamespace(record_uid='train_01',presence=True,label='water',bbox_1000=[100,200,400,500])
        self.wrong={'category':'garbage','bbox_1000':[0,0,100,100]}

    def test_corrective_training_matches_category_feedback(self):
        examples=example_set(self.sample,self.wrong,self.labels,self.settings,epoch=1)
        what,where=examples[:2]
        self.assertEqual(what['feedback']['verdict'],'reclassify')
        self.assertEqual(where['feedback'],what['feedback'])
        self.assertEqual(where['candidate']['category'],'water')
        self.assertEqual(where['answer'],'water<vis>')
        self.assertNotEqual(where['feedback']['candidate']['category'],where['candidate']['category'])

    def test_validation_fixed_and_equal_role_weight(self):
        examples=example_set(self.sample,self.wrong,self.labels,self.settings,validation=True)
        self.assertEqual(examples,example_set(self.sample,self.wrong,self.labels,self.settings,validation=True))
        for role in ('what','where','verify'):
            self.assertAlmostEqual(sum(e['weight'] for e in examples if e['role']==role),1/3)
        self.assertEqual({e['answer'] for e in examples if e['role']=='verify'},{'A','B','C'})

    def test_draws_cover_real_and_synthetic_candidates(self):
        examples=[example_set(self.sample,self.wrong,self.labels,self.settings,draw_index=i)[-1] for i in range(4)]
        self.assertIn(self.wrong,[e['candidate'] for e in examples])
        self.assertEqual({e['answer'] for e in examples},{'A','B','C'})

    def test_negative_has_no_box_supervision(self):
        negative=SimpleNamespace(record_uid='neg_01',presence=False,label=None,bbox_1000=None)
        examples=example_set(negative,self.wrong,self.labels,self.settings,validation=True)
        self.assertNotIn('where',[e['role'] for e in examples])
        self.assertEqual(examples[0]['answer'],'no_event')
        self.assertEqual({e['answer'] for e in examples if e['role']=='verify'},{'D'})
        self.assertAlmostEqual(sum(e['weight'] for e in examples),1)


if __name__=='__main__':
    unittest.main()
