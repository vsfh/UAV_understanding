"""CPU regression tests: duplicates, negatives, top-k, class AP, alignment."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from paper_small_checks import iou, multi_ap, wilson, candidate_summary
from paper_roi_review import score


def row(uid, candidates, positive=True, label='a'):
    return {'record_uid':uid,'target':{'presence':positive,'category':label if positive else None,
            'bbox_1000':[0,0,100,100] if positive else None},'candidates':candidates}


def box(score, label='a', match=True):
    return {'score':score,'category':label,'bbox_1000':[0,0,100,100] if match else [500,500,600,600]}


class Metrics(unittest.TestCase):
    def test_iou(self):
        self.assertEqual(iou([0,0,10,10],[0,0,10,10]),1)
        self.assertEqual(iou(None,[0,0,10,10]),0)

    def test_duplicate_is_false_positive(self):
        values=[row('1',[box(.9),box(.8)]),row('2',[box(.7)])]
        self.assertAlmostEqual(multi_ap(values),(1+2/3)/2)

    def test_wrong_top1_recovered_at_two(self):
        values=[row('1',[box(.9,match=False),box(.8)])]
        self.assertEqual(multi_ap(values,1),0)
        self.assertAlmostEqual(multi_ap(values,2),.5)

    def test_negative_false_positive(self):
        self.assertAlmostEqual(multi_ap([row('n',[box(.9)],False),row('p',[box(.8)])]),.5)

    def test_wrong_class_no_credit(self):
        values=[row('1',[box(.9,'b'),box(.8,'a')])]
        self.assertEqual(multi_ap(values,1,'a'),0)
        self.assertEqual(multi_ap(values,2,'a'),1)

    def test_absent_candidates_keep_denominator(self):
        self.assertEqual(multi_ap([row('1',[]),row('2',[box(.8)])]),.5)

    def test_topk_ap_not_recall(self):
        self.assertEqual(multi_ap([row('1',[box(.9,match=False),box(.8),box(.7)])]),.5)

    def test_no_class_result_is_null(self):
        self.assertIsNone(candidate_summary([row('1',[box(.9,None)])],[1])['1']['g_map50'])

    def test_wilson(self):
        lo,hi=wilson(46,560)
        self.assertAlmostEqual(lo,.0621473393399)
        self.assertAlmostEqual(hi,.1078321066929)

    def test_review_partial_and_blinded_ratings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'roi_review';root.mkdir()
            key={'manifest_hash':'test-only','records':[{'id':'R1','label':'a','weight':2,
                 'reference':[0,0,100,100],'candidate_models':{'C1':'direct','C2':'perception'}},
                 {'id':'R2','label':'a','weight':1,'reference':[0,0,100,100],
                  'candidate_models':{'C1':'perception','C2':'direct'}}]}
            (root/'private_key.json').write_text(json.dumps(key))
            draws=[];utilities=[]
            for reviewer in ['B','A']:
                for mode,paths in [('draw',draws),('utility',utilities)]:
                    responses={'R1':{'box':[0,0,100,100]},'R2':{'uncertain':True}} if mode=='draw' else {'R1':{'C1':'no','C2':'yes'},'R2':{'C1':''}}
                    p=root/f'{reviewer}_{mode}.json'
                    p.write_text(json.dumps({'reviewer':reviewer,'mode':mode,'manifest_hash':'test-only','responses':responses}));paths.append(str(p))
            with contextlib.redirect_stdout(io.StringIO()):
                score({'output':folder},SimpleNamespace(draw=draws,utility=utilities))
            result=json.loads((root/'agreement_results.json').read_text())
            self.assertEqual(result['status'],'partial')
            self.assertEqual(result['completed_pair'],1)
            self.assertEqual(result['unweighted_mean_iou'],1)
            self.assertEqual(result['utility']['perception']['yes'],2)
            self.assertEqual(result['utility']['direct']['no'],2)


if __name__=='__main__': unittest.main()
