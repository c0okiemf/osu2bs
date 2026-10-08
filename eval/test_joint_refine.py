import copy
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
from eval.joint_model import JointModel
from eval.joint_phrase import encode_events
from eval.joint_export import export_chart
from eval.joint_refine import signature,refine,common_error,select


class RefineTests(unittest.TestCase):
    def test_exact_timing_roles_pose_independence_and_export(self):
        torch.set_num_threads(2);torch.manual_seed(19);model=JointModel()
        source={'bpm':120,'duration_beats':17,'bombs':[(16,2,1)],'walls':[(0,1,0,1,0,5)],
                'notes':[(0,0,0,0,1),(0,0,1,0,8),(0,1,2,0,0),(1/3,1,3,0,8),
                         (8,0,0,0,8),(8,1,3,0,1),(16.999,0,0,0,1)]}
        changed=copy.deepcopy(source)
        changed['notes']=[(b,h,c,l+1,8 if d==8 else (d+2)%8) for b,h,c,l,d in source['notes']]
        audio={'times':np.arange(20.),'x':np.zeros((20,27))}
        with mock.patch.object(model,'hidden',wraps=model.hidden) as hidden:
            a=refine(model,source,audio)
            # The boundary event after REST sees generated hand history.
            self.assertEqual(int(hidden.call_args_list[3].args[2][0,0]),0)
        b=refine(model,changed,audio)
        self.assertTrue(a['ok']);self.assertEqual(signature(a['source']),signature(source))
        self.assertEqual(a['source']['notes'],b['source']['notes'])
        for ev in encode_events(a['source']['notes']):
            cells=[(c,l) for hand in ev['hands'] for c,l,d in hand]
            self.assertEqual(len(cells),len(set(cells)))
        with tempfile.TemporaryDirectory() as tmp:
            readback=export_chart(a['source'],tmp)
            self.assertEqual(signature(readback),signature(source))
            self.assertEqual(readback['walls'],source['walls']);self.assertEqual(readback['bombs'],source['bombs'])

    def test_common_mask_arithmetic_and_missing_opportunities(self):
        ref={'geometry':[[1]*5,[None]*5]};cand={'geometry':[[2]*5,[100]*5]}
        scales={'geometry':[2]*5};mask=[[True]*5,[False]*5]
        result=common_error(ref,cand,scales,mask)
        self.assertEqual(result['mean'],.5);self.assertEqual(result['supported_windows'],[1]*5)
        cand['geometry'][0][2]=None
        with self.assertRaisesRegex(ValueError,'missing'):common_error(ref,cand,scales,mask)

    def test_selection_is_first_admitted_or_exact_retrieval_fallback(self):
        original={'name':'retrieval'}
        rejected={'ok':True,'machine':{'admitted':False}}
        admitted={'ok':True,'machine':{'admitted':True}}
        seed,selected=select([rejected,admitted,admitted],original)
        self.assertEqual(seed,1);self.assertIs(selected,admitted)
        seed,selected=select([rejected],original)
        self.assertIsNone(seed);self.assertIs(selected,original)


if __name__=='__main__':unittest.main()
