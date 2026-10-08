import copy
import tempfile
import unittest
import numpy as np
import torch
from eval.phrase_native_position import refine,signature,direction_counters
from eval.joint_model import JointModel
from eval.joint_export import export_chart


class PositionTests(unittest.TestCase):
    def test_exact_cuts_source_pose_independence_and_native_exception(self):
        torch.set_num_threads(2);torch.manual_seed(19);model=JointModel()
        source={'bpm':120.,'duration_beats':17.,'bombs':[(16.,2,1)],'walls':[(0.,1.,0,1,0,5)],
            'notes':[(0.,0,0,0,0),(0.,1,2,0,1),(.25,0,1,0,0),(.5,0,0,0,1),(.5,0,1,0,8),
                     (8.,1,3,0,1),(16.999,0,0,0,4)]}
        changed=copy.deepcopy(source);changed['notes']=[(b,h,c,l+1,d) for b,h,c,l,d in source['notes']]
        audio={'times':np.arange(20.),'x':np.zeros((20,27))}
        a=refine(model,source,audio);b=refine(model,changed,audio)
        self.assertTrue(a['ok']);self.assertEqual(signature(a['source']),signature(source));self.assertEqual(a['source']['notes'],b['source']['notes'])
        counters=direction_counters(source);self.assertEqual(counters,direction_counters(a['source']))
        self.assertEqual(counters['bins']['fast']['same_family'],1)
        with tempfile.TemporaryDirectory() as tmp:
            saved=export_chart(a['source'],tmp)
            self.assertEqual(signature(saved),signature(source));self.assertEqual(direction_counters(saved),counters)
            self.assertEqual(saved['bombs'],source['bombs']);self.assertEqual(saved['walls'],source['walls'])


if __name__=='__main__':unittest.main()
