import copy
import tempfile
import unittest
import numpy as np
import torch
from eval.phrase_native_phase import refine,signature,direction_counters,directions
from eval.joint_model import JointModel
from eval.joint_export import export_chart


class PhaseTests(unittest.TestCase):
    def test_native_phase_source_pose_independence_and_native_exception(self):
        torch.set_num_threads(2);torch.manual_seed(19);model=JointModel()
        source={'bpm':120.,'duration_beats':17.,'bombs':[(16.,2,1)],'walls':[(0.,1.,0,1,0,5)],
            'notes':[(0.,0,0,0,0),(0.,1,2,0,1),(.25,0,1,0,0),(.5,0,0,0,1),(.5,0,1,0,8),
                     (8.,1,3,0,2),(8.25,0,0,0,3),(16.999,0,0,0,4)]}
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


    def test_diagonal_freedom_without_phase_or_lateral_escape(self):
        self.assertEqual(directions(0),(0,4,5));self.assertEqual(directions(6),(1,6,7))
        for d in (2,3,8):self.assertEqual(directions(d),(d,))
        source={'notes':[(0.,0,0,0,0),(1.,0,0,0,1),(2.,1,3,0,2),(3.,1,3,0,8)]}
        alternate={'notes':[(0.,0,0,0,5),(1.,0,0,0,6),(2.,1,3,0,2),(3.,1,3,0,8)]}
        self.assertEqual(signature(source),signature(alternate))
        for index,d in ((0,1),(2,3),(3,0)):
            changed=copy.deepcopy(source);note=list(changed['notes'][index]);note[-1]=d;changed['notes'][index]=tuple(note)
            self.assertNotEqual(signature(source),signature(changed))


if __name__=='__main__':unittest.main()
