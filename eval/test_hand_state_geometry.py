import copy
import unittest
import numpy as np
import torch
from eval import hand_state_geometry as g
from eval.joint_decode import JointState
from eval.joint_phrase import encode_events
from eval.joint_model import EMPTY,literal_slots


class GeometryStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(19)
        self.source={'bpm':120.,'duration_beats':20.,'bombs':[],'walls':[],
            'notes':[(0.,0,0,0,0),(.25,1,3,0,1),(.5,0,0,0,0),(.5,0,1,0,8),
                     (8.,1,3,0,2),(16.999,0,0,0,4)]}
        self.audio={'times':np.arange(20.),'x':np.zeros((20,27))}

    def test_history_survives_other_hand_and_rest_and_inputs_are_causal(self):
        source=self.source;events=encode_events(source['notes']);state=JointState(20.)
        state.append(events[0]);state.append(events[1]);state.rest(.4)
        x=g.features(source,self.audio,events[2],state,4.)
        self.assertEqual(x['history'][0],0);self.assertNotEqual(x['history'][3],EMPTY)
        self.assertAlmostEqual(x['age'][0],np.log1p(.25)/4)
        self.assertAlmostEqual(x['age'][1],np.log1p(.125)/4)
        changed=copy.deepcopy(events[2]);changed['hands'][0]=[(2,1,5),(3,1,8)]
        y=g.features(source,self.audio,changed,state,4.)
        for k in x:np.testing.assert_array_equal(x[k],y[k])
        state.last_by_hand[0]['notes']=((1,2,4),)
        z=g.features(source,self.audio,events[2],state,4.)
        self.assertNotEqual(x['history'],z['history'])

    def test_reachability_loss_prefix_and_rollout_pose_independence(self):
        batch=g.teacher(self.source,self.audio);model=g.GeometryModel()
        self.assertEqual(batch['history'][-1,:3].tolist(),literal_slots(encode_events(self.source['notes'])[2])[:3])
        self.assertAlmostEqual(float(batch['age'][-1,0]),np.log1p((16.999-.5)*.5)/4,places=6)
        logits=model(batch);value=g.loss(logits,batch);self.assertTrue(torch.isfinite(value));value.backward()
        altered={**batch,'slots':batch['slots'].clone()};altered['slots'][:,1:]=7
        torch.testing.assert_close(logits[:,0],model(altered)[:,0])
        changed=copy.deepcopy(self.source);changed['notes']=[(b,h,c,l+1,d) for b,h,c,l,d in self.source['notes']]
        a=g.refine(model,self.source,self.audio);b=g.refine(model,changed,self.audio)
        self.assertTrue(a['ok']);self.assertEqual(a['source']['notes'],b['source']['notes'])
        self.assertEqual(g.signature(a['source']),g.signature(self.source))
        self.assertEqual(g.direction_counters(a['source']),g.direction_counters(self.source))

    def test_mask_uses_only_exclusive_prefix_and_reserves_cells(self):
        event={'beat':0.,'hands':[[(0,0,0),(0,1,8),(3,2,1)],[(1,0,2),(1,1,3)]]}
        source={**self.source,'notes':[(0.,h,c,l,d) for h,ns in enumerate(event['hands']) for c,l,d in ns]}
        batch=g.teacher(source,self.audio);mask=g.legal_mask(batch['phase'],batch['slots'])
        expected=np.mean(np.log([30,10,30,8,7]))
        self.assertAlmostEqual(float(g.loss(torch.zeros((1,6,108)),batch)),expected,places=6)
        changed=batch['slots'].clone();changed[:,1:]=7
        torch.testing.assert_close(mask[:,0],g.legal_mask(batch['phase'],changed)[:,0])
        self.assertFalse(mask[0,0,10*9].item()) # leave two higher cells for this hand
        self.assertFalse(mask[0,3,0*9+2].item()) # other hand's occupied cell
        self.assertTrue(mask[0,3,3*9+2].item())


if __name__=='__main__':unittest.main()
