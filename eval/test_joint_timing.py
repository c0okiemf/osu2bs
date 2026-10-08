import unittest
from unittest import mock
import numpy as np
import torch
from eval.joint_model import JointModel,N_GAP,EMPTY
from eval.joint_timing import timing_support,rollout


class TimingTests(unittest.TestCase):
    def test_legal_support_tails_zero_and_nonfinite(self):
        residuals=np.zeros(N_GAP)
        for cursor,end in [(0,8),(7.999999999,8),(8,8.0000000001)]:
            for kind in (0,1,2):
                targets,mask=timing_support(cursor,end,kind,residuals)
                self.assertTrue(mask[0])
                self.assertEqual(mask[1],kind!=1)
                for i in range(2,N_GAP):
                    self.assertEqual(mask[i],cursor<targets[i]<end)
        residuals[3]=np.nan
        with self.assertRaisesRegex(ValueError,'residuals'):
            timing_support(0,8,2,residuals)
        with self.assertRaisesRegex(ValueError,'state'):
            timing_support(8,8,1,np.zeros(N_GAP))

    def test_scripted_complete_song_masks_overshoot_and_preserves_history(self):
        torch.set_num_threads(2)
        model=JointModel()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.residual.bias.fill_(8)
            model.count.bias.fill_(-1000);model.count.bias[5]=1000
            model.slot[-1].bias.fill_(-1000);model.slot[-1].bias[1]=1000
        at_boundary=torch.full((N_GAP,),-1000.);at_boundary[1]=1000
        overshoot=torch.full((N_GAP,),-1000.);overshoot[-1]=1000;overshoot[0]=0
        src={'notes':[],'events':[],'walls':[],'bombs':[],'bpm':120,'duration_beats':16}
        audio={'times':np.arange(20.),'x':np.zeros((20,27))}
        def run():
            with mock.patch.object(model.gap,'forward',side_effect=[at_boundary,overshoot]*2), \
                 mock.patch.object(model,'hidden',wraps=model.hidden) as hidden:
                r=rollout(model,src,audio,4)
                args=hidden.call_args_list[2].args
                self.assertEqual(int(args[2][0,0]),0)
                self.assertNotEqual(args[1][0,0].tolist(),[EMPTY]*6)
                return r
        a,b=run(),run()
        self.assertTrue(a['ok']);self.assertEqual(a['actions'],4)
        self.assertEqual([e['beat'] for e in a['source']['events']],[0,8])
        self.assertTrue(all(all(e['hands']) for e in a['source']['events']))
        self.assertEqual(a['source']['notes'],b['source']['notes'])
        self.assertEqual(a['timing_support']['rest_only_actions'],2)
        self.assertGreater(a['timing_support']['mean_removed_mass'],.4)

    def test_random_support_always_contains_rest_and_valid_event_times(self):
        rng=np.random.default_rng(29)
        for _ in range(1000):
            cursor=float(rng.uniform(0,500));end=cursor+float(10**rng.uniform(-10,1))
            targets,mask=timing_support(cursor,end,1,rng.normal(0,10,N_GAP))
            self.assertTrue(mask[0]);self.assertFalse(mask[1])
            self.assertTrue(all(cursor<t<end for t,ok in zip(targets[2:],mask[2:]) if ok))


if __name__=='__main__':unittest.main()
