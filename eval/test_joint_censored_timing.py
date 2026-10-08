import unittest
import numpy as np
import torch
from eval.joint_model import N_GAP,JointModel,EMPTY
from eval.joint_timing import timing_support
from eval.joint_censored_timing import censored_logits,rollout,audit_targets
from unittest import mock


class CensoredTimingTests(unittest.TestCase):
    def test_boundary_mass_is_preserved_and_in_window_odds_unchanged(self):
        logits=torch.linspace(-3,3,N_GAP);targets,allowed=timing_support(7.99,8.,1,np.zeros(N_GAP))
        for temp in (.85,1,1.15):
            result,over=censored_logits(logits,targets,allowed,8.,temp);actual=result.softmax(0)
            raw=(logits/temp).softmax(0);expected=raw.clone();expected[0]+=raw[over].sum();expected[over]=0;expected[1]=0;expected/=expected.sum()
            torch.testing.assert_close(actual,expected)
            self.assertEqual(int(torch.isfinite(result).sum()),1)
        targets,allowed=timing_support(7.,8.,0,np.zeros(N_GAP));result,over=censored_logits(logits,targets,allowed,8.,1)
        self.assertTrue(over[14]) # Exact next-boundary target belongs to REST.
        self.assertTrue(torch.equal(result[torch.tensor(allowed)][1:],logits[torch.tensor(allowed)][1:]))

    def test_no_overflow_matches_original_mask_and_zero_repeat_stays_illegal(self):
        logits=torch.arange(N_GAP,dtype=torch.float32);targets,allowed=timing_support(0.,100.,1,np.zeros(N_GAP))
        result,over=censored_logits(logits,targets,allowed,100.,1)
        self.assertFalse(over.any());self.assertFalse(torch.isfinite(result[1]))
        self.assertTrue(torch.equal(result,logits.masked_fill(~torch.tensor(allowed),-float('inf'))))

    def test_overflow_emits_only_rest_then_reconditions_without_repair(self):
        torch.set_num_threads(2);model=JointModel()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model.residual.bias.fill_(8)
            model.count.bias.fill_(-1000);model.count.bias[5]=1000
            model.slot[-1].bias.fill_(-1000);model.slot[-1].bias[1]=1000
        at_start=torch.full((N_GAP,),-1000.);at_start[1]=1000
        overflow=torch.full((N_GAP,),-1000.);overflow[-1]=1000
        source={'notes':[],'events':[],'bombs':[],'walls':[],'bpm':120,'duration_beats':16.}
        audio={'times':np.arange(20.),'x':np.zeros((20,27))}
        with mock.patch.object(model.gap,'forward',side_effect=[at_start,overflow]*2),mock.patch.object(model,'hidden',wraps=model.hidden) as hidden:
            result=rollout(model,source,audio,4)
            self.assertEqual(int(hidden.call_args_list[2].args[2][0,0]),0)
            self.assertNotEqual(hidden.call_args_list[2].args[1][0,0].tolist(),[EMPTY]*6)
        self.assertTrue(result['ok']);self.assertEqual([e['beat'] for e in result['source']['events']],[0,8])
        self.assertEqual(result['timing_support']['mean_removed_mass'],0)
        self.assertEqual(result['boundary_censoring']['mean_overflow_mass'],.5)
        self.assertTrue(audit_targets(result,result['source'])['raw_event_targets_exact'])


if __name__=='__main__':unittest.main()
