import unittest
import numpy as np
import torch
from eval.joint_likelihood import geometry_nll,select
from eval.joint_model import JointModel,EMPTY,actions,context,source_rate
from eval.joint_phrase import encode_events


class LikelihoodTests(unittest.TestCase):
    def test_batched_scores_match_incremental_raw_slot_probabilities(self):
        torch.set_num_threads(2);torch.manual_seed(23);model=JointModel().eval()
        source={'bpm':120.,'duration_beats':17.,'bombs':[],'walls':[],
                'notes':[(0,0,0,0,1),(0,0,1,0,8),(0,1,3,0,0),(1/3,1,2,1,8),
                         (8,0,0,0,0),(16.5,1,3,0,1)]}
        source['events']=encode_events(source['notes'])
        audio={'times':np.arange(10.),'x':np.zeros((10,27))};hidden=None;total=0.;count=0
        with torch.no_grad():
            for row in actions(source):
                ctx=context(audio,row['cursor'],source['bpm'],source_rate(source),18,[],[])
                h,hidden=model.hidden(torch.tensor(ctx)[None,None],torch.tensor(row['prev_slots'])[None,None],
                    torch.tensor([[row['prev_kind']]]),torch.tensor([[row['prev_gap']]],dtype=torch.float32),hidden)
                if row['gap']==0:continue
                prefix=[EMPTY]*6
                for slot,token in enumerate(row['slots']):
                    logits=model.slot_logits(h[0,0],torch.tensor(row['gap']),torch.tensor(row['count']),torch.tensor(prefix))[slot]
                    if token!=EMPTY:
                        total-=float(torch.log_softmax(logits.double(),0)[token]);count+=1
                    prefix[slot]=token
        for chunk in (1,3,256):
            score=geometry_nll(model,source,audio,chunk)
            self.assertEqual(score['tokens'],count);self.assertAlmostEqual(score['nll_sum'],total,places=5)
        self.assertEqual(geometry_nll(model,source,audio),geometry_nll(model,source,audio))

    def test_selection_rejects_unknown_and_breaks_likelihood_ties_by_seed(self):
        records=[{'ok':True,'machine':{'admitted':v}} for v in (False,True,True)]
        scores=[{'mean_nll':v} for v in (0.,2.,2.)];fallback={'original':True}
        seed,chosen=select(records,scores,fallback)
        self.assertEqual(seed,1);self.assertIs(chosen,records[1])
        self.assertIs(select(records[:1],scores[:1],fallback)[1],fallback)
        with self.assertRaises(ValueError):select(records,scores[:1],fallback)


if __name__=='__main__':unittest.main()
