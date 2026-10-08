import unittest
import torch

import groom
from learned_geometry import sample_geometry
from swing_clearance import blocked_approaches


class LearnedGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.flow = groom.Flow().eval()
        for p in self.flow.parameters(): p.data.zero_()

    def test_model_prefers_legal_alternative_instead_of_posthoc_move(self):
        self.flow.dir_head.bias.data[3] = 40
        self.flow.col_head.bias.data[3] = 20
        self.flow.lay_head.bias.data[2] = 10
        self.flow.chain_head.bias.data[0] = 30
        hidden = torch.zeros(self.flow.dir_head.in_features)
        other = [(0, 0, 0, 0, 2)]
        with torch.no_grad():
            for seed in range(10):
                d, col, layer, count = sample_geometry(self.flow, hidden, 1, 0, (3,0),
                    range(8), other, set(), torch.Generator().manual_seed(seed), .1)
                self.assertEqual((d, col, count), (3, 3, 1))
                self.assertNotEqual(layer, 0)
                self.assertFalse(blocked_approaches(other+[(0,1,col,layer,d)]))

    def test_pinned_chain_and_wall_constraints_are_joint(self):
        hidden = torch.zeros(self.flow.dir_head.in_features)
        with torch.no_grad():
            d,c,l,k=sample_geometry(self.flow,hidden,0,0,(0,0),[3],[],{3},
                                    torch.Generator().manual_seed(0),1.,followers=2)
            self.assertEqual((d,c,k),(3,0,3))
            # Unlike legacy sampling, this arm permits center-row cells.
            with self.assertRaisesRegex(ValueError,'no complete clear cut'):
                sample_geometry(self.flow,hidden,0,0,(0,0),[3],[],{0,3},
                                torch.Generator().manual_seed(0),1.,followers=2)

    def test_decoder_honors_schedule_with_no_clearance_repair(self):
        model=groom.Groomer().eval()
        for p in model.parameters(): p.data.zero_()
        schedule={s:dict(hands=(0,1),k={0:0,1:0}) for s in range(0,32,4)}
        trace={}
        raw,_=groom.groom_notes(set(schedule),40,125.,model=model,flow=self.flow,
            replay_mode='off',schedule_in=schedule,geometry_policy='learned',trace=trace,
            calibrate=False)
        self.assertEqual([(s,h) for s,h,*_ in raw],[(s,h) for s in schedule for h in (0,1)])
        self.assertFalse(blocked_approaches(raw))
        self.assertEqual(trace['cut_clearance']['events_changed'],0)
        self.assertEqual(trace['schedule_infeasible'],0)


if __name__=='__main__': unittest.main()
