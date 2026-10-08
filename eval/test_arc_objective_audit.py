import copy
import unittest
import numpy as np
from eval.arc_objective_audit import marginal,zero_arc,diagnose


class ObjectiveTests(unittest.TestCase):
    def test_exact_empirical_pair_loss(self):
        x=np.array([0.,0.,0.,20.]);d=marginal(x)
        self.assertEqual(d['median'],0.);self.assertEqual(d['zero_absolute_loss'],5.)
        self.assertEqual(d['iid_pair_absolute_loss'],float(np.abs(x[:,None]-x).mean()))
        self.assertEqual(d['iid_pair_absolute_loss'],7.5)

    def test_counterfactual_preserves_unknowns_and_marginal_ignores_order(self):
        p={'rhythm':[[1.]*5]*4,'geometry':[[1.,a,1.,1.,1.] for a in (0.,0.,20.,0.)]}
        q=copy.deepcopy(p);q['geometry'][0][1]=20.;q['geometry'][2][1]=0.
        scales={'rhythm':[1.]*5,'geometry':[1.]*5};d=diagnose(p,q,scales)
        self.assertEqual(d['original']['geometry']['components'][1],10.)
        self.assertEqual(d['arc_marginal_wasserstein'],0.)
        self.assertEqual(d['zero_arc_descriptor_only']['geometry']['components'][1],5.)
        p['geometry'][0][1]=None;z=zero_arc(p)
        self.assertIsNone(z['geometry'][0][1]);self.assertEqual(p['geometry'][2][1],20.)
        self.assertEqual(z['geometry'][2],[1.,0.,1.,1.,1.])


if __name__=='__main__':unittest.main()
