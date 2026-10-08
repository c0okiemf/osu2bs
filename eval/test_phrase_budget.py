import unittest
import numpy as np
from eval.phrase_budget import fit_ridge,predict,rhythm,features,error


class BudgetTests(unittest.TestCase):
    def test_train_boundary_constant_features_and_integer_prediction(self):
        x=np.ones((2,4));y=np.array([[.2,.3,.1],[.4,.1,.3]])
        with self.assertRaisesRegex(ValueError,'train-only'):fit_ridge(x,y,[1,1],['train','validation'])
        model=fit_ridge(x,y,[1,1],['train','train']);np.testing.assert_array_equal(model['coef'],np.zeros((4,3)))
        np.testing.assert_allclose(model['target_mean'],[.3,.2,.2])
        for arm in ('ridge','constant'):
            counts=predict(model,x,5,[4,1],arm)
            np.testing.assert_array_equal(counts,[[6,4,4],[2,1,1]])
        model['target_mean'][:]=-.1;np.testing.assert_array_equal(predict(model,x,5,[4,1],'constant'),np.zeros((2,3),int))

    def test_budget_identities_rests_tail_and_empty_audio_workload(self):
        plans=[{'start':0.,'end':8.,'counts':[2,1,3]},{'start':8.,'end':9.,'counts':[0,0,0]}]
        p=rhythm(plans,120.)
        self.assertEqual(p,[[1.5,1.25,1.,.5,0.],[0.,0.,0.,0.,1.]])
        self.assertEqual(error(p,p,{'rhythm':[1]*5,'geometry':[1]*5})['mean'],0)
        audio={'times':np.arange(10.),'x':np.zeros((10,27))}
        x=features(audio,20,21,120,6,24)
        self.assertEqual(len(x),91);self.assertEqual(x[54],3.)
        with self.assertRaises(ValueError):features(audio,0,8,0,6,24)


if __name__=='__main__':unittest.main()
