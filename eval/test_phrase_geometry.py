import unittest
import numpy as np
from eval.phrase_geometry import fit,predict


class GeometryTests(unittest.TestCase):
    def test_masked_family_weights_constant_features_and_role_boundary(self):
        x=np.ones((3,2));y=np.tile([0.,2.,4.],(5,1)).T;y[0,1]=np.nan
        heads=fit(x,y,['a','a','b'],['train']*3)
        self.assertEqual(heads[0]['target_mean'],2.5);self.assertEqual(heads[1]['target_mean'],3.)
        self.assertEqual(heads[1]['supported_windows'],2)
        for h in heads:np.testing.assert_array_equal(h['coef'],np.zeros(2))
        np.testing.assert_array_equal(predict(heads,x,'ridge'),predict(heads,x,'constant'))
        self.assertTrue((predict(heads,x,'ridge')[:,[0,3]]==1.).all())
        with self.assertRaisesRegex(ValueError,'train-only'):fit(x,y,['a','a','b'],['train','train','validation'])
        heads[2]['target_mean']=-1;self.assertTrue((predict(heads,x,'constant')[:,2]==0).all())


if __name__=='__main__':unittest.main()
