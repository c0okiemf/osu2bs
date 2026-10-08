import unittest
import numpy as np
from eval.joint_approved import approved_records,subset_bank
from eval.joint_compatible import retrieve,audit_ledger
from eval import test_joint_compatible as fixtures


class ApprovedTests(unittest.TestCase):
    def test_only_actual_approved_fit_train_sources(self):
        def record(family,role,approved,path):
            return {'family':family,'fit_role':role,'approved':approved,'sources':{'chart':{'path':path}}}
        good=record('yes','train',True,'/tmp/beat-saber-map-gen/input/bytrius/map/ExpertPlus.dat')
        rows=[good,record('val','validation',True,good['sources']['chart']['path']),
              record('general','train',False,'/tmp/general/ExpertPlus.dat')]
        self.assertEqual(approved_records(rows),[good])
        for path in ('/tmp/general/ExpertPlus.dat','/tmp/beat-saber-map-gen/input/bytrius/../excluded/map/ExpertPlus.dat'):
            with self.assertRaises(ValueError):approved_records([record('bad','train',True,path)])

    def test_subset_preserves_normalization_rows_and_literal_contract(self):
        source,audio,bank=fixtures.CompatibleTests().fixture()
        bank['temporal_x']=np.arange(33*32).reshape(33,32)
        bank['temporal_mean']=np.arange(32);bank['temporal_sd']=np.ones(32)
        filtered=subset_bank(bank,{'0','32'})
        for k in ('mean','sd','temporal_mean','temporal_sd'):self.assertIs(filtered[k],bank[k])
        for k in ('x','temporal_x'):np.testing.assert_array_equal(filtered[k],bank[k][[0,32]])
        self.assertEqual(filtered['entries'],[bank['entries'][0],bank['entries'][32]])
        result=retrieve(source,audio,filtered,.5)
        self.assertTrue(result['ok'])
        audit=audit_ledger(result['source'],result['donors'],filtered)
        self.assertEqual((audit['compressed'],audit['unknown'],audit['nonterminal_unknown_exits']),(0,0,0))
        with self.assertRaises(ValueError):subset_bank(bank,{'absent'})


if __name__=='__main__':unittest.main()
