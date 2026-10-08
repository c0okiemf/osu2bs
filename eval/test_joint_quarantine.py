import unittest
import numpy as np
from eval.joint_quarantine import quarantine_bank


class QuarantineTests(unittest.TestCase):
    def test_excluded_source_cannot_influence_refitted_similarity(self):
        entries=[{'family':f,'descriptor':np.full(55,x,dtype=np.float32),'notes':[(0.,0,0,0,0)],'entry_gap_beats':[.5,None],'exit_gap_beats':[.5,None]} for f,x in (('a',1),('excluded',1000),('b',3))]
        base={'entries':entries,'mean':np.full(55,100),'sd':np.ones(55),'x':np.zeros((3,55)),'temporal_x':np.ones((3,32))}
        result=quarantine_bank(base,{'excluded'})
        self.assertEqual([e['family'] for e in result['entries']],['a','b'])
        self.assertIs(result['entries'][0],entries[0])
        np.testing.assert_array_equal(result['mean'],np.full(55,2))
        np.testing.assert_array_equal(result['sd'],np.ones(55))
        np.testing.assert_array_equal(result['x'],np.array([[-1]*55,[1]*55]))
        self.assertEqual(set(result),{'entries','mean','sd','x'})
        with self.assertRaises(ValueError):quarantine_bank(base,{'absent'})
        with self.assertRaises(ValueError):quarantine_bank(base,{'a','b','excluded'})


if __name__=='__main__':unittest.main()
