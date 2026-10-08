import unittest
import numpy as np
from eval.joint_expanded import make_banks
from eval.joint_phrase import phrase_windows


class ExpandedTests(unittest.TestCase):
    def test_old_rows_moments_are_fixed_and_validation_is_excluded(self):
        e={'family':'old','start':0.,'notes':[(.5,0,0,0,0)],'descriptor':np.arange(55,dtype=np.float32),'entry_gap_beats':[None,None],'exit_gap_beats':[1.,None]}
        base={'entries':[e],'mean':np.ones(55),'sd':np.full(55,2.),'x':np.array([(e['descriptor']-1)/2])}
        source={'notes':[(.5,0,0,0,0),(8.5,0,0,0,1),(16.5,0,0,0,0)],'bombs':[],'walls':[],'bpm':120.,'duration_beats':24.}
        source['windows']=phrase_windows(source['notes'],[],[],24.)
        d={'family':'new','fit_role':'train','source':source,'audio':{'times':np.arange(0,12,.01),'x':np.zeros((1200,27))}}
        roles={'new':{'future_role':'train'}};banks=make_banks(base,{'old'},[d],roles)
        for bank in banks.values():self.assertIs(bank['mean'],base['mean']);self.assertIs(bank['sd'],base['sd'])
        np.testing.assert_array_equal(banks['ordered']['x'][:1],base['x'])
        self.assertEqual({x['family'] for x in banks['ordered']['entries']},{'old','new'})
        self.assertEqual(banks['ordered']['entries'][1]['exit_gap_beats'][0],8.)
        roles['new']['future_role']='validation'
        with self.assertRaises(ValueError):make_banks(base,{'old'},[d],roles)


if __name__=='__main__':unittest.main()
