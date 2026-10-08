import unittest
import numpy as np
from eval import joint_pilot as old,joint_joins as native
from eval.joint_compatible import retrieve,audit_ledger,last_directional_groups


class CompatibleTests(unittest.TestCase):
    def fixture(self):
        entries=[]
        for i in range(33):
            entries.append({'family':str(i),'start':0.,'notes':[(.5 if i==32 else 0.,0,0,0,0),(7.75,0,0,0,1)],
                            'entry_gap_beats':[.5,None],'exit_gap_beats':[.5,None]})
        bank={'entries':entries,'mean':np.zeros(55),'sd':np.ones(55),'x':np.zeros((33,55))}
        source={'bpm':120.,'duration_beats':16.,'notes':[],'events':[],'walls':[],'bombs':[]}
        audio={'times':np.arange(0,8,.01),'x':np.zeros((800,27))}
        return source,audio,bank

    def test_rank33_compatible_donor_rescues_shortlist_failure(self):
        source,audio,bank=self.fixture()
        self.assertFalse(native.retrieve(source,audio,bank,.5)['ok'])
        result=retrieve(source,audio,bank,.5)
        self.assertTrue(result['ok']);self.assertEqual(result['donors'],[('0',0.),('32',0.)])
        self.assertEqual(result['selected_raw_ranks'],[1,33])
        a=audit_ledger(result['source'],result['donors'],bank)
        self.assertEqual((a['compressed'],a['unknown'],a['nonterminal_unknown_exits']),(0,0,0))

    def test_unknown_exit_only_allowed_at_terminal_target_window(self):
        source,audio,bank=self.fixture();bank['entries'][0]['exit_gap_beats'][0]=None
        result=retrieve(source,audio,bank,.5)
        self.assertTrue(result['ok']);self.assertEqual(result['donors'][0],('1',0.))
        source['duration_beats']=8.
        self.assertEqual(retrieve(source,audio,bank,.5)['donors'],[('0',0.)])

    def test_directional_history_compression_retains_original_decisions(self):
        previous=[(0,0,0,0,1),(.1,1,3,0,0),(.5,0,0,0,0),(.5,0,1,0,0),
                  (.75,0,2,1,8),(.75,1,3,0,2),(1,1,2,1,8)]
        compressed=last_directional_groups(previous)
        for d in range(9):
            upcoming=[(1.1,0,0,0,d),(1.1,1,3,0,d)]
            self.assertEqual(old._join_ok(previous,upcoming,120),old._join_ok(compressed,upcoming,120))
        self.assertEqual(len(compressed),3)


if __name__=='__main__':unittest.main()
