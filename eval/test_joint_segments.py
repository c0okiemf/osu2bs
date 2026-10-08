import unittest
import numpy as np
from eval.joint_segments import segment_bank,materialize,retrieve,audit_ledger
from eval import joint_pilot as old


class SegmentTests(unittest.TestCase):
    def fixture(self):
        entries=[{'family':'a','start':8.*i,'notes':[(.123456789,0,0,0,0),(7.123456789,0,1,0,1)],
                  'entry_gap_beats':[1.,None],'exit_gap_beats':[1.,None]} for i in range(12)]
        pieces={'entries':entries};audio={'times':np.arange(0,60,.01),'x':np.zeros((6000,27))}
        train=[{'family':'a','fit_role':'train','source':{'bpm':120.},'audio':audio}]
        return pieces,audio,train

    def test_full_contiguous_segments_and_exact_tail(self):
        pieces,audio,train=self.fixture();bank=segment_bank(pieces,train)
        self.assertEqual(len(bank['entries']),9)
        self.assertEqual(bank['entries'][0]['donors'],[('a',0.),('a',8.),('a',16.),('a',24.)])
        source={'bpm':120.,'duration_beats':52.25,'notes':[],'events':[],'walls':[],'bombs':[]}
        result=retrieve(source,audio,bank,pieces,.5)
        self.assertTrue(result['ok']);self.assertEqual(len(result['segments']),2);self.assertEqual(len(result['donors']),7)
        self.assertLess(max(n[0] for n in result['source']['notes']),52.25)
        audit=audit_ledger(result['source'],result,bank,pieces)
        self.assertTrue(audit['native_segments_exact']);self.assertEqual(audit['compressed'],0)
        bad=[{**train[0],'fit_role':'validation'}]
        with self.assertRaises(ValueError):segment_bank(pieces,bad)

    def test_internal_authored_direction_retained_but_external_compression_rejected(self):
        pieces,audio,train=self.fixture()
        for e in pieces['entries']:e['notes']=[(b,h,c,l,0) for b,h,c,l,d in e['notes']]
        bank=segment_bank(pieces,train);lookup={(e['family'],e['start']):e for e in pieces['entries']}
        m=materialize(bank['entries'][0],lookup,0.,32.,32.,120.,[],[None,None],[None,None])
        self.assertFalse(m['reasons'])
        self.assertFalse(old._join_ok(m['notes'][:2],m['notes'][2:4],120.))
        m=materialize(bank['entries'][0],lookup,32.,64.,64.,120.,[],[32.,None],[1.,None])
        self.assertIn('native_gap_compression',m['reasons'])


if __name__=='__main__':unittest.main()
