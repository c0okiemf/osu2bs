import copy
import unittest
import numpy as np
from eval import joint_pilot as old
from eval.joint_workload import corrected_bank
from eval.joint_joins import enrich_bank,join_checks,advance,retrieve,audit_ledger
from eval.joint_phrase import encode_events


class NativeJoinTests(unittest.TestCase):
    def test_traced_compression_rejected_without_a_global_minimum(self):
        previous=[(7.998138427734375,0,2,0,1)];incoming=[(8.,0,1,2,4)]
        self.assertTrue(old._join_ok(previous,incoming,140))
        entry={'entry_gap_beats':[1.,None],'exit_gap_beats':[.5,None]}
        checks=join_checks([previous[0][0],None],[.5,None],incoming,entry)
        self.assertEqual(checks[0]['status'],'native_gap_compression')
        self.assertAlmostEqual(checks[0]['actual_gap_beats']*60000/140,.7978166852678571)
        self.assertEqual(join_checks([7.5,None],[.5,None],incoming,entry)[0]['status'],'pass')
        self.assertEqual(join_checks([None,None],[None,None],incoming,entry),[])
        self.assertEqual(join_checks([7.5,None],[None,None],incoming,entry)[0]['status'],'native_context_unknown')
        last=[7.5,None];exits=[.5,None];advance(last,exits,[],entry)
        self.assertEqual((last,exits),([7.5,None],[.5,None]))
        self.assertEqual(join_checks(last,exits,[(16.,0,1,2,4)],entry)[0]['status'],'pass')

    def test_literal_generation_and_independent_ledger_audit(self):
        notes=[(b,h,c,0,d) for b,d in ((0.,1),(.5,0),(7.5,1),(8.,0),(8.5,1),(15.5,0),(16.,1)) for h,c in ((0,0),(1,3))]
        source={'bpm':120.,'duration_beats':17.,'notes':notes,'bombs':[],'walls':[],
                'windows':[{'start':s,'end':s+8,'events':encode_events([n for n in notes if s<=n[0]<s+8])} for s in (0.,8.)]}
        audio={'times':np.arange(0,9,.01),'x':np.zeros((900,27))}
        base=corrected_bank(old.retrieval_bank([{'family':'a','fit_role':'train','source':source,'audio':audio}]))
        bank=enrich_bank(base,{'a':source});template={**source,'duration_beats':16.,'notes':[]}
        attempt=retrieve(template,audio,bank,1.,0);self.assertTrue(attempt['ok'])
        audit=audit_ledger(attempt['source'],attempt['donors'],bank)
        self.assertEqual(audit['checked_joins'],2);self.assertEqual(audit['compressed'],0);self.assertEqual(audit['unknown'],0)
        self.assertEqual(attempt['source']['notes'],notes[:-2])
        changed=copy.deepcopy(attempt['source']);changed['notes'][0]=(0.,0,2,0,1)
        with self.assertRaisesRegex(ValueError,'literal donor'):audit_ledger(changed,attempt['donors'],bank)
        failure=retrieve(template,audio,bank,1.,1)
        self.assertFalse(failure['ok']);self.assertEqual(failure['reason'],'no_native_supported_join')


if __name__=='__main__':unittest.main()
