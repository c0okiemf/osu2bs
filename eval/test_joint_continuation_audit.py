import unittest
import numpy as np
from eval import test_joint_compatible as fixtures
from eval.joint_continuation_audit import first_six,eligibility


class ContinuationTests(unittest.TestCase):
    def test_native_continuation_is_in_compatible_six_and_context_is_preserved(self):
        source,audio,bank=fixtures.CompatibleTests().fixture()
        prior=('0',0.);previous=[(7.75,0,0,0,1)];last=[7.75,None];exits=[.5,None]
        bank['entries'][32].update(family='0',start=8.)
        choices=first_six(np.arange(33),bank,prior,8.,16.,16.,120.,previous,last,exits)
        self.assertEqual(choices,[32])
        notes,reasons=eligibility(bank['entries'][32],8.,16.,16.,120.,previous,last,exits)
        self.assertEqual(reasons,[]);self.assertEqual(notes[0][0],8.5)
        bank['entries'][32]['entry_gap_beats'][0]=None
        self.assertIn('native_context_unknown',eligibility(bank['entries'][32],8.,16.,16.,120.,previous,last,exits)[1])


if __name__=='__main__':unittest.main()
