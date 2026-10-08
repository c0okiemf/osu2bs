import unittest
from eval.joint_arc_audit import window_ledger


class ArcDotTests(unittest.TestCase):
    def test_reproduces_run_and_detects_only_same_hand_dot_bridge(self):
        notes=[(0,0,0,1),(0,1,0,0),(0,1,1,1),(0,0,1,0)];times=[0.,.25,.5,.75]
        base=window_ledger(notes,times)
        self.assertEqual((base['legacy_runs'],base['dot_bridged_runs']),(1,0))
        for hand,when,expected in ((0,.375,1),(1,.375,0),(0,.25,1),(0,.75,1),(0,1.,0)):
            merged=sorted(zip(times,notes))+[(when,(hand,2,2,8))];merged.sort()
            row=window_ledger([n for t,n in merged],[t for t,n in merged])
            self.assertEqual((row['legacy_runs'],row['dot_bridged_runs']),(1,expected))
            self.assertEqual(row['legacy_runs_per_100_heads'],25.)
            self.assertEqual(row['dot_split_runs_per_100_heads'],25.*(1-expected))


if __name__=='__main__':unittest.main()
