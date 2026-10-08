import unittest
from eval.joint_approved_fit import teacher,choose_snapshot,validation_summary,gap_diagnostic


class ApprovedFitTests(unittest.TestCase):
    def test_roles_fail_before_loading_or_constructing_contexts(self):
        for role in ('validation','development',None):
            with self.assertRaisesRegex(ValueError,'only train'):teacher({},None,role)

    def test_partial_output_cannot_win_by_small_error(self):
        def good(x):return {'ok':True,'errors':{a:{'mean':x} for a in ('rhythm','geometry')}}
        summaries=[validation_summary(1000,'a',[good(.01),{'ok':False,'reason':'budget'}]),
                   validation_summary(3000,'b',[good(10),good(12)]),
                   validation_summary(6000,'c',[good(10),good(12)])]
        self.assertEqual(choose_snapshot(summaries)['update'],3000)
        self.assertEqual(summaries[0]['failures'],1)
        self.assertEqual(summaries[1]['error'],22)
        self.assertEqual(choose_snapshot([{'update':0,'failures':0,'error':None},summaries[1]])['update'],3000)

    def test_literal_simultaneity_and_tiny_gaps_remain_visible(self):
        source={'bpm':120,'notes':[(0,0,0,0,0),(0,0,1,0,8),(.00001,1,3,0,0),(.00002,0,0,0,1),(1,1,3,0,1)]}
        d=gap_diagnostic(source)
        self.assertEqual(d['same_hand_intervals'],2)
        self.assertEqual(d['under_1ms'],1)
        self.assertAlmostEqual(d['minimum_ms'],.01)


if __name__=='__main__':unittest.main()
