import unittest
from eval.joint_continuity import controls,describe


class ContinuityTests(unittest.TestCase):
    def test_equal_profiles_do_not_hide_changed_arrow_sequence(self):
        result=controls()
        self.assertTrue(result['profiles_identical'])
        self.assertEqual(result['alternating']['bins']['fast']['same_family'],0)
        self.assertEqual(result['repeated']['bins']['fast']['same_family'],6)

    def test_ambiguity_breaks_chains_but_other_hand_does_not(self):
        source={'bpm':120,'notes':[(0,0,0,0,0),(.1,1,3,0,0),(.25,0,0,1,0),
                   (.5,0,1,1,8),(.75,0,0,0,0),(1,0,0,1,0),(1,0,1,1,1),
                   (1.25,0,0,0,1),(3.25,0,0,1,1),(3.5,0,0,0,2),(3.75,0,0,1,1)]}
        result=describe(source)
        self.assertEqual(result['excluded_groups'],{'dot_only':1,'multi_head':1})
        self.assertEqual(result['bins']['fast']['opportunities'],1)
        self.assertEqual(result['bins']['fast']['same_family'],1)
        self.assertEqual(result['fast_same_family_examples'][0]['ms'],125.)
        self.assertEqual(result['bins']['recovery']['opportunities'],1)
        self.assertEqual(result['bins']['lateral_endpoint']['opportunities'],2)


if __name__=='__main__':unittest.main()
