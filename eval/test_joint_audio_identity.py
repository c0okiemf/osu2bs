import unittest
import numpy as np
from eval.joint_audio_identity import peak_correlation


class AudioIdentityTests(unittest.TestCase):
    def test_sub_half_second_shift_and_unequal_extents(self):
        rng=np.random.default_rng(17);a=rng.normal(size=5000)
        b=3*a[18:4700]+7
        r=peak_correlation(a,b)
        self.assertEqual(r['lag_samples'],18);self.assertAlmostEqual(r['correlation'],1.,places=12)
        self.assertEqual(r['overlap_samples'],len(b))
        self.assertLess(peak_correlation(a,rng.normal(size=4700))['correlation'],.2)

    def test_silence_constant_and_short_overlap_are_unknown(self):
        for a,b in ((np.ones(3000),np.ones(3000)),(np.zeros(3000),np.arange(3000)),(np.arange(100),np.arange(100))):
            self.assertIsNone(peak_correlation(a,b)['correlation'])


if __name__=='__main__':unittest.main()
