from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
from eval.joint_audio_view import self_rank,render_common


class AudioViewTests(unittest.TestCase):
    def test_rank_ties_are_disclosed(self):
        r=self_rank(np.array([[0.],[1.],[1.],[2.]]),np.array([1.]),2)
        self.assertEqual(r,{'distance':0.,'best_tied_rank':1,'worst_tied_rank':2,'stable_raw_rank':2})

    def test_float_render_preserves_time_extent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);t=np.arange(22050)/22050;y=np.sin(t*440*2*np.pi)
            sf.write(p/'original.wav',np.stack([y,y],axis=1),22050,subtype='FLOAT')
            r=render_common(p/'original.wav',p/'common.wav')
            self.assertEqual((r['samplerate'],r['channels']),(14800,1));self.assertLessEqual(abs(r['duration_delta_s']),1/14800+1/22050)
            self.assertEqual(sf.info(p/'common.wav').subtype,'FLOAT')


if __name__=='__main__':unittest.main()
