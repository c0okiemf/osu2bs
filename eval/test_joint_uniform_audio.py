import unittest
import numpy as np
from eval.joint_uniform_audio import rebuild


class UniformAudioTests(unittest.TestCase):
    def test_only_audio_coordinates_change_and_moment_membership_stays_train(self):
        entries=[{'family':'a','start':i*8.,'descriptor':np.r_[np.zeros(54),rate].astype(np.float32),'notes':[(.5,0,0,0,0)],'entry_gap_beats':[1.,None],'exit_gap_beats':[1.,None]} for i,rate in enumerate((1.,3.))]
        raw=np.stack([e['descriptor'] for e in entries]);base={'entries':entries,'mean':raw.mean(0),'sd':np.maximum(raw.std(0),1e-6)}
        source={'a':{'role':'train','source_bpm':120.}};features={'a':{'times':np.arange(800)/100,'x':np.arange(800*27,dtype=np.float32).reshape(800,27)}}
        bank,moments=rebuild(base,{'entries':entries[:1]},source,features)
        self.assertEqual(moments['keys'],[('a',0.),('a',8.)]);self.assertEqual(len(bank['entries']),1)
        self.assertIs(bank['entries'][0]['notes'],entries[0]['notes']);self.assertIs(bank['entries'][0]['entry_gap_beats'],entries[0]['entry_gap_beats'])
        np.testing.assert_array_equal(moments['descriptors'][:,-1],raw[:,-1])
        self.assertEqual(bank['mean'][-1],2.);self.assertEqual(bank['sd'][-1],1.)
        self.assertTrue(np.any(bank['entries'][0]['descriptor'][:54]!=0))
        source['a']['role']='validation'
        with self.assertRaises(ValueError):rebuild(base,base,source,features)


if __name__=='__main__':unittest.main()
