import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from eval.joint_phrase import _sha
from eval.joint_source_recovery import counterpart_proof,aligned_audio


class RecoveryTests(unittest.TestCase):
    def test_only_already_bound_exact_chart_and_metadata_authenticate(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);records=[]
            info={'_difficultyBeatmapSets':[{'_beatmapCharacteristicName':'Standard','_difficultyBeatmaps':[{'_difficulty':'ExpertPlus','_difficultyRank':9,'_beatmapFilename':'ExpertPlusStandard.dat','_noteJumpMovementSpeed':18}]}]}
            for folder,name in (('approved','ExpertPlus.dat'),('canonical','ExpertPlusStandard.dat')):
                p=root/folder;p.mkdir();(p/name).write_text('{}');(p/'Info.dat').write_text(json.dumps(info));sf.write(p/'song.wav',np.ones(1000),1000)
                records.append({'family':'fam:a','sources':{k:{'path':str(v),'sha256':_sha(v)} for k,v in (('chart',p/name),('info',p/'Info.dat'),('audio',p/'song.wav'))}})
            a,b=records;self.assertTrue(counterpart_proof(a,b,{})['ok'])
            # The copied alias itself remains unbound; no rename workaround.
            self.assertEqual(counterpart_proof(a,a,{})['reason'],'canonical_binding_unverified')
            cp=Path(b['sources']['chart']['path']);cp.write_text('{"changed":1}');b['sources']['chart']['sha256']=_sha(cp)
            self.assertEqual(counterpart_proof(a,b,{})['reason'],'different_chart_bytes')
            cp.write_text('{}');b['sources']['chart']['sha256']=_sha(cp)
            ip=Path(b['sources']['info']['path']);info['extra']=True;ip.write_text(json.dumps(info));b['sources']['info']['sha256']=_sha(ip)
            self.assertEqual(counterpart_proof(a,b,{})['reason'],'different_info_content')
            info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps']*=2
            for record in (a,b):
                ip=Path(record['sources']['info']['path']);ip.write_text(json.dumps(info));record['sources']['info']['sha256']=_sha(ip)
            self.assertEqual(counterpart_proof(a,b,{})['binding']['reason'],'ambiguous_info_entries')

    def test_zero_lag_audio_proof_rejects_shift_and_duration_change(self):
        rng=np.random.default_rng(31);x=rng.normal(size=3000);y=np.roll(x,18)
        meta=type('Info',(),{'duration':60.,'samplerate':1000})()
        a={'path':'a','sha256':'a'};b={'path':'b','sha256':'b'}
        with patch('eval.joint_source_recovery.sf.info',return_value=meta):
            self.assertTrue(aligned_audio(a,b,{'a':x,'b':x*3+5})['ok'])
            self.assertFalse(aligned_audio(a,b,{'a':x,'b':y})['ok'])
        other=type('Info',(),{'duration':60.1,'samplerate':1000})()
        with patch('eval.joint_source_recovery.sf.info',side_effect=[meta,other]):
            self.assertEqual(aligned_audio(a,b,{})['kind'],'duration_mismatch')


if __name__=='__main__':unittest.main()
