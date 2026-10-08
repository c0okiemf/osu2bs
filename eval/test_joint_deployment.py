import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from eval.joint_deployment import deployment_input,in_bpm,reference_view
from eval.joint_export import export_chart,qa_scene
from eval.joint_pilot import profile
import numpy as np


class DeploymentTests(unittest.TestCase):
    def test_physical_scene_and_reference_windows_survive_tempo_units(self):
        source={'bpm':120.,'duration_beats':16.,'notes':[(7.999999,0,0,0,1),(8.,1,3,1,8),(15.999999,0,1,2,0)],
                'bombs':[(1/3,2,0)],'walls':[(1/7,3/11,0,1,2,3),(8.,2.,3,1,0,5)]}
        changed=in_bpm(source,240.)
        self.assertEqual(changed['notes'][1][0],16.)
        self.assertEqual(reference_view(changed,source),in_bpm(source,120.))
        with tempfile.TemporaryDirectory() as tmp:
            a=qa_scene(export_chart(source,Path(tmp)/'a'));b=qa_scene(export_chart(changed,Path(tmp)/'b'))
            for field in ('notes','bombs','walls'):self.assertEqual(a[field],b[field])
        audio={'times':np.arange(9.),'rms':np.arange(9.),'onset':np.ones(9)}
        self.assertEqual(profile(source,audio),profile(reference_view(changed,source),audio))
        for bpm in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):in_bpm(source,bpm)
        with self.assertRaisesRegex(ValueError,'duration mismatch'):
            reference_view(changed,{**source,'duration_beats':17})

    def test_serving_uses_only_audio_mi_and_b0_rate_environment(self):
        class Grid:
            decision={'grid':'piecewise'}
            def time(self,step):return 150+100*step if step<10 else 1150+200*(step-10)
        with tempfile.TemporaryDirectory() as tmp:
            osu=Path(tmp)/'gen.osu';osu.write_text('fixture')
            raw=Path(tmp)/'b0.json'
            notes=[(1,0,0,0,1),(1,0,1,0,8),(1,1,3,0,0),(11,1,2,0,1)]
            raw.write_text(json.dumps({'notes':notes,'walls':[(9,2,0)]}))
            with patch('convert.parse_osu',return_value=({'_timing':[]},[],150.,150.)), \
                 patch('convert.grid_steps',return_value=(None,Grid())), \
                 patch('soundfile.info',return_value=SimpleNamespace(duration=10)) as audio_info:
                source,rate,proof=deployment_input(osu,osu,raw)
                self.assertEqual(rate,.3);self.assertEqual(source['bpm'],150.)
                self.assertEqual(source['notes'],[]);self.assertEqual(source['duration_beats'],25.)
                self.assertEqual(source['walls'],[(2.625,.75,0,1,0,5)])
                raw.write_text(json.dumps({'notes':[(b,h,3-c,2,8) for b,h,c,l,d in notes],'walls':[(9,2,0)]}))
                other,other_rate,_=deployment_input(osu,osu,raw)
                self.assertEqual(source,other);self.assertEqual(rate,other_rate)
                audio_info.return_value.duration=0
                with self.assertRaises(ValueError):deployment_input(osu,osu,raw)


if __name__=='__main__':unittest.main()
