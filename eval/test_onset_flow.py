"""Exact-source contracts and learned hand outputs, independent of old rhythm."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import convert
from onset_flow import OnsetFlow, source_onsets, onset_context, training_sequence, decode, loss, save_model, load_model
from timing import TimeGrid
from swing_clearance import blocked_approaches


class OnsetFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_dense_onsets_short_tail_and_energy(self):
        objects = [dict(t=103., end_t=103., kind='circle', x=100),
                   dict(t=127., end_t=212.5, kind='slider', x=200),
                   dict(t=127., end_t=127., kind='circle'),
                   dict(t=230., end_t=800., kind='spinner'),
                   dict(t=2300., end_t=2300., kind='circle')]
        timing = [(0., 500.), (2000., 250.)]
        a = source_onsets(objects, timing, 120.)
        self.assertEqual([e['t'] for e in a], [103., 127., 212.5, 2300.])
        self.assertEqual(len(a[1]['sources']), 2)
        self.assertEqual(a[-1]['beat_ms'], 250.)
        changed = copy.deepcopy(objects)
        for e in changed:e['x']=999;e['y']=-99;e['new_combo']=True
        self.assertEqual(a, source_onsets(changed, timing, 120.))
        self.assertGreater(onset_context(a)[0][3], onset_context(a)[-1][3])
        with self.assertRaises(ValueError):onset_context(a+a)

    def test_target_hands_not_leaked_into_current_context(self):
        events = [(0,0,1,1,0,1), (4,1,0,2,1,1), (8,0,0,1,1,1), (8,1,1,2,0,2)]
        grid = TimeGrid.uniform(125., 0., 16)
        x,y=training_sequence(events,grid)
        alternative=events[:1]+[(4,0,2,0,2,3)]+events[2:]
        z,_=training_sequence(alternative,grid)
        self.assertTrue(torch.equal(x[:2],z[:2]))
        self.assertEqual(y[:,0].tolist(),[0,1,2])
        self.assertEqual(y[0,5:].tolist(),[-100]*4)
        model=OnsetFlow(width=32)
        # Short sequences can pad to a training window without learning padding.
        xp=torch.nn.functional.pad(x,(0,0,0,2))[None]
        yp=torch.nn.functional.pad(y,(0,0,0,2),value=-100)[None]
        value=loss(model,xp,yp);self.assertTrue(torch.isfinite(value));value.backward()
        self.assertIsNotNone(model.mode_head.weight.grad)
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(x[None],y[None])[0],model(x[None],y.flip(0)[None])[0]))

    def test_learned_hand_modes_exact_export_and_schema(self):
        onsets=source_onsets([dict(t=t,end_t=t,kind='circle') for t in (103.,127.,212.5,2123.7)],[],120.)
        for mode in (0,1,2):
            torch.manual_seed(9)
            model=OnsetFlow(width=32)
            with torch.no_grad():
                model.mode_head.weight.zero_();model.mode_head.bias.fill_(-1000.);model.mode_head.bias[mode]=1000.
                model.chain_head.weight.zero_();model.chain_head.bias.copy_(torch.tensor([1000.,-1000.,-1000.]))
            notes=decode(onsets,model)
            self.assertEqual({n['t'] for n in notes},{e['t'] for e in onsets})
            self.assertEqual({n['hand'] for n in notes},{0,1} if mode==2 else {mode})
            self.assertEqual(convert.check(notes,120.,cap=float('inf')),[])
            self.assertFalse(blocked_approaches([(n['t'],n['hand'],n['col'],n['layer'],n['dir']) for n in notes]))
            with tempfile.TemporaryDirectory() as td:
                path=Path(td)/'model.pt';save_model(path,model)
                restored=load_model(path)
                self.assertEqual(notes,decode(onsets,restored))
                convert.export_charts({'ExpertPlus':(notes,[],convert.diff_spec('ExpertPlus'))},120.,dict(Title='test',Artist='test'),Path(td)/'map')
                import json
                emitted=json.loads((Path(td)/'map/ExpertPlus.dat').read_text())['_notes']
                self.assertEqual(len(notes),len(emitted))
                for n,e in zip(notes,emitted):self.assertLessEqual(abs(n['t']-e['_time']*500.),.00251)
                torch.save(dict(schema='wrong'),path)
                with self.assertRaises(ValueError):load_model(path)


    def test_exact_route_bypasses_old_pipeline_and_audio_edits(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);osu=root/'source.osu';checkpoint=root/'model.pt'
            osu.write_text("[General]\nAudioFilename:song.egg\n[Metadata]\nTitle:Test\nArtist:Test\n"
                           "[TimingPoints]\n0,500,4,2,1,100,1,0\n"
                           "[HitObjects]\n64,64,103,1,0,0:0:0:0:\n320,64,212,1,0,0:0:0:0:\n")
            torch.manual_seed(3);model=OnsetFlow(width=32)
            with torch.no_grad():
                model.mode_head.weight.zero_();model.mode_head.bias.copy_(torch.tensor([1000.,-1000.,-1000.]))
                model.chain_head.weight.zero_();model.chain_head.bias.copy_(torch.tensor([1000.,-1000.,-1000.]))
            save_model(checkpoint,model)
            captured={}
            def write(maps,bpm,meta,audio,out,shift_ms=0,end_ms=None):
                captured.update(maps=maps,shift=shift_ms,end=end_ms)
                convert.export_charts(maps,bpm,meta,out)
            with patch.object(convert,'grid_steps',side_effect=AssertionError('snapping')),                 patch.object(convert,'convert_groomed',side_effect=AssertionError('rhythm rewrite')),                 patch('groom.cached_audio_features',side_effect=AssertionError('audio energy')),                 patch.object(convert,'write_map',side_effect=write):
                self.assertEqual(convert.main(osu,root/'song.egg',root/'out',onset_checkpoint=checkpoint),0)
            self.assertEqual(captured['shift'],0);self.assertIsNone(captured['end'])
            self.assertEqual({n['t'] for n in captured['maps']['ExpertPlus'][0]},{103.,212.})
            with self.assertRaises(convert.GenerationError):
                convert.main(osu,root/'song.egg',root/'out','Easy,ExpertPlus',onset_checkpoint=checkpoint)
            with self.assertRaises(convert.GenerationError):
                convert.main(osu,root/'song.egg',root/'out',onset_checkpoint=root/'missing.pt')


if __name__=='__main__':unittest.main()
