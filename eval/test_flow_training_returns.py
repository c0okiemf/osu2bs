import math
import unittest
import torch
import groom
from timing import TimeGrid


class TrainingReturnsTests(unittest.TestCase):
    def test_native_bpm_clock_and_editor_annotations(self):
        v3 = TimeGrid.from_beatmap(120, {'bpmEvents':[{'b':2,'m':240}]},0,20)
        v2 = TimeGrid.from_beatmap(120, {'_events':[{'_time':2,'_type':100,'_floatValue':240}]},0,20)
        self.assertEqual(v3.times,v2.times)
        self.assertEqual(v3.time(8),1000)
        self.assertEqual(v3.time(12),1250)
        editor = TimeGrid.from_beatmap(120, {'_customData':{'_BPMChanges':[{'_time':2,'_BPM':240}]}},0,20)
        self.assertEqual(editor.time(12),1500)
        compact=TimeGrid.from_beatmap(100,{'bpmEvents':[{'m':120}]},0,20)
        self.assertEqual(compact.time(4),500)
        with self.assertRaises(ValueError):TimeGrid.from_beatmap(0,{},0,20)
        with self.assertRaises(ValueError):TimeGrid.from_beatmap(120,{'bpmEvents':[{'b':2,'m':0}]},0,20)

    def test_timing_matches_in_train_and_incremental_decode_with_mirroring(self):
        grid=TimeGrid.from_beatmap(120,{'bpmEvents':[{'b':2,'m':240}]},0,24)
        events=[(0,0,1,0,0,1),(0,1,0,3,0,1),(4,0,0,1,1,1),(12,1,1,2,1,1)]
        wall=torch.zeros(24)
        for ev in [events,groom.mirror_events(events)]:
            x,y=groom.events_to_xy(ev,wall,wall,grid=grid)
            base,targets=groom.events_to_xy(ev,wall,wall)
            self.assertTrue(torch.equal(x[:,:groom.NTOK],base))
            self.assertTrue(torch.equal(y,targets))
            previous={};last_any=None
            for i,(s,h,*_) in enumerate(ev):
                expected=groom.flow_time_features(grid,s,previous.get(h),last_any)
                self.assertTrue(torch.equal(x[i,groom.NTOK:],torch.tensor(expected)))
                previous[h]=s;last_any=s
            self.assertEqual(x[0,-2:].tolist(),[0,0])
            self.assertEqual(x[1,-2:].tolist(),[0,1])

    def test_elapsed_time_is_invariant_to_beat_encoding_and_sensitive_to_speed(self):
        a=TimeGrid.uniform(125,40,40);b=TimeGrid.uniform(62.5,40,80)
        x=groom.flow_time_features(a,8,4,6)
        y=groom.flow_time_features(b,16,8,12)
        self.assertEqual(x[:2],y[:2])  # absolute gaps, not the beat-duration channel
        self.assertNotEqual(x[2],y[2])
        fast=groom.flow_time_features(b,8,4,6)
        self.assertLess(fast[0],x[0])
        self.assertAlmostEqual(x[0],math.log1p(500/250))
        with self.assertRaises(ValueError):groom.flow_time_features(a,4,8,None)

    def test_architecture_shape_counts_and_incompatible_load(self):
        small=groom.Flow();large=groom.Flow(width=256);timed=groom.Flow(timing=True)
        self.assertEqual(sum(p.numel() for p in small.parameters()),1845890)
        self.assertEqual(sum(p.numel() for p in large.parameters()),3247554)
        self.assertEqual(timed.inp.in_features,groom.NTOK+5)
        with self.assertRaises(RuntimeError):large.load_state_dict(small.state_dict())
        with self.assertRaises(RuntimeError):timed.load_state_dict(small.state_dict())


if __name__=='__main__':unittest.main()
