"""Action representation and model input/output contract checks."""
import copy
import unittest

import numpy as np
import torch

from eval.joint_model import (AUDIO_DIM, CONTEXT_DIM, EMPTY, N_GAP, JointModel,
                              action_notes, actions, context, decode_gap, encode_gap,
                              losses)
from eval.joint_phrase import encode_events


class JointModelTests(unittest.TestCase):
    def test_actions_reconstruct_boundary_stacks_tiny_gaps_and_long_rests(self):
        notes = [(0, 0, 0, 0, 1), (0, 0, 1, 0, 1), (1/3, 1, 3, 2, 8),
                 (8, 0, 0, 2, 0), (8 + 2.384185791015625e-7, 1, 3, 0, 1),
                 (64, 1, 3, 2, 0)]
        rows = actions({"events": encode_events(notes), "duration_beats": 65})
        self.assertEqual(action_notes(rows), notes)
        rests = [r for r in rows if r["gap"] == 0]
        self.assertEqual([r["target_beat"] for r in rests], [8,16,24,32,40,48,56,64,65])
        boundary_note = next(r for r in rows if r["gap"] and r["target_beat"] == 8)
        self.assertEqual(boundary_note["gap"], 1)
        self.assertEqual(boundary_note["prev_kind"], 0)
        self.assertNotEqual(rests[-1]["prev_slots"], [EMPTY]*6)
        for gap in (0,1/3,2.384185791015625e-7,67.79841613769531):
            self.assertAlmostEqual(decode_gap(*encode_gap(gap)), gap, places=10)

    def test_shared_context_has_audio_lookahead_but_no_note_targets(self):
        audio = {"times": np.arange(20.), "x": np.ones((20,AUDIO_DIM))}
        a = context(audio, 0, 120, 5, 18, [(0,1,0,1,0,5)], [(0,3,2)])
        self.assertEqual(a.shape, (CONTEXT_DIM,))
        self.assertTrue(np.all(a[:8*2*AUDIO_DIM] == 0))
        self.assertEqual(a[-12:].tolist(), [1,1,1,0,0,0,0,0,0,0,0,1])
        b = context(audio, 2, 120, 5, 18, [(0,1,0,1,0,5)], [(0,3,2)])
        self.assertEqual(b[-12:].sum(), 0)

    def test_conditional_slot_does_not_see_current_or_future_target(self):
        torch.manual_seed(5)
        model = JointModel().eval()
        hidden = torch.randn(1,2,128)
        category, count = torch.ones(1,2,dtype=torch.long), torch.full((1,2),15)
        slots = torch.randint(108,(1,2,6))
        before = model.slot_logits(hidden, category, count, slots)
        changed = slots.clone()
        changed[...,3:] = (changed[...,3:] + 1) % 108
        after = model.slot_logits(hidden, category, count, changed)
        self.assertTrue(torch.equal(before[...,:4,:], after[...,:4,:]))
        self.assertFalse(torch.equal(before[...,4:,:], after[...,4:,:]))

    def test_batch_loss_backward_and_rest_only(self):
        torch.set_num_threads(2)
        model = JointModel()
        batch = {"context": torch.zeros(2,8,CONTEXT_DIM),
                 "prev": torch.full((2,8,6),EMPTY), "kind": torch.zeros(2,8,dtype=torch.long),
                 "prev_gap": torch.ones(2,8), "gap": torch.zeros(2,8,dtype=torch.long),
                 "residual": torch.zeros(2,8), "count": torch.zeros(2,8,dtype=torch.long),
                 "slots": torch.full((2,8,6),EMPTY)}
        out, state = model(batch)
        self.assertEqual(out["slots"].shape, (2,8,6,108))
        self.assertEqual(state.shape, (2,2,128))
        loss = losses(out,batch,warmup=2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.gap.weight.grad)
        batch["gap"][:,4:] = 5
        batch["count"][:,4:] = 4
        batch["slots"][:,4:,0] = 2
        out,_ = model(batch)
        loss = losses(out,batch,warmup=2)
        loss.backward()
        self.assertIsNotNone(model.slot[-1].weight.grad)

    def test_whole_and_chunked_recurrence_match(self):
        torch.manual_seed(7)
        model=JointModel().eval()
        ctx=torch.randn(1,10,CONTEXT_DIM)
        prev=torch.randint(109,(1,10,6))
        kind=torch.randint(3,(1,10))
        gap=torch.rand(1,10)
        whole, state=model.hidden(ctx,prev,kind,gap)
        first, s=model.hidden(ctx[:,:4],prev[:,:4],kind[:,:4],gap[:,:4])
        second, s=model.hidden(ctx[:,4:],prev[:,4:],kind[:,4:],gap[:,4:],s)
        self.assertTrue(torch.allclose(whole,torch.cat([first,second],1),atol=1e-6))
        self.assertTrue(torch.allclose(state,s,atol=1e-6))


if __name__ == '__main__':
    unittest.main()
