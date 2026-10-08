"""E2 ContextFlow checks.  .venv/bin/python -m unittest eval.test_e2_context -q"""
import unittest

import torch
import torch.nn.functional as F

from groom import CTX, NTOK, Flow, events_to_xy
from eval.e2_context import (CTX_DIM, SCHED_DIM, ContextFlow, ContextShim,
                             opposite_head, schedule_rows)


def fixture_events():
    ev = [(0, 0, 1, 1, 0, 1), (0, 1, 1, 2, 0, 2), (4, 0, 0, 1, 1, 1),
          (8, 1, 3, 2, 1, 1), (8, 0, 2, 1, 0, 3), (12, 1, 1, 2, 0, 1)]
    return sorted(ev, key=lambda e: (e[0], e[1]))


def xy(ev):
    T = 20
    return events_to_xy(ev, [0] * T, [0] * T, torch.zeros(T))


class TestE2(unittest.TestCase):
    def test_zero_init_matches_b0_exactly(self):
        torch.manual_seed(0)
        base = Flow().eval()
        cf = ContextFlow(base).eval()
        x, y = xy(fixture_events())
        r = schedule_rows([(s, h, k - 1) for s, h, _d, _c, _l, k in
                           fixture_events()], 125.0)
        d = F.one_hot(y[:, 0], 9).float()[None]
        c = F.one_hot(y[:, 1], 7).float()[None]
        with torch.no_grad():
            a = base(x[None], d, c)
            b = cf(x[None], r[None], d, c)
        for u, v in zip(a, b):
            self.assertTrue(torch.equal(u, v))

    def test_schedule_rows_future_only(self):
        sched = [(0, 0, 0), (0, 1, 1), (4, 0, 0)]
        r = schedule_rows(sched, 125.0)
        self.assertEqual(r.shape, (3, SCHED_DIM))
        # row 0 describes event 1 (same step: gap 0, hand 1, coincident,
        # followers 1 -> 0.5), never event 0's own followers
        self.assertEqual(r[0, :5].tolist(), [0.0, 1.0, 1.0, 0.5, 1.0])
        self.assertEqual(r[2].abs().sum().item(), 0.0)   # nothing after

    def test_opposite_head_only_for_same_step_other_hand(self):
        x, _y = xy(fixture_events())
        opp = opposite_head(x)
        self.assertGreater(opp[1].abs().sum().item(), 0)   # (0,1) after (0,0)
        self.assertEqual(opp[2].abs().sum().item(), 0)     # step 4, new step
        self.assertEqual(opp[0].abs().sum().item(), 0)     # first event

    def test_shim_maps_rows_by_index(self):
        cf = ContextFlow().eval()
        rows = torch.randn(300, SCHED_DIM)
        shim = ContextShim(cf, rows)
        seen = []
        cf.hidden = lambda x, r: seen.append(r) or torch.zeros(1, x.shape[1], 1)
        for n in range(1, 4):
            shim.hidden(torch.zeros(1, n, NTOK))
        self.assertTrue(torch.equal(seen[-1][0], rows[0:3]))
        with self.assertRaises(RuntimeError):
            shim.hidden(torch.zeros(1, 2, NTOK))           # index drift

    def test_dims(self):
        self.assertEqual(CTX_DIM, SCHED_DIM + 19)


if __name__ == "__main__":
    unittest.main()
