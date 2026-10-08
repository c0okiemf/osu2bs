"""Q1.2 tests: bounded emitted-geometry repair (V1 position-only).

Fixtures adapt motion._selfcheck's REAL Still Waiting rapid-reposition witness
(col0/top upLeft -> col2/bottom down -> col0/top up, ~2.83 grid centers) onto
the uniform test grid, embedded in a benign padded chart so global style
shares reflect a normal winner."""
import unittest

from timing import TimeGrid

import motion
from eval.quality_metrics import immutable_signature, to_ms
from quality_repair import RepairConfig, repair_winner, risk_vector

GRID = TimeGrid.uniform(125.0, 0.0, 4000)     # 120 bpm 1/4 grid
BPM, CAP = 120.0, 11.0


def _pad(start_step, n=48):
    """Benign two-hand filler: alternating hands, mixed families, 500 ms
    per-hand cadence, no flags, no pairs."""
    out = []
    dirs0 = [0, 1, 4, 1, 0, 1]     # strict back/fore alternation per hand
    dirs1 = [1, 0, 7, 0, 1, 0]
    for i in range(n):
        h = i % 2
        d = (dirs0 if h == 0 else dirs1)[(i // 2) % 6]
        c = 1 if h == 0 else 2
        out.append((start_step + 2 * i, h, c, 0 if (i // 2) % 2 else 2, d))
    return out


def witness_chart():
    """The real 3-note reposition witness at steps 4..6 + padding."""
    w = [(4, 0, 0, 2, 4), (5, 0, 2, 0, 1), (6, 0, 0, 2, 0)]
    return sorted(w + _pad(40))


class TestRepairWitness(unittest.TestCase):
    def test_witness_repaired_lower_burden_same_signature(self):
        raw = witness_chart()
        before_sig = immutable_signature(raw, [], GRID)
        before = risk_vector(raw, [], GRID)
        self.assertGreater(sum(before["flags"].values()), 0)
        r = repair_winner(raw, [], GRID, BPM, CAP)
        self.assertEqual(r.status, "repaired")
        self.assertEqual(immutable_signature(r.raw, [], GRID), before_sig)
        for e in motion.EXT_VARIANTS:
            self.assertLessEqual(r.after["flags"][e], before["flags"][e])
        self.assertLess(sum(r.after["flags"].values()),
                        sum(before["flags"].values()))
        self.assertLessEqual(r.after["opposite"], before["opposite"])
        self.assertTrue(r.witnesses)
        w = r.witnesses[0]
        self.assertIn("before", w["context"])      # preceding/following notes

    def test_slow_scaled_witness_is_byte_identical(self):
        w = [(0, 0, 0, 2, 4), (8, 0, 2, 0, 1), (16, 0, 0, 2, 0)]
        raw = sorted(w + _pad(40))
        r = repair_winner(raw, [], GRID, BPM, CAP)
        self.assertEqual(r.status, "unchanged")
        self.assertEqual(r.raw, raw)

    def test_inputs_never_mutated_and_deterministic(self):
        raw = witness_chart()
        snapshot = [tuple(n) for n in raw]
        r1 = repair_winner(raw, [], GRID, BPM, CAP)
        self.assertEqual(raw, snapshot)
        r2 = repair_winner(raw, [], GRID, BPM, CAP)
        self.assertEqual(r1.raw, r2.raw)
        self.assertEqual(r1.changed_ids, r2.changed_ids)


class TestRepairConstraints(unittest.TestCase):
    def test_pair_dodge_earns_no_credit_in_v1(self):
        # narrow <> pairs beyond the free budget, with same-hand followers:
        # a position-only layer shuffle reclassifies but cannot reduce the
        # opposite-horizontal count, so V1 must leave the chart unchanged
        pairs = []
        for k in range(5):
            s = 4 + 16 * k
            pairs += [(s, 0, 1, 0, 2), (s, 1, 2, 0, 3),
                      (s + 4, 0, 1, 0, 1), (s + 4, 1, 2, 2, 0)]
        raw = sorted(pairs + _pad(200))
        r = repair_winner(raw, [], GRID, BPM, CAP)
        self.assertEqual(r.status, "unchanged")
        self.assertEqual(r.raw, raw)

    def test_wall_and_occupied_cells_are_never_used(self):
        raw = witness_chart()
        # walls cover col 0 around the witness; another note occupies (1, 2)
        walls = [(0, 40, 0)]
        raw = sorted(raw + [(5, 1, 1, 2, 1)])
        r = repair_winner(raw, walls, GRID, BPM, CAP)
        for w in r.witnesses:
            for mv in w["moves"]:
                self.assertNotEqual(mv["to"][:2], [1, 2])
                if mv["to"][0] == 0:
                    # col 0 only legal outside the wall span — witness sits
                    # inside it, so col 0 must not appear
                    self.fail("edit used a wall-occupied column")

    def test_dot_group_moves_rigidly(self):
        # flagged head carries a dot follower: only rigid translations allowed
        w = [(4, 0, 0, 2, 4), (5, 0, 2, 0, 1), (5, 0, 2, 1, 8),
             (6, 0, 0, 2, 0)]
        raw = sorted(w + _pad(40))
        before_sig = immutable_signature(raw, [], GRID)
        r = repair_winner(raw, [], GRID, BPM, CAP)
        self.assertEqual(immutable_signature(r.raw, [], GRID), before_sig)
        if r.changed_ids:
            # find the moved head+dot and verify the relative offset survived
            heads = {i: n for i, n in enumerate(r.raw)}
            grp = [heads[i] for i in r.changed_ids
                   if heads[i][0] == 5 and heads[i][1] == 0]
            if len(grp) == 2:
                (c1, l1), (c2, l2) = (g[2:4] for g in sorted(
                    grp, key=lambda n: n[4] == 8))
                self.assertEqual((c2 - c1, l2 - l1), (0, 1))

    def test_ambiguous_multihead_group_is_pinned(self):
        # two heads on one (step, hand): every id in that group is untouchable
        w = [(4, 0, 0, 2, 4), (5, 0, 2, 0, 1), (5, 0, 3, 0, 1),
             (6, 0, 0, 2, 0)]
        raw = sorted(w + _pad(40))
        r = repair_winner(raw, [], GRID, BPM, CAP)
        amb = {i for i, n in enumerate(raw) if n[0] == 5 and n[1] == 0}
        self.assertFalse(amb & set(r.changed_ids))

    def test_edit_at_final_note(self):
        pad = _pad(4)
        last = pad[-1][0]
        w = [(last + 1, 0, 3, 0, 4), (last + 2, 0, 0, 2, 1)]
        raw = sorted(pad + w)
        r = repair_winner(raw, [], GRID, BPM, CAP)   # must not crash
        self.assertEqual(immutable_signature(r.raw, [], GRID),
                         immutable_signature(raw, [], GRID))

    def test_budget_exhaustion_returns_best_accepted(self):
        raw = witness_chart()
        cfg = RepairConfig(max_edit_frac=1.0 / 1000)      # cap = 0 edits
        r = repair_winner(raw, [], GRID, BPM, CAP, cfg)
        self.assertEqual(r.status, "unchanged")
        self.assertEqual(r.raw, raw)


def pair_chart():
    """Narrow <> pairs beyond the free budget, benign padding."""
    pairs = []
    for k in range(5):
        s = 4 + 16 * k
        pairs += [(s, 0, 1, 0, 2), (s, 1, 2, 0, 3),
                  (s + 4, 0, 1, 0, 1), (s + 4, 1, 2, 2, 0)]
    return sorted(pairs + _pad(200))


class TestPairedDirectionV2(unittest.TestCase):
    def test_v2_reduces_pairs_with_opposite_and_parity_intact(self):
        import parity
        from eval.quality_metrics import to_ms
        raw = pair_chart()
        before = risk_vector(raw, [], GRID)
        self.assertGreater(before["narrow"] + before["converging"], 0)
        sig = immutable_signature(raw, [], GRID)
        cfg = RepairConfig(variant="paired-direction")
        r = repair_winner(raw, [], GRID, BPM, CAP, cfg)
        self.assertEqual(r.status, "repaired")
        self.assertEqual(immutable_signature(r.raw, [], GRID), sig)
        self.assertLess(r.after["narrow"] + r.after["converging"],
                        before["narrow"] + before["converging"])
        self.assertLess(r.after["opposite"], before["opposite"])   # no dodge
        par = lambda rw: parity.violations(
            [(t, h, d) for t, h, _c, _l, d in to_ms(rw, GRID)])
        self.assertLessEqual(par(r.raw), par(raw))   # suffix parity intact

    def test_v1_variant_still_position_only(self):
        raw = pair_chart()
        r = repair_winner(raw, [], GRID, BPM, CAP,
                          RepairConfig(variant="position"))
        self.assertEqual(r.status, "unchanged")     # dodge rule holds


if __name__ == "__main__":
    unittest.main()
