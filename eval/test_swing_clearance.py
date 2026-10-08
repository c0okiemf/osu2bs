import unittest

import numpy as np

from parity import DIR_VEC
from swing_clearance import blocked_approaches, repair_approaches


class SwingClearanceTests(unittest.TestCase):
    def test_all_grid_positions_against_integer_geometry_oracle(self):
        # Independent ray/disk test: endpoint overlap, or positive backward
        # projection with bounded integer cross product. No unit vectors.
        for col in range(4):
            for row in range(3):
                for other_col in range(4):
                    for other_row in range(3):
                        for direction, (vx, vy) in DIR_VEC.items():
                            if direction == 8:
                                continue
                            x, y = other_col-col, other_row-row
                            expected = (x*x+y*y <= .25 or
                                (x*vx+y*vy < 0 and 4*(x*vy-y*vx)**2 <= vx*vx+vy*vy))
                            notes = [(0, 0, col, row, direction), (0, 1, other_col, other_row, 8)]
                            self.assertEqual(bool(blocked_approaches(notes)), expected, notes)

    def test_all_cut_rotations_and_color_orders(self):
        for direction, (dx, dy) in DIR_VEC.items():
            if direction == 8:
                continue
            for hand in (0, 1):
                notes = [(4., hand, 1, 1, direction), (4., 1-hand, 1-dx, 1-dy, 8)]
                self.assertEqual(blocked_approaches(notes), [(4., 0, 1)])
                notes[1] = (4., 1-hand, 1+dx, 1+dy, 8)
                self.assertEqual(blocked_approaches(notes), [])

    def test_mixed_pair_rotations_and_extra_head_cannot_bypass(self):
        inverse = {v: k for k, v in DIR_VEC.items()}
        notes = [(0., 0, 0, 0, 2), (0., 1, 1, 0, 0)]  # <^
        for _ in range(4):
            self.assertTrue(blocked_approaches(notes))
            self.assertTrue(blocked_approaches(notes + [(0., 0, 3, 2, 8)]))
            notes = [(t, h, -y, x, inverse[(-DIR_VEC[d][1], DIR_VEC[d][0])])
                     for t, h, x, y, d in notes]

    def test_geometry_timing_and_color_matter(self):
        self.assertEqual(len(blocked_approaches([(0, 0, 0, 0, 2), (0, 1, 1, 0, 3)])), 2)
        self.assertEqual(len(blocked_approaches([(0, 0, 0, 0, 2), (0, 1, 3, 0, 3)])), 2)
        for other in [(0, 1, 1, 1, 3), (0, 0, 1, 0, 3),
                      (.001, 1, 1, 0, 3)]:
            self.assertEqual(blocked_approaches([(0, 0, 0, 0, 2), other]), [])
        self.assertEqual(blocked_approaches([(0, 0, 0, 0, 3), (0, 1, 1, 0, 2)]), [])
        self.assertEqual(blocked_approaches([(0, 0, 0, 0, 8), (0, 1, 1, 0, 8)]), [])

    def test_filter_preserves_literal_notes_features_and_normalizer(self):
        from eval.phrase_clearance import filter_bank
        entries = [dict(family='a', notes=[(0, 0, 0, 0, 2), (0, 1, 1, 0, 3)]),
                   dict(family='b', notes=[(0, 0, 0, 0, 0)])]
        bank = dict(entries=entries, x=np.arange(6).reshape(2, 3), mean=np.ones(3), sd=np.ones(3))
        clean, info = filter_bank(bank)
        self.assertIs(clean['entries'][0], entries[1])
        self.assertIs(clean['mean'], bank['mean'])
        self.assertIs(clean['sd'], bank['sd'])
        np.testing.assert_array_equal(clean['x'], bank['x'][[1]])
        self.assertEqual(info['removed_entries'], 1)
        self.assertEqual(info['lost_families'], ['a'])
        with self.assertRaisesRegex(ValueError, 'no phrases'):
            filter_bank({**bank, 'entries': entries[:1], 'x': bank['x'][:1]})

    def test_repair_moves_whole_chains_and_preserves_schedule(self):
        notes = [(0, 0, 1, 0, 2), (0, 0, 0, 0, 8),
                 (0, 1, 2, 0, 3), (0, 1, 3, 0, 8)]
        fixed, stats = repair_approaches(notes)
        self.assertFalse(blocked_approaches(fixed))
        self.assertEqual(stats['events_changed'], 1)
        self.assertEqual(stats['unresolved_events'], 0)
        self.assertEqual([(t, h, d) for t, h, _, _, d in fixed],
                         [(t, h, d) for t, h, _, _, d in notes])
        for hand in (0, 1):
            deltas = {(b[2]-a[2], b[3]-a[3]) for a, b in zip(notes, fixed) if a[1] == hand}
            self.assertEqual(len(deltas), 1)
        again, stats = repair_approaches(fixed)
        self.assertEqual(again, fixed)
        self.assertEqual(stats['events_changed'], 0)

    def test_walls_and_unrepairable_groups_are_not_silently_shortened(self):
        notes = [(0, 0, 1, 0, 2), (0, 1, 2, 0, 3)]
        fixed, stats = repair_approaches(notes, [(0, 1, 0), (0, 1, 3)])
        self.assertFalse(blocked_approaches(fixed))
        self.assertTrue(all(n[2] in (1, 2) for n in fixed))
        blocked = [(0, 0, 0, 0, 3), (0, 0, 1, 0, 8), (0, 0, 2, 0, 8),
                   (0, 0, 3, 0, 8), (0, 1, 1, 0, 2)]
        fixed, stats = repair_approaches(blocked, [(0, 1, 0)])
        self.assertEqual(fixed, blocked)
        self.assertEqual(stats['unresolved_events'], 1)

    def test_converter_checks_and_rounded_export_enforce_rule(self):
        import tempfile
        from pathlib import Path
        from convert import check, export_charts, diff_spec, GenerationError
        notes = [dict(t=0., hand=0, col=0, layer=0, dir=2),
                 dict(t=0., hand=1, col=3, layer=0, dir=3)]
        self.assertTrue(any('cut-approach' in p for p in check(notes, 120.)))
        notes[1]['t'] = .0001  # distinct times collapse to the same exported beat
        self.assertFalse(any('cut-approach' in p for p in check(notes, 120.)))
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(GenerationError, 'cut-approach'):
                export_charts({'ExpertPlus': (notes, [], diff_spec('ExpertPlus'))},
                              120., dict(Title='Test', Artist='Test'), folder)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
