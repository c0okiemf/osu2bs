"""Task 3 tests: conservative geometric lower bounds.
  .venv/bin/python -m unittest qa.test_physics -q
"""
import unittest

from qa.physics import (SCENARIOS, ball_set_distance, note_center, physics,
                        reach_feasible)


def scene(notes, walls=None):
    return {"scope": None, "notes": notes, "bombs": [],
            "walls": walls or [], "settings": {}}


def n(t, li, ll, color, d=1):
    return (t, li, ll, color, d)


class TestPrimitives(unittest.TestCase):
    def test_ball_set_distance(self):
        self.assertEqual(ball_set_distance((0, 0, 0), 1.0,
                                           (5, 0, 0), 1.0), 3.0)
        self.assertEqual(ball_set_distance((0, 0, 0), 2.0,
                                           (3, 0, 0), 2.0), 0.0)

    def test_note_center_pinned_convention(self):
        x, y, z = note_center(0, 0)
        self.assertAlmostEqual(x, -0.9)
        self.assertAlmostEqual(y, 0.6)
        x2, y2, _ = note_center(3, 2)
        self.assertAlmostEqual(x2, 0.9)
        self.assertAlmostEqual(y2, 1.8)

    def test_reach_feasible_for_normal_grid(self):
        for sc in SCENARIOS:
            for li in range(4):
                for ll in range(3):
                    for hand in ("left", "right"):
                        self.assertTrue(
                            reach_feasible(note_center(li, ll), sc, hand))


class TestBounds(unittest.TestCase):
    def test_rotating_stationary_grip_is_not_forced_wrist_travel(self):
        # two adjacent same-hand notes: relaxed sets overlap -> LB zero
        r = physics(scene([n(1.0, 1, 0, 1), n(1.2, 2, 0, 1)]))
        self.assertEqual(r["grip_speed"]["max_lower_bound"], 0.0)
        self.assertEqual(r["grip_speed"]["violations"], [])

    def test_one_ik_collision_is_not_a_certificate(self):
        r = physics(scene([n(1.0, 1, 0, 1), n(1.05, 2, 2, 0)]))
        self.assertFalse(r["unavoidable_collision"])

    def test_relaxation_too_loose_returns_zero_honestly(self):
        # max grid distance 1.8m << 2*relaxed radius -> bound stays 0
        r = physics(scene([n(1.0, 0, 0, 1), n(1.05, 3, 2, 1)]))
        self.assertEqual(r["grip_speed"]["max_lower_bound"], 0.0)

    def test_synthetic_teleport_fires_bound(self):
        # inject an out-of-grid synthetic note far away via direct centers
        from qa.physics import grip_speed_lower_bound
        lb = grip_speed_lower_bound((0, 1, 0), (6, 1, 0), dt=0.1)
        self.assertGreater(lb, 20.0)

    def test_close_grip_doubles_allowed(self):
        r = physics(scene([n(1.0, 1, 0, 0), n(1.0, 2, 0, 1)]))
        self.assertEqual(r["grip_speed"]["violations"], [])
        self.assertFalse(r["unavoidable_collision"])

    def test_time_translation_invariance(self):
        a = physics(scene([n(1.0, 1, 0, 1), n(1.4, 2, 1, 1)]))
        b = physics(scene([n(101.0, 1, 0, 1), n(101.4, 2, 1, 1)]))
        self.assertEqual(a["grip_speed"], b["grip_speed"])

    def test_three_height_sensitivity_reported(self):
        r = physics(scene([n(1.0, 1, 0, 1)]))
        self.assertEqual(sorted(r["scenarios"]), ["1.4", "1.7", "2.0"])


class TestWalls(unittest.TestCase):
    def test_full_width_simultaneous_walls_empty_corridor(self):
        walls = [(1.0, 0, 0, 2.0, 2), (1.0, 2, 0, 2.0, 2)]  # cols 0-3 blocked
        r = physics(scene([n(1.5, 1, 0, 1)], walls=walls))
        self.assertTrue(r["corridor"]["empty_intervals"])

    def test_partial_walls_leave_corridor(self):
        walls = [(1.0, 0, 0, 2.0, 2)]                        # cols 0-1 only
        r = physics(scene([n(1.5, 3, 0, 1)], walls=walls))
        self.assertFalse(r["corridor"]["empty_intervals"])

    def test_crouch_wall_contaminates_unknown(self):
        walls = [(1.0, 0, 1, 2.0, 4)]                        # type 1 crouch
        r = physics(scene([n(1.5, 1, 0, 1)], walls=walls))
        self.assertTrue(r["corridor"]["unknown_intervals"])


if __name__ == "__main__":
    unittest.main()
