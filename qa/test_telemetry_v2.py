"""Task 2 tests: validated training records — window hygiene, rotation
integrity, miss-hand semantics, mirroring.
  .venv/bin/python -m unittest qa.test_telemetry_v2 -q
"""
import math
import unittest

from qa.telemetry_v2 import (TELEMETRY_VERSION, mirror_pose, slerp,
                             window_record)


def replay_fixture(n=120, dt=0.01, gap_at=None, gap_seconds=0.0,
                   quat_norm=1.0, full_turn=False, nonfinite=False):
    frames = []
    t = 0.0
    for i in range(n):
        if gap_at is not None and i == gap_at:
            t += gap_seconds
        ang = 4 * math.pi * (i / n) if full_turn else 0.0
        q = (0.0, math.sin(ang / 2) * quat_norm, 0.0,
             math.cos(ang / 2) * quat_norm)
        x = float("nan") if (nonfinite and i == n // 2) else 0.3 + 0.001 * i
        frames.append((t, 90, ((0.0, 1.6, 0.0), (0, 0, 0, 1)),
                       ((-0.3, 1.2, 0.5), (0, 0, 0, 1)),
                       ((x, 1.2, 0.5), q)))
        t += dt
    notes = [{"event_time": n * dt / 2, "event_type": 0,
              "cut": {"saberType": 1}, "line_index": 2, "line_layer": 0,
              "color": 1, "cut_direction": 1}]
    return {"info": {"height": 1.7, "leftHanded": False},
            "frames": frames, "notes": notes}


def fixture_alignment():
    return {"aligned": [(0, 0)],
            "chart": [(0.6, 2, 0, 1, 1)]}       # (t, li, ll, color, dir)


class TestWindowHygiene(unittest.TestCase):
    def test_valid_window_supported(self):
        r = window_record(replay_fixture(), fixture_alignment(), 0)
        self.assertTrue(r["supported"])
        self.assertEqual(r["telemetry_version"], TELEMETRY_VERSION)
        self.assertGreaterEqual(r["n_frames"], 16)

    def test_tracking_gap_is_not_interpolated_into_support(self):
        r = window_record(replay_fixture(gap_at=60, gap_seconds=0.08),
                          fixture_alignment(), 0)
        self.assertFalse(r["supported"])
        self.assertEqual(r["reason"], "tracking_gap")

    def test_nonfinite_pose_rejected(self):
        r = window_record(replay_fixture(nonfinite=True),
                          fixture_alignment(), 0)
        self.assertFalse(r["supported"])
        self.assertEqual(r["reason"], "nonfinite_pose")

    def test_bad_quaternion_norm_rejected(self):
        r = window_record(replay_fixture(quat_norm=0.5),
                          fixture_alignment(), 0)
        self.assertFalse(r["supported"])
        self.assertEqual(r["reason"], "quaternion_norm")

    def test_unobserved_endpoint_rejected(self):
        fx = replay_fixture(n=50)                # window end beyond capture
        fx["notes"][0]["event_time"] = 0.45
        r = window_record(fx, {"aligned": [(0, 0)],
                               "chart": [(0.45, 2, 0, 1, 1)]}, 0)
        self.assertFalse(r["supported"])
        self.assertEqual(r["reason"], "unobserved_endpoint")

    def test_too_few_frames_rejected(self):
        r = window_record(replay_fixture(n=120, dt=0.06),
                          fixture_alignment(), 0)
        # 0.8s window at ~16fps -> <16 frames within +-400ms? dt=.06 ->
        # ~13 frames in the window
        self.assertFalse(r["supported"])
        self.assertIn(r["reason"], ("too_few_frames", "tracking_gap"))


class TestRotation(unittest.TestCase):
    def test_quaternion_full_turn_does_not_disappear(self):
        r = window_record(replay_fixture(full_turn=True),
                          fixture_alignment(), 0)
        self.assertTrue(r["supported"])
        # fixture spins 4pi over 1.2s; the +-0.4s window sees ~2/3 of it
        self.assertGreater(r["targets"]["rotation_rad"], math.pi)

    def test_slerp_shortest_path(self):
        a = (0, 0, 0, 1)
        b = (0, math.sin(0.2), 0, math.cos(0.2))
        m = slerp(a, b, 0.5)
        self.assertAlmostEqual(m[1], math.sin(0.1), places=5)
        self.assertAlmostEqual(sum(x * x for x in m), 1.0, places=6)


class TestMissAndMirror(unittest.TestCase):
    def test_miss_uses_chart_hand_not_default_right(self):
        fx = replay_fixture()
        fx["notes"][0] = {"event_time": 0.6, "event_type": 2,
                          "line_index": 2, "line_layer": 0, "color": 0,
                          "cut_direction": 1}          # miss, LEFT color
        r = window_record(fx, {"aligned": [(0, 0)],
                               "chart": [(0.6, 2, 0, 0, 1)]}, 0)
        self.assertEqual(r["hand"], "left")
        self.assertEqual(r["outcome"], "miss")

    def test_mirror_pose_reflects_rotation_matrix(self):
        # a rotation about +y mirrored across x should invert its yaw sign
        q = (0.0, math.sin(0.3), 0.0, math.cos(0.3))
        pos, mq = mirror_pose((0.5, 1.0, 0.3), q)
        self.assertEqual(pos[0], -0.5)
        self.assertAlmostEqual(mq[1], -q[1], places=6)
        self.assertAlmostEqual(sum(x * x for x in mq), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
