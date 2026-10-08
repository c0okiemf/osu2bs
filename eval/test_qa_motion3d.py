"""qa.motion3d tests (synthetic parsed replays).
  .venv/bin/python -m unittest eval.test_qa_motion3d -q
"""
import math
import unittest

from qa.motion3d import (MOTION3D_VERSION, N_SAMPLES, canonicalize,
                         swing_trajectories)


def synthetic_parsed(height=1.7, left_handed=False, n=80, dt=0.02,
                     arc=True):
    frames = []
    for i in range(n):
        t = i * dt
        ang = t * math.pi
        right = ((0.4 + (0.3 * math.cos(ang) if arc else 0.0)),
                 1.2 + (0.3 * math.sin(ang) if arc else 0.0), 0.5)
        frames.append((t, 90, ((0.0, 1.6, 0.0), (0, 0, 0, 1)),
                       ((-0.4, 1.2, 0.5), (0, 0, 0, 1)),
                       (right, (0, 0, 0, 1))))
    notes = [{"note_id": 0, "event_time": n * dt / 2, "spawn_time": 0.0,
              "event_type": 0,
              "cut": {"saberType": 1, "saberSpeed": 10.0}}]
    return {"info": {"height": height, "leftHanded": left_handed},
            "frames": frames, "notes": notes}


class TestCanonicalize(unittest.TestCase):
    def test_height_scaling(self):
        c = canonicalize(synthetic_parsed(height=2.0))
        self.assertAlmostEqual(c["height_scale"], 1.7 / 2.0)
        self.assertTrue(c["height_known"])
        # head y 1.6 scales
        self.assertAlmostEqual(c["frames"][0][1][1], 1.6 * 1.7 / 2.0,
                               places=5)

    def test_missing_height_stays_unknown(self):
        c = canonicalize(synthetic_parsed(height=0.0))
        self.assertFalse(c["height_known"])
        self.assertIsNone(c["height_scale"])
        self.assertAlmostEqual(c["frames"][0][1][1], 1.6, places=5)

    def test_mirroring_swaps_hands_and_flips_x(self):
        a = canonicalize(synthetic_parsed(left_handed=False))
        b = canonicalize(synthetic_parsed(left_handed=True))
        # canonical-right of the mirrored replay == x-flipped physical LEFT
        self.assertAlmostEqual(b["frames"][0][3][0],
                               -a["frames"][0][2][0], places=5)
        self.assertTrue(b["mirrored"])


class TestSwings(unittest.TestCase):
    def test_arc_swing_extraction(self):
        p = synthetic_parsed()
        recs, meta = swing_trajectories(p, [(0, 0)])
        self.assertEqual(meta["motion3d_version"], MOTION3D_VERSION)
        r = recs[0]
        self.assertTrue(r["supported"])
        self.assertEqual(len(r["resampled"]), N_SAMPLES)
        self.assertGreater(r["path_m"], 0.1)
        self.assertLess(r["straightness"], 0.9)     # arc, not a line
        self.assertGreater(r["vertical_extent"], 0.2)

    def test_straight_line_high_straightness(self):
        p = synthetic_parsed(arc=False)
        # replace right track with a moving straight line
        frames = []
        for i, (t, fps, head, left, right) in enumerate(p["frames"]):
            frames.append((t, fps, head, left,
                           ((0.4, 1.0 + 0.01 * i, 0.5), (0, 0, 0, 1))))
        p["frames"] = frames
        recs, _ = swing_trajectories(p, [(0, 0)])
        self.assertGreater(recs[0]["straightness"], 0.99)

    def test_thin_window_unsupported(self):
        p = synthetic_parsed(n=4)
        recs, _ = swing_trajectories(p, [(0, 0)])
        self.assertFalse(recs[0]["supported"])
        self.assertNotIn("path_m", recs[0])


if __name__ == "__main__":
    unittest.main()
