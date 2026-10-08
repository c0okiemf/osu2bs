"""Task 2 tests: descriptive expression profiles with honest counterexamples.
  .venv/bin/python -m unittest eval.test_expression_profile -q
Fixtures pin descriptor ARITHMETIC, never an expected quality judgment.
"""
import math
import random
import unittest

from eval.expression_profile import profile, reference

# emitted tuples: (hand, col, layer, dir); times supplied separately (s)


def fixture_times(n=32, dt=0.25):
    return [i * dt for i in range(n)]


def low_vertical_stream(n=32):
    """Repeated low up-downs: alternating hands, rows 0, vertical cuts."""
    notes, times = [], []
    for i in range(n):
        notes.append((i % 2, 1 + i % 2, 0, 0 if i % 2 else 1))
        times.append(i * 0.25)
    return notes, times


def curved_centers(n=13, hand=0, mirror=False):
    """Same-hand heads along a circular arc: consistent small turns."""
    notes, times = [], []
    for i in range(n):
        ang = i * math.radians(35)
        c = round(1.5 + 1.4 * math.cos(ang), 3)
        l = round(1.0 + 0.9 * math.sin(ang), 3)
        if mirror:
            c = 3 - c
        # cut direction follows the path around the circle (varied cuts)
        notes.append((hand, c, l, (0, 5, 2, 7, 1, 4, 3, 6)[i % 8]))
        times.append(i * 0.3)
    return notes, times


def single_hand_fixture(n=16):
    return [(0, 1, 0, 1) for _ in range(n)], fixture_times(n)


def doubles_fixture(varied=True, n=12):
    """Simultaneous opposite-hand pairs; varied or one repeated token."""
    notes, times = [], []
    for i in range(n):
        t = i * 0.5
        if varied:
            d0, d1 = (0, 1, 2, 3)[i % 4], (1, 0, 3, 2)[i % 4]
            l = i % 3
        else:
            d0, d1, l = 1, 1, 0
        notes.append((0, 1, l, d0))
        times.append(t)
        notes.append((1, 2, l, d1))
        times.append(t)
    return notes, times


class TestSupport(unittest.TestCase):
    def test_no_double_opportunities_are_unknown(self):
        r = profile(*single_hand_fixture())
        self.assertFalse(r["double_vocabulary"]["supported"])
        self.assertIsNone(r["double_vocabulary"]["entropy"])

    def test_unknown_scene_does_not_look_easy(self):
        r = profile([], [], scene_status="chart_unknown")
        self.assertEqual(r["status"], "unknown")
        self.assertIsNone(r["features"])

    def test_empty_supported_chart_is_unsupported_not_perfect(self):
        r = profile([], [])
        self.assertEqual(r["status"], "unsupported")


class TestSpatialAndVertical(unittest.TestCase):
    def test_low_vertical_share_detects_boring_stream(self):
        r = profile(*low_vertical_stream())
        f = r["features"]["spatial"]
        self.assertGreater(f["low_vertical_share"], 0.99)
        self.assertEqual(f["row_occupancy"]["0"], 1.0)

    def test_curved_path_has_low_low_vertical_share(self):
        r = profile(*curved_centers())
        self.assertLess(r["features"]["spatial"]["low_vertical_share"], 0.5)

    def test_displacement_quantiles_grid_units(self):
        notes = [(0, 0, 0, 1), (0, 3, 0, 1), (0, 3, 2, 1), (0, 3, 2, 2),
                 (0, 0, 2, 3)]
        r = profile(notes, [0.0, 0.5, 1.0, 1.5, 2.0])
        d = r["features"]["spatial"]["disp"]
        self.assertAlmostEqual(d["p90"], 3.0, places=6)   # largest col jump

    def test_under_four_heads_is_unsupported(self):
        r = profile([(0, 0, 0, 1), (0, 3, 0, 1), (0, 3, 2, 1)],
                    [0.0, 0.5, 1.0])
        self.assertEqual(r["status"], "unsupported")


class TestArcs(unittest.TestCase):
    def test_curved_centers_form_arc_runs(self):
        r = profile(*curved_centers())
        a = r["features"]["arcs"]
        self.assertGreater(a["runs_per_100_heads"], 0)
        self.assertGreater(a["mean_abs_turn_deg"], 20)

    def test_mirror_preserves_unsigned_arcs(self):
        a = profile(*curved_centers())["features"]["arcs"]
        b = profile(*curved_centers(mirror=True))["features"]["arcs"]
        self.assertAlmostEqual(a["runs_per_100_heads"],
                               b["runs_per_100_heads"], places=6)
        self.assertAlmostEqual(a["mean_abs_turn_deg"],
                               b["mean_abs_turn_deg"], places=4)

    def test_straight_stream_has_no_arc_runs(self):
        notes = [(0, i % 4, 0, 1) for i in range(16)]
        r = profile(notes, fixture_times(16))
        self.assertEqual(r["features"]["arcs"]["runs_per_100_heads"], 0.0)

    def test_gap_over_one_second_breaks_runs(self):
        notes, times = curved_centers(9)
        times = times[:4] + [t + 5.0 for t in times[4:]]     # gap after 4
        few = profile(notes, times)["features"]["arcs"]["runs_per_100_heads"]
        full = profile(*curved_centers(9))["features"]["arcs"][
            "runs_per_100_heads"]
        self.assertLess(few, full)

    def test_zero_length_vectors_rejected(self):
        notes = [(0, 1, 1, 1)] * 8                     # stationary: no arcs
        r = profile(notes, fixture_times(8))
        self.assertEqual(r["features"]["arcs"]["runs_per_100_heads"], 0.0)


class TestDoubles(unittest.TestCase):
    def test_varied_vs_repeated_vocabulary(self):
        varied = profile(*doubles_fixture(varied=True))
        repeated = profile(*doubles_fixture(varied=False))
        dv, dr = (x["double_vocabulary"] for x in (varied, repeated))
        self.assertTrue(dv["supported"] and dr["supported"])
        self.assertGreater(dv["distinct_tokens"], dr["distinct_tokens"])
        self.assertGreater(dv["entropy"], dr["entropy"])
        self.assertGreater(dr["concentration"], dv["concentration"])

    def test_shuffle_can_raise_entropy_without_quality_claim(self):
        notes, times = doubles_fixture(varied=False)
        rng = random.Random(0)
        shuf = [(h, rng.randrange(4), rng.randrange(3), rng.randrange(9))
                for h, _c, _l, _d in notes]
        r = profile(shuf, times)
        self.assertNotIn("quality", r)
        self.assertNotIn("fun", r)


class TestInvariances(unittest.TestCase):
    def test_time_translation_invariant(self):
        notes, times = curved_centers()
        a = profile(notes, times)["features"]
        b = profile(notes, [t + 123.4 for t in times])["features"]
        self.assertEqual(a["arcs"], b["arcs"])
        self.assertEqual(a["spatial"], b["spatial"])

    def test_followers_counted_separately_not_as_heads(self):
        notes, times = curved_centers()
        with_dots = notes + [(0, 2, 1, 8)]
        r = profile(with_dots, times + [times[-1]])
        self.assertEqual(r["counts"]["followers"], 1)
        self.assertEqual(r["counts"]["heads"], len(notes))


class TestReference(unittest.TestCase):
    def test_family_balanced_no_quality_field(self):
        profs = {"famA": [profile(*curved_centers())],
                 "famB": [profile(*low_vertical_stream())] * 3}
        ref = reference(profs)
        self.assertEqual(ref["n_families"], 2)
        self.assertNotIn("quality", json_str(ref))
        self.assertIn("iqr", json_str(ref))


def json_str(d):
    import json
    return json.dumps(d)


if __name__ == "__main__":
    unittest.main()
