"""Construction-known fixtures for the time-only difficulty vector (Task 2).
  .venv/bin/python -m unittest eval.test_difficulty_vector -v
"""
import json
import tempfile
import unittest
from pathlib import Path

from eval.difficulty_vector import (TOL, grouped_events, _hand_gaps_ms,
                                    measure_scene, nearest_rank)
from eval.visibility import Note


def scene(times_hands, direction=1):
    notes = []
    for k, (t, h) in enumerate(times_hands):
        col = 0 if h == 0 else 3
        notes.append(Note(f"n{k}", float(t), col, 0, h, direction))
    return {"notes": notes, "chart_unknown": False, "ignored": {}}


def add_dot(sc, i):
    n = sc["notes"][i]
    col = 1 if n.col != 1 else 2
    sc["notes"].append(Note(n.id + "d", n.hit_s, col, n.row, n.color, 8))
    return sc


def add_directional(sc, i):
    n = sc["notes"][i]
    col = 1 if n.col != 1 else 2
    sc["notes"].append(Note(n.id + "x", n.hit_s, col, n.row, n.color, 2))
    return sc


class TestGrouping(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(nearest_rank([100, 200, 300, 400], 0.10), 100)
        self.assertEqual(nearest_rank([100, 200, 300, 400], 0.50), 200)
        self.assertEqual(nearest_rank([100, 200, 300, 400], 0.90), 400)
        self.assertIsNone(nearest_rank([], 0.5))

    def test_near_coincident_two_clusters_not_transitive(self):
        g = grouped_events([Note("a", 0.0, 0, 0, 0, 1),
                            Note("b", 0.75e-6, 1, 0, 0, 1),
                            Note("c", 1.5e-6, 2, 0, 0, 1)])
        self.assertEqual(len(g["events"]), 2)   # {0,.75e-6} then {1.5e-6}

    def test_simultaneous_opposite_hands_one_instant(self):
        g = grouped_events(scene([(1.0, 0), (1.0, 1)])["notes"])
        self.assertEqual(len(g["events"]), 2)
        self.assertEqual(len(g["instants"]), 1)
        self.assertEqual(g["instants"][0]["hands"], [0, 1])


class TestMeasure(unittest.TestCase):
    def test_equal_rate_different_coordination(self):
        a = scene([(i * .25, i % 2) for i in range(8)])
        b = scene([(t, h) for t in (0, .5, 1, 1.75) for h in (0, 1)])
        va, vb = measure_scene(a), measure_scene(b)
        self.assertAlmostEqual(va["rates"]["grouped_per_s"],
                               vb["rates"]["grouped_per_s"])
        self.assertEqual(va["coincidence_share"], 0)
        self.assertEqual(vb["coincidence_share"], 1)

    def test_equal_rate_different_burst(self):
        a = scene([(t, 0) for t in (0, .5, 1, 1.5, 2, 2.5, 3, 4)])
        b = scene([(t, 0) for t in (0, .1, .2, .3, .4, .5, .6, 4)])
        va, vb = measure_scene(a), measure_scene(b)
        self.assertEqual(va["rates"]["grouped_per_s"], 2)
        self.assertEqual(vb["rates"]["grouped_per_s"], 2)
        self.assertEqual(va["burst_2s"]["combined"]["rate"], 2)
        self.assertEqual(vb["burst_2s"]["combined"]["rate"], 3.5)

    def test_dots_raw_up_grouped_unchanged(self):
        base = scene([(t, 0) for t in (0, .5, 1)])
        v0 = measure_scene(base)
        add_dot(base, 1)                       # same-hand dot at t=.5
        v1 = measure_scene(base)
        self.assertEqual(v1["counts"]["raw_colored_notes"], 4)
        self.assertEqual(v1["counts"]["dots"], 1)
        self.assertEqual(v1["counts"]["grouped_events"],
                         v0["counts"]["grouped_events"])
        self.assertEqual(v1["rates"]["grouped_per_s"], v0["rates"]["grouped_per_s"])

    def test_multi_direction_group_flagged(self):
        sc = scene([(t, 0) for t in (0, .5, 1)])
        add_directional(sc, 1)                 # second arrow same hand/time
        v = measure_scene(sc)
        self.assertEqual(v["counts"]["grouped_events"], 3)
        self.assertEqual(v["counts"]["multi_direction_groups"], 1)

    def test_all_dot_group_direction_unknown(self):
        v = measure_scene(scene([(0, 0), (.5, 0)], direction=8))
        self.assertEqual(v["counts"]["dot_only_groups"], 2)
        self.assertEqual(v["counts"]["directional_notes"], 0)

    def test_empty_and_one_instant_null_rates(self):
        self.assertEqual(measure_scene(scene([]))["status"], "empty")
        v = measure_scene(scene([(1.0, 0)]))
        self.assertIsNone(v["rates"])          # D=0

    def test_hand_gaps_quantiles(self):
        v = measure_scene(scene([(t, 0) for t in (0, .1, .3, .6, 1.0)]))
        g = v["hand_gaps_ms"]["left"]
        self.assertAlmostEqual(g["p10"], 100, places=6)
        self.assertAlmostEqual(g["p50"], 200, places=6)
        self.assertAlmostEqual(g["p90"], 400, places=6)

    def test_zero_gap_fails_metric(self):
        # two same-hand events at the same time (defensive guard, never clamp)
        events = [{"t_s": 1.0, "hand": 0, "ids": ["a"], "n_directional": 1, "n_dots": 0},
                  {"t_s": 1.0, "hand": 0, "ids": ["b"], "n_directional": 1, "n_dots": 0}]
        self.assertEqual(_hand_gaps_ms(events, 0)["failed"], "nonpositive_gap")

    def test_occupied_gaps_long_share(self):
        v = measure_scene(scene([(t, 0) for t in (0, 1, 1.5, 3)]))
        og = v["occupied_gaps"]
        self.assertEqual(og["count_ge_1s"], 2)                 # gaps 1.0 and 1.5
        self.assertAlmostEqual(og["long_gap_share"], 2.5 / 3)

    def test_exact_2s_boundary_excluded(self):
        # burst window (t-2,t] at t=2 excludes the event exactly at t=0
        v = measure_scene(scene([(t, 0) for t in (0, 2)]))
        # at t=2 the window (0,2] contains only the t=2 event -> count 1
        self.assertEqual(v["burst_2s"]["combined"]["count"], 1)

    def test_imbalance_needs_8_events(self):
        v = measure_scene(scene([(i * .1, 0) for i in range(4)]))  # <8 events
        self.assertIsNone(v["imbalance_8s"])

    def test_mirrored_doubles_zero_imbalance(self):
        v = measure_scene(scene([(t, h) for t in (0, .5, 1, 1.5) for h in (0, 1)]))
        self.assertEqual(v["imbalance_8s"]["value"], 0.0)


class TestInvariance(unittest.TestCase):
    def _times(self):
        return [(t, h) for t in (0, .25, .5, .75, 1.0, 1.25) for h in (0, 1)]

    def test_permutation_invariant(self):
        base = self._times()
        import random
        shuffled = list(base)
        random.Random(1).shuffle(shuffled)
        a = measure_scene(scene(base))["rates"]["grouped_per_s"]
        b = measure_scene(scene(shuffled))["rates"]["grouped_per_s"]
        self.assertAlmostEqual(a, b, places=9)

    def test_time_translation_invariant(self):
        base = measure_scene(scene(self._times()))
        shifted = measure_scene(scene([(t + 1000.123, h) for t, h in self._times()]))
        self.assertAlmostEqual(base["rates"]["grouped_per_s"],
                               shifted["rates"]["grouped_per_s"], places=9)
        self.assertAlmostEqual(base["occupied_gaps"]["long_gap_share"] or 0,
                               shifted["occupied_gaps"]["long_gap_share"] or 0, places=9)

    def test_mirror_swaps_hands(self):
        v = measure_scene(scene([(0, 0), (.5, 0), (.5, 1)]))
        vm = measure_scene(scene([(0, 1), (.5, 1), (.5, 0)]))
        self.assertEqual(v["counts"]["grouped_events_L"],
                         vm["counts"]["grouped_events_R"])
        self.assertEqual(v["coincidence_share"], vm["coincidence_share"])

    def test_130_260_bpm_same_seconds(self):
        # identical physical timeline encoded at 130 vs 260 BPM
        tmp = Path(tempfile.mkdtemp())
        from eval.visibility import load_scene

        def chart(name, beats_hands, bpm):
            notes = [{"_time": b, "_type": h, "_lineIndex": 0 if h == 0 else 3,
                      "_lineLayer": 0, "_cutDirection": 1} for b, h in beats_hands]
            p = tmp / name
            p.write_text(json.dumps({"_version": "2.0.0", "_notes": notes}))
            import hashlib
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            return {"id": name, "role": "reference", "chart": str(p), "sha256": sha,
                    "bpm": bpm, "origin_s": 0}
        # 130 BPM: beat = 60/130 s; 260 BPM doubles the beat numbers for same secs
        a = measure_scene(load_scene(chart("a.dat", [(0, 0), (2, 0), (4, 1)], 130.0),
                                     "standard"))
        b = measure_scene(load_scene(chart("b.dat", [(0, 0), (4, 0), (8, 1)], 260.0),
                                     "standard"))
        self.assertAlmostEqual(a["span_s"]["duration_s"], b["span_s"]["duration_s"],
                               places=8)
        self.assertAlmostEqual(a["rates"]["grouped_per_s"],
                               b["rates"]["grouped_per_s"], places=8)

    def test_unknown_scene_no_rate(self):
        v = measure_scene({"chart_unknown": True, "scope": "ambiguous_timing", "ignored": {}})
        self.assertEqual(v["status"], "unknown")
        self.assertNotIn("rates", v)


if __name__ == "__main__":
    unittest.main()
