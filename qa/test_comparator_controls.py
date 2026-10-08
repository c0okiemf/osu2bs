"""Comparator Task 5 tests: frozen controls + independent expectations.
  .venv/bin/python -m unittest qa.test_comparator_controls -q
"""
import unittest

from qa.certificates import contradictions, verify_certificate
from qa.comparator_controls import (collapse_window, eligible_windows,
                                    gain_scene_audio, make_cases,
                                    mirror_pair, per_hand_spans,
                                    roundtrip_scene, score_case,
                                    time_origin_shift)

WARN = {"speed_warning": None}


def rich_scene(seconds=60, step=0.4):
    notes = []
    i = 0
    t = 1.0
    while t < seconds:
        notes.append((round(t, 3), i % 4, i % 3, i % 2, (i * 3) % 8))
        t += step
        i += 1
    return {"scope": None, "chart_sha256": "aa" * 32,
            "info_sha256": "bb" * 32, "notes": notes, "bombs": [],
            "walls": [], "settings": {"njs": 18}}


def audio_fixture(dur=70):
    return {"supported": True, "duration_s": dur,
            "frames": {"times": [i * 0.1 for i in range(dur * 10)]},
            "beat_times": [float(i) for i in range(dur)],
            "per_second": {"intensity_rms": [0.2] * dur,
                           "harmonic_rms": [0.3] * dur,
                           "percussive_rms": [0.1] * dur}}


CONTRACT = {"label": "comparator-v1"}


class TestCollapse(unittest.TestCase):
    def test_contraction_in_summed_per_hand_spans(self):
        sc = rich_scene()
        core = eligible_windows(sc, "s")[0]
        src = per_hand_spans(sc, core)
        self.assertGreater(src["h_sum"], 0)
        self.assertGreater(src["v_sum"], 0)
        mut, chk = collapse_window(sc, core)
        self.assertTrue(chk["valid"])
        self.assertEqual(chk["mut_spans"], {"h_sum": 0.0, "v_sum": 0.0})
        # audio-free invariants: timestamps, hands, directions preserved
        a = [(t, c, d) for t, li, ll, c, d in sc["notes"]]
        b = [(t, c, d) for t, li, ll, c, d in mut["notes"]]
        self.assertEqual(a, b)

    def test_combined_two_hand_span_cannot_contract(self):
        sc = rich_scene()
        core = eligible_windows(sc, "s")[0]
        mut, _ = collapse_window(sc, core)
        cols = [li for t, li, ll, c, d in mut["notes"]
                if core[0] <= t < core[1]]
        self.assertEqual(max(cols) - min(cols), 3)   # 0 and 3 both used

    def test_duplicate_same_hand_occupancy_rejects_construction(self):
        sc = rich_scene()
        core = eligible_windows(sc, "s")[0]
        # two same-hand notes at one timestamp inside the core
        t0 = core[0] + 1.0
        sc["notes"] += [(t0, 1, 1, 0, 1), (t0, 2, 2, 0, 1)]
        sc["notes"].sort()
        mut, chk = collapse_window(sc, core)
        self.assertIsNone(mut)
        self.assertEqual(chk["reason"], "duplicate_same_hand_occupancy")


class TestEligibility(unittest.TestCase):
    def test_hash_ranked_nonoverlapping_cap(self):
        wins = eligible_windows(rich_scene(seconds=600), "s")
        self.assertLessEqual(len(wins), 40)
        for i, a in enumerate(wins):
            for b in wins[i + 1:]:
                self.assertLessEqual(min(a[1], b[1]) - max(a[0], b[0]), 0)

    def test_short_family_is_insufficient(self):
        short = rich_scene(seconds=4)
        r = make_cases("fam:x", short, audio_fixture(), CONTRACT)
        self.assertEqual(r["status"], "INSUFFICIENT")


class TestNegativesAndCases(unittest.TestCase):
    def test_full_family_manifest(self):
        r = make_cases("fam:x", rich_scene(seconds=400), audio_fixture(),
                       CONTRACT)
        self.assertEqual(r["status"], "OK")
        self.assertEqual(len(r["structural"]), 10)
        self.assertEqual(len(r["wall"]), 10)
        self.assertEqual(len(r["expression"]), 5)
        self.assertEqual(len(r["benign"]), 5)
        # structural negatives actually detected by the typed checker
        for c in r["structural"]:
            m = contradictions(c["scene"], WARN)
            self.assertEqual(score_case(c["expected"], m,
                                        [])["detected"], True)
        # wall negatives all independently certified pre-use
        for c in r["wall"]:
            self.assertTrue(verify_certificate(c["scene"],
                                               c["certificate"])["valid"])
            m = contradictions(c["scene"], WARN)
            self.assertTrue(score_case(c["expected"], m, [])["detected"])

    def test_wall_layouts_alternate(self):
        r = make_cases("fam:x", rich_scene(seconds=400), audio_fixture(),
                       CONTRACT)
        n_walls = [len(c["scene"]["walls"]) for c in r["wall"]]
        self.assertIn(1, n_walls)
        self.assertIn(2, n_walls)


class TestTransforms(unittest.TestCase):
    def test_serialization_roundtrip_machine_equal(self):
        sc = rich_scene()
        self.assertEqual(contradictions(sc, WARN),
                         contradictions(roundtrip_scene(sc), WARN))

    def test_time_origin_shift_moves_chart_and_audio_together(self):
        sc, au = rich_scene(), audio_fixture()
        sc2, au2 = time_origin_shift(sc, au, 2.0)
        self.assertAlmostEqual(sc2["notes"][0][0], sc["notes"][0][0] + 2)
        self.assertEqual(len(au2["per_second"]["intensity_rms"]),
                         len(au["per_second"]["intensity_rms"]) + 2)
        self.assertAlmostEqual(au2["beat_times"][0],
                               au["beat_times"][0] + 2)

    def test_gain_scales_audio_evidence(self):
        au = gain_scene_audio(audio_fixture(), -6.0)
        self.assertAlmostEqual(au["per_second"]["intensity_rms"][0],
                               0.2 * 10 ** (-0.3), places=6)

    def test_mirror_pairs_scene_and_reference(self):
        sc = rich_scene()
        ref = {"rel_path_24": [[0.5, 1.0, 0.2]]}
        m_sc, m_ref = mirror_pair(sc, ref)
        self.assertEqual(m_sc["notes"][0][1], 3 - sc["notes"][0][1])
        self.assertEqual(m_ref["rel_path_24"][0][0], -0.5)


class TestScoring(unittest.TestCase):
    def test_wall_detection_needs_reason_and_overlap(self):
        exp = {"kind": "wall_negative", "status": "MODEL_CONTRADICTION",
               "interval_s": [10.0, 12.0]}
        missing_audio = {"status": "UNKNOWN", "model": [],
                         "unknowns": [{"type":
                                       "speed_evidence_unknown"}]}
        self.assertFalse(score_case(exp, missing_audio, [])["detected"])
        wrong_interval = {"status": "MODEL_CONTRADICTION",
                          "model": [{"interval_s": [50.0, 51.0]}]}
        self.assertFalse(score_case(exp, wrong_interval,
                                    [])["detected"])
        right = {"status": "MODEL_CONTRADICTION",
                 "model": [{"interval_s": [11.0, 12.5]}]}
        self.assertTrue(score_case(exp, right, [])["detected"])

    def test_generic_more_motion_earns_zero(self):
        exp = {"kind": "expression_pair", "prefer": "source",
               "localized_to": [10.0, 18.0]}
        generic = {"pair_verdict": "LEFT_BETTER", "preferred": "source",
                   "claims": [{"text": "more movement is better"}]}
        self.assertEqual(score_case(exp, {}, [generic])["credit"], 0)
        localized = {"pair_verdict": "LEFT_BETTER",
                     "preferred": "source",
                     "claims": [{"interval_s": [11.0, 13.0],
                                 "axis": "vocabulary"}]}
        self.assertEqual(score_case(exp, {}, [localized])["credit"], 1)

    def test_benign_tie_without_invented_defect(self):
        exp = {"kind": "benign_pair", "pair": "TIE"}
        tie = {"pair_verdict": "TIE", "claims": []}
        self.assertTrue(score_case(exp, {}, [tie])["preserved"])
        invented = {"pair_verdict": "TIE",
                    "claims": [{"change_specific_defect": True}]}
        self.assertFalse(score_case(exp, {}, [invented])["preserved"])
        self.assertFalse(score_case(exp, {}, [])["preserved"])


if __name__ == "__main__":
    unittest.main()
