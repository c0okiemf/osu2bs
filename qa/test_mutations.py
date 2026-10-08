"""Task 6 tests: red-team fixtures and controls.
  .venv/bin/python -m unittest qa.test_mutations -q
"""
import unittest

from qa.mutations import RECIPES, mutate, select_windows


def chart_fixture(n=64, varied=True):
    notes = []
    for i in range(n):
        d = (0, 1, 2, 3, 4, 5)[i % 6] if varied else 1
        li = (0, 1, 2, 3)[i % 4] if varied else 1
        notes.append({"_time": 2.0 + i * 0.5, "_type": i % 2,
                      "_lineIndex": li, "_lineLayer": i % 3 if varied else 0,
                      "_cutDirection": d})
    return {"_version": "2.0.0", "_notes": notes, "_obstacles": []}


class TestTaxonomy(unittest.TestCase):
    def test_cosmetic_shift_is_not_automatically_bad(self):
        m = mutate(chart_fixture(), None, "cosmetic_shift", 0)
        self.assertNotIn(m["expected"], {"MUST_HARD_FAIL", "MUST_NOT_PASS"})
        self.assertEqual(m["expected"], "CHALLENGE")

    def test_speed_mutant_needs_certificate(self):
        m = mutate(chart_fixture(), None, "compress_recovery", 0)
        if not m.get("certificate"):
            self.assertEqual(m["expected"], "CHALLENGE")

    def test_structural_invalid_is_hard_fail(self):
        m = mutate(chart_fixture(), None, "nonfinite_time", 0)
        self.assertEqual(m["expected"], "MUST_HARD_FAIL")

    def test_duplicate_occupancy_is_hard_fail(self):
        m = mutate(chart_fixture(), None, "duplicate_occupancy", 0)
        self.assertEqual(m["expected"], "MUST_HARD_FAIL")

    def test_wall_corridor_negative_is_certified(self):
        m = mutate(chart_fixture(), None, "empty_corridor", 0)
        self.assertEqual(m["expected"], "MUST_NOT_PASS")
        self.assertTrue(m["certificate"])
        self.assertIn("empty_intervals", m["certificate"])

    def test_preserve_class_time_translation(self):
        m = mutate(chart_fixture(), None, "time_translate", 0)
        self.assertEqual(m["expected"], "MUST_PRESERVE")
        self.assertEqual(len(m["chart"]["_notes"]),
                         len(chart_fixture()["_notes"]))


class TestLocality(unittest.TestCase):
    def test_mechanical_collapse_changes_only_target_window(self):
        src = chart_fixture(varied=True)
        m = mutate(src, None, "mechanical_collapse", 0)
        if m["expected"] == "SKIP":
            self.skipTest(m["rationale"])
        a, b = src["_notes"], m["chart"]["_notes"]
        w0, w1 = m["interval_beats"]
        outside_same = [x for x in a if not (w0 <= x["_time"] < w1)] == \
            [x for x in b if not (w0 <= x["_time"] < w1)]
        self.assertTrue(outside_same)
        inside = [x for x in b if w0 <= x["_time"] < w1]
        motifs = {(x["_lineIndex"], x["_lineLayer"], x["_cutDirection"])
                  for x in inside}
        self.assertLessEqual(len(motifs), 2)      # collapsed to one motif

    def test_diagonal_substitution_only_touches_directions(self):
        src = chart_fixture()
        m = mutate(src, None, "diagonal_substitution", 0)
        for a, b in zip(src["_notes"], m["chart"]["_notes"]):
            self.assertEqual(a["_time"], b["_time"])
            self.assertEqual(a["_lineIndex"], b["_lineIndex"])


class TestSelectors(unittest.TestCase):
    def test_six_windows_deterministic_nonoverlapping(self):
        src = chart_fixture(n=200)
        a = select_windows(src, seed="s1")
        b = select_windows(src, seed="s1")
        self.assertEqual(a, b)
        self.assertLessEqual(len(a), 6)
        for (s1, e1), (s2, e2) in zip(a, a[1:]):
            self.assertLessEqual(e1, s2)

    def test_all_recipes_declare_class(self):
        for name, rec in RECIPES.items():
            self.assertIn(rec["class"],
                          {"MUST_HARD_FAIL", "MUST_NOT_PASS", "CHALLENGE",
                           "MUST_PRESERVE"}, name)


if __name__ == "__main__":
    unittest.main()
