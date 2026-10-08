"""Comparator Task 7 tests: stage scoring and failure branches.
  .venv/bin/python -m unittest qa.test_comparator_run -q
"""
import unittest

from qa.comparator_run import score_stage


def rec(**over):
    base = {"provenance_ok": True,
            "structural": [True] * 10,
            "human_hard_fail": False, "abstain_ok": True,
            "wall": [True] * 10, "certificate_fixtures_ok": True,
            "retention": {"pooled": 0.95, "equal_player": 0.94,
                          "share_supported": 0.93, "run_gate": True},
            "benign_machine": {"exact": [True, True],
                               "soft": [True, True],
                               "added_structural": False},
            "expression": {"judged": True,
                           "credits": [1, 1, 1, 1, 0]},
            "benign_judged": {"preserved": [1, 1, 1, 1, 1]},
            "consistency": {"agreements": [1] * 10},
            "evidence_honesty_ok": True}
    base.update(over)
    return base


def two_fams(a=None, b=None):
    return {"fam:a": rec(**(a or {})), "fam:b": rec(**(b or {}))}


class TestScoring(unittest.TestCase):
    def test_all_rows_pass(self):
        r = score_stage(two_fams())
        self.assertEqual(r["status"], "PASS")
        self.assertTrue(all(g["pass"] for g in r["gates"].values()))

    def test_pending_without_judgments(self):
        fams = two_fams(a={"expression": {"judged": False}},
                        b={"expression": {"judged": False}})
        r = score_stage(fams)
        self.assertEqual(r["status"], "PENDING_ADJUDICATION")
        self.assertIsNone(r["gates"]["quality_sensitivity"]["pass"])

    def test_each_row_fails_with_stage_and_dimension(self):
        breaks = {
            "provenance": {"provenance_ok": False},
            "structural_scope": {"structural": [True] * 9 + [False]},
            "retention": {"retention": {"pooled": 0.85,
                                        "equal_player": 0.94,
                                        "share_supported": 0.93,
                                        "run_gate": True}},
            "certified_negatives": {"wall": [True] * 8 + [False, False]},
            "benign_machine": {"benign_machine":
                               {"exact": [True, False], "soft": [True],
                                "added_structural": False}},
            "quality_sensitivity": {"expression":
                                    {"judged": True,
                                     "credits": [1, 1, 0, 0, 0]}},
            "benign_judge": {"benign_judged":
                             {"preserved": [1, 1, 1, 0, 0]}},
            "judge_consistency": {"consistency":
                                  {"agreements": [1] * 7 + [0] * 3}},
            "evidence_honesty": {"evidence_honesty_ok": False}}
        for dim, over in breaks.items():
            r = score_stage(two_fams(a=over), stage="development")
            self.assertEqual(r["status"], "FAILED", dim)
            self.assertEqual(r["first_failing"]["stage"], "development")
            self.assertEqual(r["first_failing"]["dimension"], dim)

    def test_human_hard_fail_breaks_structural_gate(self):
        r = score_stage(two_fams(a={"human_hard_fail": True}))
        self.assertEqual(r["first_failing"]["dimension"],
                         "structural_scope")

    def test_empty_class_is_never_a_pass(self):
        r = score_stage(two_fams(a={"wall": [],
                                    "certificate_fixtures_ok": False}))
        self.assertEqual(r["first_failing"]["dimension"],
                         "certified_negatives")
        r2 = score_stage(two_fams(a={"expression": {"judged": True,
                                                    "credits": []}}))
        self.assertEqual(r2["first_failing"]["dimension"],
                         "quality_sensitivity")

    def test_unknowns_stay_in_denominators(self):
        # 5 pairs, one INSUFFICIENT counted as 0 credit: 4/5 still passes
        # per family but pooled must use all 10
        fams = two_fams(a={"expression": {"judged": True,
                                          "credits": [1, 1, 1, 1, 0]}},
                        b={"expression": {"judged": True,
                                          "credits": [1, 1, 1, 0, 0]}})
        r = score_stage(fams)
        self.assertEqual(r["gates"]["quality_sensitivity"]["pooled"],
                         0.7)
        self.assertEqual(r["first_failing"]["dimension"],
                         "quality_sensitivity")

    def test_pooled_pass_cannot_hide_failing_family(self):
        fams = two_fams(a={"consistency": {"agreements": [1] * 10}},
                        b={"consistency": {"agreements":
                                           [1] * 7 + [0] * 3}})
        r = score_stage(fams)
        self.assertEqual(r["first_failing"]["dimension"],
                         "judge_consistency")


if __name__ == "__main__":
    unittest.main()
