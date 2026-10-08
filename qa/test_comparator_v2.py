"""comparator-v2 machine-only scoring checks.
  .venv/bin/python -m unittest qa.test_comparator_v2 -q
"""
import unittest

from qa.comparator_v2 import chart_verdict, score


def rec(**over):
    r = {"provenance_ok": True, "structural": [True] * 10,
         "human_hard_fail": False, "wall": [True] * 10,
         "certificates_verified": True,
         "benign": {"serialization_exact": True, "time_origin": True,
                    "mirror": True},
         "retention": {"pooled": 1.0, "equal_player": 1.0,
                       "share_supported": 0.95, "clustered_runs": 0}}
    r.update(over)
    return r


OK = {"status": "NO_CONTRADICTION_FOUND", "unknowns": []}


class TestScore(unittest.TestCase):
    def test_pass_and_quality_never_passes(self):
        r = score({"a": rec(), "b": rec()}, "development")
        self.assertEqual(r["status"], "PASS")
        self.assertIsNone(r["gates"]["quality_sensitivity"]["pass"])
        self.assertEqual(r["gates"]["judge_consistency"]["status"],
                         "QUALITY_NOT_EVALUATED")

    def test_clustered_run_is_reported_not_failed(self):
        ret = {"pooled": 1.0, "equal_player": 1.0, "share_supported": 0.99,
               "clustered_runs": 1}
        r = score({"a": rec(retention=ret)}, "development")
        self.assertEqual(r["status"], "PASS")
        self.assertEqual(r["gates"]["retention"]["support_unknown_runs"],
                         {"a": 1})

    def test_each_machine_gate_fails(self):
        cases = {"provenance": {"provenance_ok": False},
                 "structural_scope": {"human_hard_fail": True},
                 "retention": {"retention": {"pooled": 0.8,
                                             "equal_player": 1.0,
                                             "share_supported": 0.95,
                                             "clustered_runs": 0}},
                 "certified_negatives": {"wall": [True] * 8 + [False] * 2},
                 "benign_machine": {"benign": {"serialization_exact": True,
                                               "time_origin": False,
                                               "mirror": True}}}
        for dim, over in cases.items():
            r = score({"a": rec(), "b": rec(**over)}, "fresh")
            self.assertEqual(r["first_failing"]["dimension"], dim, dim)

    def test_empty_stage_fails(self):
        self.assertEqual(score({}, "fresh")["status"], "FAILED")


class TestChartVerdict(unittest.TestCase):
    def test_verdicts(self):
        sup = {"share_supported": 0.95, "run_gate": True}
        self.assertEqual(chart_verdict(OK, sup),
                         "MACHINE_PASS_QUALITY_NOT_EVALUATED")
        self.assertEqual(chart_verdict(OK, {**sup, "run_gate": False}),
                         "SUPPORT_UNKNOWN")
        self.assertEqual(chart_verdict(OK, {**sup, "share_supported": 0.8}),
                         "SUPPORT_UNKNOWN")
        self.assertEqual(chart_verdict(
            {"status": "STRUCTURAL_CONTRADICTION", "unknowns": []}, sup),
            "HARD_FAIL")
        self.assertEqual(chart_verdict(
            {"status": "MODEL_CONTRADICTION", "unknowns": []}, sup),
            "REGENERATE_MODEL_CONTRADICTION")


if __name__ == "__main__":
    unittest.main()
