"""Blind pair staging/ingest/scoring + calibration scoring checks.
  .venv/bin/python -m unittest qa.test_comparator_pairs -q
"""
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from qa import comparator_calib as cc
from qa import comparator_pairs as cp


def _custodian(root):
    cases = {
        "p-e": {"family": "fam:a", "kind": "expr", "core_s": [10.0, 18.0],
                "source_side_primary": "left",
                "expected": {"prefer": "source"}},
        "p-b": {"family": "fam:a", "kind": "benign", "core_s": [30.0, 38.0],
                "source_side_primary": "right",
                "expected": {"pair": "TIE"}}}
    p = Path(root) / "cust.json"
    p.write_text(json.dumps({"cases": cases, "completed": True}))
    return p


def _pair(cid, verdict, claims=()):
    return {"case_id": cid, "pair_verdict": verdict, "claims": list(claims)}


LOCAL = {"interval_s": [11.0, 13.0], "axis": "vocabulary",
         "text": "range removed", "evidence_ref": "left"}


class TestPairs(unittest.TestCase):
    def _patched(self, td):
        return unittest.mock.patch.multiple(
            cp, JUDGE_DIR=Path(td) / "j", CUSTODIAN_P=_custodian(td),
            OUT_ROOT=Path(td),
            _identity=lambda o: {"orientation": o})

    def test_side_assignment_is_deterministic(self):
        self.assertEqual(cp._source_left("p-x"), cp._source_left("p-x"))

    def test_ingest_rejects_bad_verdict_and_bad_interval(self):
        with tempfile.TemporaryDirectory() as td, self._patched(td):
            with self.assertRaises(ValueError):
                cp.ingest_response(json.dumps([_pair("p-e", "MAYBE")]),
                                   ["p-e"], "primary")
            bad = dict(LOCAL, interval_s=[13.0, 11.0])
            with self.assertRaises(ValueError):
                cp.ingest_response(
                    json.dumps([_pair("p-e", "LEFT_BETTER", [bad])]),
                    ["p-e"], "primary")

    def test_first_valid_is_final_and_scoring(self):
        with tempfile.TemporaryDirectory() as td, self._patched(td):
            cp.ingest_response(json.dumps(
                [_pair("p-e", "LEFT_BETTER", [LOCAL]), _pair("p-b", "TIE")]),
                ["p-e", "p-b"], "primary")
            # a later contradicting reply for the same case is ignored
            cp.ingest_response(json.dumps([_pair("p-e", "TIE")]),
                               ["p-e"], "primary")
            # reversed: source now on the right -> RIGHT_BETTER agrees
            cp.ingest_response(json.dumps([_pair("p-e", "RIGHT_BETTER")]),
                               ["p-e"], "reversed")
            r = cp.score_pairs()["per_family"]["fam:a"]
            self.assertEqual(r["expr_credits"], [1])
            self.assertEqual(r["benign_preserved"], [1])
            self.assertEqual(r["consistency"], [1])


def _calib_verdict(side, pref, claims=()):
    return {"verdict": {"observed_loss": {"side": side,
                                          "claims": list(claims)},
                        "overall_preference": pref}}


class TestCalibScore(unittest.TestCase):
    def test_recognition_preference_and_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "plan.json").write_text(json.dumps(
                {"judged": ["p-e", "p-b"], "short_circuit": []}))
            j = root / "judgments" / "main"
            j.mkdir(parents=True)
            # source on the LEFT in primary -> mutant is RIGHT
            (j / "p-e.primary.json").write_text(json.dumps(
                _calib_verdict("RIGHT", "TIE", [dict(LOCAL)])))
            # reversed: mutant shown on the LEFT; normalized back = RIGHT
            (j / "p-e.reversed.json").write_text(json.dumps(
                _calib_verdict("LEFT", "TIE")))
            (j / "p-b.primary.json").write_text(json.dumps(
                _calib_verdict("NONE", "TIE")))
            with unittest.mock.patch.multiple(
                    cc, CAL=root, PLAN_P=root / "plan.json",
                    CUSTODIAN_P=_custodian(td)):
                r = cc.score()
        self.assertEqual(r["recog"][:2], (1, 1))
        self.assertEqual(r["no_false_loss"][:2], (1, 1))
        self.assertEqual(r["pref"][:2], (0, 1))      # TIE != source
        self.assertEqual(r["q1_consist"][:2], (1, 1))

    def test_norm_side(self):
        self.assertEqual(cc._norm_side("LEFT", True), "RIGHT")
        self.assertEqual(cc._norm_side("NONE", True), "NONE")
        self.assertEqual(cc._norm_side("LEFT", False), "LEFT")


if __name__ == "__main__":
    unittest.main()
