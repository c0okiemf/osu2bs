"""Q2 admission/selection semantics (synthetic, fast)."""
import unittest

from quality import Candidate, admit_and_select, gates_vs_baseline


def _rv(flags=0.0, narrow=0, broad=0, conv=0, opp=0, p95=10.0,
        vert=0.5, longrun=0.3, lat=0.1):
    return {"flags": {e: int(flags) for e in (0.0, 0.25, 0.5, 0.75)},
            "flags_p1000": {str(e): flags for e in (0.0, 0.25, 0.5, 0.75)},
            "narrow": narrow, "broad": broad, "converging": conv,
            "opposite": opp, "offaxis": 0, "reverse": 0, "unknown_dots": 0,
            "p95": p95, "style": {"vert": vert, "longrun": longrun,
                                  "lat": lat},
            "neighborhood_total": 0}


def _c(cid, rv, score=0.0, honored=True):
    return Candidate(cid=cid, raw=[], walls=[], rv=rv, score=score,
                     honored=honored)


class TestAdmitAndSelect(unittest.TestCase):
    def test_all_rejected_keeps_valid_baseline(self):
        b0 = _c("b0", _rv(flags=5))
        bad = [_c("x", _rv(flags=9), honored=True),          # flags regress
               _c("y", _rv(flags=1), honored=False)]         # not honored
        r = admit_and_select(b0, bad)
        self.assertEqual(r.selected.cid, "b0")
        self.assertEqual(sorted(r.rejections), ["x", "y"])
        self.assertIn("not_honored", r.rejections["y"])

    def test_risk_first_selection_ignores_score(self):
        b0 = _c("b0", _rv(flags=5), score=99.0)
        better = _c("a", _rv(flags=0), score=-99.0)          # far lower J
        r = admit_and_select(b0, [better])
        self.assertEqual(r.selected.cid, "a")                # score never leads

    def test_score_breaks_ties_only_when_motion_equivalent(self):
        b0 = _c("b0", _rv(flags=5))
        a = _c("a", _rv(flags=0.0), score=1.0)
        b = _c("b", _rv(flags=0.5), score=9.0)               # within 1 flag/1k
        r = admit_and_select(b0, [a, b])
        self.assertEqual(r.selected.cid, "b")                # tie-break by score
        c = _c("c", _rv(flags=0.0, narrow=1), score=99.0)    # pairs differ
        r2 = admit_and_select(b0, [a, c])
        self.assertEqual(r2.selected.cid, "a")               # no tie-break

    def test_every_rejection_reported_with_reasons(self):
        b0 = _c("b0", _rv())
        cand = _c("z", _rv(narrow=2, p95=100.0))
        r = admit_and_select(b0, [cand])
        self.assertEqual(set(r.rejections["z"]) & {"narrow", "p95"},
                         {"narrow", "p95"})
        self.assertEqual(r.accounting["n_candidates"], 1)
        self.assertEqual(r.accounting["n_admitted"], 1)      # baseline only

    def test_gates_vs_baseline_style_not_worse_rule(self):
        base = _rv(vert=0.9)                                 # B0 already over
        ok = gates_vs_baseline(_rv(vert=0.9), base)
        self.assertNotIn("style", ok)
        worse = gates_vs_baseline(_rv(vert=0.95), base)
        self.assertIn("style", worse)


if __name__ == "__main__":
    unittest.main()
