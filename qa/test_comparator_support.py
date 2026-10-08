"""Comparator Task 3 tests: raw-reference support + honest denominators.
  .venv/bin/python -m unittest qa.test_comparator_support -q
"""
import unittest
import unittest.mock

from qa import comparator_support as cs


def report_for_flags(flags, times):
    """Fixture helper exercising the PRODUCTION unsupported-run reducer."""
    return cs.run_gate(flags, times)


class TestRunGate(unittest.TestCase):
    def test_four_heads_over_two_seconds_violates(self):
        self.assertFalse(report_for_flags([False] * 4,
                                          [0, 1, 2, 2.01])["run_gate"])

    def test_three_heads_cannot_trigger(self):
        self.assertTrue(report_for_flags([False] * 3,
                                         [0, 1, 3])["run_gate"])

    def test_exactly_two_second_run_allowed(self):
        self.assertTrue(report_for_flags([False] * 4,
                                         [0, 0.5, 1, 2.0])["run_gate"])

    def test_initial_and_terminal_runs_count(self):
        r = report_for_flags([False, False, False, False, True],
                             [0, 1, 2, 2.5, 3])
        self.assertFalse(r["run_gate"])
        self.assertEqual(r["runs"][0]["n_heads"], 4)


class TestFreeze(unittest.TestCase):
    def _rows(self, fam, n, dist, diverse=True):
        return [{"family": fam, "window_id": f"{fam}:{i}",
                 "distance": dist, "diverse": diverse} for i in range(n)]

    def test_empty_rows_refused(self):
        with self.assertRaises(ValueError):
            cs.freeze_support([], {"identity": "qa-train-v2"})

    def test_max_family_p95(self):
        rows = self._rows("a", 100, 3.0) + self._rows("b", 100, 5.0)
        t = cs.freeze_support(rows, {"identity": "qa-train-v2"})
        self.assertEqual(t["T_support"], 5.0)
        self.assertIn("provenance", t)

    def test_insufficient_diverse_retrieval_stops(self):
        rows = (self._rows("a", 85, 3.0)
                + self._rows("a", 15, None, diverse=False))
        with self.assertRaisesRegex(RuntimeError,
                                    "DATA_SUPPORT_INSUFFICIENT"):
            cs.freeze_support(rows, {"identity": "qa-train-v2"})


class _OkValidator:
    def valid(self, row):
        return True, None


class _BadValidator:
    def valid(self, row):
        return False, "hash_mismatch"


def scene_fixture(n=10):
    return {"scope": None, "chart_sha256": "aa" * 32,
            "notes": [(float(i), i % 4, 0, i % 2, 1) for i in range(n)],
            "bombs": [], "walls": [], "settings": {"njs": 18}}


def fake_retrieve(distances):
    def _r(bank, q, exclude=None):
        t = float(q[0])                # descriptor stub carries the time
        d = distances(t)
        if d is None:
            return {"status": "insufficient_support", "neighbours": [],
                    "support_distance": None}
        return {"status": "ok", "support_distance": d,
                "neighbours": [{"traj_ref": "x#0", "content_hash": "h",
                                "family": f"f{i}", "player": f"p{i}",
                                "targets": None} for i in range(5)]}
    return _r


class TestMeasure(unittest.TestCase):
    def _measure(self, distances, validator, n=10):
        with unittest.mock.patch("qa.neighbours.retrieve",
                                 fake_retrieve(distances)), \
             unittest.mock.patch("qa.neighbours.descriptor",
                                 lambda sc, i: [sc["notes"][i][0]]):
            return cs.measure_support(scene_fixture(n), None,
                                      {"T_support": 5.0},
                                      None, validator=validator)

    def test_exactly_ninety_percent_passes(self):
        r = self._measure(lambda t: 9.0 if t == 5.0 else 1.0,
                          _OkValidator())
        self.assertEqual(r["share_supported"], 0.9)
        self.assertTrue(r["machine_support_pass"])
        self.assertEqual(r["per_head"][5]["reason"],
                         "distance_above_cutoff")

    def test_invalid_reference_cannot_earn_support(self):
        r = self._measure(lambda t: 1.0, _BadValidator())
        self.assertEqual(r["heads_supported"], 0)
        self.assertTrue(r["evidence_errors"])
        self.assertFalse(r["machine_support_pass"])
        self.assertTrue(all(h["reason"] == "reference_hash_mismatch"
                            for h in r["per_head"]))

    def test_insufficient_diversity_is_unsupported_not_zero(self):
        r = self._measure(lambda t: None, _OkValidator())
        self.assertEqual(r["heads_supported"], 0)
        self.assertTrue(all(h["distance"] is None
                            for h in r["per_head"]))

    def test_zero_denominator_is_not_success(self):
        r = self._measure(lambda t: 1.0, _OkValidator(), n=0)
        self.assertIsNone(r["share_supported"])
        self.assertFalse(r["machine_support_pass"])

    def test_diagnostic_values_cannot_affect_report(self):
        with unittest.mock.patch("qa.neighbours.retrieve",
                                 fake_retrieve(lambda t: 1.0)), \
             unittest.mock.patch("qa.neighbours.descriptor",
                                 lambda sc, i: [sc["notes"][i][0]]):
            a = cs.measure_support(scene_fixture(), None,
                                   {"T_support": 5.0}, None,
                                   validator=_OkValidator())
            sc = scene_fixture()
            sc["model_quantiles"] = [1, 2, 3]
            sc["mixture_nll"] = -0.5
            b = cs.measure_support(sc, None, {"T_support": 5.0}, None,
                                   validator=_OkValidator())
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
