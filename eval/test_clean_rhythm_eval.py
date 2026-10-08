"""Tests for clean-rhythm A/B reduction: shared rest bins + acceptance gates.
  .venv/bin/python -m unittest eval.test_clean_rhythm_eval -v
"""
import unittest

from eval.clean_rhythm_eval import (evaluate_acceptance, gate_song, rest_pass,
                                    shared_rest_bins)


class TestSharedRestBins(unittest.TestCase):
    def test_spec_example(self):
        r = shared_rest_bins([{"t": 0., "dir": 1}, {"t": 2000., "dir": 0},
                              {"t": 2100., "dir": 8}], 0., 4500.)
        self.assertEqual(r["counts"], [1, 1])      # dot at 2100 not counted
        self.assertEqual(r["quiet_ids"], [0, 1])
        self.assertEqual(r["n_bins"], 2)           # trailing partial excluded

    def test_boundary_belongs_to_next_bin(self):
        r = shared_rest_bins([{"t": 2000., "dir": 1}], 0., 4000.)
        self.assertEqual(r["counts"], [0, 1])      # t=2000 -> bin 1, not 0

    def test_shared_origin_not_first_note(self):
        # bins anchored at start_ms, not the first note
        a = shared_rest_bins([{"t": 1000., "dir": 1}], 0., 4000.)
        b = shared_rest_bins([{"t": 3000., "dir": 1}], 0., 4000.)
        self.assertEqual(a["counts"], [1, 0])
        self.assertEqual(b["counts"], [0, 1])

    def test_empty_domain(self):
        self.assertEqual(shared_rest_bins([], 0., 1000.)["n_bins"], 0)


class TestRestPass(unittest.TestCase):
    def test_share_and_retention(self):
        self.assertTrue(rest_pass(list(range(10)), list(range(9)), 100))
        self.assertFalse(rest_pass(list(range(10)), list(range(8)), 100))
        # same share but wholly moved rests fail retention
        self.assertFalse(rest_pass(list(range(10)), list(range(10, 20)), 100))

    def test_empty_domain_fails(self):
        self.assertFalse(rest_pass([], [], 0))

    def test_a_empty_retention_na(self):
        self.assertTrue(rest_pass([], [], 100))    # share 0, retention N/A


def _metrics(valid=True, in_band=True, dup=0.0, flags=30.0, narrow=0, conv=0,
             run=8, quiet=(list(range(5)), list(range(5)), 50)):
    return {"valid": valid, "in_band": in_band,
            "dup8_winner": dup, "dup8_six_mean": dup,
            "flags_by_ext": {e: (flags, flags) for e in ("0.0", "0.25", "0.5", "0.75")},
            "narrow_winner": narrow, "narrow_six_mean": narrow,
            "conv_winner": conv, "conv_six_mean": conv,
            "longest_run_winner": run, "longest_run_six_mean": run,
            "quiet": {"a_ids": quiet[0], "b_ids": quiet[1], "n_bins": quiet[2]}}


class TestGateSong(unittest.TestCase):
    def test_clean_pass(self):
        self.assertEqual(gate_song(_metrics(), _metrics()), [])

    def test_dup_regression(self):
        a, b = _metrics(dup=0.0), _metrics(dup=2.0)   # > max(0,1.0)
        self.assertIn("dup8", gate_song(a, b))

    def test_flags_within_slack(self):
        a, b = _metrics(flags=30.0), _metrics(flags=33.0)  # +3 <= max(5,3)
        self.assertNotIn("repositioning@0.0", gate_song(a, b))

    def test_flags_regression(self):
        a, b = _metrics(flags=30.0), _metrics(flags=40.0)  # +10 > max(5,3)
        self.assertIn("repositioning@0.0", gate_song(a, b))

    def test_hand_monopoly_slack_to_12(self):
        a, b = _metrics(run=5), _metrics(run=12)      # <= max(5,12)
        self.assertNotIn("hand_monopoly", gate_song(a, b))
        self.assertIn("hand_monopoly", gate_song(_metrics(run=5), _metrics(run=13)))

    def test_invalid_fails(self):
        self.assertIn("validity", gate_song(_metrics(), _metrics(valid=False)))


class TestEvaluateAcceptance(unittest.TestCase):
    def test_accept_when_all_clean(self):
        rec = {"integrity_ok": True, "learned": True,
               "songs": {"s1": {"a": _metrics(), "b": _metrics()}}}
        self.assertEqual(evaluate_acceptance(rec)["status"], "AB_ACCEPTED")

    def test_not_accepted_on_one_song(self):
        rec = {"integrity_ok": True, "learned": True,
               "songs": {"s1": {"a": _metrics(), "b": _metrics()},
                         "s2": {"a": _metrics(dup=0.0), "b": _metrics(dup=3.0)}}}
        self.assertEqual(evaluate_acceptance(rec)["status"], "AB_NOT_ACCEPTED")

    def test_incomplete_when_not_learned(self):
        rec = {"integrity_ok": True, "learned": False, "songs": {}}
        self.assertEqual(evaluate_acceptance(rec)["status"], "INCOMPLETE")

    def test_inconclusive_baseline(self):
        rec = {"integrity_ok": True, "learned": True,
               "songs": {"s1": {"inconclusive": True}}}
        self.assertEqual(evaluate_acceptance(rec)["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
