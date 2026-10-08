"""Task 4 tests: actual B0-policy checkpoint comparison.
  .venv/bin/python -m unittest eval.test_expressive_eval -q
"""
import unittest

from eval.expressive_eval import (admit, packet_coverage, select_arm,
                                  window_flags)


def candidate(critic=0.0, flags=0, cid="c", eligible=True):
    return {"id": cid, "critic": critic, "production_key": (True, critic),
            "eligible": eligible,
            "metrics": {"flags_by_ext": {"0.0": flags, "0.25": flags,
                                         "0.5": flags, "0.75": flags}}}


def metrics(flags=0, narrow=0, conv=0, windows=None):
    return {"flags_by_ext": {"0.0": flags, "0.25": flags, "0.5": flags,
                             "0.75": flags},
            "narrow": narrow, "conv": conv, "windows_4s": windows or {}}


CONTRACT = {"flags_rule": "max(1.25*B0, B0+10)", "pair_slack": 2,
            "window_slack": 2}


class TestCeilings(unittest.TestCase):
    def test_boundary_admits_just_above_rejects(self):
        b0 = metrics(flags=40)
        lim = max(1.25 * 40, 40 + 10)                       # 50
        self.assertTrue(admit(metrics(flags=int(lim)), b0,
                              CONTRACT)["admitted"])
        rej = admit(metrics(flags=int(lim) + 1), b0, CONTRACT)
        self.assertFalse(rej["admitted"])
        self.assertTrue(any("flags" in r for r in rej["reasons"]))

    def test_low_b0_gets_absolute_slack(self):
        b0 = metrics(flags=0)
        self.assertTrue(admit(metrics(flags=10), b0, CONTRACT)["admitted"])
        self.assertFalse(admit(metrics(flags=11), b0, CONTRACT)["admitted"])

    def test_narrow_and_conv_assessed_separately(self):
        b0 = metrics(narrow=1, conv=5)
        self.assertTrue(admit(metrics(narrow=3, conv=7), b0,
                              CONTRACT)["admitted"])
        self.assertFalse(admit(metrics(narrow=4, conv=0), b0,
                               CONTRACT)["admitted"])
        self.assertFalse(admit(metrics(narrow=0, conv=8), b0,
                               CONTRACT)["admitted"])

    def test_window_ceiling(self):
        b0 = metrics(windows={"0": 1, "1": 0})
        ok = admit(metrics(windows={"0": 3, "1": 2}), b0, CONTRACT)
        self.assertTrue(ok["admitted"])
        bad = admit(metrics(windows={"1": 3}), b0, CONTRACT)
        self.assertFalse(bad["admitted"])

    def test_broad_opposite_reported_not_rejected(self):
        b0 = metrics()
        c = metrics()
        c["broad"], c["opposite"] = 99, 99
        self.assertTrue(admit(c, b0, CONTRACT)["admitted"])


class TestSelectors(unittest.TestCase):
    def test_ceilings_are_not_rank_scores(self):
        a = candidate(critic=4, flags=8, cid="a")
        b = candidate(critic=3, flags=0, cid="b")
        # Applies to E2 floor-only selector, not E1's unchanged B0 pick.
        self.assertEqual(select_arm([a, b], selector="critic_floor")["id"],
                         "a")

    def test_critic_floor_ignores_ineligible(self):
        a = candidate(critic=9, cid="a", eligible=False)
        b = candidate(critic=1, cid="b")
        self.assertEqual(select_arm([a, b], selector="critic_floor")["id"],
                         "b")

    def test_b0_selector_uses_production_key(self):
        a = candidate(cid="a")
        a["production_key"] = (True, -2.0)
        b = candidate(cid="b")
        b["production_key"] = (True, -1.0)
        self.assertEqual(select_arm([a, b], selector="b0")["id"], "b")

    def test_critic_floor_tie_breaks_by_canonical_id(self):
        a = candidate(critic=2.0, cid="s0")
        b = candidate(critic=2.0, cid="s1")
        self.assertEqual(select_arm([b, a], selector="critic_floor")["id"],
                         "s0")

    def test_no_eligible_candidates_is_explicit(self):
        a = candidate(cid="a", eligible=False)
        r = select_arm([a], selector="critic_floor")
        self.assertIsNone(r["id"])
        self.assertEqual(r["status"], "fallback_b0")


class TestCoverage(unittest.TestCase):
    def test_missing_attempt_is_not_success(self):
        self.assertEqual(packet_coverage(required=6, completed=5),
                         "INCOMPLETE")
        self.assertEqual(packet_coverage(required=6, completed=6), "COMPLETE")


class TestWindowFlags(unittest.TestCase):
    def test_flags_binned_by_4s_from_origin(self):
        # two same-hand notes 150 ms apart, far cells -> one flagged
        # transition landing in the window of its END time
        notes = [(4900.0, 0, 0, 0, 1), (5050.0, 0, 3, 2, 1)]
        w = window_flags(notes, ext=0.0)
        self.assertEqual(sum(w.values()), 1)
        self.assertEqual(list(w), [str(int(5050.0 // 4000))])

    def test_slow_transitions_unflagged(self):
        notes = [(0.0, 0, 0, 0, 1), (2000.0, 0, 3, 2, 1)]
        self.assertEqual(sum(window_flags(notes, ext=0.0).values()), 0)


if __name__ == "__main__":
    unittest.main()
