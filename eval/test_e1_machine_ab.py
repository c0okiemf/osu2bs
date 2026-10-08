"""Preregistered decision rule check.  .venv/bin/python -m unittest eval.test_e1_machine_ab -q"""
import unittest

from eval.e1_machine_ab import decide


def row(p1, p2, p3, guard=0.1, verdict="MACHINE_PASS_QUALITY_NOT_EVALUATED",
        share=0.95):
    return {"props": {"P1": p1, "P2": p2, "P3": p3}, "guard": guard,
            "verdict": verdict, "share_supported": share}


class TestDecide(unittest.TestCase):
    def test_directions(self):
        d = decide(row(1.0, 0.5, 3.0), row(2.0, 0.4, 3.5))
        self.assertTrue(d["P1"] and d["P2"] and d["P3"] and d["guard"])
        d = decide(row(1.0, 0.5, 3.0), row(0.5, 0.6, 2.0))
        self.assertFalse(d["P1"] or d["P2"] or d["P3"])

    def test_fallback_is_no_change(self):
        d = decide(row(1.0, 0.5, 3.0), None)
        self.assertFalse(d["P1"] or d["P2"] or d["P3"])
        self.assertTrue(d["guard"] and d["protect"])

    def test_guard_and_protection(self):
        self.assertFalse(decide(row(1, .5, 3, guard=.10),
                                row(2, .4, 4, guard=.13))["guard"])
        self.assertFalse(decide(row(1, .5, 3),
                                row(2, .4, 4, verdict="HARD_FAIL"))["protect"])
        self.assertFalse(decide(row(1, .5, 3, share=.95),
                                row(2, .4, 4, share=.89))["protect"])


if __name__ == "__main__":
    unittest.main()
