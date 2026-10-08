"""Task 5 tests: blind player feedback is the first real acceptance.
  .venv/bin/python -m unittest eval.test_expressive_feedback -q
"""
import unittest

from eval.expressive_feedback import blind_order, decide_early, resolve_wins


def rating(pair_id="p0", overall="left", discomfort=False, rid=None):
    return {"rating_id": rid or f"r-{pair_id}", "pair_id": pair_id,
            "overall": overall, "musical_flow": None,
            "arm_path_enjoyment": None, "doubles_enjoyment": None,
            "discomfort": discomfort, "comment": ""}


def ratings_for(pattern, discomfort=False):
    """W/L/T pattern already resolved against the NEW model."""
    out = []
    for i, ch in enumerate(pattern):
        out.append({"pair_id": f"p{i}",
                    "resolved": {"W": "win", "L": "loss",
                                 "T": "tie"}[ch],
                    "discomfort": discomfort and i == 0})
    return out


class TestDecision(unittest.TestCase):
    def test_four_wins_one_loss_one_tie_advances(self):
        self.assertTrue(decide_early(ratings_for("WWWWLT"))["advance"])

    def test_missing_or_discomfort_never_advances(self):
        self.assertFalse(decide_early(ratings_for("WWWWW"))["advance"])
        self.assertFalse(decide_early(ratings_for("WWWWWW",
                                                  discomfort=True))
                         ["advance"])

    def test_ties_are_not_wins(self):
        self.assertFalse(decide_early(ratings_for("WWWTTT"))["advance"])

    def test_two_losses_fail(self):
        self.assertFalse(decide_early(ratings_for("WWWWLL"))["advance"])

    def test_empty_is_waiting(self):
        d = decide_early([])
        self.assertFalse(d["advance"])
        self.assertEqual(d["status"], "WAITING_FOR_PLAYTEST")


class TestBlinding(unittest.TestCase):
    def test_side_order_randomized_but_deterministic(self):
        a = [blind_order(f"p{i}", seed=7) for i in range(20)]
        b = [blind_order(f"p{i}", seed=7) for i in range(20)]
        self.assertEqual(a, b)
        self.assertTrue(any(x == "swapped" for x in a))
        self.assertTrue(any(x == "normal" for x in a))
        self.assertNotEqual(a, [blind_order(f"p{i}", seed=8)
                                for i in range(20)])


class TestResolution(unittest.TestCase):
    def _pairs(self):
        return {"p0": {"sides": "normal", "fallback": False},
                "p1": {"sides": "swapped", "fallback": False},
                "p2": {"sides": "normal", "fallback": True}}

    def test_swapped_sides_resolve_correctly(self):
        res = resolve_wins([rating("p0", overall="right"),
                            rating("p1", overall="right")], self._pairs())
        # p0 normal: right = NEW side loses? convention: left=B0, right=new
        # under "normal"; swapped inverts. p0 right -> win; p1 right -> loss
        by = {r["pair_id"]: r["resolved"] for r in res}
        self.assertEqual(by["p0"], "win")
        self.assertEqual(by["p1"], "loss")

    def test_fallback_pair_cannot_be_new_model_win(self):
        res = resolve_wins([rating("p2", overall="right")], self._pairs())
        self.assertEqual(res[0]["resolved"], "fallback")

    def test_duplicate_rating_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            resolve_wins([rating("p0", rid="x"), rating("p0", rid="x")],
                         self._pairs())

    def test_unknown_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown pair"):
            resolve_wins([rating("p9")], self._pairs())

    def test_tie_stays_tie(self):
        res = resolve_wins([rating("p0", overall="tie")], self._pairs())
        self.assertEqual(res[0]["resolved"], "tie")


if __name__ == "__main__":
    unittest.main()
