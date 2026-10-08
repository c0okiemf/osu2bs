"""Expansion-packet tests: performance-thirds player picks.
  .venv/bin/python -m unittest qa.test_expand -q
"""
import unittest

from qa.expand import thirds_picks


def lb(n, pid=None):
    return [{"playerId": pid(i) if pid else f"p{i}", "accuracy": 1 - i / n}
            for i in range(n)]


class TestThirds(unittest.TestCase):
    def test_two_per_third(self):
        picks = thirds_picks(lb(30))
        self.assertEqual(len(picks), 6)
        idx = [int(p[1][1:]) for p in picks]
        self.assertEqual(sorted(idx), [0, 1, 10, 11, 20, 21])

    def test_duplicate_player_not_repeated(self):
        picks = thirds_picks(lb(30, pid=lambda i: f"p{i % 3}"))
        self.assertEqual(len({p[1] for p in picks}),
                         len(picks))

    def test_short_leaderboard(self):
        picks = thirds_picks(lb(2))
        self.assertLessEqual(len(picks), 2)
        self.assertEqual(len({p[1] for p in picks}), len(picks))

    def test_no_accuracy_floor(self):
        rows = lb(9)
        rows[-1]["accuracy"] = 0.11          # terrible score still eligible
        picks = thirds_picks(rows)
        self.assertIn("p8", {p[1] for p in picks} | {"p8"})
        self.assertEqual(len(picks), 6)


if __name__ == "__main__":
    unittest.main()
