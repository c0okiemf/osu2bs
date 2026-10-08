"""Chart-latent Task 3 tests: cross-fitted local predictions.
  .venv/bin/python -m unittest qa.test_chart_latent_train -q
"""
import random
import unittest

import torch

from qa.chart_latent import _bag_windows
from qa.chart_latent_train import (cache_oof_predictions, fold_partition,
                                   make_local_folds)
from qa.latent_validation import FOLD_SEED_BASE


def pack(players):
    n = len(players)
    return {"x": torch.zeros(n, 3), "y": torch.zeros(n, 10),
            "mask": torch.ones(n, 10, dtype=torch.bool),
            "players": players}


class TestFolds(unittest.TestCase):
    def test_six_folds_with_seeds(self):
        fams = [f"fam:{i:02d}" for i in range(24)]
        folds = make_local_folds(fams)
        self.assertEqual(len(folds), 6)
        self.assertEqual([f["seed"] for f in folds],
                         [FOLD_SEED_BASE + k for k in range(6)])
        self.assertTrue(all(f["updates"] == 500 for f in folds))

    def test_held_player_cannot_influence_scaler(self):
        packs = {"fam:A": pack(["p1", "p2", "p1"]),
                 "fam:H": pack(["p2", "p3", "p4"])}
        kept = fold_partition(packs, held={"fam:H"},
                              removed_players={"p2", "p3", "p4"})
        self.assertNotIn("fam:H", kept)
        self.assertEqual(kept["fam:A"], [0, 2])   # p2's row removed

    def test_final_base_cannot_substitute_for_fold(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            cache_oof_predictions({"kind": "final"}, "fam:X",
                                  pack(["p1"]))
        with self.assertRaisesRegex(ValueError, "partition"):
            cache_oof_predictions({"kind": "latent-fold",
                                   "held": ["fam:Y"]}, "fam:X",
                                  pack(["p1"]))


class TestBagSampling(unittest.TestCase):
    def _store(self, players, times):
        return {"players": players, "event_times": times}

    def test_bag_needs_three_players_eight_separated_windows(self):
        players = (["p1"] * 20 + ["p2"] * 20 + ["p3"] * 20)
        times = [i * 1.0 for i in range(20)] * 3
        rows = _bag_windows(self._store(players, times),
                            random.Random(0))
        self.assertEqual(len(rows), 24)
        by_p = {}
        for r in rows:
            by_p.setdefault(players[r], []).append(times[r])
        self.assertEqual(len(by_p), 3)
        for ts in by_p.values():
            ts = sorted(ts)
            self.assertEqual(len(ts), 8)
            self.assertTrue(all(b - a >= 0.8 for a, b in zip(ts, ts[1:])))

    def test_insufficient_family_returns_none_not_duplicates(self):
        players = ["p1"] * 30 + ["p2"] * 30
        times = [i * 1.0 for i in range(30)] * 2
        self.assertIsNone(_bag_windows(self._store(players, times),
                                       random.Random(0)))
        # crowded windows (<0.8s apart) cannot be stretched into a bag
        players3 = ["p1"] * 10 + ["p2"] * 10 + ["p3"] * 10
        times3 = [i * 0.1 for i in range(10)] * 3
        self.assertIsNone(_bag_windows(self._store(players3, times3),
                                       random.Random(0)))


if __name__ == "__main__":
    unittest.main()
