"""Feasibility-recipe tests .
  .venv/bin/python -m unittest qa.test_feasibility -q
"""
import math
import unittest

from qa.feasibility import (family_min_factors, required_factor,
                            weighted_min_factor)


class TestRequiredFactor(unittest.TestCase):
    def test_below_and_above_median(self):
        self.assertAlmostEqual(required_factor(0.5, 1.0, 2.0, 4.0), 1.5)
        self.assertAlmostEqual(required_factor(5.0, 1.0, 2.0, 4.0), 1.5)

    def test_target_at_median_needs_zero(self):
        self.assertEqual(required_factor(2.0, 1.0, 2.0, 4.0), 0.0)

    def test_zero_width_side_is_explicitly_uncoverable(self):
        self.assertEqual(required_factor(1.0, 2.0, 2.0, 4.0), math.inf)
        self.assertEqual(required_factor(2.0, 2.0, 2.0, 4.0), 0.0)


class TestWeightedQuantile(unittest.TestCase):
    def test_exact_empirical_with_ties(self):
        # 10 values: 85% quantile = smallest v with cum weight >= 8.5
        vals = [0.1] * 8 + [1.7, 1.7]
        self.assertEqual(weighted_min_factor(vals, [1.0] * 10), 1.7)
        vals = [0.1] * 9 + [1.7]
        self.assertEqual(weighted_min_factor(vals, [1.0] * 10), 0.1)

    def test_weights_shift_the_quantile(self):
        self.assertEqual(weighted_min_factor([0.1, 1.9], [0.9, 0.1]), 0.1)
        self.assertEqual(weighted_min_factor([0.1, 1.9], [0.5, 0.5]), 1.9)

    def test_inf_when_uncoverable_mass_blocks(self):
        self.assertEqual(weighted_min_factor([0.1, math.inf],
                                             [0.5, 0.5]), math.inf)


class TestEqualPlayer(unittest.TestCase):
    def test_players_weighted_equally(self):
        # player A: 9 easy obs; player B: 1 hard obs -> equal-player
        # aggregation needs B covered (0.5 mass), pooled does not
        obs = ([{"player": "a", "component": "c", "f_req": 0.2}] * 9
               + [{"player": "b", "component": "c", "f_req": 1.8}])
        pooled, equal = family_min_factors(obs)
        self.assertEqual(pooled, 0.2)
        self.assertEqual(equal, 1.8)


if __name__ == "__main__":
    unittest.main()
