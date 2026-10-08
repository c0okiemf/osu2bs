"""Recentred-variant tests .
  .venv/bin/python -m unittest qa.test_recentered -q
"""
import unittest

import torch

from qa.model import TARGET_NAMES, mixture_quantiles
from qa.recentered import neighbour_delta

N_T = len(TARGET_NAMES)


def nn_fixture(status="ok", masked_j=None):
    nbs = []
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        t = torch.full((N_T,), v)
        m = torch.ones(N_T, dtype=torch.bool)
        if masked_j is not None:
            m[masked_j] = False
        nbs.append({"targets": t, "mask": m})
    return {"status": status, "neighbours": nbs if status == "ok" else []}


class TestDelta(unittest.TestCase):
    def test_median_of_observed_minus_model_median(self):
        d, unk = neighbour_delta(nn_fixture(), torch.full((N_T,), 2.0))
        self.assertEqual(unk, [])
        self.assertTrue(torch.allclose(d, torch.full((N_T,), 1.0)))

    def test_masked_neighbour_components_are_not_data(self):
        d, unk = neighbour_delta(nn_fixture(masked_j=3),
                                 torch.zeros(N_T))
        self.assertEqual(unk, [TARGET_NAMES[3]])
        self.assertEqual(float(d[3]), 0.0)        # uncorrected, unknown
        self.assertEqual(float(d[0]), 3.0)

    def test_unsupported_retrieval_leaves_window_uncorrected(self):
        d, unk = neighbour_delta(nn_fixture(status="insufficient_support"),
                                 torch.ones(N_T))
        self.assertTrue(torch.equal(d, torch.zeros(N_T)))
        self.assertEqual(unk, list(TARGET_NAMES))


class TestLocationShift(unittest.TestCase):
    def test_uniform_mean_shift_moves_all_quantiles_by_delta(self):
        lw = torch.log(torch.tensor([[0.3, 0.7]]))
        mu = torch.tensor([[[-2.0], [3.0]]])
        ls = torch.log(torch.tensor([[[0.5], [1.5]]]))
        q0 = mixture_quantiles(lw, mu, ls)
        delta = 0.37
        q1 = mixture_quantiles(lw, mu + delta, ls)
        self.assertTrue(torch.allclose(q1, q0 + delta, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
