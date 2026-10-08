"""Chart-latent Task 4 tests: exact shared-latent adapter.
  .venv/bin/python -m unittest qa.test_chart_latent -q
"""
import math
import unittest

import torch

from qa.chart_latent import ChartLatentAdapter, constant_prior
from qa.model import N_TARGETS

T = N_TARGETS


def adapter_fixture(seed=7):
    torch.manual_seed(seed)
    a = ChartLatentAdapter(torch.ones(T)).double()
    with torch.no_grad():
        a.W += torch.randn(3, 21, dtype=torch.float64) * 0.3
        a.u += torch.randn(3, T, dtype=torch.float64) * 0.3
        a.v += torch.randn(3, T, dtype=torch.float64) * 0.3
    return a


def tiny_bag(n=3, k=2, seed=11):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)
    lw = torch.log_softmax(rnd(1, n, k), -1)
    mu = rnd(1, n, k, T)
    ls = rnd(1, n, k, T) * 0.3
    y = rnd(1, n, T)
    mask = torch.rand(1, n, T, generator=g) > 0.3
    mask[0, 0, 0] = True
    ctx = rnd(1, 21)
    return lw, mu, ls, y, mask, ctx


def enumerated_shared_latent_nll(adapter, lw, mu, ls, y, mask, ctx):
    log_pi = adapter.prior(ctx)[0]
    d, av = adapter.delta(), adapter.a()
    per_z = []
    for z in range(3):
        tot = 0.0
        for i in range(y.shape[1]):
            comps = []
            for kk in range(mu.shape[2]):
                lp = 0.0
                for t in range(T):
                    if not bool(mask[0, i, t]):
                        continue
                    m = float(mu[0, i, kk, t] + d[z, t])
                    sd = math.exp(float(ls[0, i, kk, t] + av[z, t]))
                    v = float(y[0, i, t])
                    lp += (-0.5 * ((v - m) / sd) ** 2
                           - math.log(sd) - 0.5 * math.log(2 * math.pi))
                comps.append(float(lw[0, i, kk]) + lp)
            mx = max(comps)
            tot += mx + math.log(sum(math.exp(c - mx) for c in comps))
        per_z.append(float(log_pi[z]) + tot)
    mx = max(per_z)
    chart_lp = mx + math.log(sum(math.exp(v - mx) for v in per_z))
    return -chart_lp / float(mask.sum())


class TestLikelihood(unittest.TestCase):
    def test_prior_is_paid_once_per_chart_bag(self):
        a = adapter_fixture()
        bag = tiny_bag()
        got = float(a.bag_nll(*bag))
        want = enumerated_shared_latent_nll(a, *bag)
        self.assertLess(abs(got - want), 1e-7)

    def test_zero_observed_components_rejected(self):
        a = adapter_fixture()
        lw, mu, ls, y, mask, ctx = tiny_bag()
        with self.assertRaises(ValueError):
            a.bag_nll(lw, mu, ls, y, torch.zeros_like(mask), ctx)

    def test_frozen_base_receives_no_gradients(self):
        a = adapter_fixture()
        bag = tiny_bag()
        loss = a.bag_nll(*bag) + a.regularizer()
        loss.backward()
        self.assertFalse(bag[1].requires_grad)
        self.assertIsNotNone(a.W.grad)
        self.assertIsNotNone(a.u.grad)


class TestServing(unittest.TestCase):
    def test_marginal_weights_normalize(self):
        a = adapter_fixture()
        lw, mu, ls, _y, _m, ctx = tiny_bag(k=4)
        lw12, mu12, ls12 = a.marginal(lw[0], mu[0], ls[0], ctx[0])
        self.assertEqual(mu12.shape[1], 12)
        self.assertTrue(torch.allclose(torch.logsumexp(lw12, dim=1),
                                       torch.zeros(lw12.shape[0],
                                                   dtype=torch.float64),
                                       atol=1e-8))

    def test_targets_cannot_change_serving_predictions(self):
        a = adapter_fixture()
        lw, mu, ls, y, _m, ctx = tiny_bag()
        q1 = a.quantiles(lw[0], mu[0], ls[0], ctx[0])
        y += 100.0
        q2 = a.quantiles(lw[0], mu[0], ls[0], ctx[0])
        self.assertTrue(torch.equal(q1, q2))

    def test_twelve_component_cdf_matches_independent_integration(self):
        a = adapter_fixture()
        lw, mu, ls, _y, _m, ctx = tiny_bag(n=1, k=4)
        q = a.quantiles(lw[0], mu[0], ls[0], ctx[0])
        lw12, mu12, ls12 = a.marginal(lw[0], mu[0], ls[0], ctx[0])
        w = lw12.exp()[0]
        for pi_, p in ((0, 0.1), (1, 0.5), (2, 0.9)):
            t = float(q[0, pi_, 0])
            cdf = sum(float(w[c]) * 0.5 * (1 + math.erf(
                (t - float(mu12[0, c, 0]))
                / (math.exp(float(ls12[0, c, 0])) * math.sqrt(2))))
                for c in range(12))
            self.assertLess(abs(cdf - p), 1e-7)

    def test_chart_sample_uses_one_state(self):
        a = adapter_fixture()
        lw, mu, ls, _y, _m, ctx = tiny_bag(n=6)
        g = torch.Generator().manual_seed(3)
        s = a.sample_chart(lw[0], mu[0], ls[0], ctx[0], g)
        self.assertEqual(len(set(s["latent_state_per_note"])), 1)

    def test_shared_state_correlates_notes_within_a_chart(self):
        a = ChartLatentAdapter(torch.ones(T)).double()
        with torch.no_grad():
            a.u[0, :] = -5.0                     # strong, distinct states
            a.u[2, :] = 5.0
        lw = torch.zeros(2, 1, dtype=torch.float64)
        mu = torch.zeros(2, 1, T, dtype=torch.float64)
        ls = torch.full((2, 1, T), -2.0, dtype=torch.float64)
        ctx = torch.zeros(21, dtype=torch.float64)
        g = torch.Generator().manual_seed(9)
        xs, ys = [], []
        for _ in range(300):
            s = a.sample_chart(lw, mu, ls, ctx, g)
            xs.append(float(s["samples"][0, 0]))
            ys.append(float(s["samples"][1, 0]))
        xt, yt = torch.tensor(xs), torch.tensor(ys)
        corr = float(((xt - xt.mean()) * (yt - yt.mean())).mean()
                     / (xt.std() * yt.std()))
        self.assertGreater(corr, 0.5)


class TestInitAndAblation(unittest.TestCase):
    def test_symmetry_broken_at_init(self):
        a = ChartLatentAdapter(torch.ones(T))
        d = a.delta()
        self.assertLess(float(d[0, 0]), 0.0)
        self.assertEqual(float(d[1, 0]), 0.0)
        self.assertGreater(float(d[2, 0]), 0.0)
        self.assertAlmostEqual(float(d[2, 0]), 0.25, places=5)
        self.assertEqual(float(d[0, 7]), 0.0)    # non-intensity targets

    def test_constant_prior_uses_training_contexts_only(self):
        a = adapter_fixture()
        ctxs = [torch.randn(21, dtype=torch.float64) for _ in range(5)]
        p = constant_prior(a, ctxs)
        self.assertTrue(torch.allclose(p["p"].sum(),
                                       torch.tensor(1.0,
                                                    dtype=torch.float64)))
        self.assertEqual(p["source_roles"], {"qa_train"})


if __name__ == "__main__":
    unittest.main()
