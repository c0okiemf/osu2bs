"""Task 4 tests: conditional kinematics mixture.
  .venv/bin/python -m unittest qa.test_model -q
"""
import math
import unittest

import torch

from qa.model import (FEATURE_NAMES, N_TARGETS, MotionMixture, evaluate_map,
                      map_features, mixture_nll, mixture_quantiles,
                      target_from_window)


def window_fixture(missing_rotation=False):
    t = {"path_m": 1.2, "speed_p90": 3.0, "accel_p90": 40.0,
         "lateral_extent": 0.5, "vertical_extent": 0.7,
         "rotation_rad": 2.0, "angular_speed_p90": 9.0,
         "other_path_m": 0.4, "head_path_m": 0.1,
         "min_hand_separation": 0.3}
    if missing_rotation:
        t["rotation_rad"] = None
        t["angular_speed_p90"] = None
    return {"supported": True, "outcome": "good", "targets": t}


def scene_fixture(n=12):
    notes = [(0.5 * i, i % 4, i % 3, i % 2, 1) for i in range(n)]
    return {"scope": None, "notes": notes, "bombs": [], "walls": [],
            "settings": {"njs": 18, "offset_beats": 0.0}}


def audio_fixture(dur=10):
    return {"supported": True, "duration_s": dur,
            "frames": {"times": [i * 0.023 for i in range(int(dur / .023))],
                       "onset_strength": [0.5] * int(dur / .023)},
            "per_second": {"harmonic_rms": [0.1] * dur,
                           "percussive_rms": [0.2] * dur}}


class TestTargets(unittest.TestCase):
    def test_log1p_transform_and_mask(self):
        y, mask = target_from_window(window_fixture())
        self.assertEqual(len(y), N_TARGETS)
        self.assertTrue(all(mask))
        self.assertAlmostEqual(y[0].item(), math.log1p(1.2), places=5)

    def test_missing_rotation_is_not_zero_rotation_loss(self):
        y, mask = target_from_window(window_fixture(missing_rotation=True))
        self.assertFalse(mask[5])
        self.assertFalse(mask[6])
        self.assertTrue(mask[0])


class TestFeatures(unittest.TestCase):
    def test_no_forbidden_fields(self):
        for banned in ("player", "hmd", "controller", "skill", "rank",
                       "outcome", "pose", "critic", "score"):
            self.assertFalse(any(banned in f for f in FEATURE_NAMES),
                             banned)

    def test_shape_and_padding_masks(self):
        x = map_features(scene_fixture(), audio_fixture(), 0,
                         {"height": 1.7, "height_known": True,
                          "left_handed": False})
        self.assertEqual(len(x), len(FEATURE_NAMES))
        self.assertTrue(torch.isfinite(x).all())
        # first note has no predecessors: prev masks zero
        prev_mask = [f for f in FEATURE_NAMES if f.startswith("prev1_ok")]
        self.assertEqual(x[FEATURE_NAMES.index("prev1_ok")].item(), 0.0)

    def test_unknown_height_masked_not_default(self):
        x = map_features(scene_fixture(), audio_fixture(), 3,
                         {"height": None, "height_known": False,
                          "left_handed": False})
        i = FEATURE_NAMES.index("body_height")
        k = FEATURE_NAMES.index("body_height_known")
        self.assertEqual(x[i].item(), 0.0)
        self.assertEqual(x[k].item(), 0.0)


class TestMixture(unittest.TestCase):
    def test_masked_nll_and_clamps(self):
        m = MotionMixture(n_in=len(FEATURE_NAMES))
        x = torch.rand(4, len(FEATURE_NAMES))
        w, mu, ls = m(x)
        self.assertTrue((ls >= -4).all() and (ls <= 2).all())
        y = torch.rand(4, N_TARGETS)
        mask = torch.ones(4, N_TARGETS, dtype=torch.bool)
        mask[:, 5] = False
        nll = mixture_nll(w, mu, ls, y, mask)
        self.assertTrue(torch.isfinite(nll))

    def test_fully_masked_row_contributes_nothing(self):
        m = MotionMixture(n_in=len(FEATURE_NAMES))
        x = torch.rand(2, len(FEATURE_NAMES))
        w, mu, ls = m(x)
        y = torch.zeros(2, N_TARGETS)
        mask = torch.zeros(2, N_TARGETS, dtype=torch.bool)
        mask[0] = True
        a = mixture_nll(w, mu, ls, y, mask)
        b = mixture_nll(w[:1], mu[:1], ls[:1], y[:1], mask[:1])
        self.assertAlmostEqual(a.item(), b.item(), places=5)


def standard_normal_mixture():
    return (torch.zeros(1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))


def unequal_mixture_fixture():
    lw = torch.log(torch.tensor([[0.3, 0.7]]))
    mu = torch.tensor([[[-2.0], [3.0]]])
    ls = torch.log(torch.tensor([[[0.5], [1.5]]]))
    return lw, mu, ls


def _independent_cdf(t, ws, mus, sds):
    return sum(w * 0.5 * (1 + math.erf((t - m) / (s * math.sqrt(2))))
               for w, m, s in zip(ws, mus, sds))


class TestMixtureQuantiles(unittest.TestCase):
    def test_standard_normal_p90(self):
        q = mixture_quantiles(*standard_normal_mixture())
        self.assertLess(abs(float(q[0, 2, 0]) - 1.2815515655), 1e-6)
        self.assertLess(abs(float(q[0, 1, 0])), 1e-7)

    def test_gate_quantiles_do_not_depend_on_rng(self):
        p = unequal_mixture_fixture()
        a = mixture_quantiles(*p)
        torch.manual_seed(17)
        self.assertTrue(torch.equal(a, mixture_quantiles(*p)))

    def test_shifted_scaled_normal(self):
        lw = torch.zeros(1, 1)
        mu = torch.full((1, 1, 1), 5.0)
        ls = torch.log(torch.full((1, 1, 1), 3.0))
        q = mixture_quantiles(lw, mu, ls)
        self.assertLess(abs(float(q[0, 2, 0]) - (5 + 3 * 1.2815515655)),
                        1e-6)

    def test_multimodal_matches_independent_integration(self):
        q = mixture_quantiles(*unequal_mixture_fixture())
        for pi, p in enumerate((0.1, 0.5, 0.9)):
            f = _independent_cdf(float(q[0, pi, 0]), [0.3, 0.7],
                                 [-2.0, 3.0], [0.5, 1.5])
            self.assertLess(abs(f - p), 1e-7)
        # bimodal with a heavy right mode is asymmetric about the median
        p10, p50, p90 = (float(q[0, i, 0]) for i in range(3))
        self.assertNotAlmostEqual(p90 - p50, p50 - p10, places=2)

    def test_extreme_allowed_scales(self):
        lw = torch.log(torch.tensor([[0.5, 0.5]]))
        mu = torch.tensor([[[0.0], [10.0]]])
        ls = torch.tensor([[[-4.0], [2.0]]])       # clamp bounds
        q = mixture_quantiles(lw, mu, ls)
        for pi, p in enumerate((0.1, 0.5, 0.9)):
            f = _independent_cdf(float(q[0, pi, 0]), [0.5, 0.5],
                                 [0.0, 10.0],
                                 [math.exp(-4), math.exp(2)])
            self.assertLess(abs(f - p), 1e-7)

    def test_batch_order_invariance(self):
        lw = torch.log(torch.tensor([[0.3, 0.7], [0.9, 0.1]]))
        mu = torch.tensor([[[-2.0], [3.0]], [[0.0], [8.0]]])
        ls = torch.zeros(2, 2, 1)
        q = mixture_quantiles(lw, mu, ls)
        r = mixture_quantiles(lw.flip(0), mu.flip(0), ls.flip(0))
        self.assertTrue(torch.equal(q.flip(0), r))

    def test_nonfinite_parameters_fail_explicitly(self):
        lw, mu, ls = unequal_mixture_fixture()
        mu = mu.clone()
        mu[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            mixture_quantiles(lw, mu, ls)


class TestServing(unittest.TestCase):
    def test_map_only_prediction_has_no_observed_nll(self):
        m = MotionMixture(n_in=len(FEATURE_NAMES))
        out = evaluate_map(scene_fixture(), audio_fixture(), m,
                           {"height": 1.7, "height_known": True,
                            "left_handed": False})
        self.assertNotIn("observed_nll", out)
        self.assertIn("quantiles", out[0] if isinstance(out, list)
                      else out)


if __name__ == "__main__":
    unittest.main()
