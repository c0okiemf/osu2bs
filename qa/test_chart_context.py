"""Chart-latent Task 2 tests: identity-free chart context.
  .venv/bin/python -m unittest qa.test_chart_context -q
"""
import unittest

import torch

from qa.chart_context import (FIELD_ORDER, ContextRecord, chart_context,
                              fit_context_scaler, schema_sha, standardize)


def audio_fixture(dur=10):
    return {"supported": True, "duration_s": dur,
            "per_second": {"intensity_rms": [0.2] * dur,
                           "harmonic_rms": [0.3] * dur,
                           "percussive_rms": [0.1] * dur}}


def scene_fixture(notes=None):
    return {"scope": None, "chart_sha256": "aa" * 32,
            "info_sha256": "bb" * 32, "bombs": [], "walls": [],
            "settings": {"njs": 18},
            "notes": notes if notes is not None else [
                (1.0, 1, 0, 0, 1), (1.0, 2, 0, 1, 1),
                (2.0, 0, 1, 0, 0), (5.0, 3, 2, 1, 8)]}


def pair_free_scene():
    return scene_fixture(notes=[(1.0, 1, 0, 0, 1), (2.0, 0, 1, 0, 0),
                                (3.0, 2, 0, 1, 4), (5.0, 3, 2, 1, 8)])


def f(rec, name):
    return rec.fields[name]


class TestObservables(unittest.TestCase):
    def test_same_hand_followers_group_to_one_head(self):
        sc = scene_fixture(notes=[(1.0, 1, 0, 0, 1), (1.0, 2, 0, 0, 1),
                                  (3.0, 0, 0, 1, 0)])
        rec = chart_context(sc, audio_fixture())
        self.assertEqual(rec.audit["n_heads"], 2)

    def test_anchored_coincidence(self):
        sc = scene_fixture(notes=[(1.0, 1, 0, 0, 1), (1.0005, 2, 0, 1, 1),
                                  (3.0, 0, 0, 0, 0), (3.5, 3, 0, 1, 0)])
        rec = chart_context(sc, audio_fixture())
        # occupied {1.0, 1.0005, 3.0, 3.5}: first two coincident
        self.assertAlmostEqual(f(rec, "coincidence_mean") * 10 / 8,
                               2 / 4, places=6)

    def test_partial_final_window_is_duration_weighted(self):
        rec = chart_context(scene_fixture(), audio_fixture(dur=10))
        # 4 heads in w0 (8s), none in w1 (2s): mean = (0.5*8 + 0*2)/10
        self.assertAlmostEqual(f(rec, "head_rate_mean"), 0.4, places=6)
        self.assertAlmostEqual(f(rec, "head_rate_p90"), 0.5, places=6)

    def test_transitions_over_long_rests_excluded(self):
        rec = chart_context(scene_fixture(), audio_fixture())
        # L: 1.0->2.0 (dx |0-1|/3, dy |1-0|/2); R: 1.0->5.0 gap>2s excluded
        self.assertAlmostEqual(f(rec, "dx_mean"), (1 / 3) * 0.8, places=6)
        self.assertAlmostEqual(f(rec, "dy_mean"), 0.5 * 0.8, places=6)

    def test_long_gap_pairs_attribute_to_later_window(self):
        sc = scene_fixture(notes=[(7.9, 1, 0, 0, 1), (8.4, 2, 0, 1, 1)])
        rec = chart_context(sc, audio_fixture(dur=10))
        # pair (7.9 -> 8.4): 0.5s NOT > 0.5 -> not long; later window w1
        self.assertEqual(f(rec, "long_gap_share_mean"), 0.0)
        sc2 = scene_fixture(notes=[(7.9, 1, 0, 0, 1), (8.6, 2, 0, 1, 1)])
        rec2 = chart_context(sc2, audio_fixture(dur=10))
        # 0.7s gap assigned to w1 (weight 2 of 10)
        self.assertAlmostEqual(f(rec2, "long_gap_share_mean"), 0.2,
                               places=6)

    def test_under_four_heads_is_zero_with_audit(self):
        sc = scene_fixture(notes=[(1.0, 1, 0, 0, 1), (2.0, 2, 0, 1, 1)])
        rec = chart_context(sc, audio_fixture())
        self.assertEqual(f(rec, "gram4_max_mean"), 0.0)
        self.assertEqual(rec.audit["windows_under4_heads"],
                         rec.audit["n_windows"])

    def test_no_audio_bit_and_zero_audio_fields(self):
        rec = chart_context(scene_fixture(), None)
        self.assertEqual(f(rec, "audio_available"), 0.0)
        self.assertEqual(f(rec, "audio_rms_rel_mean"), 0.0)
        self.assertEqual(f(rec, "harmonic_share_mean"), 0.0)

    def test_harmonic_share(self):
        rec = chart_context(scene_fixture(), audio_fixture())
        self.assertAlmostEqual(f(rec, "harmonic_share_mean"),
                               0.3 / (0.4 + 1e-6), places=4)


class TestIdentityFreedom(unittest.TestCase):
    def test_telemetry_and_metadata_cannot_change_context(self):
        sc, au = scene_fixture(), audio_fixture()
        original = chart_context(sc, au).tensor()
        sc["mapper"] = "untrusted-family-name"
        sc["player_accuracy"] = .99
        sc["observed_target_mean"] = [100.] * 10
        self.assertTrue(torch.equal(original,
                                    chart_context(sc, au).tensor()))

    def test_mirror_preserves_unsigned_context(self):
        from qa.train import mirror_scene
        sc, au = pair_free_scene(), audio_fixture()
        a = chart_context(sc, au).tensor()
        b = chart_context(mirror_scene(sc), au).tensor()
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        # paired chart: every non-sequential field still invariant
        sc2 = scene_fixture()
        a2 = chart_context(sc2, au)
        b2 = chart_context(mirror_scene(sc2), au)
        for i, name in enumerate(FIELD_ORDER):
            if name.startswith("gram4_max"):
                continue
            self.assertAlmostEqual(a2.tensor()[i].item(),
                                   b2.tensor()[i].item(), places=6,
                                   msg=name)

    def test_schema_and_tensor_shape(self):
        rec = chart_context(scene_fixture(), audio_fixture())
        self.assertEqual(len(rec.tensor()), 21)
        self.assertEqual(rec.schema_sha256, schema_sha())
        self.assertTrue(torch.isfinite(rec.tensor()).all())


class TestScaler(unittest.TestCase):
    def test_floor_and_clip_diagnostics(self):
        rows = [torch.zeros(21), torch.zeros(21)]
        rows[0][0], rows[1][0] = 1.0, 1.0        # constant dim -> floor
        sc = fit_context_scaler(rows)
        self.assertGreaterEqual(float(sc["sd"][0]), 1e-3)
        z, clipped = standardize(torch.full((21,), 100.0), sc)
        self.assertTrue((z <= 4.0).all())
        self.assertIn(FIELD_ORDER[1], clipped)

    def test_scaler_accepts_context_records(self):
        recs = [chart_context(scene_fixture(), audio_fixture()),
                chart_context(pair_free_scene(), audio_fixture())]
        sc = fit_context_scaler(recs)
        self.assertEqual(sc["n_families"], 2)


if __name__ == "__main__":
    unittest.main()
