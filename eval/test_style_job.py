"""Q4 style vocabulary + conditioned-flow tests (synthetic fixtures only).
  .venv/bin/python -m unittest eval.test_style_job -v
"""
import unittest

import numpy as np
import torch

from style import (DESCRIPTORS, adjusted_rand, assign_style, fit_residualizer,
                   residualize, support_table, _kmeans, MIN_FAMS, SEP_SD)


class TestResidualization(unittest.TestCase):
    def test_linear_confound_removed(self):
        rng = np.random.RandomState(0)
        conf = rng.rand(200, 4)
        noise = rng.randn(200, 12) * 0.1
        desc = 3.0 * conf[:, [0]] + noise          # dim-wise confounded
        beta = fit_residualizer(desc, conf)
        resid = residualize(desc, conf, beta)
        for j in range(resid.shape[1]):
            r = np.corrcoef(resid[:, j], conf[:, 0])[0, 1]
            self.assertLess(abs(r), 0.1, f"dim {j} still confounded")

    def test_residualizer_is_train_only_deterministic(self):
        rng = np.random.RandomState(1)
        conf, desc = rng.rand(50, 4), rng.rand(50, 12)
        b1 = fit_residualizer(desc, conf)
        b2 = fit_residualizer(desc.copy(), conf.copy())
        self.assertTrue(np.allclose(b1, b2))


class TestClustering(unittest.TestCase):
    def _blobs(self):
        rng = np.random.RandomState(2)
        a = rng.randn(60, 12) * 0.2 + np.r_[np.ones(3) * 3, np.zeros(9)]
        b = rng.randn(60, 12) * 0.2 - np.r_[np.ones(3) * 3, np.zeros(9)]
        tiny = rng.randn(5, 12) * 0.2 + np.r_[np.zeros(9), np.ones(3) * 6]
        return np.vstack([a, b, tiny])

    def test_deterministic_fixed_seed(self):
        z = self._blobs()
        k1, k2 = _kmeans(z, 4), _kmeans(z, 4)
        self.assertTrue(np.allclose(k1.cluster_centers_, k2.cluster_centers_))
        self.assertTrue((k1.labels_ == k2.labels_).all())

    def test_support_rules(self):
        z = self._blobs()
        km = _kmeans(z, 4)
        sup = support_table(z, km.labels_, 4)
        exposed = [c for c, s in sup.items() if s["exposed"]]
        for c in exposed:
            self.assertGreaterEqual(sup[c]["n"], MIN_FAMS)
            self.assertGreaterEqual(len(sup[c]["sep_dims"]), 2)
        # the 5-family blob can never be exposed
        small = [c for c, s in sup.items() if 0 < s["n"] < MIN_FAMS]
        for c in small:
            self.assertFalse(sup[c]["exposed"])

    def test_adjusted_rand_bounds(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(adjusted_rand(a, a), 1.0)
        self.assertLess(adjusted_rand(a, np.array([0, 1, 0, 1, 0, 1])), 0.5)


class TestAssignment(unittest.TestCase):
    def test_neutral_when_no_styles(self):
        vocab = {"styles": {}, "beta": [[0] * 12] * 5,
                 "mu": [0] * 12, "sd": [1] * 12}
        self.assertEqual(assign_style(np.zeros(12), np.zeros(4), vocab),
                         "neutral")

    def test_numeric_inputs_only(self):
        """Assignment consumes descriptor + confound vectors, nothing else —
        no artist/title/genre channel exists."""
        vocab = {"styles": {"s1": {"centroid": [1.0] * 12},
                            "s2": {"centroid": [-1.0] * 12}},
                 "beta": [[0.0] * 12] * 5, "mu": [0.0] * 12, "sd": [1.0] * 12}
        near_s1 = assign_style(np.ones(12) * 0.9, np.zeros(4), vocab)
        self.assertEqual(near_s1, "s1")
        near_s2 = assign_style(-np.ones(12) * 0.9, np.zeros(4), vocab)
        self.assertEqual(near_s2, "s2")

    def test_descriptor_names_have_no_metadata(self):
        for banned in ("genre", "artist", "title", "song", "mapper"):
            self.assertFalse(any(banned in d for d in DESCRIPTORS), banned)


class TestCondFlow(unittest.TestCase):
    def _events(self):
        # (step, hand, dir, col, layer, chain)
        return [(0, 0, 1, 1, 0, 1), (2, 1, 0, 2, 0, 1), (4, 0, 1, 0, 0, 1),
                (4, 1, 0, 3, 0, 1), (8, 0, 1, 1, 1, 1), (12, 1, 0, 2, 1, 1)]

    def test_future_geometry_never_enters_conditions(self):
        """Upcoming human geometry is a target, never input: changing a
        future event's dir/col/layer leaves every condition row and every
        earlier base token unchanged."""
        from cond_flow import cond_block
        from groom import events_to_xy
        ev1 = self._events()
        ev2 = [e if e[0] < 8 else (e[0], e[1], 8, 3, 2, 1) for e in ev1]
        sh = [(e[0], e[1]) for e in ev1]
        times = [e[0] * 125.0 for e in ev1]
        intent = torch.zeros(1, 12)
        work = torch.zeros(10)
        c1 = cond_block(sh, times, intent, work)
        c2 = cond_block([(e[0], e[1]) for e in ev2], times, intent, work)
        self.assertTrue(torch.equal(c1, c2))
        wl = wr = [False] * 16
        x1, _ = events_to_xy(ev1, wl, wr)
        x2, _ = events_to_xy(ev2, wl, wr)
        self.assertTrue(torch.equal(x1[:4], x2[:4]))   # rows before change

    def test_bpm_reencoding_invariant_gaps(self):
        """Doubled step indices at halved step_ms = the same real times ->
        identical gap buckets."""
        from cond_flow import cond_block, N_GAP
        ev = self._events()
        sh1 = [(e[0], e[1]) for e in ev]
        sh2 = [(e[0] * 2, e[1]) for e in ev]
        times = [e[0] * 125.0 for e in ev]              # identical ms
        intent, work = torch.zeros(1, 12), torch.zeros(10)
        c1 = cond_block(sh1, times, intent, work)
        c2 = cond_block(sh2, times, intent, work)
        self.assertTrue(torch.equal(c1[:, :3 * N_GAP], c2[:, :3 * N_GAP]))

    def test_shim_matches_direct_forward_and_schema_guard(self):
        from cond_flow import (CondFlowShim, ConditionedFlow, NCOND)
        from groom import NTOK
        torch.manual_seed(0)
        m = ConditionedFlow()
        m.eval()
        x = torch.rand(1, 5, NTOK)
        cond = torch.rand(5, NCOND)
        shim = CondFlowShim(m, cond)
        with torch.no_grad():
            h1 = shim.hidden(x)
            h2 = m.hidden(torch.cat([x, cond[None]], -1),
                          torch.zeros(1, dtype=torch.long))
        self.assertTrue(torch.allclose(h1, h2))
        with self.assertRaises(AssertionError):        # schedule too short
            shim.hidden(torch.rand(1, 6, NTOK))

    def test_style_dropout_is_neutral_noop_at_k1(self):
        from cond_flow import ConditionedFlow, NCOND
        from groom import NTOK
        torch.manual_seed(1)
        m = ConditionedFlow(n_styles=1)
        m.eval()
        x = torch.rand(2, 4, NTOK + NCOND)
        d_oh = torch.zeros(2, 4, 9)
        c_oh = torch.zeros(2, 4, 7)
        with torch.no_grad():
            a = m(x, d_oh, c_oh, None)
            b = m(x, d_oh, c_oh, torch.zeros(2, dtype=torch.long))
        for ta, tb in zip(a, b):
            self.assertTrue(torch.equal(ta, tb))

    def test_expected_risk_orders_fast_jumps_above_stays(self):
        from cond_flow import expected_risk, DCOL, DLAY
        aux = torch.tensor([[[100.0, 1.0, 0.0, -1.0, 0.0]]])  # dt=100ms
        far = torch.full((1, 1, DCOL), -1e9)
        far[..., 6] = 10.0                                    # dcol=+3 mass
        stay = torch.full((1, 1, DCOL), -1e9)
        stay[..., 3] = 10.0                                   # dcol=0 mass
        lay = torch.zeros(1, 1, DLAY)
        self.assertGreater(float(expected_risk(far, lay, aux)),
                           float(expected_risk(stay, lay, aux)))


class TestJobContext(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.out = self._td.name
        self.addCleanup(self._td.cleanup)
        from pathlib import Path
        self.songs = []
        for i, content in enumerate((b"audio-one", b"audio-two",
                                     b"audio-three")):
            p = Path(self.out) / f"song{i}.egg"
            p.write_bytes(content)
            self.songs.append(str(p))

    def test_stable_audio_hashing_not_process_hash(self):
        from pathlib import Path
        from job_context import audio_key, stable_seed
        dup = Path(self.out) / "copy.egg"
        dup.write_bytes(b"audio-one")
        self.assertEqual(audio_key(self.songs[0]), audio_key(dup))
        self.assertEqual(stable_seed(audio_key(self.songs[0])),
                         stable_seed(audio_key(dup)))

    def test_explicit_seed_binds_then_conflicts(self):
        from job_context import JobManifest
        m = JobManifest(self.out)
        r = m.resolve(self.songs[0], explicit_seed=42)
        self.assertEqual(r["seed"], 42)
        self.assertEqual(m.resolve(self.songs[0])["seed"], 42)   # retry
        with self.assertRaises(ValueError):
            m.resolve(self.songs[0], explicit_seed=7)

    def test_restart_append_reorder(self):
        from job_context import JobManifest
        m1 = JobManifest(self.out)
        first = {s: m1.resolve(s) for s in self.songs[:2]}
        m2 = JobManifest(self.out)                     # restart
        for s in reversed(self.songs):                 # reorder + append
            r = m2.resolve(s)
            if s in first:
                self.assertEqual(r, dict(first[s], key=r["key"]))

    def test_multiple_difficulties_extend(self):
        from job_context import JobManifest
        m = JobManifest(self.out)
        m.resolve(self.songs[0], difficulties=("Expert",))
        r = m.resolve(self.songs[0], difficulties=("ExpertPlus",))
        self.assertEqual(r["difficulties"], ["Expert", "ExpertPlus"])

    def test_two_concurrent_creators_same_content_same_binding(self):
        import tempfile
        from job_context import JobManifest
        with tempfile.TemporaryDirectory() as other:
            a = JobManifest(self.out).resolve(self.songs[0])
            b = JobManifest(other).resolve(self.songs[0])
            self.assertEqual((a["seed"], a["style"]), (b["seed"], b["style"]))

    def test_zip_only_cleanup_keeps_manifest(self):
        from pathlib import Path
        from job_context import JobManifest, MANIFEST_NAME
        m = JobManifest(self.out)
        m.resolve(self.songs[0])
        mapdir = Path(self.out) / "song0 (mapped)"
        mapdir.mkdir()
        (mapdir / "Info.dat").write_text("{}")
        (Path(self.out) / "song0.zip").write_bytes(b"zipzip")
        removed = m.cleanup_zip_only(self.out)
        self.assertEqual(removed, ["song0 (mapped)"])
        self.assertTrue((Path(self.out) / MANIFEST_NAME).exists())
        self.assertTrue((Path(self.out) / "song0.zip").exists())
        m2 = JobManifest(self.out)                     # rebind survives
        self.assertEqual(m2.resolve(self.songs[0])["seed"],
                         m.resolve(self.songs[0])["seed"])


if __name__ == "__main__":
    unittest.main()
