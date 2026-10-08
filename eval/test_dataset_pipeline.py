"""Task 1 tests: verified prepared-dataset reuse (bit-exact, fail-closed).
  .venv/bin/python -m unittest eval.test_dataset_pipeline -v
"""
import random
import tempfile
import unittest
from pathlib import Path

import torch

from eval.prepared_dataset import (PreparedDatasetError, canonical_payload_bytes,
                                   dataset_digest, read_prepared, write_prepared)


def _rows(seed=0):
    """Two 7-field rows (inp,pres,events,wallL,wallR,weight,family_key) from one
    family — freshly allocated tensors each call (equal values)."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for k in range(2):
        n = 40 + k
        rows.append((torch.rand(n, 14, generator=g),
                     torch.zeros(n, 2), torch.zeros(0),
                     torch.zeros(n, dtype=torch.bool),
                     torch.zeros(n, dtype=torch.bool), 0.5, "famA"))
    return rows


class TestPreparedDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.ident = {"schema": 1, "manifest_snapshot_sha256": "abc",
                      "inventory": {"n": 2}, "consumed_features": {},
                      "code_hashes": {"groom.py": "h"}}

    def test_canonical_bytes_equal_for_separately_allocated_equal_tensors(self):
        a, b = _rows(0), _rows(0)                 # independent allocations, equal
        self.assertIsNot(a[0][0], b[0][0])
        st = random.Random(0).getstate()
        self.assertEqual(canonical_payload_bytes((a, None, st)),
                         canonical_payload_bytes((b, None, st)))

    def test_roundtrip_digest_and_bytes(self):
        tr, val = _rows(1), _rows(2)
        rng = random.Random(0)
        rng.shuffle(tr)
        state = rng.getstate()
        write_prepared(self.tmp, tr, val, state, self.ident)
        a, b, restored = read_prepared(self.tmp, self.ident)
        self.assertEqual(dataset_digest(a, b, restored), dataset_digest(tr, val, state))
        self.assertEqual(canonical_payload_bytes((a, b, restored)),
                         canonical_payload_bytes((tr, val, state)))
        # bit-exact tensor recovery
        self.assertTrue(torch.equal(a[0][0], tr[0][0]))

    def test_identity_mismatch_rejected(self):
        write_prepared(self.tmp, _rows(1), _rows(2), random.Random(0).getstate(),
                       self.ident)
        stale = dict(self.ident, manifest_snapshot_sha256="different")
        with self.assertRaises(PreparedDatasetError):
            read_prepared(self.tmp, stale)

    def test_absent_marker_rejected(self):
        with self.assertRaises(PreparedDatasetError):
            read_prepared(self.tmp, self.ident)          # nothing written

    def test_corrupt_payload_rejected(self):
        write_prepared(self.tmp, _rows(1), _rows(2), random.Random(0).getstate(),
                       self.ident)
        (self.tmp / "dataset.pt").write_bytes(b"garbage")
        with self.assertRaises(Exception):               # load or checksum fails
            read_prepared(self.tmp, self.ident)

    def test_checksum_mismatch_rejected(self):
        write_prepared(self.tmp, _rows(1), _rows(2), random.Random(0).getstate(),
                       self.ident)
        # tamper the payload with a DIFFERENT valid torch archive
        torch.save((_rows(9), _rows(8), random.Random(1).getstate()),
                   self.tmp / "dataset.pt")
        with self.assertRaises(PreparedDatasetError):
            read_prepared(self.tmp, self.ident)


class TestOrderedExecutorSeam(unittest.TestCase):
    """The map_executor seam must produce a dataset byte-identical to serial,
    regardless of worker completion order."""

    def setUp(self):
        import groom
        self.groom = groom
        self.tmp = Path(tempfile.mkdtemp())
        self.mapdir = self.tmp / "maps"
        self.mapdir.mkdir()
        for name in ("a", "b", "c"):     # non-lexicographic-friendly names
            (self.mapdir / name).mkdir()
        self._orig = (groom.load_map_all, groom.held_out)

        def fake_load(d, *, feature_provider=None):
            g = torch.Generator().manual_seed(abs(hash(d.name)) % (2 ** 31))
            return {"ExpertPlus": (torch.rand(20, groom.IN_CH, generator=g),
                                   torch.zeros(20, 2), torch.zeros(0),
                                   torch.zeros(20, dtype=torch.bool),
                                   torch.zeros(20, dtype=torch.bool))}
        groom.load_map_all = fake_load
        groom.held_out = lambda d: False

    def tearDown(self):
        self.groom.load_map_all, self.groom.held_out = self._orig

    def test_reversed_completion_matches_serial(self):
        serial = self.groom.load_dataset([self.mapdir])

        def rev_executor(dirs):
            dirs = list(dirs)
            done = {d: self.groom.load_map_all(d) for d in reversed(dirs)}
            return [done[d] for d in dirs]        # returned in INPUT order

        par = self.groom.load_dataset([self.mapdir], map_executor=rev_executor)
        self.assertEqual(len(serial), len(par))
        for a, b in zip(serial, par):
            self.assertTrue(torch.equal(a[0], b[0]))   # inp bit-identical
            self.assertEqual(a[5], b[5])               # weight identical


class _FakeFut:
    def __init__(self, v):
        self._v = v

    def result(self):
        return self._v


class _FakeExecutor:
    """Runs submitted work in-process, in submit order (deterministic tests)."""
    def submit(self, fn, arg):
        return _FakeFut(fn(arg))

    def shutdown(self):
        pass


class TestPoolInternals(unittest.TestCase):
    def test_provider_delta_beats_snapshot_and_miss(self):
        import groom
        import eval.map_load_pool as pool
        tmp = Path(tempfile.mkdtemp())
        song = tmp / "song.egg"
        song.write_bytes(b"audio-bytes")
        key = groom._feat_key(str(song), [0.0, 10.0, 20.0])
        pool._WCACHE = {key: torch.zeros(3, groom.N_AUDIO)}
        prov = pool._provider_factory({})
        self.assertEqual(len(prov(str(song), [0.0, 10.0, 20.0])), 3)
        with self.assertRaises(groom.FeatureMiss):
            prov(str(song), [0.0, 10.0])                    # wrong length -> miss
        # delta overrides the stale snapshot
        prov2 = pool._provider_factory({key: torch.ones(3, groom.N_AUDIO)})
        self.assertEqual(float(prov2(str(song), [0.0, 10.0, 20.0]).sum()),
                         3 * groom.N_AUDIO)
        pool._WCACHE = {}

    def test_auto_workers_bounds(self):
        from eval.map_load_pool import auto_workers
        self.assertEqual(auto_workers(0), 1)
        self.assertLessEqual(auto_workers(1000), 8)
        self.assertEqual(auto_workers(2), min(2, auto_workers(1000)))


class TestPrewarmOrchestration(unittest.TestCase):
    """Dedup + canonical order + parent-hit-on-stale-miss + retry, with an
    in-process fake executor and mocked parse/feature so we can count exactly."""

    def setUp(self):
        import groom
        import eval.map_load_pool as pool
        self.groom, self.pool_mod = groom, pool
        self._orig = (groom.load_map_all, groom.audio_features, groom._feat_key)
        self.GRID = [0.0, 10.0, 20.0]
        self.calls = {"feat": 0}
        # deterministic key per (audio, len): so aliases share a key
        groom._feat_key = lambda p, st: f"K:{p}:{len(st)}"

        def fake_feat(path, st):
            self.calls["feat"] += 1
            t = torch.full((len(st), groom.N_AUDIO), float(hash(path) % 7 + 1))
            return t
        groom.audio_features = fake_feat

        # d1->audioA, d2->audioB (two unique keys), d3-> no audio (ok);
        # d4-> audioA again (val, stale worker)
        self.aud = {"d1": "audioA", "d2": "audioB", "d4": "audioA"}

        def fake_lma(d, *, feature_provider=None):
            name = Path(d).name
            if name in self.aud:
                feats = feature_provider(self.aud[name], self.GRID)  # may raise
                return {"ExpertPlus": ("S", name, float(feats[0, 0]))}
            return {"ExpertPlus": ("S", name)}
        groom.load_map_all = fake_lma

    def tearDown(self):
        (self.groom.load_map_all, self.groom.audio_features,
         self.groom._feat_key) = self._orig

    def _pool(self):
        p = self.pool_mod.OrderedMapPool(2, "/nonexistent-cache.pt")
        p._ex = _FakeExecutor()
        p._parent = {}
        return p

    def test_dedup_order_parent_hit_and_retry(self):
        p = self._pool()
        train = p.map([Path(x) for x in ("d1", "d2", "d3")])
        val = p.map([Path(x) for x in ("d4",)])
        # two unique keys computed once each; val's audioA is a parent hit
        self.assertEqual(self.calls["feat"], 2)
        self.assertEqual(p.stats["unique_features_computed"], 2)
        self.assertEqual(p.stats["parent_feature_calls"], 0)
        self.assertEqual(p.stats["parent_hits_on_stale_miss"], 1)
        # order preserved + slots filled (d1,d2 via retry, d3 direct)
        self.assertEqual([s["ExpertPlus"][1] for s in train], ["d1", "d2", "d3"])
        self.assertEqual(val[0]["ExpertPlus"][1], "d4")


class TestPoolPersistence(unittest.TestCase):
    def _pool(self, cache):
        import eval.map_load_pool as pool
        p = pool.OrderedMapPool(2, cache)
        p._ex = _FakeExecutor()
        return p

    def test_publish_once_and_sync_feats(self):
        import groom
        import torch
        cache = Path(tempfile.mkdtemp()) / "feats.pt"
        p = self._pool(cache)
        p._parent = {"K": torch.ones(3, groom.N_AUDIO)}
        p._dirty = True
        groom._FEATS.clear()
        p.close(publish=True)
        self.assertTrue(cache.exists())
        self.assertEqual(p.stats["cache_writes"], 1)
        self.assertIn("K", torch.load(cache))
        self.assertIn("K", groom._FEATS)             # parent memory synced
        groom._FEATS.clear()

    def test_no_write_when_clean(self):
        cache = Path(tempfile.mkdtemp()) / "feats.pt"
        p = self._pool(cache)
        p._dirty = False
        p.close(publish=True)
        self.assertFalse(cache.exists())
        self.assertEqual(p.stats["cache_writes"], 0)

    def test_no_publish_on_exception_exit(self):
        import torch
        cache = Path(tempfile.mkdtemp()) / "feats.pt"
        p = self._pool(cache)
        p._parent = {"K": torch.ones(1, 4)}
        p._dirty = True
        p.__exit__(ValueError, ValueError("boom"), None)   # failure -> no publish
        self.assertFalse(cache.exists())


class TestClassify(unittest.TestCase):
    ALL = {"cache": True, "pool": True, "checkpoint_cache": True,
           "checkpoint_pool": True}

    def _c(self, exact, repeat=True, sp=12.0, pp=3.0, live=12.0, cached=0.1,
           orig=24.0, opt=3.1, cache_only=12.1):
        from eval.pipeline_speed import classify
        return classify(exact, repeat, sp, pp, live, cached, orig, opt, cache_only)

    def test_accept_cache_and_pool(self):
        self.assertEqual(self._c(self.ALL), "CACHE_AND_POOL_ACCEPTED")

    def test_not_repeatable_inconclusive(self):
        self.assertEqual(self._c(self.ALL, repeat=False), "INCONCLUSIVE")

    def test_inexact_cache_not_accepted(self):
        self.assertEqual(self._c(dict(self.ALL, cache=False)), "NOT_ACCEPTED")

    def test_pool_inexact_falls_back_cache_only(self):
        self.assertEqual(self._c(dict(self.ALL, checkpoint_pool=False)),
                         "CACHE_ONLY_ACCEPTED")

    def test_pool_slow_falls_back_cache_only(self):
        self.assertEqual(self._c(self.ALL, pp=10.0, opt=10.1),  # 1.2x < 1.5
                         "CACHE_ONLY_ACCEPTED")

    def test_cache_reuse_too_slow_not_accepted(self):
        # cached load not <=25% of live, and pool also fails -> nothing accepted
        self.assertEqual(self._c(dict(self.ALL, checkpoint_pool=False),
                                 cached=9.0), "NOT_ACCEPTED")


if __name__ == "__main__":
    unittest.main()
