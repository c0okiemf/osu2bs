"""Task 1 tests: safe experiment output + rhythm-model injection. No GPU
training inside unit tests (a 1-epoch/1-step CPU smoke only).
  .venv/bin/python -m unittest eval.test_clean_rhythm -v
"""
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

import convert
import groom
from eval.clean_rhythm import guard_checkpoint_destination


class TestCheckpointGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run = self.tmp / "run"
        self.run.mkdir()

    def test_protected_path_rejected_without_write(self):
        protected = self.tmp / "groom.pt"
        protected.write_bytes(b"production-sentinel")
        before = protected.read_bytes()
        with self.assertRaises(ValueError):
            guard_checkpoint_destination(protected, self.run, [protected])
        self.assertEqual(protected.read_bytes(), before)

    def test_symlink_to_protected_rejected(self):
        protected = self.tmp / "groom.pt"
        protected.write_bytes(b"x")
        link = self.run / "ck.pt"
        link.symlink_to(protected)
        with self.assertRaises(ValueError):
            guard_checkpoint_destination(link, self.run, [protected])

    def test_hardlink_to_protected_rejected(self):
        protected = self.tmp / "groom.pt"
        protected.write_bytes(b"x")
        link = self.run / "ck.pt"
        os_link = link
        import os
        os.link(protected, os_link)
        with self.assertRaises(ValueError):
            guard_checkpoint_destination(link, self.run, [protected])

    def test_outside_run_rejected(self):
        with self.assertRaises(ValueError):
            guard_checkpoint_destination(self.tmp / "elsewhere.pt", self.run,
                                         [self.tmp / "groom.pt"])

    def test_existing_file_rejected(self):
        p = self.run / "ck.pt"
        p.write_bytes(b"already")
        with self.assertRaises(ValueError):
            guard_checkpoint_destination(p, self.run, [])

    def test_fresh_path_ok(self):
        rp = guard_checkpoint_destination(self.run / "ck.pt", self.run, [])
        self.assertEqual(rp, (self.run / "ck.pt").resolve())


def _sample(n=600):
    inp = torch.zeros(n, groom.IN_CH)
    inp[::8, 0] = 1.0
    pres = torch.zeros(n, 2)
    pres[::8, 0] = 1.0
    wl = torch.zeros(n, dtype=torch.bool)
    wr = torch.zeros(n, dtype=torch.bool)
    return (inp, pres, torch.zeros(0), wl, wr, 1.0)


class TestTinyTrainer(unittest.TestCase):
    def test_cpu_smoke_writes_only_experiment_path(self):
        tmp = Path(tempfile.mkdtemp())
        protected = tmp / "groom.pt"
        protected.write_bytes(b"production-sentinel")
        before = protected.read_bytes()
        out = tmp / "run" / "ck.pt"
        hist = tmp / "run" / "history.json"
        tr, val = [_sample()], [_sample()]
        h = groom.train_rhythm(tr, val, "cpu", random.Random(0), out_path=out,
                               history_path=hist, max_epochs=1, steps_per_epoch=1)
        self.assertTrue(out.exists())
        # loadable raw Groomer state
        m = groom.Groomer()
        m.load_state_dict(torch.load(out, map_location="cpu"))
        self.assertIn("initial_val_loss", h)
        self.assertEqual(len(h["epochs"]), 1)
        self.assertEqual(h["updates"], 1)
        self.assertTrue(all(k in h["epochs"][0]
                            for k in ("train_loss", "val_loss", "note_f1")))
        self.assertTrue(json.loads(hist.read_text())["epochs"])
        self.assertEqual(protected.read_bytes(), before)   # shipped untouched


class _Stop(Exception):
    pass


class TestModelInjection(unittest.TestCase):
    def setUp(self):
        from timing import TimeGrid
        self.grid = TimeGrid.uniform(10.0, 0.0, 200)
        self._orig_gs = convert.grid_steps
        self._orig_gn = groom.groom_notes
        convert.grid_steps = lambda *a, **k: ({0, 4, 8}, 200, 10.0, 0.0, self.grid)

    def tearDown(self):
        convert.grid_steps = self._orig_gs
        groom.groom_notes = self._orig_gn

    def _run(self, **kw):
        recorded = []

        def fake_gn(steps, T, step_ms, offset, model=None, **k):
            recorded.append(model)
            raise _Stop()
        groom.groom_notes = fake_gn
        with self.assertRaises(_Stop):
            convert.convert_groomed([], 174.0, 0.0, collect=[], **kw)
        return recorded

    def test_injected_model_forwarded(self):
        sentinel = object()
        rec = self._run(rhythm_model=sentinel)
        self.assertEqual(rec, [sentinel])          # exact object, first call

    def test_omitted_injection_forwards_none(self):
        rec = self._run()
        self.assertEqual(rec, [None])


class TestPrepareTrainGuards(unittest.TestCase):
    def test_family_mass_preserved_after_crop_filter(self):
        # family "A" had base weight 1.0 across two tiers; one tier dropped by the
        # CROP filter -> the survivor must renorm back to the full 1.0, not stay 0.5
        from groom import _renorm_by_family
        surv = [(None, None, None, None, None, 0.5, "A")]
        out = _renorm_by_family(surv, {"A": 1.0})
        self.assertAlmostEqual(out[0][5], 1.0, places=9)

    def test_train_aborts_on_manifest_snapshot_tamper(self):
        import eval.clean_rhythm as cr
        tmp = Path(tempfile.mkdtemp())
        run = tmp / "run"
        run.mkdir()
        (run / "corpus_manifest.snapshot.json").write_text('{"version": 2}')
        rec = {"manifest_snapshot_sha256": "0" * 64,   # deliberately wrong
               "protected_digests": cr._protected_digests(),
               "feat_cache": str(run / "feats_cache.pt")}
        (run / "run.json").write_text(json.dumps(rec))
        with self.assertRaises((AssertionError, SystemExit)):
            cr.train(run)

    def test_prepared_dataset_branch_skips_all_loading(self):
        import eval.clean_rhythm as cr
        import groom
        from eval.prepared_dataset import write_prepared
        tmp = Path(tempfile.mkdtemp())
        run = tmp / "run"
        run.mkdir()
        snap = run / "corpus_manifest.snapshot.json"
        snap.write_text('{"version": 2}')
        (run / "feats_cache.pt").write_bytes(b"feat")
        rec = {"seed": cr.SEED, "manifest_snapshot_sha256": _sha256_file(snap),
               "code_hashes": cr._code_hashes(),
               "protected_digests": cr._protected_digests(),
               "feat_cache": str(run / "feats_cache.pt"),
               "inventory": {"n_train_charts": 2}}
        (run / "run.json").write_text(json.dumps(rec))
        tr, val = _sample_rows(), _sample_rows()
        write_prepared(run, tr, val, random.Random(cr.SEED).getstate(),
                       cr._dataset_identity(run, rec))
        # any live-loading path must NOT be touched in the cached branch
        orig = {n: getattr(groom, n) for n in
                ("_split", "load_dataset", "load_map_all", "cached_audio_features",
                 "train_rhythm")}

        def boom(*a, **k):
            raise AssertionError("live loader called in cached branch")
        captured = {}

        def fake_train(t, v, dev, rng, *, out_path, history_path, max_epochs,
                       steps_per_epoch):
            captured.update(n_tr=len(t), n_val=len(v), max_epochs=max_epochs,
                            steps=steps_per_epoch)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(groom.Groomer().state_dict(), out_path)
            Path(history_path).write_text('{}')
            return {"initial_val_loss": 1.0, "best_val_loss": 0.5,
                    "best_epoch": 0, "updates": 1}
        try:
            for n in ("_split", "load_dataset", "load_map_all",
                      "cached_audio_features"):
                setattr(groom, n, boom)
            groom.train_rhythm = fake_train
            cr.train(run, use_prepared_dataset=True)
        finally:
            for n, f in orig.items():
                setattr(groom, n, f)
        self.assertEqual(captured["n_tr"], len(tr))    # exact prepared rows
        self.assertEqual(captured["max_epochs"], 200)  # unchanged budget
        self.assertEqual(captured["steps"], 30)


def _sha256_file(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sample_rows():
    return [(torch.rand(40, groom.IN_CH), torch.zeros(40, 2), torch.zeros(0),
             torch.zeros(40, dtype=torch.bool), torch.zeros(40, dtype=torch.bool),
             0.5, "famA")]


if __name__ == "__main__":
    unittest.main()
