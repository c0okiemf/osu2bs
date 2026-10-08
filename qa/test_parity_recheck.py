"""Task 4 tests (feature-parity repair): frozen correction run.
  .venv/bin/python -m unittest qa.test_parity_recheck -q
"""
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from qa import parity_recheck as pr


def forbidden_call(*a, **k):
    raise AssertionError("_fit must not be called for an existing fold")


def _fold_artifact(path, fold=0):
    art = {"state": {"w": torch.zeros(1)}, "scaler": None, "updates": 500,
           "seed": pr.SEED + fold, "held": ["fam:x"],
           "removed_players": [], "macro_nll_recorded": 0.0,
           "macro_nll_recomputed": 0.0}
    torch.save(art, path)
    return art


def _entry(path, fold=0, updates=500, sha=None):
    e = {"fold": fold, "path": str(path), "updates": updates,
         "held": ["fam:x"], "recorded_macro_nll": 0.0}
    if sha:
        e["sha256"] = sha
    return e


class TestFoldRecovery(unittest.TestCase):
    def test_existing_fold_does_not_retrain(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fold0.pt"
            _fold_artifact(p)
            with unittest.mock.patch("qa.train._fit", forbidden_call):
                art = pr.load_or_recover_fold(_entry(p))
            self.assertEqual(art["updates"], 500)

    def test_wrong_artifact_hash_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fold0.pt"
            _fold_artifact(p)
            with self.assertRaisesRegex(ValueError, "hash"):
                pr.load_or_recover_fold(_entry(p, sha="00" * 32))

    def test_wrong_seed_or_updates_identity_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fold0.pt"
            _fold_artifact(p, fold=0)
            bad = _entry(p, fold=1)          # artifact seed is SEED+0
            with self.assertRaisesRegex(ValueError, "identity"):
                pr.load_or_recover_fold(bad)

    def test_recovery_cannot_reselect_updates(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "fold9.pt"
            with self.assertRaisesRegex(ValueError, "reselect"):
                pr.load_or_recover_fold(_entry(missing, updates=2000))


class TestMaskedSpans(unittest.TestCase):
    def test_masked_zero_is_not_a_reference_target(self):
        self.assertGreater(
            pr.observed_span([0., 2., 3.], [False, True, True]), 0)
        self.assertEqual(
            pr.observed_values([0., 2., 3.], [False, True, True]),
            [2., 3.])

    def test_all_masked_span_is_none_not_zero(self):
        self.assertIsNone(pr.observed_span([0., 0.], [False, False]))

    def test_exact_quantile_interpolation_pinned(self):
        span = pr.observed_span([1., 2., 3., 4.], [True] * 4)
        self.assertAlmostEqual(span, 3.97 - 1.03, places=6)


def _fam_result(pooled, equal, wide, n=100):
    cov, sharp = {}, {}
    for f in pr.INFLATION_GRID:
        cov[str(f)] = {"pooled": pooled, "equal_player": equal}
        sharp[str(f)] = {"wide_share": wide}
    return {"n_selected": n, "coverage": cov, "sharpness": sharp,
            "support": {"nn_ok": 0, "nn_insufficient": n}}


class TestDisposition(unittest.TestCase):
    def test_global_factor_needs_every_family_on_both_aggregations(self):
        res = {"a": _fam_result(0.9, 0.9, 0.1),
               "b": _fam_result(0.9, 0.84, 0.1)}   # equal-player fails
        self.assertIsNone(pr.choose_global_factor(res))
        d = pr.disposition(res)
        self.assertEqual(d["status"], "CALIBRATION_FAILED_V2")

    def test_missing_family_denominator_is_insufficient(self):
        res = {"a": _fam_result(0.9, 0.9, 0.1),
               "b": _fam_result(None, None, 0.0, n=0)}
        self.assertEqual(pr.disposition(res)["status"], "INSUFFICIENT")

    def test_insufficient_nn_support_does_not_block_predictive_verdict(self):
        res = {"a": _fam_result(0.9, 0.9, 0.1)}   # nn_ok == 0 everywhere
        d = pr.disposition(res)
        self.assertEqual(d["status"], "CALIBRATION_REPAIRED")
        self.assertEqual(d["factor"], 1.0)

    def test_sharpness_gates_each_family(self):
        res = {"a": _fam_result(0.9, 0.9, 0.25)}
        self.assertEqual(pr.disposition(res)["status"],
                         "CALIBRATION_FAILED_V2")


class TestSampleFreeze(unittest.TestCase):
    def test_altered_sample_on_resume_refused(self):
        import hashlib
        ids = ["fam:a:x:1:0", "fam:a:x:2:1"]
        sm = {"ids": ids, "sha256": hashlib.sha256(
            "\n".join(ids).encode()).hexdigest()}
        pr.verify_sample(sm)                     # intact -> fine
        with self.assertRaisesRegex(ValueError, "sample"):
            pr.verify_sample({"ids": ids + ["fam:a:x:3:0"],
                              "sha256": sm["sha256"]})


if __name__ == "__main__":
    unittest.main()
