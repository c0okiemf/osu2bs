"""Task 1 tests: run identity, approved provenance, reservation exclusion,
stale-cache rejection, atomic resume.
  .venv/bin/python -m pytest eval/test_expressive_manifest.py -q   (or unittest)
"""
import json
import unittest
from pathlib import Path

from eval.expressive_manifest import (family_strata, freeze_run, load_run,
                                      partial_completed, reserve_families,
                                      save_partial)

APPROVED_DIR = "/home/x/beat-saber-map-gen/input/bytrius/b (Song - M)"
REJECTED_DIR = "/home/x/beat-saber-map-gen/input/excluded/r (Bad - M)"


def fixture_corpus(rep="general/a", approved_alias=None, rejected_alias=None,
                   drop_field=None):
    maps = [{"dir": f"/home/x/beatsaver/{rep}", "family": "family:a",
             "split": "train", "eligible": "ok", "family_rep": True}]
    if approved_alias:
        maps.append({"dir": APPROVED_DIR, "family": "family:a",
                     "split": "train", "eligible": "ok", "family_rep": False})
    if rejected_alias:
        maps.append({"dir": REJECTED_DIR, "family": "family:r",
                     "split": "train", "eligible": "ok", "family_rep": True})
    maps.append({"dir": "/home/x/beatsaver/zz", "family": "family:g",
                 "split": "train", "eligible": "ok", "family_rep": True})
    if drop_field:
        maps[0] = {k: v for k, v in maps[0].items() if k != drop_field}
    return {"maps": maps}


def fixture_config(seed=1):
    return {"seed": seed, "panel": ["s1", "s2"], "attempts": 6,
            "protected": {"flow.pt": "ab" * 32}}


def fixture_record(completed=True):
    return {"artifact_sha256": "cd" * 32, "completed": completed}


class TestStrata(unittest.TestCase):
    def test_family_approval_survives_representative_choice(self):
        c = fixture_corpus(rep="general/a", approved_alias="approved/b")
        s = family_strata(c)
        self.assertEqual(s["approved"], ["family:a"])
        self.assertNotIn("family:a", s["general"])

    def test_plain_families_are_general_not_negative(self):
        s = family_strata(fixture_corpus())
        self.assertIn("family:a", s["general"])
        self.assertIn("family:g", s["general"])
        self.assertEqual(s["rejected"], [])

    def test_explicit_rejection_has_provenance(self):
        s = family_strata(fixture_corpus(rejected_alias=True))
        self.assertEqual(s["rejected"], ["family:r"])
        self.assertNotIn("family:r", s["general"])
        self.assertNotIn("family:r", s["approved"])

    def test_minimal_fixture_cannot_bypass_validation(self):
        with self.assertRaises(ValueError):
            family_strata(fixture_corpus(drop_field="eligible"))
        with self.assertRaises(ValueError):
            family_strata({"maps": []})


class TestRunIdentity(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.addCleanup(self._td.cleanup)

    def test_freeze_and_reload(self):
        r1 = freeze_run(self.root, fixture_config(seed=1))
        r2 = load_run(self.root)
        self.assertEqual(r1["identity"], r2["identity"])

    def test_partial_identity_cannot_be_reused(self):
        freeze_run(self.root, fixture_config(seed=1))
        save_partial(self.root, "song:seed0", fixture_record())
        with self.assertRaisesRegex(ValueError, "identity"):
            freeze_run(self.root, fixture_config(seed=2))

    def test_refreeze_same_config_is_idempotent(self):
        a = freeze_run(self.root, fixture_config(seed=1))
        b = freeze_run(self.root, fixture_config(seed=1))
        self.assertEqual(a["identity"], b["identity"])

    def test_completed_partial_never_recomputed(self):
        freeze_run(self.root, fixture_config())
        self.assertFalse(partial_completed(self.root, "song:seed0"))
        save_partial(self.root, "song:seed0", fixture_record(completed=True))
        self.assertTrue(partial_completed(self.root, "song:seed0"))
        with self.assertRaisesRegex(ValueError, "completed"):
            save_partial(self.root, "song:seed0",
                         {"artifact_sha256": "ee" * 32, "completed": True})

    def test_incomplete_partial_can_resume(self):
        freeze_run(self.root, fixture_config())
        save_partial(self.root, "k", fixture_record(completed=False))
        self.assertFalse(partial_completed(self.root, "k"))
        save_partial(self.root, "k", fixture_record(completed=True))
        self.assertTrue(partial_completed(self.root, "k"))

    def test_partial_carries_run_identity(self):
        run = freeze_run(self.root, fixture_config())
        save_partial(self.root, "k", fixture_record())
        rec = json.loads((self.root / "partial" / "k.json").read_text())
        self.assertEqual(rec["config_sha"], run["identity"])


class TestReservation(unittest.TestCase):
    def _fams(self, n, split="test"):
        return [{"dir": f"/x/{i}", "family": f"fam:{i:03d}", "split": split,
                 "eligible": "ok", "family_rep": True} for i in range(n)]

    def test_reserved_excludes_prior_panels_and_is_deterministic(self):
        corpus = {"maps": self._fams(40)}
        prior = {f"fam:{i:03d}" for i in range(10)}
        a = reserve_families(corpus, exclude=prior, n=12, salt="20260923")
        b = reserve_families(corpus, exclude=prior, n=12, salt="20260923")
        self.assertEqual(a["families"], b["families"])
        self.assertEqual(len(a["families"]), 12)
        self.assertFalse(set(a["families"]) & prior)
        self.assertEqual(len(a["playtest"]), 8)
        self.assertEqual(a["playtest"], a["families"][:8])

    def test_shortage_is_reported_not_padded(self):
        corpus = {"maps": self._fams(14)}
        prior = {f"fam:{i:03d}" for i in range(6)}
        r = reserve_families(corpus, exclude=prior, n=12, salt="20260923")
        self.assertEqual(r["status"], "SHORTAGE")
        self.assertEqual(r["available"], 8)


if __name__ == "__main__":
    unittest.main()
