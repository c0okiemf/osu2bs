"""Task 1 tests: frozen roles, access policy, safe identity storage.
  .venv/bin/python -m unittest qa.test_contract -q
"""
import json
import unittest
from pathlib import Path

from qa.contract import (Contract, EVALUATOR_MODULES, FORBIDDEN_IMPORTS,
                         atomic_record, canonical_bytes, load_contract,
                         open_role, sanitized_record, scan_imports)


def fixture_contract(tmp, frozen=False):
    root = Path(tmp)
    for role in ("qa_train", "qa_calib", "seal"):
        d = root / role
        d.mkdir(parents=True, exist_ok=True)
        rec = d / "r1.json"
        rec.write_text(json.dumps({"player_token": "t1", "ok": True}))
    c = {"version": 1, "frozen": frozen,
         "roles": {r: {"root": str(root / r),
                       "families": [f"fam:{r}"], "players": [f"tok-{r}"],
                       "files": {"r1.json": _sha_text(
                           json.dumps({"player_token": "t1", "ok": True}))}}
                   for r in ("qa_train", "qa_calib", "seal")}}
    p = root / "contract.json"
    p.write_text(json.dumps(c, indent=1))
    return p


def _sha_text(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


def bsor_fixture(player_name="identity-sentinel"):
    return {"info": {"score": 1, "modifiers": "", "height": 1.7},
            "identity": {"playerID": "76561198_secret",
                         "playerName": player_name, "platform": "steam"},
            "frames": [(0.0, 90, ((0, 1.6, 0), (0, 0, 0, 1)),
                        ((0, 1, 0), (0, 0, 0, 1)),
                        ((0, 1, 0), (0, 0, 0, 1)))],
            "notes": [], "score_id": 12345}


class TestAccessPolicy(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = self._td.name
        self.addCleanup(self._td.cleanup)

    def test_confirm_requires_frozen_identity(self):
        c = load_contract(fixture_contract(self.tmp, frozen=False))
        with self.assertRaises(PermissionError):
            open_role(c, "seal", "confirm")

    def test_frozen_confirm_allowed(self):
        c = load_contract(fixture_contract(self.tmp, frozen=True))
        recs = open_role(c, "seal", "confirm")
        self.assertEqual(len(recs), 1)

    def test_role_purpose_matrix(self):
        c = load_contract(fixture_contract(self.tmp, frozen=True))
        open_role(c, "qa_train", "fit")
        open_role(c, "qa_train", "reference")
        open_role(c, "qa_calib", "calibrate")
        for role, purpose in (("qa_calib", "fit"), ("qa_calib", "reference"),
                              ("seal", "fit"), ("qa_train", "confirm"),
                              ("gen", "fit")):
            with self.assertRaises(PermissionError, msg=(role, purpose)):
                open_role(c, role, purpose)

    def test_tampered_content_refused(self):
        p = fixture_contract(self.tmp, frozen=True)
        c = load_contract(p)
        rec = Path(self.tmp) / "qa_train" / "r1.json"
        rec.write_text(json.dumps({"player_token": "t1", "ok": False}))
        with self.assertRaisesRegex(ValueError, "hash"):
            open_role(c, "qa_train", "fit")

    def test_path_traversal_refused(self):
        p = fixture_contract(self.tmp, frozen=True)
        c = json.loads(Path(p).read_text())
        c["roles"]["qa_train"]["files"] = {"../../etc/passwd": "0" * 64}
        Path(p).write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError, "path"):
            open_role(load_contract(p), "qa_train", "fit")

    def test_cross_role_overlap_refused(self):
        p = fixture_contract(self.tmp, frozen=True)
        c = json.loads(Path(p).read_text())
        c["roles"]["qa_calib"]["families"] = ["fam:qa_train"]
        Path(p).write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_contract(p)


class TestSanitization(unittest.TestCase):
    def test_source_is_not_anonymized_by_parsing(self):
        derived = sanitized_record(bsor_fixture(), player_token="tok-abc")
        raw = canonical_bytes(derived)
        self.assertNotIn(b"identity-sentinel", raw)
        self.assertNotIn(b"76561198_secret", raw)
        self.assertNotIn("score_id", derived)
        self.assertNotIn("identity", derived)
        self.assertEqual(derived["player_token"], "tok-abc")
        self.assertEqual(len(derived["frames"]), 1)

    def test_sanitized_keeps_provenance_hash(self):
        d = sanitized_record(bsor_fixture(), player_token="t",
                             source_sha256="ab" * 32)
        self.assertEqual(d["source_sha256"], "ab" * 32)


class TestIsolation(unittest.TestCase):
    def test_evaluator_modules_never_import_generator(self):
        bad = scan_imports(EVALUATOR_MODULES, FORBIDDEN_IMPORTS)
        self.assertEqual(bad, [], f"forbidden imports: {bad}")


class TestAtomicRecords(unittest.TestCase):
    def test_atomic_and_identity_bound(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            atomic_record(td, "k1", {"x": 1}, identity="idA")
            rec = json.loads((Path(td) / "k1.json").read_text())
            self.assertEqual(rec["identity"], "idA")
            with self.assertRaisesRegex(ValueError, "identity"):
                atomic_record(td, "k1", {"x": 2}, identity="idB")


if __name__ == "__main__":
    unittest.main()
