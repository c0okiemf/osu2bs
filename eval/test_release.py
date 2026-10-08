"""Q5 scoped-release tests.
  .venv/bin/python -m unittest eval.test_release -v      (fixtures)
  .venv/bin/python -m eval.test_release --exact           (bundle exactness:
      bundle_song(still_waiting) must equal the stored q5-composition
      evaluation output bit for bit — serving == evaluation)
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPolicyFile(unittest.TestCase):
    def test_absent_policy_means_b0(self):
        import quality_policy as qp
        orig = qp.POLICY_FILE
        try:
            qp.POLICY_FILE = ROOT / "nonexistent_policy.json"
            self.assertEqual(qp.active_policy(), "b0")
        finally:
            qp.POLICY_FILE = orig

    def test_invalid_policy_means_b0(self):
        import tempfile
        import quality_policy as qp
        orig = qp.POLICY_FILE
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as f:
                f.write("{broken")
                qp.POLICY_FILE = Path(f.name)
            self.assertEqual(qp.active_policy(), "b0")
        finally:
            qp.POLICY_FILE = orig

    def test_rollback_flips_policy(self):
        import tempfile
        import quality_policy as qp
        orig = qp.POLICY_FILE
        try:
            td = tempfile.mkdtemp()
            qp.POLICY_FILE = Path(td) / "quality_policy.json"
            qp.POLICY_FILE.write_text(json.dumps({"policy": "bundle-v1"}))
            self.assertEqual(qp.active_policy(), "bundle-v1")
            qp.rollback()
            self.assertEqual(qp.active_policy(), "b0")
        finally:
            qp.POLICY_FILE = orig


class TestWrongVersion(unittest.TestCase):
    def test_tampered_manifest_refuses(self):
        import tempfile
        import quality_policy as qp
        if not qp.MANIFEST.exists():
            self.skipTest("bundle not installed")
        orig = qp.MANIFEST
        try:
            man = json.loads(orig.read_text())
            k = sorted(man["artifacts"])[0]
            man["artifacts"][k] = "0" * 64
            with tempfile.NamedTemporaryFile("w", suffix=".json",
                                             delete=False) as f:
                json.dump(man, f)
                qp.MANIFEST = Path(f.name)
            with self.assertRaises(RuntimeError):
                qp._verify_artifacts()
        finally:
            qp.MANIFEST = orig

    def test_installed_artifacts_verify(self):
        import quality_policy as qp
        if not qp.MANIFEST.exists():
            self.skipTest("bundle not installed")
        man = qp._verify_artifacts()
        self.assertEqual(man["policy"], "bundle-v1")
        self.assertIn("c_confirmation", man)      # the FAIL ships disclosed
        self.assertTrue(any("variety" in n for n in man["non_claims"]))


def _exactness():
    """bundle_song output must be BIT-IDENTICAL to the stored q5-composition
    evaluation partial for still_waiting (same seeds, same config) — the
    promoted policy is exactly the evaluated one."""
    import torch
    torch.set_num_threads(4)
    from convert import parse_osu
    import quality_policy as qp
    part = json.loads((ROOT / "experiments/quality-v1/q5-composition/partial/"
                       "panel_still_waiting.json").read_text())
    from eval.quality_panel import dev_entries
    e = next(x for x in dev_entries() if x["song"] == "still_waiting")
    audio = ROOT / e["audio"]
    _m, objects, bpm, offset = parse_osu(ROOT / e["osu"])
    notes, walls, info = qp.bundle_song(objects, bpm, offset, str(audio))
    want_raw = [tuple(int(v) for v in n) for n in part["bundle"]["notes"]]
    # the partial stores step-tuples; convert to output dict form via the
    # same grid the bundle used
    from convert import grid_steps
    _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
    want_notes = [{"t": grid.time(s), "hand": h, "col": c, "layer": l,
                   "dir": d} for s, h, c, l, d in want_raw]
    assert info["source"] == part["source"], (info["source"], part["source"])
    assert notes == want_notes, "bundle notes differ from evaluated output"
    print(f"exactness OK: {info['source']}, {len(notes)} notes bit-identical "
          "to the q5-composition evaluation")


if __name__ == "__main__":
    import sys
    if "--exact" in sys.argv:
        _exactness()
    else:
        unittest.main()
