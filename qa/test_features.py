"""Task 1 tests (feature-parity repair): one shared evidence provider.
  .venv/bin/python -m pytest qa/test_features.py -q
"""
import unittest
from pathlib import Path

import torch

from qa.features import (audio_cache_key, calibration_features,
                         feature_schema_sha, features_for, records_for_role,
                         replay_view, training_features)
from qa.model import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent


def audio_indices():
    return [i for i, n in enumerate(FEATURE_NAMES)
            if n.startswith("audio_")]


def scene_fixture():
    return {"scope": None, "chart_sha256": "aa" * 32,
            "info_sha256": "bb" * 32,
            "notes": [(1.0, 1, 0, 1, 1), (1.5, 2, 1, 0, 0),
                      (2.0, 0, 2, 1, 4)],
            "bombs": [], "walls": [], "settings": {"njs": 18}}


def evidence_fixture(nonzero_audio=True):
    n = 40
    times = [0.1 * i for i in range(n)]
    ons = [0.7 + 0.01 * i for i in range(n)] if nonzero_audio else [0.0] * n
    ps = [0.5, 0.6, 0.4, 0.3] if nonzero_audio else [0.0] * 4
    return {"family": "fam:test", "policy": "audio",
            "audio": {"supported": True,
                      "frames": {"times": times, "onset_strength": ons},
                      "per_second": {"harmonic_rms": ps,
                                     "percussive_rms": ps}}}


def record_fixture(scene=None, profile=None, chart_index=0):
    return {"family": "fam:test", "player_token": "p" * 16,
            "replay_sha": "cc" * 32, "chart_index": chart_index,
            "window_index": 0, "scene": scene or scene_fixture(),
            "profile": profile or {"height": 1.8, "height_known": True,
                                   "left_handed": False},
            "window": {"supported": True, "outcome": "good"}}


class TestAudioParity(unittest.TestCase):
    def test_calibration_has_the_same_audio_features_as_training(self):
        rec, evidence = record_fixture(), evidence_fixture(True)
        a = training_features(rec, evidence)
        b = calibration_features(rec, evidence)
        self.assertTrue(torch.equal(a, b))
        self.assertGreater(int(torch.count_nonzero(a[audio_indices()])), 0)

    def test_evidence_cannot_silently_become_none(self):
        with self.assertRaises(ValueError):
            features_for(record_fixture(), None)
        with self.assertRaisesRegex(ValueError, "family"):
            features_for(record_fixture(),
                         {**evidence_fixture(), "family": "fam:other"})
        # zero audio under an explicit no-audio policy is allowed
        x = features_for(record_fixture(),
                         {"family": "fam:test", "policy": "no_audio:missing",
                          "audio": None})
        self.assertEqual(int(torch.count_nonzero(x[audio_indices()])), 0)
        # audio=None under an "audio" policy is a contract violation
        with self.assertRaisesRegex(ValueError, "policy"):
            features_for(record_fixture(),
                         {"family": "fam:test", "policy": "audio",
                          "audio": None})

    def test_no_caller_builds_features_with_none_audio(self):
        for rel in ("qa/train.py", "qa/calibrate.py", "qa/diagnose.py"):
            src = (ROOT / rel).read_text()
            self.assertNotIn("map_features(sc, None", src, rel)
            self.assertNotIn("map_features(scene, None", src, rel)
            self.assertIn("features_for", src, rel)


class TestGeometryAndBody(unittest.TestCase):
    def test_mirrored_replay_uses_mirrored_scene_view(self):
        from qa.train import mirror_scene
        sc = scene_fixture()
        m = mirror_scene(sc)
        view, profile = replay_view(sc, m, {"height": 1.7}, True)
        self.assertIs(view, m)
        self.assertTrue(profile["left_handed"])
        # note 0: li=1 c=1 -> mirrored li=2 c=0; x flips sign, hand flips
        ev = evidence_fixture()
        xa = features_for(record_fixture(scene=sc), ev)
        xb = features_for(record_fixture(scene=m), ev)
        ix = FEATURE_NAMES.index("note_x")
        ih = FEATURE_NAMES.index("hand_right")
        self.assertAlmostEqual(float(xa[ix]), -float(xb[ix]), places=6)
        self.assertNotEqual(float(xa[ih]), float(xb[ih]))

    def test_missing_height_stays_unknown(self):
        view, profile = replay_view(scene_fixture(), scene_fixture(),
                                    {"height": None}, False)
        self.assertFalse(profile["height_known"])
        x = features_for(record_fixture(profile=profile),
                         evidence_fixture())
        self.assertEqual(float(x[FEATURE_NAMES.index("body_height")]), 0.0)
        self.assertEqual(
            float(x[FEATURE_NAMES.index("body_height_known")]), 0.0)


class TestRolesAndCaches(unittest.TestCase):
    def test_role_denial(self):
        with self.assertRaises(PermissionError):
            next(records_for_role("seal", "confirm"))
        with self.assertRaises(PermissionError):
            next(records_for_role("qa_train", "confirm"))
        with self.assertRaises(PermissionError):
            next(records_for_role("gen", "fit"))

    def test_audio_cache_key_binds_bytes_and_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p1, p2 = Path(td) / "a.ogg", Path(td) / "b.ogg"
            p1.write_bytes(b"AUDIO-ONE")
            p2.write_bytes(b"AUDIO-TWO")
            k1 = audio_cache_key(p1, Path(td))
            k2 = audio_cache_key(p2, Path(td))
            k3 = audio_cache_key(p1, Path(td), n_sections=13)
            self.assertNotEqual(k1, k2)
            self.assertNotEqual(k1, k3)

    def test_schema_sha_pins_channel_order(self):
        s = feature_schema_sha()
        self.assertEqual(len(s), 64)
        self.assertEqual(s, feature_schema_sha())


if __name__ == "__main__":
    unittest.main()
