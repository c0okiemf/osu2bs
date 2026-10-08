"""qa.audio tests (synthetic click+tone track).
  .venv/bin/python -m unittest eval.test_qa_audio -q
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

from qa.audio import evidence


def _click_track(tmp):
    """20 s: 2 Hz clicks over a quiet tone, loud second half."""
    p = Path(tmp) / "t.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i",
         "sine=frequency=220:duration=20",
         "-f", "lavfi", "-i",
         "anoisesrc=color=pink:duration=20:amplitude=0.02",
         "-filter_complex",
         "[0:a]volume='if(gte(t,10),1.0,0.15)':eval=frame[a0];"
         "[a0][1:a]amix=inputs=2[out]",
         "-map", "[out]", str(p)], check=True)
    return p


class TestEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.audio = _click_track(cls._td.name)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_shapes_and_determinism_and_cache(self):
        with tempfile.TemporaryDirectory() as cd:
            a = evidence(self.audio, cache_dir=cd)
            b = evidence(self.audio, cache_dir=cd)     # cache hit
            self.assertEqual(a, b)
            self.assertAlmostEqual(a["duration_s"], 20.0, delta=0.5)
            n = int(a["duration_s"])
            for k, v in a["per_second"].items():
                self.assertEqual(len(v), n, k)
            self.assertGreaterEqual(len(a["section_bounds"]), 1)
            self.assertGreater(a["tempo_bpm"], 0)

    def test_intensity_reflects_loud_half(self):
        a = evidence(self.audio)
        rms = a["per_second"]["intensity_rms"]
        first, second = rms[2:9], rms[12:19]
        self.assertGreater(sum(second) / len(second),
                           2 * sum(first) / len(first))

    def test_no_quality_or_fun_fields(self):
        import json
        s = json.dumps(evidence(self.audio))
        self.assertNotIn('"quality"', s)
        self.assertNotIn('"fun"', s)



class TestCacheIdentity(unittest.TestCase):
    def test_distinct_n_sections_distinct_keys(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td, \
                tempfile.TemporaryDirectory() as cd:
            audio = _click_track(td)
            evidence(audio, cache_dir=cd, n_sections=8)
            evidence(audio, cache_dir=cd, n_sections=12)
            keys = list(Path(cd).glob("*.json"))
            self.assertEqual(len(keys), 2)

    def test_short_silent_audio_explicit_support(self):
        import subprocess, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f",
                            "lavfi", "-i", "anullsrc=d=1", str(p)],
                           check=True)
            r = evidence(p)
            self.assertFalse(r["supported"])
            self.assertEqual(r["reason"], "too_short_or_silent")

if __name__ == "__main__":
    unittest.main()
