"""Fail-closed export test (QA architecture 2026-09-24): a difficulty that
fails final checks is never written; no passing difficulty means no export.
  .venv/bin/python -m unittest eval.test_failclosed -q
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import convert


def _tiny_osu(tmp):
    p = Path(tmp) / "gen.osu"
    hits = "\n".join(f"{64 + (i % 4) * 128},192,{1000 + i * 250},1,0"
                     for i in range(64))
    p.write_text("osu file format v14\n\n[General]\nAudioFilename: a.mp3\n"
                 "\n[TimingPoints]\n1000,500,4,2,0,100,1,0\n\n[HitObjects]\n"
                 + hits + "\n")
    a = Path(tmp) / "a.mp3"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=20", "-q:a", "5",
                    str(a)], check=True)
    return p, a


class TestFailClosed(unittest.TestCase):
    def test_all_failing_checks_export_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            osu, audio = _tiny_osu(td)
            out = Path(td) / "out"
            with mock.patch.object(convert, "GROOM_PT") as gp, \
                    mock.patch.object(convert, "check",
                                      return_value=["synthetic failure"]):
                gp.exists.return_value = False        # legacy path, no models
                with self.assertRaises(convert.GenerationError):
                    convert.main(str(osu), str(audio), out,
                                 diffs="ExpertPlus")
            self.assertFalse((out / "Info.dat").exists(),
                             "failing map must never be written")

    def test_passing_checks_still_export(self):
        with tempfile.TemporaryDirectory() as td:
            osu, audio = _tiny_osu(td)
            out = Path(td) / "out"
            with mock.patch.object(convert, "GROOM_PT") as gp:
                gp.exists.return_value = False
                rc = convert.main(str(osu), str(audio), out,
                                  diffs="ExpertPlus")
            self.assertEqual(rc, 0)
            self.assertTrue((out / "Info.dat").exists())


if __name__ == "__main__":
    unittest.main()
