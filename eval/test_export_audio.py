"""Real codec regression: a WAV/Opus file named .egg must never reach the game."""
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import convert


class AudioExportTests(unittest.TestCase):
    def make_audio(self, path, codec="pcm_s16le", container="wav", rate=14800):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency=440:duration=4:sample_rate={rate}",
                        "-c:a", codec, "-f", container, str(path)], check=True)

    def maps(self):
        return {"ExpertPlus": ([dict(t=1000., hand=0, col=1, layer=0, dir=1)],
                               [], convert.diff_spec("ExpertPlus"))}

    def test_disguised_wave_and_opus_normalized_in_real_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for suffix, codec, container, rate in [
                    ("egg", "pcm_s16le", "wav", 14800),
                    ("ogg", "pcm_s16le", "wav", 14800),
                    ("egg", "libopus", "ogg", 48000)]:
                with self.subTest(codec=codec, suffix=suffix):
                    src = root / f"{codec}.{suffix}"
                    self.make_audio(src, codec, container, rate)
                    out = root / f"map-{codec}-{suffix}"
                    convert.write_map(self.maps(), 120., dict(Title="test", Artist="test"), src, out)
                    self.assertAlmostEqual(convert.validate_audio(out / "song.egg"), 4., places=2)
                    self.assertEqual((out / "song.egg").read_bytes()[:4], b"OggS")
                    with zipfile.ZipFile(out.with_suffix(".zip")) as z:
                        self.assertIsNone(z.testzip())
                        self.assertEqual(z.read("song.egg"), (out / "song.egg").read_bytes())
                        self.assertEqual(json.loads(z.read("ExpertPlus.dat"))["_notes"][0]["_time"], 2.)

    def test_valid_vorbis_preserved_and_in_place_conversion_supported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / "song.egg"
            self.make_audio(src)
            convert.write_audio(src, root)
            before = src.read_bytes()
            convert.write_audio(src, root)
            self.assertEqual(src.read_bytes(), before)
            convert.write_audio(src, root / "copy")
            self.assertEqual((root / "copy/song.egg").read_bytes(), before)

    def test_bad_input_or_failed_transcode_cannot_publish_or_replace_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / "bad.egg"
            src.write_bytes(b"OggS dummy")
            out = root / "map"
            with self.assertRaises(convert.GenerationError):
                convert.write_map(self.maps(), 120., dict(Title="test", Artist="test"), src, out)
            self.assertFalse((out / "Info.dat").exists())
            self.assertFalse(out.with_suffix(".zip").exists())
            self.make_audio(src)
            convert.write_audio(src, out)
            before = (out / "song.egg").read_bytes()
            real_run = subprocess.run
            def fail_encode(cmd, **kwargs):
                if cmd[0] == "ffmpeg" and "libvorbis" in cmd:
                    raise subprocess.CalledProcessError(1, cmd)
                return real_run(cmd, **kwargs)
            with patch.object(convert.subprocess, "run", side_effect=fail_encode):
                with self.assertRaises(convert.GenerationError):
                    convert.write_audio(src, out)
            self.assertEqual((out / "song.egg").read_bytes(), before)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["song.egg"])

    def test_review_packager_rejects_mislabeled_audio_before_replacing_zip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); out = root / "map"; out.mkdir()
            convert.export_charts(self.maps(), 120., dict(Title="test", Artist="test"), out)
            self.make_audio(out / "song.egg")
            archive = root / "review.zip"; archive.write_bytes(b"existing artifact")
            with self.assertRaises(convert.GenerationError):
                convert.package_map(out, archive)
            self.assertEqual(archive.read_bytes(), b"existing artifact")

    def test_shift_and_cut_always_start_from_pristine_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root / "source.egg"
            self.make_audio(src)
            for shift, end, expected in [(500, None, 4.5), (-500, None, 3.5), (0, 1500, 1.5)]:
                with self.subTest(shift=shift, end=end):
                    out = root / f"map-{shift}-{end}"
                    convert.write_audio(src, out, shift, end)
                    convert.write_audio(out / "song.egg", out, shift, end)
                    self.assertAlmostEqual(convert.validate_audio(out / "song.egg"), expected, places=2)
                    self.assertEqual((out / "song_orig.egg").read_bytes(), src.read_bytes())


if __name__ == "__main__":
    unittest.main()
