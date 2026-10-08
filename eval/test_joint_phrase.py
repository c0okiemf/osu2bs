"""Focused checks for source fidelity and fail-closed experiment inputs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.joint_phrase import (UnsupportedSource, _sha, _validate_raw,
                               decode_events, decoder_restrictions, encode_events,
                               inventory_records, phrase_windows, verify_sources)


class JointPhraseTests(unittest.TestCase):
    def test_roundtrip_stacks_dots_and_non_grid_timing(self):
        notes = [(0, 0, 0, 0, 1), (0, 0, 1, 0, 1), (0, 0, 2, 0, 8),
                 (1 / 3, 1, 3, 2, 8), (8, 0, 0, 2, 0)]
        self.assertEqual(decode_events(json.loads(json.dumps(encode_events(notes)))), notes)

    def test_windows_keep_empty_tail_boundary_and_environment(self):
        notes = [(0, 0, 0, 0, 1), (8, 1, 3, 2, 0), (24.5, 0, 0, 0, 1)]
        windows = phrase_windows(notes, [(8, 1, 1)], [(7, 3, 0, 1, 0, 5)], 25)
        self.assertEqual(len(windows), 4)
        self.assertEqual([len(w["events"]) for w in windows], [1, 1, 0, 1])
        self.assertEqual(windows[-1]["end"], 25)
        self.assertEqual(windows[0]["bomb_ids"], [])
        self.assertEqual(windows[1]["bomb_ids"], [0])
        self.assertEqual([w["wall_ids"] for w in windows], [[0], [0], [], []])
        self.assertEqual(windows[3]["previous"]["beat"], 8)
        self.assertEqual(decode_events([e for w in windows for e in w["events"]]), notes)
        self.assertEqual(len(phrase_windows([], [], [], 17)), 3)
        self.assertEqual(phrase_windows([], [], [], 0), [])
        with self.assertRaisesRegex(UnsupportedSource, "outside_audio"):
            phrase_windows([(25, 0, 0, 0, 1)], [], [], 25)

    def test_invalid_numbers_capacity_and_occupancy_fail(self):
        for value in (float("nan"), float("inf"), True, -1):
            with self.subTest(value=value), self.assertRaises(UnsupportedSource):
                encode_events([(value, 0, 0, 0, 1)])
        for note in [(0, True, 0, 0, 1), (0, 0, 0.0, 0, 1), (0, 0, 0, 0, 9)]:
            with self.assertRaises(UnsupportedSource):
                encode_events([note])
        with self.assertRaisesRegex(UnsupportedSource, "duplicate_occupancy"):
            encode_events([(0, 0, 1, 0, 1), (0, 1, 1, 0, 0)])
        with self.assertRaisesRegex(UnsupportedSource, "slot_capacity"):
            encode_events([(0, 0, c, 0, 1) for c in range(4)])

    def test_scope_before_shared_reader_can_drop_notes(self):
        for raw in ({"version": "4.0.0", "colorNotes": []},
                    {"version": "3.0.0", "sliders": [{}]},
                    {"_version": "2.0.0", "_events": [{"_type": 14}]},
                    {"_version": "2.0.0", "_notes": [{"_type": 7}]}):
            with self.subTest(raw=raw), self.assertRaises(UnsupportedSource):
                _validate_raw(raw)
        self.assertEqual(_validate_raw({"_version": "2.0.0", "_notes": []}), "2.0.0")
        raw = {"version": "3.0.0", "colorNotes": [
            {"b": 0, "c": 0, "x": 1, "y": 1, "d": 0, "a": 15}]}
        with self.assertRaisesRegex(UnsupportedSource, "angle_offset"):
            _validate_raw(raw)

    def test_restrictions_are_witnessed_not_repaired(self):
        notes = [(0, 0, 0, 0, 1), (0, 0, 1, 0, 1), (0, 1, 3, 0, 8),
                 (1 / 3, 0, 3, 1, 0), (2, 1, 1, 1, 1)]
        original = list(notes)
        r = decoder_restrictions(notes, 120, [(0, 1, 0, 2, 2, 3)], [(1, 1, 1)])
        for k in ("multiple_directional_heads", "standalone_dots", "off_quarter_grid",
                  "cross_body_outer_lane", "central_middle_cell", "wall_not_exportable",
                  "bomb_not_exportable"):
            self.assertGreater(r["counts"][k], 0)
            self.assertTrue(r["witnesses"][k])
        self.assertEqual(notes, original)

    def test_sources_reject_content_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x"
            p.write_text("one")
            r = {"family": "a", "sources": {"chart": {"path": str(p), "sha256": _sha(p)}}}
            verify_sources(r)
            p.write_text("two")
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                verify_sources(r)

    def test_read_source_preserves_v2_and_v3_environment(self):
        import numpy as np
        import soundfile as sf
        from eval.joint_phrase import read_source
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio, info, chart = (root / p for p in ("song.wav", "Info.dat", "ExpertPlus.dat"))
            sf.write(audio, np.zeros(44100), 22050)
            info.write_text(json.dumps({"_beatsPerMinute": 120,
                "_difficultyBeatmapSets": [{"_beatmapCharacteristicName": "Standard",
                  "_difficultyBeatmaps": [{"_beatmapFilename": chart.name,
                    "_difficulty": "ExpertPlus", "_difficultyRank": 9,
                    "_noteJumpMovementSpeed": 18}]}]}))
            sources = [
                {"_version": "2.0.0", "_notes": [
                    {"_time": 0, "_type": 0, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 1},
                    {"_time": 1, "_type": 3, "_lineIndex": 1, "_lineLayer": 1, "_cutDirection": 8}],
                 "_obstacles": [{"_time": 1, "_duration": 2, "_lineIndex": 0, "_width": 2, "_type": 1}]},
                {"version": "3.0.0", "colorNotes": [{"b": 0, "c": 0, "x": 0, "y": 0, "d": 1}],
                 "bombNotes": [{"b": 1, "x": 1, "y": 1}],
                 "obstacles": [{"b": 1, "d": 2, "x": 0, "w": 2, "y": 2, "h": 3}]}]
            for raw in sources:
                chart.write_text(json.dumps(raw))
                record = {"family": "a", "sources": {k: {"path": str(p), "sha256": _sha(p)}
                          for k, p in (("chart", chart), ("info", info), ("audio", audio))}}
                got = read_source(record)
                self.assertEqual(got["notes"], [(0, 0, 0, 0, 1)])
                self.assertEqual(got["bombs"], [(1, 1, 1)])
                self.assertEqual(got["walls"], [(1, 2, 0, 2, 2, 3)])
                self.assertEqual(got["duration_beats"], 4)

    def test_split_overlap_fails_before_source_access(self):
        maps = [{"family": f"fam:{i}", "dir": "/missing", "split": "train",
                 "eligible": "ok", "family_rep": True, "charts": {"ExpertPlus": {}}}
                for i in range(12)]
        with self.assertRaisesRegex(ValueError, "overlap"):
            inventory_records({"maps": maps}, {"families": maps}, {})

    def test_resume_reuses_complete_partials_and_preserves_sources(self):
        from eval.joint_phrase import run_audit
        record = {"family": "fam:a", "role": "train", "sources": {}}
        run = {"identity": "same", "config": {"records": [record],
               "protected": {}}}
        with tempfile.TemporaryDirectory() as d, \
                patch("eval.joint_phrase.freeze_inventory", return_value=run), \
                patch("eval.joint_phrase.PROTECTED", ()), \
                patch("eval.joint_phrase.read_source", side_effect=UnsupportedSource("unsupported")) as read:
            first = run_audit(Path(d))
            self.assertEqual(read.call_count, 1)
            self.assertEqual(run_audit(Path(d)), first)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(first["status"], "INSUFFICIENT_SUPPORTED_DATA")


if __name__ == "__main__":
    unittest.main()
