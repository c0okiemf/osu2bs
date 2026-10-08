"""Task 2 tests: independent scene + monotone unique alignment.
  .venv/bin/python -m unittest qa.test_scene -q
"""
import json
import tempfile
import unittest
from pathlib import Path

from qa.scene import align, read_scene


def _mk(tmp, notes=None, version="v2", extra=None):
    d = Path(tmp)
    if version == "v2":
        raw = {"_version": "2.0.0", "_obstacles": [],
               "_notes": notes if notes is not None else [
                   {"_time": 2.0, "_type": 0, "_lineIndex": 1,
                    "_lineLayer": 0, "_cutDirection": 1},
                   {"_time": 4.0, "_type": 1, "_lineIndex": 2,
                    "_lineLayer": 0, "_cutDirection": 0}]}
        if extra:
            raw.update(extra)
    else:
        raw = {"version": "3.0.0",
               "colorNotes": [{"b": 2.0, "x": 1, "y": 0, "c": 0, "d": 1},
                              {"b": 4.0, "x": 2, "y": 0, "c": 1, "d": 0}]}
        if extra:
            raw.update(extra)
    (d / "chart.dat").write_text(json.dumps(raw))
    (d / "Info.dat").write_text(json.dumps({
        "_beatsPerMinute": 120,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{
                "_beatmapFilename": "chart.dat",
                "_difficulty": "ExpertPlus",
                "_noteJumpMovementSpeed": 18,
                "_noteJumpStartBeatOffset": 0.0}]}]}))
    return d / "chart.dat", d / "Info.dat"


def _ev(t, li, ll, c, d, etype=0):
    return {"event_time": t, "line_index": li, "line_layer": ll,
            "color": c, "cut_direction": d, "event_type": etype}


class TestScene(unittest.TestCase):
    def test_v2_v3_equivalence(self):
        with tempfile.TemporaryDirectory() as a, \
                tempfile.TemporaryDirectory() as b:
            s2 = read_scene(*_mk(a, version="v2"))
            s3 = read_scene(*_mk(b, version="v3"))
            self.assertIsNone(s2["scope"])
            self.assertIsNone(s3["scope"])
            self.assertEqual(s2["notes"], s3["notes"])
            self.assertEqual(s2["notes"][0][0], 1.0)     # 2 beats @120

    def test_unsupported_v3_mechanics_scope_fail(self):
        with tempfile.TemporaryDirectory() as a:
            s = read_scene(*_mk(a, version="v3",
                                extra={"sliders": [{"b": 1}]}))
            self.assertEqual(s["scope"], "unsupported_v3_mechanics")

    def test_settings_bound_from_info(self):
        with tempfile.TemporaryDirectory() as a:
            s = read_scene(*_mk(a))
            self.assertEqual(s["settings"]["njs"], 18)
            self.assertEqual(s["settings"]["characteristic"], "Standard")


class TestAlign(unittest.TestCase):
    def _scene(self, tmp):
        return read_scene(*_mk(tmp))

    def test_basic_monotone(self):
        with tempfile.TemporaryDirectory() as a:
            sc = self._scene(a)
            al = align(sc, [_ev(1.02, 1, 0, 0, 1), _ev(2.05, 2, 0, 1, 0)])
            self.assertEqual(al["aligned"], [(0, 0), (1, 1)])
            self.assertEqual(al["ambiguous"], 0)

    def test_repeated_attributes_monotone_disambiguation(self):
        with tempfile.TemporaryDirectory() as a:
            notes = [{"_time": b, "_type": 0, "_lineIndex": 1,
                      "_lineLayer": 0, "_cutDirection": 1}
                     for b in (2.0, 2.2, 2.4)]     # 3 identical notes
            sc = read_scene(*_mk(a, notes=notes))
            al = align(sc, [_ev(1.0, 1, 0, 0, 1), _ev(1.1, 1, 0, 0, 1),
                            _ev(1.2, 1, 0, 0, 1)])
            self.assertEqual(al["aligned"], [(0, 0), (1, 1), (2, 2)])

    def test_left_handed_mirror_matching(self):
        with tempfile.TemporaryDirectory() as a:
            sc = self._scene(a)
            # mirrored replay: line 3-1=2, color 1, dir 1
            al = align(sc, [_ev(1.0, 2, 0, 1, 1)], left_handed=True)
            self.assertEqual(al["aligned"], [(0, 0)])
            self.assertTrue(al["mirrored"])

    def test_unmatched_and_residuals(self):
        with tempfile.TemporaryDirectory() as a:
            sc = self._scene(a)
            al = align(sc, [_ev(9.0, 3, 2, 1, 8)])
            self.assertEqual(al["unmatched"], 1)
            self.assertIsNone(al["timing_residual_p95"])


if __name__ == "__main__":
    unittest.main()


class TestModchartScope(unittest.TestCase):
    def _uniform(self, n=25):
        return [{"_time": 2.0 + 0.5 * i, "_type": 1, "_lineIndex": 2,
                 "_lineLayer": 0, "_cutDirection": 8} for i in range(n)]

    def test_uniform_dot_with_customdata_is_scope_failure(self):
        with tempfile.TemporaryDirectory() as td:
            dat, info = _mk(td, notes=self._uniform(),
                            extra={"_customData": {"_x": 1}})
            self.assertEqual(read_scene(dat, info)["scope"],
                             "unsupported_suspected_modchart")

    def test_customdata_alone_is_not_a_modchart_claim(self):
        with tempfile.TemporaryDirectory() as td:
            dat, info = _mk(td, extra={"_customData": {"_x": 1}})
            self.assertIsNone(read_scene(dat, info)["scope"])

    def test_uniform_dots_without_customdata_pass(self):
        with tempfile.TemporaryDirectory() as td:
            dat, info = _mk(td, notes=self._uniform())
            self.assertIsNone(read_scene(dat, info)["scope"])
