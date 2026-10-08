"""Tests for the wall-corridor panel audit (Task 3). Constructed fixtures only,
never the sealed corpus.
  .venv/bin/python -m unittest eval.test_wall_audit -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.wall_audit import audit


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


V2_CHART = {"_version": "2.0.0", "_notes": [], "_obstacles": [
    {"_time": 0, "_duration": 2, "_lineIndex": 0, "_width": 1, "_type": 0},
    {"_time": 4, "_duration": 2, "_lineIndex": 3, "_width": 1, "_type": 0},
]}
# a chart with a crouch wall -> partial (unknown) coverage
V2_UNKNOWN = {"_version": "2.0.0", "_notes": [], "_obstacles": [
    {"_time": 0, "_duration": 2, "_lineIndex": 0, "_width": 1, "_type": 0},
    {"_time": 1, "_duration": 2, "_lineIndex": 1, "_width": 2, "_type": 1},
]}


class TestAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _write_chart(self, name, obj):
        p = self.tmp / name
        p.write_text(json.dumps(obj))
        return p

    def _panel(self, entries, **extra):
        p = self.tmp / "panel.json"
        doc = {"schema_version": 1, "frozen": "test", "missing": [],
               "entries": entries}
        doc.update(extra)
        p.write_text(json.dumps(doc))
        return p

    def _entry(self, id, chart, bpm=120.0, role="generated"):
        # generated entries bind BPM to a results.json-style timing source
        ts = self.tmp / (Path(chart).stem + ".results.json")
        ts.write_text(json.dumps({"bpm": bpm}))
        return {"id": id, "role": role, "chart": str(chart), "sha256": _sha(chart),
                "bpm": bpm, "origin_s": 0, "timing_source": str(ts),
                "timing_source_sha256": _sha(ts)}

    def test_clean_panel_certifies(self):
        c = self._write_chart("c.dat", V2_CHART)
        panel = self._panel([self._entry("generated/a", c)])
        code, rep = audit(panel, self.tmp / "out", make_plots=False)
        self.assertEqual(code, 0)
        self.assertEqual(rep["certification"], "CERTIFIED")
        self.assertEqual(len(rep["maps"]), 1)

    def test_one_flipped_byte_uncertifies_nonzero(self):
        c = self._write_chart("c.dat", V2_CHART)
        panel = self._panel([self._entry("generated/a", c)])
        # tamper AFTER recording the hash in the panel
        c.write_text(c.read_text() + " ")
        code, rep = audit(panel, self.tmp / "out2", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertEqual(rep["certification"], "UNCERTIFIED")
        self.assertTrue(any(f["reason"] == "chart_sha256_mismatch"
                            for f in rep["integrity"]["failures"]))
        self.assertEqual(rep["maps"], [])   # no analysis on a failed panel

    def test_missing_chart_uncertifies(self):
        c = self._write_chart("c.dat", V2_CHART)
        entry = self._entry("generated/a", c)
        c.unlink()
        panel = self._panel([entry])
        code, rep = audit(panel, self.tmp / "out3", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "chart_missing"
                            for f in rep["integrity"]["failures"]))

    def test_unknown_geometry_is_partial_not_clearance(self):
        c = self._write_chart("u.dat", V2_UNKNOWN)
        panel = self._panel([self._entry("generated/u", c)])
        _, rep = audit(panel, self.tmp / "out4", make_plots=False)
        m = rep["maps"][0]
        self.assertGreaterEqual(m["n_unknown_objects"], 1)
        self.assertEqual(m["radii"]["0.25"]["coverage"], "partial")

    def test_deterministic_report_bytes(self):
        c = self._write_chart("c.dat", V2_CHART)
        panel = self._panel([self._entry("generated/a", c)])
        audit(panel, self.tmp / "ra", make_plots=False)
        audit(panel, self.tmp / "rb", make_plots=False)
        a = (self.tmp / "ra" / "report.json").read_bytes()
        b = (self.tmp / "rb" / "report.json").read_bytes()
        self.assertEqual(a, b)

    def test_report_has_no_wall_clock_runtime(self):
        c = self._write_chart("c.dat", V2_CHART)
        panel = self._panel([self._entry("generated/a", c)])
        _, rep = audit(panel, self.tmp / "out5", make_plots=False)
        blob = json.dumps(rep)
        self.assertNotIn("wall_clock", blob)
        self.assertNotIn("runtime", blob)
        # runtime lives in the separate resources log instead
        self.assertTrue((self.tmp / "out5" / "resources.log").exists())

    def test_width1_generated_zero_forced_travel(self):
        c = self._write_chart("c.dat", V2_CHART)   # two width-1 outer walls
        panel = self._panel([self._entry("generated/a", c)])
        _, rep = audit(panel, self.tmp / "out6", make_plots=False)
        self.assertTrue(rep["aggregate"]["generated_zero_forced_travel_nominal"])

    # --- Task 0 repair: coverage contamination + panel validation ---

    def test_zero_width_wall_not_full_coverage(self):
        # the exact bug from review's review: invalid object, coverage != full
        chart = {"_version": "2.0.0", "_notes": [], "_obstacles": [
            {"_time": 0, "_duration": 2, "_lineIndex": 0, "_width": 0, "_type": 0}]}
        c = self._write_chart("z.dat", chart)
        panel = self._panel([self._entry("generated/z", c)])
        _, rep = audit(panel, self.tmp / "outz", make_plots=False)
        m = rep["maps"][0]
        self.assertEqual(m["n_invalid"], 1)
        for rad in ("0.15", "0.25", "0.35"):
            self.assertNotEqual(m["radii"][rad]["coverage"], "full")

    def test_unresolvable_timestamp_chart_unknown(self):
        chart = {"_version": "2.0.0", "_notes": [], "_obstacles": [
            {"_time": float("nan"), "_duration": 2, "_lineIndex": 0,
             "_width": 1, "_type": 0}]}
        # json can't hold NaN; write it raw
        p = self.tmp / "nan.dat"
        p.write_text('{"_version":"2.0.0","_notes":[],"_obstacles":[{"_time":NaN,'
                     '"_duration":2,"_lineIndex":0,"_width":1,"_type":0}]}')
        panel = self._panel([self._entry("generated/nan", p)])
        _, rep = audit(panel, self.tmp / "outn", make_plots=False)
        self.assertTrue(rep["maps"][0]["chart_unknown"])

    def test_empty_panel_uncertified(self):
        p = self.tmp / "empty.json"
        p.write_text(json.dumps({"schema_version": 1, "entries": []}))
        code, rep = audit(p, self.tmp / "oute", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertEqual(rep["certification"], "UNCERTIFIED")

    def test_duplicate_id_uncertified(self):
        c = self._write_chart("c.dat", V2_CHART)
        panel = self._panel([self._entry("generated/a", c),
                             self._entry("generated/a", c)])
        code, rep = audit(panel, self.tmp / "outd", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "duplicate_id"
                            for f in rep["integrity"]["failures"]))

    def test_bpm_not_bound_to_source_uncertified(self):
        c = self._write_chart("c.dat", V2_CHART)
        e = self._entry("generated/a", c, bpm=120.0)
        # timing source (the chart itself here) has no bpm -> unresolvable;
        # use a results.json-style source with a DIFFERENT bpm to force mismatch
        ts = self.tmp / "results.json"
        ts.write_text(json.dumps({"bpm": 999.0}))
        e["timing_source"] = str(ts)
        e["timing_source_sha256"] = _sha(ts)
        panel = self._panel([e])
        code, rep = audit(panel, self.tmp / "outb", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "bpm_not_bound_to_source"
                            for f in rep["integrity"]["failures"]))

    def test_nonpositive_bpm_uncertified(self):
        c = self._write_chart("c.dat", V2_CHART)
        panel = self._panel([self._entry("generated/a", c, bpm=0.0)])
        code, rep = audit(panel, self.tmp / "outp", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "nonfinite_or_nonpositive_bpm"
                            for f in rep["integrity"]["failures"]))

    def test_declared_missing_entries_uncertified(self):
        c = self._write_chart("c.dat", V2_CHART)
        code, rep = audit(self._panel([self._entry("generated/a", c)],
                                      missing=["generated/gone"]),
                          self.tmp / "outm", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "declared_missing_entries"
                            for f in rep["integrity"]["failures"]))

    def test_expected_membership_dropped_uncertified(self):
        c = self._write_chart("c.dat", V2_CHART)
        code, rep = audit(self._panel(
            [self._entry("generated/a", c)],
            expected_ids=["generated/a", "generated/b"]),
            self.tmp / "outem", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "missing_expected_entry"
                            for f in rep["integrity"]["failures"]))


if __name__ == "__main__":
    unittest.main()
