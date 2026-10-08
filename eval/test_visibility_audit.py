"""Tests for the E01 visibility panel audit runner (Task 3). Constructed
fixtures only.
  .venv/bin/python -m unittest eval.test_visibility_audit -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.visibility_audit import (COV_METRIC, DUR_METRICS, _plot_selection,
                                   audit, converged)


def _m7(cov=0.5, dur=0.1):
    return {COV_METRIC: cov, **{k: dur for k in DUR_METRICS}}


def _fake_map(mid, role, strata, witnesses):
    return {"id": mid, "role": role, "standard": {"status": "ok",
            "witness_strata": strata, "witnesses": witnesses}}


class TestConvergenceGate(unittest.TestCase):
    def test_identical_converges(self):
        self.assertTrue(converged(_m7(), _m7()))

    def test_mean_coverage_out_of_bound_fails(self):
        self.assertFalse(converged(_m7(0.5), _m7(0.53)))   # 0.03 > 0.02

    def test_one_duration_metric_out_of_bound_fails(self):
        base, fine = _m7(), _m7()
        fine = dict(fine, **{DUR_METRICS[3]: base[DUR_METRICS[3]] + 0.03})
        self.assertFalse(converged(base, fine))            # 30 ms > 20 ms

    def test_within_bounds_converges(self):
        base = _m7(0.5, 0.10)
        fine = dict(_m7(0.51, 0.10), **{DUR_METRICS[0]: 0.118})  # 18 ms
        self.assertTrue(converged(base, fine))


class TestPlotSelection(unittest.TestCase):
    def _wit(self, tid, cov, term):
        return {"target_id": tid, "max_coverage": cov, "terminal_clear_s": term,
                "samples": [{"start_s": 0, "end_s": 0.01, "coverage": cov,
                             "marker_coverage": 0.0}]}

    def test_benign_never_from_uniform(self):
        maps = [_fake_map("g/a", "generated",
                {"banned_hides_nothing": [], "hidden_by_banned": [],
                 "hidden_only_nonbanned": ["note:5"], "uniform_valid": ["note:1"]},
                [self._wit("note:5", 0.9, 0.2), self._wit("note:1", 0.0, 0.6)])]
        recs = {r["category"]: r for r in _plot_selection(maps)}
        self.assertTrue(recs["benign_centre"]["absent"])   # not filled from uniform
        self.assertEqual(recs["noncentre_hiding"]["target_id"], "note:5")
        self.assertEqual(recs["noncentre_hiding"]["role"], "target_hidden_by_nonbanned")
        self.assertEqual(recs["recovery"]["target_id"], "note:5")  # >=.5 & term>=.05
        self.assertEqual(recs["noncentre_hiding"]["camera_eye"], [0.0, 1.60, -0.65])
        self.assertEqual(recs["noncentre_hiding"]["settings"]["njs"], 16.0)

    def test_benign_uses_banned_hides_nothing(self):
        maps = [_fake_map("g/a", "generated",
                {"banned_hides_nothing": ["note:2"], "hidden_by_banned": [],
                 "hidden_only_nonbanned": [], "uniform_valid": ["note:1"]},
                [self._wit("note:2", 0.0, 0.6), self._wit("note:1", 0.0, 0.6)])]
        rec = {r["category"]: r for r in _plot_selection(maps)}["benign_centre"]
        self.assertFalse(rec["absent"])
        self.assertEqual(rec["target_id"], "note:2")
        self.assertEqual(rec["role"], "outgoing_occluder_hides_nothing")
        self.assertEqual(rec["outgoing_targets_hidden"], 0)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _note(t, col, row, typ=0, d=1):
    return {"_time": t, "_type": typ, "_lineIndex": col, "_lineLayer": row,
            "_cutDirection": d}


# a near-axis same-cell stack in a NON-banned cell (col2,row2) with a 0.1s gap:
# the target is hidden by a non-banned occluder (probed ~0.98 coverage). bpm120
# -> beat4=2.0s target, beat3.8=1.9s blocker; plus a lone far note.
STACK = {"_version": "2.0.0", "_obstacles": [], "_notes": [
    _note(4, 2, 2), _note(3.8, 2, 2), _note(8, 3, 0)]}


class TestVisibilityAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _chart(self, obj, name="c.dat"):
        p = self.tmp / name
        p.write_text(json.dumps(obj))
        return p

    def _entry(self, id, chart, bpm=120.0):
        ts = self.tmp / (Path(chart).stem + ".results.json")
        ts.write_text(json.dumps({"bpm": bpm}))
        return {"id": id, "role": "generated", "chart": str(chart),
                "sha256": _sha(chart), "bpm": bpm, "origin_s": 0,
                "timing_source": str(ts), "timing_source_sha256": _sha(ts),
                "authored": {"status": "unknown", "reason": "test"}}

    def _panel(self, entries, **extra):
        p = self.tmp / "panel.json"
        doc = {"schema_version": 1, "standard_scenario":
               {"njs": 16.0, "lifetime_s": 0.6}, "entries": entries}
        doc.update(extra)
        p.write_text(json.dumps(doc))
        return p

    def test_clean_panel_certifies_and_classifies_occluder(self):
        c = self._chart(STACK)
        code, rep = audit(self._panel([self._entry("generated/a", c)]),
                          self.tmp / "o1", make_plots=False)
        self.assertEqual(code, 0)
        self.assertEqual(rep["certification"], "CERTIFIED")
        st = rep["maps"][0]["standard"]
        self.assertGreaterEqual(st["n_hidden_ge_0.5"], 1)
        # the occluder is a non-banned cell -> not counted as banned-caused
        self.assertEqual(st["n_hidden_by_banned_occluder"], 0)
        self.assertGreaterEqual(st["n_hidden_only_nonbanned"], 1)

    def test_deterministic_report(self):
        c = self._chart(STACK)
        panel = self._panel([self._entry("generated/a", c)])
        audit(panel, self.tmp / "ra", make_plots=False)
        audit(panel, self.tmp / "rb", make_plots=False)
        self.assertEqual((self.tmp / "ra" / "report.json").read_bytes(),
                         (self.tmp / "rb" / "report.json").read_bytes())

    def test_no_wall_clock_runtime_in_report(self):
        c = self._chart(STACK)
        _, rep = audit(self._panel([self._entry("generated/a", c)]),
                       self.tmp / "o2", make_plots=False)
        self.assertNotIn("wall_clock", json.dumps(rep))
        self.assertTrue((self.tmp / "o2" / "resources.log").exists())

    def test_empty_panel_uncertified(self):
        p = self.tmp / "e.json"
        p.write_text(json.dumps({"schema_version": 1, "entries": []}))
        code, rep = audit(p, self.tmp / "o3", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertEqual(rep["certification"], "UNCERTIFIED")

    def test_tampered_chart_uncertified(self):
        c = self._chart(STACK)
        e = self._entry("generated/a", c)
        c.write_text(c.read_text() + " ")   # change after hashing
        code, rep = audit(self._panel([e]), self.tmp / "o4", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "chart_sha256_mismatch"
                            for f in rep["integrity"]["failures"]))

    def test_duplicate_id_uncertified(self):
        c = self._chart(STACK)
        code, rep = audit(self._panel([self._entry("generated/a", c),
                                       self._entry("generated/a", c)]),
                          self.tmp / "o5", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "duplicate_id"
                            for f in rep["integrity"]["failures"]))

    def test_declared_missing_entries_uncertified(self):
        c = self._chart(STACK)
        code, rep = audit(self._panel([self._entry("generated/a", c)],
                                      missing=["generated/gone"]),
                          self.tmp / "om", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "declared_missing_entries"
                            for f in rep["integrity"]["failures"]))

    def test_expected_membership_dropped_entry_uncertified(self):
        c = self._chart(STACK)
        code, rep = audit(self._panel(
            [self._entry("generated/a", c)],
            expected_ids=["generated/a", "generated/b"],
            expected_counts={"generated": 2, "reference": 0}),
            self.tmp / "oe", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "missing_expected_entry"
                            for f in rep["integrity"]["failures"]))

    def test_nonfinite_origin_uncertified(self):
        c = self._chart(STACK)
        e = self._entry("generated/a", c)
        e["origin_s"] = float("inf")
        # json can't serialize inf via our _panel; write manually
        p = self.tmp / "oo.json"
        p.write_text(json.dumps({"schema_version": 1, "entries": [e]}).replace(
            "Infinity", "1e999"))
        code, rep = audit(p, self.tmp / "oo", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "nonfinite_origin"
                            for f in rep["integrity"]["failures"]))

    def test_standard_scenario_mismatch_uncertified(self):
        c = self._chart(STACK)
        code, rep = audit(self._panel([self._entry("generated/a", c)],
                                      standard_scenario={"njs": 99.0, "lifetime_s": 0.6}),
                          self.tmp / "os", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "standard_scenario_mismatch"
                            for f in rep["integrity"]["failures"]))

    def test_authored_record_altered_uncertified(self):
        chart = self.tmp / "ExpertPlus.dat"
        chart.write_text(json.dumps(STACK))
        info = self.tmp / "Info.dat"
        info.write_text(json.dumps({"_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
                "_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 20,
                "_noteJumpStartBeatOffset": -0.5}]}]}))
        e = self._entry("generated/a", chart)
        e["authored"] = {"status": "verified", "njs": 20.0, "offset_beats": -0.5,
                         "info_sha256": _sha(info)}
        code, _ = audit(self._panel([e]), self.tmp / "oa1", make_plots=False)
        self.assertEqual(code, 0)                 # live binding matches -> ok
        e["authored"] = dict(e["authored"], njs=99.0)   # tamper recorded NJS
        code, rep = audit(self._panel([e]), self.tmp / "oa2", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertTrue(any(f["reason"] == "authored_record_altered"
                            for f in rep["integrity"]["failures"]))

    def test_ambiguous_bpm_chart_unknown_still_certifies(self):
        chart = json.loads(json.dumps(STACK))
        chart["_customData"] = {"_BPMChanges": [{"_BPM": 120, "_time": 1},
                                                {"_BPM": 150, "_time": 2}]}
        c = self._chart(chart, "amb.dat")
        code, rep = audit(self._panel([self._entry("generated/a", c)]),
                          self.tmp / "o6", make_plots=False)
        self.assertEqual(code, 0)   # integrity ok; the chart itself is unknown
        self.assertEqual(rep["maps"][0]["standard"]["status"], "chart_unknown")


if __name__ == "__main__":
    unittest.main()
