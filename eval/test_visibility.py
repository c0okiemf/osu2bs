"""E01 visibility contract tests (Task 1): HJD jump settings, scene loading,
and Info.dat authored-setting binding. Oracles are hand-computed.
  .venv/bin/python -m unittest eval.test_visibility -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.visibility import (STD_LIFETIME_S, STD_NJS, Face, Note,
                             covered_fraction, jump_settings, load_scene,
                             project_face, resolve_authored)
from eval.visibility_oracle import ray_fraction

O = (0.0, 0.0, 0.0)
TARGET = Face("target", -1, 1, -1, 1, 4)   # projects to [-.25,.25]^2 from O


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TestJumpSettings(unittest.TestCase):
    def test_spec_example(self):
        # BPM120 / NJS16 / offset -0.5 -> h=1.5, T=0.75s, half 12m, full 24m
        js = jump_settings(120.0, 16.0, -0.5)
        self.assertAlmostEqual(js["hjd_beats"], 1.5, places=12)
        self.assertAlmostEqual(js["lifetime_s"], 0.75, places=12)
        self.assertAlmostEqual(js["half_distance_m"], 12.0, places=12)
        self.assertAlmostEqual(js["full_distance_m"], 24.0, places=12)

    def test_quarter_beat_clamp(self):
        # a large negative offset clamps h at 0.25 beats
        js = jump_settings(120.0, 16.0, -100.0)
        self.assertAlmostEqual(js["hjd_beats"], 0.25, places=12)

    def test_halving_boundary(self):
        # at NJS 16 / BPM 120, h0 halves 4->2 (2*16*0.5*4=64>35.998; =32 stops)
        js = jump_settings(120.0, 16.0, 0.0)
        self.assertAlmostEqual(js["hjd_beats"], 2.0, places=12)
        # a slower song keeps h0 at 4 (2*16*(60/240)*4 = 32 <= 35.998)
        js2 = jump_settings(240.0, 16.0, 0.0)
        self.assertAlmostEqual(js2["hjd_beats"], 4.0, places=12)

    def test_invalid_inputs_raise(self):
        for bad in [(0.0, 16.0, 0.0), (120.0, 0.0, 0.0), (120.0, -1.0, 0.0),
                    (120.0, 16.0, float("nan")), (float("inf"), 16.0, 0.0)]:
            with self.assertRaises(ValueError):
                jump_settings(*bad)


V2 = {"_version": "2.0.0", "_obstacles": [], "_notes": [
    {"_time": 2, "_type": 0, "_lineIndex": 1, "_lineLayer": 0, "_cutDirection": 1},
    {"_time": 2, "_type": 1, "_lineIndex": 2, "_lineLayer": 1, "_cutDirection": 8},
    {"_time": 4, "_type": 3, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 0},
]}


class TestLoadScene(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _chart(self, obj, name="c.dat"):
        p = self.tmp / name
        p.write_text(json.dumps(obj))
        return p

    def _entry(self, chart, bpm=120.0, authored=None, role="reference"):
        return {"id": "x", "role": role, "chart": str(chart), "sha256": _sha(chart),
                "bpm": bpm, "origin_s": 0, "timing_source": str(chart),
                "timing_source_sha256": _sha(chart),
                "authored": authored or {"status": "unknown", "reason": "n/a"}}

    def test_standard_mode(self):
        s = load_scene(self._entry(self._chart(V2)), "standard")
        self.assertEqual(s["njs"], STD_NJS)
        self.assertEqual(s["lifetime_s"], STD_LIFETIME_S)
        self.assertEqual(s["settings_status"], "standard")
        self.assertEqual(len(s["notes"]), 2)                 # bomb excluded
        self.assertEqual(s["ignored"]["bombs"], 1)
        # note hit time in seconds: 2 beats @120bpm = 1.0s
        self.assertAlmostEqual(s["notes"][0].hit_s, 1.0, places=12)
        self.assertIn(Note("note:1", 1.0, 2, 1, 1, 8), s["notes"])

    def test_authored_verified_uses_hjd(self):
        # authored mode now verifies the binding LIVE against Info.dat, so a
        # matching Info file must exist and the record must match it
        chart = self._chart(V2, "ExpertPlus.dat")
        info = self.tmp / "Info.dat"
        info.write_text(json.dumps({"_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
                "_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 16,
                "_noteJumpStartBeatOffset": -0.5}]}]}))
        e = self._entry(chart, authored={"status": "verified", "njs": 16.0,
                        "offset_beats": -0.5, "info_sha256": _sha(info)})
        s = load_scene(e, "authored")
        self.assertEqual(s["settings_status"], "verified")
        self.assertEqual(s["njs"], 16.0)
        self.assertAlmostEqual(s["lifetime_s"], 0.75, places=12)  # HJD at bpm120

    def test_authored_altered_record_rejected(self):
        # Info says njs 16 but the panel record claims 99 -> altered -> unknown
        chart = self._chart(V2, "ExpertPlus.dat")
        info = self.tmp / "Info.dat"
        info.write_text(json.dumps({"_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
                "_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 16,
                "_noteJumpStartBeatOffset": -0.5}]}]}))
        e = self._entry(chart, authored={"status": "verified", "njs": 99.0,
                        "offset_beats": -0.5, "info_sha256": _sha(info)})
        s = load_scene(e, "authored")
        self.assertEqual(s["settings_status"], "unknown")
        self.assertEqual(s["settings_reason"], "authored_record_altered")

    def test_authored_unknown_no_fallback(self):
        e = self._entry(self._chart(V2),
                        authored={"status": "unknown", "reason": "no_saved_jump_settings"})
        s = load_scene(e, "authored")
        self.assertEqual(s["settings_status"], "unknown")
        self.assertIsNone(s["njs"])
        self.assertTrue(s["chart_unknown"])

    def test_ambiguous_bpm_chart_unknown(self):
        chart = dict(V2)
        chart = json.loads(json.dumps(V2))
        chart["_customData"] = {"_BPMChanges": [{"_BPM": 120, "_time": 4},
                                                {"_BPM": 130, "_time": 8}]}
        s = load_scene(self._entry(self._chart(chart, "bpm.dat")), "standard")
        self.assertTrue(s["chart_unknown"])
        self.assertTrue(s["scope"].startswith("ambiguous_timing"))

    def test_invalid_and_transform_notes_mark_chart_unknown(self):
        chart = {"_version": "2.0.0", "_obstacles": [], "_notes": [
            {"_time": 0, "_type": 0, "_lineIndex": 9, "_lineLayer": 0, "_cutDirection": 1},
            {"_time": 0, "_type": 0, "_lineIndex": 0, "_lineLayer": 0,
             "_cutDirection": 1, "_customData": {"_position": [1, 1]}},
            {"_time": 0, "_type": False, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 1},
        ]}
        s = load_scene(self._entry(self._chart(chart, "inv.dat")), "standard")
        self.assertEqual(len(s["invalid"]), 2)      # out_of_grid + noncolor bool
        self.assertEqual(len(s["unsupported"]), 1)  # custom transform
        self.assertTrue(s["chart_unknown"])         # any bad note -> unknown scene

    def test_dropped_unknown_blocker_does_not_clear_target(self):
        # review R1 exact fixture: a transformed earlier blocker + a later valid
        # target must NOT trace as a clear target
        chart = {"_version": "2.0.0", "_notes": [
            {"_time": 1, "_type": 0, "_lineIndex": 1, "_lineLayer": 2,
             "_cutDirection": 1, "_customData": {"_position": [0, 2]}},
            {"_time": 1.2, "_type": 0, "_lineIndex": 1, "_lineLayer": 2,
             "_cutDirection": 1}]}
        s = load_scene(self._entry(self._chart(chart, "drop.dat"), bpm=60.0), "standard")
        self.assertTrue(s["chart_unknown"])
        self.assertEqual(trace_target(s, "note:1", TCFG)["status"], "chart_unknown")

    def test_bad_cut_direction_is_invalid(self):
        chart = {"_version": "2.0.0", "_notes": [
            {"_time": 0, "_type": 0, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 99}]}
        s = load_scene(self._entry(self._chart(chart, "dir.dat")), "standard")
        self.assertTrue(s["chart_unknown"])
        self.assertEqual(s["invalid"][0]["reason"], "bad_cut_direction")

    def test_duplicate_cell_occupancy_is_invalid(self):
        chart = {"_version": "2.0.0", "_notes": [
            {"_time": 1, "_type": 0, "_lineIndex": 1, "_lineLayer": 1, "_cutDirection": 1},
            {"_time": 1, "_type": 1, "_lineIndex": 1, "_lineLayer": 1, "_cutDirection": 1}]}
        s = load_scene(self._entry(self._chart(chart, "dup.dat")), "standard")
        self.assertTrue(s["chart_unknown"])
        self.assertTrue(any(x["reason"] == "duplicate_cell_occupancy"
                            for x in s["invalid"]))

    def test_rotation_event_marks_scene_unknown(self):
        chart = {"_version": "2.0.0", "_notes": [
            {"_time": 1, "_type": 0, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 1}],
            "_events": [{"_time": 0, "_type": 14, "_value": 0}]}  # value 0 = 60L
        s = load_scene(self._entry(self._chart(chart, "rot.dat")), "standard")
        self.assertTrue(s["chart_unknown"])
        self.assertTrue(s["scope"].startswith("unsupported_transform"))


class TestResolveAuthored(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _info(self, beatmaps):
        p = self.tmp / "Info.dat"
        p.write_text(json.dumps({"_difficultyBeatmapSets": [
            {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": beatmaps}]}))
        return p

    def _chart(self, name, body="{}"):
        p = self.tmp / name
        p.write_text(body)
        return p

    def test_exact_filename_hash_binds(self):
        c = self._chart("ExpertPlus.dat", '{"_version":"2.0.0"}')
        self._info([{"_beatmapFilename": "ExpertPlus.dat",
                     "_noteJumpMovementSpeed": 20, "_noteJumpStartBeatOffset": -0.5}])
        r = resolve_authored(c, _sha(c))
        self.assertEqual(r["status"], "verified")
        self.assertEqual((r["njs"], r["offset_beats"]), (20.0, -0.5))

    def test_hash_mismatch_unknown(self):
        c = self._chart("ExpertPlus.dat", '{"_version":"2.0.0"}')
        self._info([{"_beatmapFilename": "ExpertPlus.dat",
                     "_noteJumpMovementSpeed": 20}])
        r = resolve_authored(c, "deadbeef" * 8)   # wrong hash
        self.assertEqual(r["status"], "unknown")
        self.assertEqual(r["reason"], "info_filename_hash_mismatch")

    def test_no_matching_filename_unknown(self):
        # Spaceman case: chart is ExpertPlus.dat, Info only names ExpertPlusStandard.dat
        c = self._chart("ExpertPlus.dat", '{"_version":"2.0.0"}')
        self._info([{"_beatmapFilename": "ExpertPlusStandard.dat",
                     "_noteJumpMovementSpeed": 19}])
        r = resolve_authored(c, _sha(c))
        self.assertEqual(r["reason"], "no_info_entry_binds_this_filename")

    def test_ambiguous_multiple_entries_unknown(self):
        c = self._chart("ExpertPlus.dat", '{"_version":"2.0.0"}')
        self._info([{"_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 20},
                    {"_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": 18}])
        r = resolve_authored(c, _sha(c))
        self.assertEqual(r["reason"], "ambiguous_info_entries")

    def test_absent_njs_unknown(self):
        c = self._chart("ExpertPlus.dat", '{"_version":"2.0.0"}')
        self._info([{"_beatmapFilename": "ExpertPlus.dat"}])   # no NJS
        r = resolve_authored(c, _sha(c))
        self.assertEqual(r["reason"], "absent_or_invalid_njs_offset")


class TestGeometry(unittest.TestCase):
    def _cf(self, blockers):
        return covered_fraction(TARGET, blockers, O)

    def test_target_alone_zero(self):
        self.assertEqual(self._cf([]), 0.0)
        # a centre-cell-like note alone still occludes nothing
        self.assertEqual(covered_fraction(Face("c", -.25, .25, -.25, .25, 4), [], O), 0.0)

    def test_full_cover(self):
        b = Face("b", -0.5, 0.5, -0.5, 0.5, 2)   # projects to [-.25,.25]^2
        self.assertAlmostEqual(self._cf([b]), 1.0, places=9)

    def test_half_cover(self):
        b = Face("b", 0, 0.5, -0.5, 0.5, 2)
        self.assertAlmostEqual(self._cf([b]), 0.5, places=9)

    def test_quarter_cover(self):
        b = Face("b", 0, 0.5, 0, 0.5, 2)
        self.assertAlmostEqual(self._cf([b]), 0.25, places=9)

    def test_farther_blocker_zero(self):
        b = Face("b", -0.5, 0.5, -0.5, 0.5, 8)   # behind the target
        self.assertEqual(self._cf([b]), 0.0)

    def test_equal_depth_no_occlusion(self):
        b = Face("b", -0.5, 0.5, -0.5, 0.5, 4)   # coplanar with target
        self.assertEqual(self._cf([b]), 0.0)

    def test_disjoint_and_tangent_zero(self):
        # projects entirely outside the target face
        b = Face("b", 2, 3, -0.5, 0.5, 2)
        self.assertEqual(self._cf([b]), 0.0)
        edge = Face("e", 0.5, 1.0, -0.5, 0.5, 2)  # left edge touches target right edge
        self.assertEqual(self._cf([edge]), 0.0)

    def test_two_disjoint_halves_full(self):
        b1 = Face("b1", -0.5, 0, -0.5, 0.5, 2)
        b2 = Face("b2", 0, 0.5, -0.5, 0.5, 2)
        self.assertAlmostEqual(self._cf([b1, b2]), 1.0, places=9)

    def test_two_identical_halves_half(self):
        b = Face("b", -0.5, 0, -0.5, 0.5, 2)
        self.assertAlmostEqual(self._cf([b, b]), 0.5, places=9)

    def test_overlapping_union_three_quarters(self):
        b1 = Face("b1", -0.5, 0.5, -0.5, 0, 2)   # bottom half
        b2 = Face("b2", -0.5, 0, -0.5, 0.5, 2)   # left half
        self.assertAlmostEqual(self._cf([b1, b2]), 0.75, places=9)

    def test_target_ignored_by_id(self):
        # the target listed as its own blocker must not self-occlude
        self.assertEqual(covered_fraction(TARGET, [TARGET], O), 0.0)

    def test_duplicate_id_different_geometry_raises(self):
        with self.assertRaises(ValueError):
            self._cf([Face("d", 0, 0.5, 0, 0.5, 2), Face("d", -0.5, 0, -0.5, 0, 2)])

    def test_project_nonpositive_depth_raises(self):
        with self.assertRaises(ValueError):
            project_face(Face("x", -1, 1, -1, 1, 0.0), O)

    def test_off_axis_alignment_from_displaced_eye(self):
        # an outer (non-banned, col-3-ish) blocker aligns onto the target when
        # the eye is shifted right: its projection [-.125,.125] overlaps the
        # target's [-.475,.025], so location alone is not the occlusion story
        eye = (0.9, 0.0, 0.0)
        b = Face("b", 0.65, 1.15, -0.25, 0.25, 2)
        self.assertGreater(covered_fraction(TARGET, [b], eye), 0.0)


class TestOracleAgreement(unittest.TestCase):
    SCENES = {
        "full": [Face("b", -0.5, 0.5, -0.5, 0.5, 2)],
        "half": [Face("b", 0, 0.5, -0.5, 0.5, 2)],
        "quarter": [Face("b", 0, 0.5, 0, 0.5, 2)],
        "overlap075": [Face("b1", -0.5, 0.5, -0.5, 0, 2),
                       Face("b2", -0.5, 0, -0.5, 0.5, 2)],
        "two_halves": [Face("b1", -0.5, 0, -0.5, 0.5, 2),
                       Face("b2", 0, 0.5, -0.5, 0.5, 2)],
    }

    def test_oracle_matches_analytic(self):
        for name, blk in self.SCENES.items():
            a = covered_fraction(TARGET, blk, O)
            r = ray_fraction(TARGET, blk, O, samples=400)
            self.assertLess(abs(a - r), 0.01, f"{name}: analytic {a} vs oracle {r}")

    def test_invariance_mirror_translate_scale(self):
        blk = self.SCENES["overlap075"]
        base = covered_fraction(TARGET, blk, O)

        def mir(f):
            return Face(f.id, -f.x1, -f.x0, f.y0, f.y1, f.z)
        self.assertAlmostEqual(
            covered_fraction(mir(TARGET), [mir(b) for b in blk],
                             (-O[0], O[1], O[2])), base, places=9)

        def tr(f, d):
            return Face(f.id, f.x0 + d[0], f.x1 + d[0], f.y0 + d[1], f.y1 + d[1], f.z + d[2])
        d = (3.0, -2.0, 5.0)
        self.assertAlmostEqual(
            covered_fraction(tr(TARGET, d), [tr(b, d) for b in blk],
                             (O[0] + d[0], O[1] + d[1], O[2] + d[2])), base, places=9)

        def sc(f, k):
            return Face(f.id, f.x0 * k, f.x1 * k, f.y0 * k, f.y1 * k, f.z * k)
        k = 2.5
        self.assertAlmostEqual(
            covered_fraction(sc(TARGET, k), [sc(b, k) for b in blk],
                             tuple(c * k for c in O)), base, places=9)


from eval.visibility import trace_target

TCFG = {"eye": (0.0, 1.60, -0.65), "face_side_m": 0.5, "step_s": 0.010}


def _scene(notes, njs=16.0, T=0.6):
    return {"notes": notes, "njs": njs, "lifetime_s": T}


class TestTraces(unittest.TestCase):
    # from a central eye (x=0), only near-axis, near-in-time stacks occlude;
    # a col2/row2 stack at a 0.1s gap gives ~0.98 coverage (probed), col2/row1
    # is a banned cell.
    def test_isolated_target_fully_clear(self):
        s = _scene([Note("t", 2.0, 2, 2, 0, 1)])
        m = trace_target(s, "t", TCFG)
        self.assertEqual(m["max_coverage"], 0.0)
        self.assertAlmostEqual(m["terminal_clear_s"], 0.6, places=6)
        self.assertAlmostEqual(m["early_clear_s"], 0.6, places=6)
        self.assertEqual(m["time_ge_0.5_s"], 0.0)

    def test_nonbanned_stack_blocks_then_clears_on_removal(self):
        # blocker hits at 1.9 (0.1s before target), same non-banned cell.
        s = _scene([Note("t", 2.0, 2, 2, 0, 1), Note("b", 1.9, 2, 2, 0, 1)])
        m = trace_target(s, "t", TCFG, detail=True)
        self.assertGreaterEqual(m["max_coverage"], 0.5)
        self.assertFalse(any(c["banned"] for c in m["hidden_occluder_cells"]))
        # blocker removed at 1.9 -> every later sample is fully clear
        self.assertTrue(all(s2["coverage"] == 0.0 for s2 in m["samples"]
                            if s2["start_s"] >= 1.9 - 1e-9))
        self.assertAlmostEqual(m["terminal_clear_s"], 0.1, places=9)
        # complete denominator: sample cells tile the whole approach interval
        self.assertAlmostEqual(sum(s2["end_s"] - s2["start_s"]
                                   for s2 in m["samples"]), 0.6, places=9)

    def test_banned_occluder_classified_banned(self):
        s = _scene([Note("t", 2.0, 2, 1, 0, 1), Note("b", 1.94, 2, 1, 0, 1)])
        m = trace_target(s, "t", TCFG)
        self.assertGreaterEqual(m["max_coverage"], 0.5)
        self.assertTrue(any(c["banned"] for c in m["hidden_occluder_cells"]))

    def test_blocker_before_spawn_no_occlusion(self):
        s = _scene([Note("t", 2.0, 2, 2, 0, 1), Note("b", 1.3, 2, 2, 0, 1)])
        m = trace_target(s, "t", TCFG)
        self.assertEqual(m["max_coverage"], 0.0)   # blocker gone before spawn

    def test_coincident_hit_not_a_blocker(self):
        s = _scene([Note("t", 2.0, 2, 2, 0, 1), Note("b", 2.0, 2, 2, 0, 1)])
        m = trace_target(s, "t", TCFG)
        self.assertEqual(m["max_coverage"], 0.0)   # equal depth, never nearer

    def test_substep_lifetime_runs(self):
        m = trace_target(_scene([Note("t", 1.0, 2, 2, 0, 1)], T=0.005), "t", TCFG)
        self.assertEqual(m["status"], "ok")

    def test_njs_effect_at_fixed_T_is_small(self):
        # coverage is NOT njs-invariant (the eye z-offset breaks pure depth
        # scaling) but the effect at fixed T is minor and higher NJS need not
        # improve visibility -- a sensitivity to report, not an invariance
        notes = [Note("t", 2.0, 2, 2, 0, 1), Note("b", 1.9, 2, 2, 0, 1)]
        a = trace_target(_scene(notes, njs=12.0), "t", TCFG)
        b = trace_target(_scene(notes, njs=20.0), "t", TCFG)
        self.assertLess(abs(a["area_time_integral"] - b["area_time_integral"]), 0.05)

    def test_settings_unknown_scene(self):
        s = {"notes": [Note("t", 2.0, 2, 2, 0, 1)], "njs": None, "lifetime_s": None}
        self.assertEqual(trace_target(s, "t", TCFG)["status"], "settings_unknown")


class TestFrozenPanel(unittest.TestCase):
    def test_panel_matches_spec(self):
        panel = json.loads((Path(__file__).parent / "visibility_panel.json").read_text())
        self.assertEqual(panel["schema_version"], 1)
        self.assertEqual(panel["counts"],
                         {"generated": 5, "reference": 4, "authored_verified": 3})
        ids = [e["id"] for e in panel["entries"]]
        self.assertEqual(len(ids), len(set(ids)))       # unique
        for e in panel["entries"]:
            self.assertTrue(Path(e["chart"]).exists())
            self.assertEqual(_sha(e["chart"]), e["sha256"])   # frozen hash holds


if __name__ == "__main__":
    unittest.main()
