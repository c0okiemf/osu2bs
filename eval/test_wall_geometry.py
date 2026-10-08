"""Construction-known tests for the wall-corridor diagnostic (A06/D01).

Oracles come from the design fixture table, not a second call to the solver.
  .venv/bin/python -m unittest eval.test_wall_geometry -v
"""
import math
import unittest

from eval.wall_geometry import Wall, analyze_walls, normalize_v2, safe_intervals


def _norm(walls, unknown=None):
    return {"walls": list(walls), "unknown": list(unknown or []),
            "invalid": [], "chart_unknown": False, "scope": "wall_only"}


def _tot_len(intervals):
    return sum(hi - lo for lo, hi in intervals)


def _v2(*obstacles, version="2.0.0"):
    return {"_version": version, "_obstacles": list(obstacles)}


def _ob(t, dur, li, w, typ=0):
    return {"_time": t, "_duration": dur, "_lineIndex": li, "_width": w,
            "_type": typ}


class TestNormalizeV2(unittest.TestCase):
    def test_time_conversion_exact(self):
        raw = _v2(_ob(2, 4, 0, 2, 0))
        r = normalize_v2(raw, seconds_per_beat=0.5, origin_s=7.0)
        self.assertEqual(r["walls"], [Wall("wall:0", 8.0, 10.0, 0.0, 2.0)])
        self.assertFalse(r["chart_unknown"])
        self.assertEqual(r["unknown"], [])
        self.assertEqual(r["invalid"], [])
        self.assertEqual(r["scope"], "wall_only")

    def test_supported_full_height_outer(self):
        r = normalize_v2(_v2(_ob(0, 2, 3, 1, 0)), 0.5, 0.0)
        self.assertEqual(r["walls"], [Wall("wall:0", 0.0, 1.0, 3.0, 4.0)])

    def test_ids_retain_raw_index(self):
        # a crouch wall between two full-height walls: ids stay wall:0/1/2
        r = normalize_v2(_v2(_ob(0, 1, 0, 1, 0), _ob(0, 1, 1, 1, 1),
                             _ob(0, 1, 3, 1, 0)), 1.0, 0.0)
        self.assertEqual([w.id for w in r["walls"]], ["wall:0", "wall:2"])
        self.assertEqual(r["unknown"][0]["id"], "wall:1")

    def test_crouch_wall_is_unknown_with_known_times(self):
        r = normalize_v2(_v2(_ob(4, 2, 1, 2, 1)), 0.5, 0.0)
        self.assertEqual(r["walls"], [])
        u = r["unknown"][0]
        self.assertEqual((u["start_s"], u["end_s"]), (2.0, 3.0))
        self.assertFalse(r["chart_unknown"])

    def test_extended_geometry_out_of_domain_is_unknown(self):
        # width pushes x1 beyond lane 4 -> extended geometry, known times
        r = normalize_v2(_v2(_ob(0, 1, 2, 3, 0)), 1.0, 0.0)
        self.assertEqual(r["walls"], [])
        self.assertEqual(r["unknown"][0]["reason"],
                         "extended_or_out_of_domain_lane")

    def test_fractional_lane_is_unknown(self):
        r = normalize_v2(_v2(_ob(0, 1, 1.5, 1, 0)), 1.0, 0.0)
        self.assertEqual(r["walls"], [])
        self.assertEqual(r["unknown"][0]["reason"],
                         "extended_or_out_of_domain_lane")

    def test_zero_duration_is_invalid(self):
        r = normalize_v2(_v2(_ob(0, 0, 0, 1, 0)), 1.0, 0.0)
        self.assertEqual(r["walls"], [])
        self.assertEqual(r["invalid"][0]["reason"], "nonpositive_duration")

    def test_zero_width_is_invalid(self):
        r = normalize_v2(_v2(_ob(0, 1, 0, 0, 0)), 1.0, 0.0)
        self.assertEqual(r["invalid"][0]["reason"], "nonpositive_width")

    def test_nan_and_inf_are_invalid(self):
        r = normalize_v2(_v2(_ob(float("nan"), 1, 0, 1, 0),
                             _ob(0, float("inf"), 0, 1, 0)), 1.0, 0.0)
        self.assertEqual(r["walls"], [])
        self.assertEqual(len(r["invalid"]), 2)
        self.assertTrue(all(x["reason"] == "nonfinite_or_missing_field"
                            for x in r["invalid"]))

    def test_missing_field_is_invalid(self):
        r = normalize_v2(_v2({"_time": 0, "_duration": 1, "_lineIndex": 0,
                              "_type": 0}), 1.0, 0.0)  # no _width
        self.assertEqual(r["invalid"][0]["reason"],
                         "nonfinite_or_missing_field")

    def test_bool_is_not_a_number(self):
        r = normalize_v2(_v2(_ob(True, 1, 0, 1, 0)), 1.0, 0.0)
        self.assertEqual(r["invalid"][0]["reason"],
                         "nonfinite_or_missing_field")

    def test_v3_version_marks_chart_unknown(self):
        r = normalize_v2({"_version": "3.2.0", "obstacles": []}, 0.5, 0.0)
        self.assertTrue(r["chart_unknown"])
        self.assertEqual(r["walls"], [])
        self.assertTrue(r["scope"].startswith("unsupported_schema"))

    def test_missing_version_marks_chart_unknown(self):
        r = normalize_v2({"_obstacles": [_ob(0, 1, 0, 1, 0)]}, 0.5, 0.0)
        self.assertTrue(r["chart_unknown"])
        self.assertEqual(r["walls"], [])

    def test_bad_timing_provenance_marks_chart_unknown(self):
        self.assertTrue(normalize_v2(_v2(_ob(0, 1, 0, 1, 0)), 0.0, 0.0)
                        ["chart_unknown"])
        self.assertTrue(normalize_v2(_v2(_ob(0, 1, 0, 1, 0)),
                                     float("nan"), 0.0)["chart_unknown"])

    # --- Task 0 repair: fail-closed scope (review review 2026-09-20) ---

    def test_type_false_is_invalid_not_full_height(self):
        # bool _type must NOT be read as full-height (_type=0)
        r = normalize_v2(_v2(_ob(0, 1, 0, 1, False)), 1.0, 0.0)
        self.assertEqual(r["walls"], [])
        self.assertEqual(r["invalid"][0]["reason"], "noninteger_type")

    def test_noninteger_type_is_invalid(self):
        r = normalize_v2(_v2(_ob(0, 1, 0, 1, 1.5)), 1.0, 0.0)
        self.assertEqual(r["invalid"][0]["reason"], "noninteger_type")

    def test_unresolvable_timestamp_marks_chart_unknown(self):
        r = normalize_v2(_v2(_ob(float("nan"), 1, 0, 1, 0)), 1.0, 0.0)
        self.assertTrue(r["chart_unknown"])

    def test_overflow_timestamp_is_invalid_and_chart_unknown(self):
        r = normalize_v2(_v2(_ob(1e308, 1e308, 0, 1, 0)), 1e308, 0.0)
        self.assertTrue(r["chart_unknown"])

    def test_obstacle_custom_transform_is_unknown(self):
        o = _ob(0, 1, 0, 1, 0)
        o["_customData"] = {"_position": [1.0, 0.0], "_scale": [2.0, 1.0]}
        r = normalize_v2(_v2(o), 1.0, 0.0)
        self.assertEqual(r["walls"], [])
        self.assertEqual(r["unknown"][0]["reason"], "custom_geometry_transform")

    def test_obstacle_cosmetic_customdata_is_supported(self):
        # cosmetic-only _customData (color) must NOT reject the wall
        o = _ob(0, 1, 0, 1, 0)
        o["_customData"] = {"_color": [1.0, 0.0, 0.0]}
        r = normalize_v2(_v2(o), 1.0, 0.0)
        self.assertEqual(len(r["walls"]), 1)

    def test_ambiguous_bpm_changes_mark_chart_unknown(self):
        raw = _v2(_ob(0, 1, 0, 1, 0))
        raw["_customData"] = {"_BPMChanges": [{"_BPM": 120.0, "_time": 4},
                                              {"_BPM": 128.0, "_time": 8}]}
        r = normalize_v2(raw, 0.5, 0.0)
        self.assertTrue(r["chart_unknown"])
        self.assertTrue(r["scope"].startswith("ambiguous_timing"))

    def test_empty_editor_metadata_is_supported(self):
        raw = _v2(_ob(0, 1, 0, 1, 0))
        raw["_customData"] = {"_BPMChanges": [], "_bookmarks": [], "_time": 0}
        r = normalize_v2(raw, 1.0, 0.0)
        self.assertFalse(r["chart_unknown"])
        self.assertEqual(len(r["walls"]), 1)

    def test_rotation_event_marks_chart_unknown(self):
        raw = _v2(_ob(0, 1, 0, 1, 0))
        raw["_events"] = [{"_time": 2, "_type": 14, "_value": 3}]  # lane rotation
        r = normalize_v2(raw, 0.5, 0.0)
        self.assertTrue(r["chart_unknown"])

    def test_zero_value_rotation_event_is_unsupported(self):
        # rotation value 0 is a 60-degree LEFT rotation in the legacy enum,
        # NOT a no-op -> any type 14/15 event is unsupported (review 2026-09-20)
        raw = _v2(_ob(0, 1, 0, 1, 0))
        raw["_events"] = [{"_time": 2, "_type": 14, "_value": 0}]
        self.assertTrue(normalize_v2(raw, 1.0, 0.0)["chart_unknown"])

    def test_lighting_events_are_supported(self):
        raw = _v2(_ob(0, 1, 0, 1, 0))
        raw["_events"] = [{"_time": 2, "_type": 0, "_value": 1}]  # lighting
        self.assertFalse(normalize_v2(raw, 1.0, 0.0)["chart_unknown"])

    def test_legacy_type100_bpm_event_unsupported(self):
        raw = _v2(_ob(0, 1, 0, 1, 0))
        raw["_events"] = [{"_time": 2, "_type": 100, "_value": 0, "_floatValue": 128}]
        self.assertTrue(normalize_v2(raw, 1.0, 0.0)["chart_unknown"])

    def test_130_and_260_bpm_same_physical_walls(self):
        # identical physical timeline at 130 vs 260 BPM: a beat at 130 BPM lasts
        # 2x a beat at 260 BPM, so the 260 chart doubles _time/_duration.
        spb130 = 60.0 / 130.0
        spb260 = 60.0 / 260.0
        a = normalize_v2(_v2(_ob(4, 2, 0, 1, 0)), spb130, 0.0)["walls"][0]
        b = normalize_v2(_v2(_ob(8, 4, 0, 1, 0)), spb260, 0.0)["walls"][0]
        self.assertTrue(math.isclose(a.start_s, b.start_s, abs_tol=1e-9))
        self.assertTrue(math.isclose(a.end_s, b.end_s, abs_tol=1e-9))


class TestSafeIntervals(unittest.TestCase):
    R = 0.25

    def test_no_walls(self):
        self.assertEqual(safe_intervals([], self.R), [(0.25, 3.75)])

    def test_outer_left(self):
        w = Wall("l", 0.0, 1.0, 0.0, 1.0)
        self.assertEqual(safe_intervals([w], self.R), [(1.25, 3.75)])

    def test_outer_right(self):
        w = Wall("r", 0.0, 1.0, 3.0, 4.0)
        self.assertEqual(safe_intervals([w], self.R), [(0.25, 2.75)])

    def test_both_outer_overlap_not_empty(self):
        wl = Wall("l", 0.0, 1.0, 0.0, 1.0)
        wr = Wall("r", 0.0, 1.0, 3.0, 4.0)
        self.assertEqual(safe_intervals([wl, wr], self.R), [(1.25, 2.75)])

    def test_central_wall_two_intervals_not_averaged(self):
        c = Wall("c", 0.0, 1.0, 1.0, 3.0)
        self.assertEqual(safe_intervals([c], self.R), [(0.25, 0.75), (3.25, 3.75)])

    def test_full_span_walls_empty(self):
        # left half [0,2] and right half [2,4] together cover the lane
        wl = Wall("l", 0.0, 1.0, 0.0, 2.0)
        wr = Wall("r", 0.0, 1.0, 2.0, 4.0)
        self.assertEqual(safe_intervals([wl, wr], self.R), [])

    def test_tangent_singleton_points(self):
        # r=0.5: wall[0,1] excludes (-0.5,1.5); wall[2,3] excludes (1.5,3.5).
        # The exclusions touch at 1.5 (open endpoints stay safe) -> singleton
        # [1.5,1.5]; the right wall is tangent to the domain edge -> [3.5,3.5].
        wl = Wall("l", 0.0, 1.0, 0.0, 1.0)
        wr = Wall("r", 0.0, 1.0, 2.0, 3.0)
        self.assertEqual(safe_intervals([wl, wr], 0.5), [(1.5, 1.5), (3.5, 3.5)])

    def test_duplicate_and_nested_union_preserves_geometry(self):
        a = Wall("a", 0.0, 1.0, 0.0, 2.0)
        dup = Wall("dup", 0.0, 1.0, 0.0, 2.0)
        nested = Wall("nest", 0.0, 1.0, 0.0, 1.0)
        self.assertEqual(safe_intervals([a, dup, nested], self.R),
                         safe_intervals([a], self.R))

    def test_radius_monotone_shrinkage(self):
        c = Wall("c", 0.0, 1.0, 1.0, 3.0)
        lens = [_tot_len(safe_intervals([c], r)) for r in (0.15, 0.25, 0.35)]
        self.assertGreater(lens[0], lens[1])
        self.assertGreater(lens[1], lens[2])

    def test_radius_over_two_empty_domain(self):
        self.assertEqual(safe_intervals([], 2.5), [])

    def test_mirror_symmetry(self):
        left = safe_intervals([Wall("l", 0.0, 1.0, 0.0, 1.0)], self.R)
        right = safe_intervals([Wall("r", 0.0, 1.0, 3.0, 4.0)], self.R)
        mirrored = sorted((4.0 - hi, 4.0 - lo) for lo, hi in right)
        self.assertEqual(left, mirrored)

    def test_source_order_invariance(self):
        wl = Wall("l", 0.0, 1.0, 0.0, 1.0)
        wr = Wall("r", 0.0, 1.0, 3.0, 4.0)
        self.assertEqual(safe_intervals([wl, wr], self.R),
                         safe_intervals([wr, wl], self.R))


class TestAnalyzeTravel(unittest.TestCase):
    R = 0.25

    def _trs(self, walls, unknown=None):
        return analyze_walls(_norm(walls, unknown), self.R)["transitions"]

    def test_alternate_outer_no_forced_dodge_zero_gap(self):
        wl = Wall("l", 0.0, 1.0, 0.0, 1.0)
        wr = Wall("r", 1.0, 2.0, 3.0, 4.0)
        tr = self._trs([wl, wr])
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0]["distance_lane"], 0.0)
        self.assertEqual(tr[0]["speed_lane_s"], 0.0)

    def test_alternate_outer_no_forced_dodge_small_gap(self):
        wl = Wall("l", 0.0, 1.0, 0.0, 1.0)
        wr = Wall("r", 1.1, 2.0, 3.0, 4.0)
        tr = self._trs([wl, wr])
        self.assertEqual(tr[0]["distance_lane"], 0.0)

    def test_half_switch_gap_0p1_lower_bound_5(self):
        left = Wall("left", 0.0, 1.0, 0.0, 2.0)
        right = Wall("right", 1.1, 2.0, 2.0, 4.0)
        tr = analyze_walls(_norm([left, right]), 0.25)["transitions"][0]
        self.assertLess(abs(tr["distance_lane"] - 0.5), 1e-9)
        self.assertLess(abs(tr["speed_lane_s"] - 5.0), 1e-9)
        self.assertEqual(tr["reason"], "lower_bound")

    def test_half_switch_gap_1s_lower_bound_half(self):
        left = Wall("left", 0.0, 1.0, 0.0, 2.0)
        right = Wall("right", 2.0, 3.0, 2.0, 4.0)   # 1s gap
        tr = self._trs([left, right])[0]
        self.assertLess(abs(tr["distance_lane"] - 0.5), 1e-9)
        self.assertLess(abs(tr["speed_lane_s"] - 0.5), 1e-9)

    def test_half_switch_same_time_discontinuous(self):
        left = Wall("left", 0.0, 1.0, 0.0, 2.0)
        right = Wall("right", 1.0, 2.0, 2.0, 4.0)   # zero gap
        tr = self._trs([left, right])[0]
        self.assertLess(abs(tr["distance_lane"] - 0.5), 1e-9)
        self.assertIsNone(tr["speed_lane_s"])
        self.assertEqual(tr["reason"], "discontinuous_model_constraint")

    def test_pairwise_zero_does_not_prove_global(self):
        # safe sets [0.25,0.75] -> [0.25,2.75] -> [2.25,3.75]; consecutive pairs
        # intersect (distance 0) but the three share no common position.
        a = Wall("a", 0.0, 1.0, 1.0, 4.0)
        b = Wall("b", 1.0, 2.0, 3.0, 4.0)
        c = Wall("c", 2.0, 3.0, 0.0, 2.0)
        res = analyze_walls(_norm([a, b, c]), 0.25)
        self.assertEqual([s["safe_intervals"] for s in res["segments"]],
                         [[(0.25, 0.75)], [(0.25, 2.75)], [(2.25, 3.75)]])
        for tr in res["transitions"]:
            self.assertEqual(tr["distance_lane"], 0.0)
        # first and last share no position -> no global zero-motion field exists
        self.assertNotIn("global_feasible", res)

    def test_single_wall_no_transition_no_initial_pose(self):
        self.assertEqual(self._trs([Wall("a", 0.0, 1.0, 0.0, 1.0)]), [])

    def test_empty_corridor_endpoint_is_skipped(self):
        # full-cover segment [1,2] between two outer walls
        pre = Wall("pre", 0.0, 1.0, 0.0, 1.0)
        el = Wall("el", 1.0, 2.0, 0.0, 2.0)
        er = Wall("er", 1.0, 2.0, 2.0, 4.0)
        post = Wall("post", 2.0, 3.0, 3.0, 4.0)
        tr = self._trs([pre, el, er, post])
        self.assertTrue(any(t["reason"] == "empty_endpoint" for t in tr))

    def test_empty_corridor_duration_exact(self):
        # overlap for exactly 0.2s -> empty_model_s == 0.2
        el = Wall("el", 0.0, 1.2, 0.0, 2.0)
        er = Wall("er", 1.0, 2.0, 2.0, 4.0)     # overlap [1.0,1.2)
        res = analyze_walls(_norm([el, er]), 0.25)
        self.assertLess(abs(res["counts"]["empty_model_s"] - 0.2), 1e-9)

    def test_unknown_between_skips_transition(self):
        wl = Wall("l", 0.0, 1.0, 0.0, 2.0)
        wr = Wall("r", 2.0, 3.0, 2.0, 4.0)
        crouch = [{"id": "u", "reason": "unsupported_type:1",
                   "start_s": 1.2, "end_s": 1.8}]
        tr = self._trs([wl, wr], unknown=crouch)
        self.assertEqual(tr[0]["reason"], "unknown_geometry")
        self.assertIsNone(tr[0]["distance_lane"])

    def test_time_translation_invariance(self):
        base = [Wall("l", 0.0, 1.0, 0.0, 2.0), Wall("r", 1.1, 2.0, 2.0, 4.0)]
        shift = [Wall("l", 10.0, 11.0, 0.0, 2.0), Wall("r", 11.1, 12.0, 2.0, 4.0)]
        t0 = self._trs(base)[0]
        t1 = self._trs(shift)[0]
        self.assertLess(abs(t0["distance_lane"] - t1["distance_lane"]), 1e-9)
        self.assertLess(abs(t0["speed_lane_s"] - t1["speed_lane_s"]), 1e-9)

    def test_grouped_events_end_and_start_same_time(self):
        # A ends and B starts at t=1 simultaneously; grouped handling must yield
        # two clean segments, not a spurious overlap.
        a = Wall("a", 0.0, 1.0, 0.0, 1.0)
        b = Wall("b", 1.0, 2.0, 3.0, 4.0)
        res = analyze_walls(_norm([a, b]), 0.25)
        self.assertEqual(len(res["segments"]), 2)
        self.assertEqual(res["segments"][0]["wall_ids"], ["a"])
        self.assertEqual(res["segments"][1]["wall_ids"], ["b"])

    def test_submillisecond_gap(self):
        wl = Wall("l", 0.0, 1.0, 0.0, 2.0)
        wr = Wall("r", 1.0003, 2.0, 2.0, 4.0)   # 0.3 ms gap
        tr = self._trs([wl, wr])[0]
        self.assertLess(abs(tr["gap_s"] - 0.0003), 1e-9)
        self.assertEqual(tr["reason"], "lower_bound")

    def test_130_260_bpm_same_physical_analysis(self):
        a = normalize_v2(_v2(_ob(0, 2, 0, 2, 0), _ob(4, 2, 2, 2, 0)),
                         60.0 / 130.0, 0.0)
        b = normalize_v2(_v2(_ob(0, 4, 0, 2, 0), _ob(8, 4, 2, 2, 0)),
                         60.0 / 260.0, 0.0)
        ra = analyze_walls(a, 0.25)
        rb = analyze_walls(b, 0.25)
        self.assertEqual([s["safe_intervals"] for s in ra["segments"]],
                         [s["safe_intervals"] for s in rb["segments"]])

    def test_chart_unknown_yields_no_segments(self):
        r = analyze_walls({"walls": [], "unknown": [], "invalid": [],
                           "chart_unknown": True, "scope": "unsupported_schema:3"},
                          0.25)
        self.assertEqual(r["segments"], [])
        self.assertTrue(r["chart_unknown"])

    def test_no_nan_or_inf_in_output(self):
        import json
        res = analyze_walls(_norm([Wall("l", 0.0, 1.0, 0.0, 2.0),
                                   Wall("r", 1.0, 2.0, 2.0, 4.0)]), 0.25)
        json.dumps(res, allow_nan=False)   # raises if Infinity/NaN leaked


if __name__ == "__main__":
    unittest.main()
