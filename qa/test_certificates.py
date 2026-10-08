"""Comparator Task 2 tests: typed contradictions + independent
certificates.
  .venv/bin/python -m unittest qa.test_certificates -q
"""
import json
import tempfile
import unittest
from pathlib import Path

from qa.certificates import (CHECKER_VERSION, contradictions,
                             load_speed_warning, make_reach_certificate,
                             make_wall_certificate, verify_certificate)

WARN = {"speed_warning": {"value": 14.07, "sha256": "x" * 64}}


def scene(notes=None, walls=None, scope=None, declared=None):
    s = {"scope": scope, "chart_sha256": "aa" * 32,
         "info_sha256": "bb" * 32, "bombs": [], "walls": walls or [],
         "settings": {"njs": 18},
         "notes": notes if notes is not None else
         [(1.0, 1, 0, 0, 1), (2.0, 2, 0, 1, 0)]}
    if declared is not None:
        s["declared_chart_sha256"] = declared
    return s


class TestStructural(unittest.TestCase):
    def test_nonfinite_and_bool_and_missing(self):
        r = contradictions(scene(notes=[(float("inf"), 1, 0, 0, 1),
                                        (1.0, True, 0, 0, 1),
                                        (2.0, 1, None, 0, 1)]), WARN)
        kinds = {i["kind"] for i in r["structural"]}
        self.assertEqual(r["status"], "STRUCTURAL_CONTRADICTION")
        self.assertEqual(kinds, {"nonfinite", "bool_as_int",
                                 "missing_required"})

    def test_negative_wall_duration(self):
        r = contradictions(scene(walls=[(1.0, 0, 0, -1.0, 4)]), WARN)
        self.assertTrue(any(i["kind"] == "malformed_duration"
                            for i in r["structural"]))

    def test_asset_mismatch(self):
        r = contradictions(scene(declared="cc" * 32), WARN)
        self.assertTrue(any(i["kind"] == "asset_mismatch"
                            for i in r["structural"]))

    def test_nonfinite_raw_time_is_scope_failure(self):
        from qa.scene import read_scene
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "c.dat").write_text(json.dumps(
                {"_version": "2.0.0", "_obstacles": [],
                 "_notes": [{"_time": 1e400, "_type": 0, "_lineIndex": 1,
                             "_lineLayer": 0, "_cutDirection": 1}]}))
            (d / "Info.dat").write_text(json.dumps(
                {"_beatsPerMinute": 120, "_difficultyBeatmapSets": []}))
            self.assertEqual(read_scene(d / "c.dat", d / "Info.dat")
                             ["scope"], "invalid_note_time")


class TestUnknowns(unittest.TestCase):
    def test_duplicate_occupancy_is_unknown_not_structural(self):
        dup = scene(notes=[(1.0, 1, 0, 0, 1), (1.0, 1, 0, 0, 1),
                           (2.0, 2, 0, 1, 0)])
        r = contradictions(dup, WARN)
        self.assertTrue(r["unknowns"])
        self.assertFalse(r["structural"])
        self.assertEqual(r["unknowns"][0]["type"], "duplicate_occupancy")
        self.assertEqual(r["status"], "NO_CONTRADICTION_FOUND")

    def test_unsupported_mechanics_scope_is_unknown(self):
        r = contradictions(scene(scope="unsupported_v3_mechanics"), WARN)
        self.assertEqual(r["status"], "UNKNOWN")
        self.assertEqual(r["unknowns"][0]["type"],
                         "unsupported_mechanics")

    def test_missing_speed_artifact_is_unknown_never_default(self):
        r = contradictions(scene(), {"speed_warning": None})
        self.assertTrue(any(u["type"] == "speed_evidence_unknown"
                            for u in r["unknowns"]))
        self.assertFalse(r["warnings"])

    def test_load_speed_warning_missing_file(self):
        self.assertIsNone(load_speed_warning("/nonexistent/t.json"))

    def test_wide_arc_scene_has_no_structural(self):
        wide = scene(notes=[(1.0, 0, 0, 0, 4), (1.3, 3, 2, 0, 5),
                            (1.6, 0, 2, 0, 4)])
        r = contradictions(wide, WARN)
        self.assertFalse(r["structural"])


class TestWallCertificates(unittest.TestCase):
    def test_single_covering_wall_detected_and_verified(self):
        sc = scene(walls=[(1.0, 0, 0, 2.0, 4)])
        r = contradictions(sc, WARN)
        self.assertEqual(r["status"], "MODEL_CONTRADICTION")
        self.assertTrue(r["model"][0]["verified"])

    def test_two_jointly_covering_walls(self):
        sc = scene(walls=[(1.0, 0, 0, 2.0, 2), (1.0, 2, 0, 2.0, 2)])
        cert = make_wall_certificate(sc, (1.0, 3.0))
        self.assertTrue(verify_certificate(sc, cert)["valid"])

    def test_one_lane_gap_forged_certificate_invalid(self):
        gap = scene(walls=[(1.0, 0, 0, 2.0, 3)])       # col 3 free
        forged = make_wall_certificate(gap, (1.0, 3.0))
        v = verify_certificate(gap, forged)
        self.assertFalse(v["valid"])
        self.assertIn("not covered", v["reason"])
        self.assertEqual(contradictions(gap, WARN)["status"],
                         "NO_CONTRADICTION_FOUND")

    def test_tangent_endpoint_half_open_continuity(self):
        sc = scene(walls=[(0.0, 0, 0, 1.0, 4), (1.0, 0, 0, 1.0, 4)])
        cert = make_wall_certificate(sc, (0.0, 2.0))
        self.assertTrue(verify_certificate(sc, cert)["valid"])

    def test_zero_duration_wall_covers_nothing(self):
        sc = scene(walls=[(1.0, 0, 0, 0.0, 4)])
        cert = make_wall_certificate(sc, (1.0, 1.1))
        self.assertFalse(verify_certificate(sc, cert)["valid"])

    def test_crouch_wall_inside_claim_leaves_domain(self):
        sc = scene(walls=[(1.0, 0, 0, 2.0, 4), (1.5, 0, 1, 0.5, 4)])
        cert = make_wall_certificate(sc, (1.0, 3.0))
        v = verify_certificate(sc, cert)
        self.assertFalse(v["valid"])
        self.assertIn("non-full-height", v["reason"])

    def test_tampering_invalidates(self):
        sc = scene(walls=[(1.0, 0, 0, 2.0, 4)])
        cert = make_wall_certificate(sc, (1.0, 3.0))
        wrong_src = {**cert, "source": {"chart_sha256": "ff" * 32}}
        self.assertIn("source hash",
                      verify_certificate(sc, wrong_src)["reason"])
        wrong_iv = {**cert, "interval_s": [1.0, 5.0]}
        self.assertFalse(verify_certificate(sc, wrong_iv)["valid"])
        wrong_as = {**cert, "assumptions": {**cert["assumptions"],
                                            "stance_columns": [0, 1]}}
        self.assertIn("stance domain",
                      verify_certificate(sc, wrong_as)["reason"])
        wrong_v = {**cert, "checker_version": CHECKER_VERSION + 1}
        self.assertIn("version",
                      verify_certificate(sc, wrong_v)["reason"])


class TestReachCertificates(unittest.TestCase):
    def test_standard_grid_zero_bound_counterexample(self):
        # stationary-wrist/rotating-saber: containing-ball distance is 0
        # across the whole standard grid, so ANY positive claimed minimum
        # speed must be refused
        sc = scene()
        ok = make_reach_certificate(sc, (0, 0), (3, 2), 0.1, 0.0)
        self.assertTrue(verify_certificate(sc, ok)["valid"])
        bad = make_reach_certificate(sc, (0, 0), (3, 2), 0.1, 1.0)
        v = verify_certificate(sc, bad)
        self.assertFalse(v["valid"])
        self.assertIn("independent bound", v["reason"])

    def test_invalid_dt(self):
        sc = scene()
        bad = make_reach_certificate(sc, (0, 0), (3, 2), 0.0, 0.0)
        self.assertIn("dt", verify_certificate(sc, bad)["reason"])


class TestRecipesV2(unittest.TestCase):
    def test_duplicate_occupancy_is_must_abstain_in_v2_only(self):
        from qa.mutations import RECIPES, recipes_v2
        self.assertEqual(RECIPES["duplicate_occupancy"]["class"],
                         "MUST_HARD_FAIL")        # history preserved
        self.assertEqual(recipes_v2()["duplicate_occupancy"]["class"],
                         "MUST_ABSTAIN")


if __name__ == "__main__":
    unittest.main()
