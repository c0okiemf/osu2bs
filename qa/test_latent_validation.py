"""Chart-latent Task 1 tests.
  .venv/bin/python -m unittest qa.test_latent_validation -q
"""
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from qa import latent_validation as lv


def forbidden_call(*a, **k):
    raise AssertionError("kinematics/payload loader invoked during "
                         "metadata-only reservation")


def _mk_dir(tmp, name, notes=None, custom=False):
    d = Path(tmp) / name
    d.mkdir()
    raw = {"_version": "2.0.0", "_obstacles": [],
           "_notes": notes if notes is not None else [
               {"_time": 2.0 + 0.5 * i, "_type": i % 2, "_lineIndex": i % 4,
                "_lineLayer": 0, "_cutDirection": 1} for i in range(30)]}
    if custom:
        raw["_customData"] = {"_x": 1}
    (d / "chart.dat").write_text(json.dumps(raw))
    (d / "Info.dat").write_text(json.dumps({
        "_beatsPerMinute": 120,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": [{"_beatmapFilename": "chart.dat",
                                     "_difficulty": "ExpertPlus"}]}]}))
    return d


def modchart_notes(n=30):
    return [{"_time": 2.0 + 0.5 * i, "_type": 1, "_lineIndex": 2,
             "_lineLayer": 0, "_cutDirection": 8} for i in range(n)]


def fake_prober(fam, map_dir):
    return {"family": fam, "dir": str(map_dir), "hash": "AB" * 20,
            "leaderboard": 1, "difficulty": "ExpertPlus",
            "n_modifier_free": 50, "usable": True}


class TestReservation(unittest.TestCase):
    def _reserve(self, pool, prober=fake_prober):
        with tempfile.TemporaryDirectory() as td:
            with unittest.mock.patch.object(lv, "OUT_DIR", Path(td)), \
                 unittest.mock.patch.object(lv, "RESERVE_P",
                                            Path(td) / "r.json"), \
                 unittest.mock.patch("qa.telemetry_v2.window_record",
                                     forbidden_call), \
                 unittest.mock.patch("qa.bsor.parse_bsor",
                                     forbidden_call):
                return lv.reserve_validation(design={"stub": True},
                                             pool=pool, prober=prober)

    def test_reservation_does_not_inspect_kinematics(self):
        with tempfile.TemporaryDirectory() as md:
            pool = [(f"fam:t{i}", str(_mk_dir(md, f"m{i}")))
                    for i in range(6)]
            r = self._reserve(pool)
            self.assertEqual(r["status"], "RESERVED")
            self.assertEqual(len(r["families"]), 4)

    def test_modchart_is_scope_ineligible(self):
        # 3 good + 1 modchart: the modchart MUST be examined, rejected
        # chart-side, and the shortfall reported honestly
        with tempfile.TemporaryDirectory() as md, \
                tempfile.TemporaryDirectory() as td:
            bad = _mk_dir(md, "bad", notes=modchart_notes(), custom=True)
            pool = [("fam:bad0", str(bad))] + \
                [(f"fam:t{i}", str(_mk_dir(md, f"m{i}")))
                 for i in range(3)]
            rp = Path(td) / "r.json"
            with unittest.mock.patch.object(lv, "OUT_DIR", Path(td)), \
                 unittest.mock.patch.object(lv, "RESERVE_P", rp), \
                 unittest.mock.patch("qa.telemetry_v2.window_record",
                                     forbidden_call):
                with self.assertRaisesRegex(RuntimeError,
                                            "INVENTORY_INCOMPLETE"):
                    lv.reserve_validation(design={"stub": True},
                                          pool=pool, prober=fake_prober)
            r = json.loads(rp.read_text())
            reasons = {c["family"]: c["reason"] for c in r["census"]
                       if not c.get("usable")}
            self.assertIn("modchart", reasons["fam:bad0"])

    def test_shortfall_is_inventory_incomplete(self):
        with tempfile.TemporaryDirectory() as md:
            pool = [(f"fam:t{i}", str(_mk_dir(md, f"m{i}")))
                    for i in range(2)]
            with self.assertRaisesRegex(RuntimeError,
                                        "INVENTORY_INCOMPLETE"):
                self._reserve(pool)

    def test_deterministic_hash_order(self):
        with tempfile.TemporaryDirectory() as md:
            pool = [(f"fam:t{i}", str(_mk_dir(md, f"m{i}")))
                    for i in range(8)]
            a = self._reserve(pool)
            b = self._reserve(list(reversed(pool)))
            self.assertEqual([x["family"] for x in a["families"]],
                             [x["family"] for x in b["families"]])


class TestRolesAndFetchGuard(unittest.TestCase):
    def test_role_isolation(self):
        from qa.features import records_for_role
        for purpose in ("fit", "reference", "calibrate", "confirm"):
            with self.assertRaises(PermissionError):
                next(records_for_role("qa_validate_latent", purpose))
        # validate is permitted (no data yet -> empty iterator, no error)
        list(records_for_role("qa_validate_latent", "validate",
                              families=set()))

    def test_fresh_payloads_require_candidate_freeze(self):
        with self.assertRaises(PermissionError):
            lv.fetch_validation(reservation={"status": "RESERVED",
                                             "families": []},
                                candidate_freeze=None)
        with self.assertRaises(PermissionError):
            lv.fetch_validation(reservation={"status": "RESERVED",
                                             "families": []},
                                candidate_freeze="/nonexistent/freeze")

    def test_payload_caps(self):
        self.assertEqual(lv.MAX_PAYLOADS, 24)
        self.assertEqual(lv.MAX_PAYLOADS_PER_FAMILY, 6)
        self.assertEqual(lv.N_VALIDATE, 4)


class TestFolds(unittest.TestCase):
    def test_six_disjoint_folds_of_four(self):
        fams = [f"fam:{i:02d}" for i in range(24)]
        folds = lv.latent_folds(fams)
        self.assertEqual(len(folds), 6)
        self.assertTrue(all(len(f) == 4 for f in folds))
        self.assertEqual(sorted(sum(folds, [])), sorted(fams))
        self.assertEqual(folds, lv.latent_folds(list(reversed(fams))))


if __name__ == "__main__":
    unittest.main()
