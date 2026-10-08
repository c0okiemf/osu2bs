"""Tests for the difficulty-vector panel validation + runner (Tasks 1, 3).
Constructed fixtures only.
  .venv/bin/python -m unittest eval.test_difficulty_audit -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from eval.difficulty_audit import (_selected_pairs, _summaries, _witnesses,
                                   audit, rate_close, validate_panel)
from eval.difficulty_vector import measure_scene
from eval.visibility import Note


def mk_chart(cid, cohort, family, selected, times_hands, label="x",
             label_status="generated_target", rank=None):
    notes = [Note(f"{cid}n{k}", float(t), 0 if h == 0 else 3, 0, h, 1)
             for k, (t, h) in enumerate(times_hands)]
    sc = {"notes": notes, "chart_unknown": False, "ignored": {}}
    return {"id": cid, "cohort": cohort, "family": family, "label": label,
            "label_status": label_status, "authored_rank": rank,
            "selected": selected, "measure": measure_scene(sc)}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


CHART = {"_version": "2.0.0", "_obstacles": [], "_notes": [
    {"_time": 0, "_type": 0, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 1},
    {"_time": 1, "_type": 1, "_lineIndex": 3, "_lineLayer": 0, "_cutDirection": 1}]}


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def valid_replay_panel(tmp):
    chart = _write(tmp / "chart.dat", CHART)
    res = _write(tmp / "results.json", {"bpm": 120.0, "e2e": [
        {"mode": "off", "dat": "chart.dat", "seed": 0, "selected": True}]})
    e = {"id": "generated/x/chart", "role": "generated", "cohort": "replay_off",
         "chart": str(chart), "sha256": _sha(chart), "bpm": 120.0, "origin_s": 0,
         "timing_source": str(res), "timing_source_sha256": _sha(res),
         "family": "generated/x", "label": "chart",
         "label_status": "generated_target", "authored_rank": None,
         "selected": True, "seed": 0, "source_kind": "results",
         "source_id": str(res)}
    panel = {"schema_version": 1, "missing": [], "expected_ids": [e["id"]],
             "expected_counts": {"replay_off": 1}, "entries": [e]}
    return panel, e


def bound_rank9_panel(tmp, custom_label="Expert++"):
    chart = _write(tmp / "ExpertPlus2.dat", CHART)
    info = _write(tmp / "Info.dat", {"_difficultyBeatmapSets": [{
        "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
            "_difficulty": "ExpertPlus", "_difficultyRank": 9,
            "_beatmapFilename": "ExpertPlus2.dat", "_noteJumpMovementSpeed": 20,
            "_noteJumpStartBeatOffset": 0.0,
            "_customData": {"_difficultyLabel": custom_label}}]}]})
    bench = _write(tmp / "benchmark.json", {"songs": [{
        "id": "latest/x", "bpm": 192.0, "dir": str(tmp),
        "difficulties": {"ExpertPlus2.dat": _sha(chart)}}]})
    e = {"id": "historical/x/ExpertPlus2", "role": "reference",
         "cohort": "historical_export", "chart": str(chart), "sha256": _sha(chart),
         "bpm": 192.0, "origin_s": 0, "timing_source": str(bench),
         "timing_source_sha256": _sha(bench), "family": "latest/x",
         "label": custom_label, "label_status": "verified", "authored_rank": 9,
         "info_sha256": _sha(info), "selected": None, "seed": None,
         "source_kind": "benchmark", "source_id": "latest/x"}
    panel = {"schema_version": 1, "missing": [], "expected_ids": [e["id"]],
             "expected_counts": {"historical_export": 1}, "entries": [e]}
    return panel, e


class TestValidatePanel(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_valid_replay_panel_ok(self):
        panel, _ = valid_replay_panel(self.tmp)
        good, failures = validate_panel(panel)
        self.assertEqual(failures, [])
        self.assertEqual(len(good), 1)

    def test_custom_label_does_not_create_rank(self):
        panel, _ = bound_rank9_panel(self.tmp, custom_label="Expert++")
        entries, failures = validate_panel(panel)
        self.assertEqual(failures, [])
        self.assertEqual(entries[0]["authored_rank"], 9)   # from Info, not "++"
        self.assertEqual(entries[0]["label"], "Expert++")

    def test_missing_expected_ids_is_failure(self):
        panel, _ = valid_replay_panel(self.tmp)
        del panel["expected_ids"]
        self.assertTrue(validate_panel(panel)[1])

    def test_deleted_entry_but_membership_remains(self):
        panel, _ = valid_replay_panel(self.tmp)
        panel["expected_ids"] = panel["expected_ids"] + ["generated/x/gone"]
        self.assertTrue(any(f["reason"] == "missing_expected_entry"
                            for f in validate_panel(panel)[1]))

    def test_declared_missing_is_failure(self):
        panel, _ = valid_replay_panel(self.tmp)
        panel["missing"] = ["generated/x/gone"]
        self.assertTrue(any(f["reason"] == "declared_missing_entries"
                            for f in validate_panel(panel)[1]))

    def test_duplicate_id(self):
        panel, e = valid_replay_panel(self.tmp)
        panel["entries"] = [e, dict(e)]
        panel["expected_counts"] = {"replay_off": 2}
        self.assertTrue(any(f["reason"] == "duplicate_id"
                            for f in validate_panel(panel)[1]))

    def test_nonfinite_bpm(self):
        panel, e = valid_replay_panel(self.tmp)
        e["bpm"] = 0.0
        self.assertTrue(any(f["reason"] == "nonfinite_bpm"
                            for f in validate_panel(panel)[1]))

    def test_altered_chart_hash(self):
        panel, e = valid_replay_panel(self.tmp)
        e["sha256"] = "0" * 64
        self.assertTrue(any(f["reason"] == "chart_sha256_mismatch"
                            for f in validate_panel(panel)[1]))

    def test_altered_bpm_despite_source(self):
        panel, e = valid_replay_panel(self.tmp)
        e["bpm"] = 130.0   # source results.json still says 120
        self.assertTrue(any(f["reason"] == "bpm_not_bound_to_source"
                            for f in validate_panel(panel)[1]))

    def test_selected_flag_mismatch(self):
        panel, e = valid_replay_panel(self.tmp)
        e["selected"] = False   # results record says True
        self.assertTrue(any(f["reason"] == "selected_flag_mismatch"
                            for f in validate_panel(panel)[1]))

    def test_missing_chart(self):
        panel, e = valid_replay_panel(self.tmp)
        Path(e["chart"]).unlink()
        self.assertTrue(any(f["reason"] == "chart_missing"
                            for f in validate_panel(panel)[1]))

    def test_label_status_altered(self):
        panel, e = bound_rank9_panel(self.tmp)
        e["label_status"] = "benchmark_label_unverified"   # live says verified
        self.assertTrue(any(f["reason"] == "label_status_altered"
                            for f in validate_panel(panel)[1]))

    def test_onesaber_binding_rejected(self):
        # a OneSaber set with the matching filename must NOT authenticate
        from eval.difficulty_audit import authored_label
        chart = _write(self.tmp / "ExpertPlus.dat", CHART)
        _write(self.tmp / "Info.dat", {"_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [{
                "_beatmapFilename": "ExpertPlus.dat", "_difficultyRank": 9,
                "_noteJumpMovementSpeed": 18}]}]})
        lab = authored_label(str(chart), _sha(chart), "human")
        self.assertEqual(lab["label_status"], "benchmark_label_unverified")

    def test_altered_label_text_rejected(self):
        panel, e = bound_rank9_panel(self.tmp, custom_label="Expert++")
        e["label"] = "Expert"   # source says Expert++
        self.assertTrue(any(f["reason"] == "label_text_altered"
                            for f in validate_panel(panel)[1]))

    def test_missing_info_sha256_on_verified_fails(self):
        panel, e = bound_rank9_panel(self.tmp)
        del e["info_sha256"]
        self.assertTrue(any(f["reason"] == "missing_info_sha256"
                            for f in validate_panel(panel)[1]))

    def test_tampered_info_bytes_fail(self):
        # a byte change to Info that does NOT alter the resolved rank/label must
        # still fail via the frozen Info hash (review provenance blocker)
        panel, e = bound_rank9_panel(self.tmp)
        info = self.tmp / "Info.dat"
        obj = json.loads(info.read_text())
        obj["_customData"] = {"_editors": {"note": "tampered"}}   # cosmetic byte change
        info.write_text(json.dumps(obj))
        self.assertTrue(any(f["reason"] == "info_hash_altered"
                            for f in validate_panel(panel)[1]))

    def test_unmatched_alias_stays_unverified(self):
        # Info binds a different filename -> live unknown; a "verified" claim fails
        chart = _write(self.tmp / "ExpertPlus.dat", CHART)
        _write(self.tmp / "Info.dat", {"_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
                "_beatmapFilename": "ExpertPlusStandard.dat",
                "_noteJumpMovementSpeed": 19}]}]})
        bench = _write(self.tmp / "benchmark.json", {"songs": [{
            "id": "latest/x", "bpm": 192.0, "dir": str(self.tmp),
            "difficulties": {"ExpertPlus.dat": _sha(chart)}}]})
        e = {"id": "historical/x/ExpertPlus", "role": "reference",
             "cohort": "historical_export", "chart": str(chart),
             "sha256": _sha(chart), "bpm": 192.0, "origin_s": 0,
             "timing_source": str(bench), "timing_source_sha256": _sha(bench),
             "family": "latest/x", "label": "ExpertPlus",
             "label_status": "verified", "authored_rank": 9, "selected": None,
             "seed": None, "source_kind": "benchmark", "source_id": "latest/x"}
        panel = {"schema_version": 1, "missing": [], "expected_ids": [e["id"]],
                 "expected_counts": {"historical_export": 1}, "entries": [e]}
        self.assertTrue(any(f["reason"] == "label_status_altered"
                            for f in validate_panel(panel)[1]))


class TestRunnerPure(unittest.TestCase):
    def test_rate_close_predicate(self):
        self.assertTrue(rate_close(5, 5.5))
        self.assertFalse(rate_close(5, 6))
        self.assertFalse(rate_close(None, 5))
        self.assertFalse(rate_close(0, 0))

    def test_six_candidates_one_song(self):
        charts = [mk_chart(f"generated/s/c{i}", "replay_off", "generated/s",
                           i == 0, [(t, t % 2) for t in range(10)]) for i in range(6)]
        s = _summaries(charts)
        self.assertEqual(list(s["generated_by_song"]), ["generated/s"])
        self.assertEqual(s["generated_by_song"]["generated/s"]["n_candidates"], 6)
        self.assertEqual(s["generated_by_song"]["generated/s"]["selected_id"],
                         "generated/s/c0")

    def test_two_rc_versions_one_family(self):
        charts = [mk_chart("human/reality_check_25f/ExpertPlus", "human",
                           "reality_check", False, [(i * .3, i % 2) for i in range(12)]),
                  mk_chart("human/reality_check_3741/ExpertPlus", "human",
                           "reality_check", False, [(i * .3, i % 2) for i in range(12)])]
        s = _summaries(charts)
        self.assertEqual(list(s["human_by_family"]), ["reality_check"])
        self.assertEqual(len(s["human_by_family"]["reality_check"]), 2)

    def test_no_eligible_pairs_when_rates_far(self):
        charts = [mk_chart("generated/a/c0", "replay_off", "generated/a", True,
                           [(i * .2, i % 2) for i in range(20)]),   # fast
                  mk_chart("generated/b/c0", "replay_off", "generated/b", True,
                           [(i * 2.0, i % 2) for i in range(6)])]   # slow
        self.assertEqual(_selected_pairs(charts), [])

    def test_witness_only_from_selected(self):
        # a larger NON-selected candidate must not become the imbalance witness
        sel = mk_chart("generated/a/c0", "replay_off", "generated/a", True,
                       [(i * .25, 0) for i in range(12)])            # all-left, high imbalance
        big = mk_chart("generated/a/c1", "replay_off", "generated/a", False,
                       [(i * .25, 0) for i in range(40)])
        w = _witnesses([sel, big], [])
        self.assertEqual(w["max_imbalance"]["raster_id"], "generated/a/c0")

    def test_peak_rate_witness_uses_peak_hand_window(self):
        # two rate-close selected songs; the witness must report the peak HAND's
        # rate/window, not the combined-rate peak
        a = mk_chart("generated/a/c0", "replay_off", "generated/a", True,
                     [(i * .25, 0) for i in range(9)])            # left-heavy burst
        b = mk_chart("generated/b/c0", "replay_off", "generated/b", True,
                     [(i * .25, i % 2) for i in range(9)])        # balanced
        pairs = _selected_pairs([a, b])
        self.assertTrue(pairs)                                    # rate-close
        w = _witnesses([a, b], pairs)["peak_rate_pair"]
        self.assertIn(w["peak_hand"], ("left", "right"))
        rc = {c["id"]: c for c in (a, b)}[w["raster_id"]]["measure"]["burst_2s"]
        self.assertEqual(w["peak_hand_rate"], rc[w["peak_hand"]]["rate"])
        self.assertGreaterEqual(w["peak_hand_rate"], rc["combined"]["rate"])

    def test_missing_witness_category_not_applicable(self):
        # no selected charts -> every witness category is not_applicable
        charts = [mk_chart("generated/a/c1", "replay_off", "generated/a", False,
                           [(i * .25, i % 2) for i in range(12)])]
        w = _witnesses(charts, [])
        self.assertEqual(w["max_imbalance"], "not_applicable")
        self.assertEqual(w["longest_gap"], "not_applicable")


@unittest.skipUnless(
    all((Path(__file__).resolve().parent.parent / row["chart"]).is_file()
        for row in json.loads((Path(__file__).parent / "difficulty_panel.json").read_text())["entries"]),
    "requires the local historical map corpus",
)
class TestShippedPanel(unittest.TestCase):
    def test_frozen_panel_valid_and_counted(self):
        panel = json.loads((Path(__file__).parent / "difficulty_panel.json").read_text())
        good, failures = validate_panel(panel)
        self.assertEqual(failures, [])
        self.assertEqual(len(good), 45)
        from collections import Counter
        self.assertEqual(Counter(e["cohort"] for e in good),
                         {"replay_off": 30, "human": 11, "historical_export": 4})

    def test_audit_certifies_and_deterministic(self):
        panel = str(Path(__file__).parent / "difficulty_panel.json")
        tmp = Path(tempfile.mkdtemp())
        c1, r1 = audit(panel, tmp / "a", make_plots=False)
        c2, r2 = audit(panel, tmp / "b", make_plots=False)
        self.assertEqual(c1, 0)
        self.assertEqual(r1["certification"], "CERTIFIED")
        self.assertEqual(r1["coverage"]["supported"], 44)   # numb unknown
        self.assertEqual((tmp / "a" / "report.json").read_bytes(),
                         (tmp / "b" / "report.json").read_bytes())

    def test_audit_integrity_failure_no_measurement(self):
        panel = json.loads((Path(__file__).parent / "difficulty_panel.json").read_text())
        panel["schema_version"] = 2   # break it
        tmp = Path(tempfile.mkdtemp())
        p = tmp / "bad.json"
        p.write_text(json.dumps(panel))
        code, rep = audit(p, tmp / "o", make_plots=False)
        self.assertNotEqual(code, 0)
        self.assertEqual(rep["certification"], "UNCERTIFIED")
        self.assertEqual(rep["charts"], [])
        self.assertEqual(rep["summaries"], {})


if __name__ == "__main__":
    unittest.main()
