"""Q1.1 tests: panel membership fail-closed + C sealing, gate-boundary metric
fixtures (anti-gaming), and comparison-verdict semantics."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from timing import TimeGrid

import eval.quality_panel as qp
from eval.quality_eval import (INCOMPLETE, NON_REGRESSION_PASS, QUALITY_FAIL,
                               QUALITY_PASS, q1_verdict)
from eval.quality_metrics import (four_gram_stats, immutable_signature,
                                  opposite_horizontal, quality_metrics,
                                  threshold_neighborhood, to_ms,
                                  unknown_dot_burden, unrounded_transitions)

GRID = TimeGrid.uniform(125.0, 0.0, 400)          # 120 bpm, 1/4 grid
BEAT = 500.0


def _ms(raw):
    return to_ms(raw, GRID)


# ---------- panel membership: fail closed, never shrink ----------

def _fixture_tree(tmp, *, break_sha=False, drop_mi=False, dup_family=False):
    """Tiny synthetic panel inputs wired through the module constants."""
    root = Path(tmp)
    (root / "songs").mkdir()
    osu = root / "songs" / "a.osu"
    osu.write_text("osu file version stub")
    audio = root / "songs" / "a.mp3"
    audio.write_bytes(b"AUDIO")
    clean = {"entries": [{
        "song": "alpha", "cohort": "sentinel",
        "osu": "songs/a.osu", "audio": "songs/a.mp3",
        "osu_sha256": ("0" * 64) if break_sha else qp._sha(osu),
        "audio_sha256": qp._sha(audio)}]}
    cp = root / "clean_panel.json"
    cp.write_text(json.dumps(clean))
    fam_dir = root / "corpus" / "123ab (Beta - X)"
    fam_dir.mkdir(parents=True)
    (fam_dir / "song.egg").write_bytes(b"EGG")
    mi = root / "pilot-mi" / "fam_123ab"
    mi.mkdir(parents=True)
    if not drop_mi:
        (mi / "gen.osu").write_text("mi stub")
        (mi / "provenance.json").write_text("{}")
    chosen = [{"family": "fam:123ab", "song": "beta", "split": "train",
               "genre": "rock", "dir": str(fam_dir)}]
    if dup_family:
        chosen.append(dict(chosen[0], song="beta two"))
    pilot = root / "pilot-subset.json"
    pilot.write_text(json.dumps({"chosen": chosen}))
    manifest = {"maps": [
        {"dir": str(fam_dir), "family": "fam:123ab", "split": "train",
         "eligible": "ok", "audio_file": "song.egg"},
        # 24+ usable test families for C
        *[{"dir": str(fam_dir), "family": f"fam:test{i:02d}", "split": "test",
           "eligible": "ok", "audio_file": "song.egg"} for i in range(26)],
        {"dir": str(root / "missing"), "family": "fam:gone", "split": "test",
         "eligible": "ok", "audio_file": "song.egg"},
    ]}
    mf = root / "manifest.json"
    mf.write_text(json.dumps(manifest))
    return dict(CLEAN_PANEL=cp, PILOT=pilot, MI_DIR=root / "pilot-mi",
                MANIFEST=mf, ROOT=root, PANEL=root / "quality_panel.json")


def _patched(fix):
    return mock.patch.multiple(qp, **fix)


class TestPanel(unittest.TestCase):
    def test_build_and_load_roundtrip_and_confirm_seal(self):
        with tempfile.TemporaryDirectory() as t:
            fix = _fixture_tree(t)
            with _patched(fix):
                p = qp.build_panel()
                self.assertEqual(len(p["dev"]), 2)
                self.assertEqual(len(p["confirm"]["families"]), qp.N_CONFIRM)
                # an unusable family never lands in C silently: it is either
                # recorded as excluded or ordered past the cutoff
                self.assertNotIn("fam:gone",
                                 [c["family"] for c in p["confirm"]["families"]])
                self.assertEqual(len(qp.dev_entries()), 2)
                with self.assertRaises(qp.PanelError):
                    qp.confirm_families()               # sealed
                self.assertTrue(qp.confirm_families(qp.UNSEAL_TOKEN)["sealed"])

    def test_hash_deviation_fails(self):
        with tempfile.TemporaryDirectory() as t:
            with _patched(_fixture_tree(t, break_sha=True)):
                with self.assertRaises(qp.PanelError):
                    qp.build_panel()

    def test_missing_saved_mi_fails_not_shrinks(self):
        with tempfile.TemporaryDirectory() as t:
            with _patched(_fixture_tree(t, drop_mi=True)):
                with self.assertRaises(qp.PanelError):
                    qp.build_panel()

    def test_duplicate_family_fails(self):
        with tempfile.TemporaryDirectory() as t:
            with _patched(_fixture_tree(t, dup_family=True)):
                with self.assertRaises(qp.PanelError):
                    qp.build_panel()

    def test_load_verifies_sources(self):
        with tempfile.TemporaryDirectory() as t:
            fix = _fixture_tree(t)
            with _patched(fix):
                qp.build_panel()
                (fix["ROOT"] / "songs" / "a.osu").write_text("tampered")
                with self.assertRaises(qp.PanelError):
                    qp.load_panel()


# ---------- gate-boundary metric fixtures (anti-gaming) ----------

class TestMetricFixtures(unittest.TestCase):
    def test_pair_hidden_by_row_or_column_still_counts(self):
        # equal columns: motion.lr_doubles skips the pair entirely, but the
        # position-independent opposite-horizontal count must still see it
        raw = [(8, 0, 2, 0, 2), (8, 1, 2, 2, 3)]
        notes = _ms(raw)
        self.assertEqual(opposite_horizontal(notes), 1)
        import motion
        lr = motion.lr_doubles(notes)
        self.assertEqual(len(lr["broad"]), 0)      # the historical metric miss

    def test_broad_slow_sweep_is_not_flagged(self):
        # one-grid-unit move over a full second: no flag at any extent,
        # no neighborhood burden
        raw = [(0, 0, 0, 0, 0), (8, 0, 1, 0, 0)]  # 1000 ms apart
        notes = _ms(raw)
        for e in (0.0, 0.25, 0.5, 0.75):
            self.assertEqual(
                len([x for x in unrounded_transitions(notes, e)
                     if x["ms"] <= 200 and x["dist"] >= 2.0]), 0)
            self.assertEqual(threshold_neighborhood(notes, e)["total_flags"], 0)

    def test_threshold_dodge_visible_in_neighborhood(self):
        # engineered transition with dist in [1.9, 2.0) at 125 ms: the frozen
        # 2.0-threshold evaluator misses it; the 3x3 neighborhood must not
        raw = [(0, 0, 0, 0, 3), (1, 0, 2, 1, 0)]
        notes = _ms(raw)
        tr = unrounded_transitions(notes, 0.25)
        self.assertEqual(len(tr), 1)
        self.assertTrue(1.9 <= tr[0]["dist"] < 2.0,
                        f"fixture dist {tr[0]['dist']} not in [1.9, 2.0)")
        self.assertLessEqual(tr[0]["ms"], 180)
        import motion
        self.assertEqual(motion.flag_stats(notes, 0.25)["flags"], 0)
        self.assertGreater(
            threshold_neighborhood(notes, 0.25)["total_flags"], 0)

    def test_vertical_stream_is_valid_not_a_repair_site(self):
        # Rap God-like intentional vertical vocabulary: high token
        # concentration is REPORTED, but there are no flags and no pairs —
        # nothing here licenses moving notes
        raw = [(2 * i, 0, 1, 0 if i % 2 else 2, 0 if i % 2 else 1)
               for i in range(40)]
        notes = _ms(raw)
        m = quality_metrics(raw, [], GRID, 120.0)
        self.assertGreater(m["four_gram"]["max_4gram_share"], 0.4)
        for e in ("0.0", "0.25", "0.5", "0.75"):
            self.assertEqual(m["flags_by_ext"][e], 0.0)
        self.assertEqual(m["narrow"] + m["converging"] + m["broad"], 0)

    def test_unknown_stack_is_burden_not_zero(self):
        # two heads + one dot in a single (step, hand) group: dot_stacks
        # cannot classify it; the burden metric must not report zero
        raw = [(4, 0, 0, 0, 0), (4, 0, 1, 0, 1), (4, 0, 2, 0, 8)]
        self.assertEqual(unknown_dot_burden(_ms(raw)), 1)
        import motion
        ds = motion.dot_stacks(_ms(raw))
        self.assertEqual(ds["forward"] + ds["reverse"] + ds["offaxis"], 0)

    def test_signature_covers_identity_not_geometry(self):
        raw = [(0, 0, 0, 0, 0), (4, 1, 3, 2, 1), (4, 1, 3, 1, 8)]
        walls = [(10, 4, 0)]
        base = immutable_signature(raw, walls, GRID)
        moved = [(0, 0, 2, 1, 3), (4, 1, 1, 0, 0), (4, 1, 2, 2, 8)]
        self.assertEqual(immutable_signature(moved, walls, GRID), base)
        self.assertNotEqual(immutable_signature(raw[:2], walls, GRID), base)
        rehand = [(0, 1, 0, 0, 0)] + raw[1:]
        self.assertNotEqual(immutable_signature(rehand, walls, GRID), base)
        redot = [(0, 0, 0, 0, 8)] + raw[1:]      # head -> dot role change
        self.assertNotEqual(immutable_signature(redot, walls, GRID), base)
        self.assertNotEqual(immutable_signature(raw, [(10, 5, 0)], GRID), base)


# ---------- verdict semantics ----------

def _m(flags, narrow=0, conv=0, broad=0, opp=0):
    return {"flags_by_ext": {"0.0": flags, "0.25": flags, "0.5": flags,
                             "0.75": flags},
            "narrow": narrow, "converging": conv, "broad": broad,
            "opposite_horizontal": opp}


class TestVerdicts(unittest.TestCase):
    def test_integrity_never_quality_pass(self):
        b = {"f1": _m(10.0, narrow=6, conv=6)}
        a = {"f1": _m(2.0, narrow=1, conv=1)}
        v = q1_verdict(b, a, integrity_ok=False, gates_ok=True)
        self.assertEqual(v["status"], QUALITY_FAIL)

    def test_non_regression_without_positive_target(self):
        b = {"f1": _m(10.0, narrow=6, conv=6)}
        a = {"f1": _m(9.5, narrow=6, conv=6)}     # barely better, not 25/20%
        v = q1_verdict(b, a, integrity_ok=True, gates_ok=True)
        self.assertEqual(v["status"], NON_REGRESSION_PASS)

    def test_quality_pass_needs_positive_everywhere(self):
        b = {"f1": _m(10.0, narrow=6, conv=6), "f2": _m(4.0, narrow=3)}
        a = {"f1": _m(2.0, narrow=1, conv=1), "f2": _m(1.0, narrow=1)}
        v = q1_verdict(b, a, integrity_ok=True, gates_ok=True)
        self.assertEqual(v["status"], QUALITY_PASS)

    def test_broad_increase_is_fail(self):
        b = {"f1": _m(10.0, narrow=6, conv=6, broad=1)}
        a = {"f1": _m(2.0, narrow=1, conv=1, broad=3)}
        v = q1_verdict(b, a, integrity_ok=True, gates_ok=True)
        self.assertEqual(v["status"], QUALITY_FAIL)

    def test_opposite_horizontal_increase_is_fail(self):
        b = {"f1": _m(10.0, narrow=6, conv=6, opp=1)}
        a = {"f1": _m(2.0, narrow=1, conv=1, opp=4)}
        v = q1_verdict(b, a, integrity_ok=True, gates_ok=True)
        self.assertEqual(v["status"], QUALITY_FAIL)

    def test_incomplete_on_family_mismatch(self):
        v = q1_verdict({"f1": _m(1.0)}, {}, integrity_ok=True, gates_ok=True)
        self.assertEqual(v["status"], INCOMPLETE)


class TestConvertSeam(unittest.TestCase):
    """Q1.3 integration: the quality_profile output-transform seam on ONE real
    panel song (full decodes — the slow, run-sparingly class)."""
    SONG = "still_waiting"

    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(4)
        from convert import convert_groomed, parse_osu
        e = next(x for x in json.loads(
            (Path(__file__).parent / "clean_rhythm_panel.json").read_text())
            ["entries"] if x["song"] == cls.SONG)
        root = Path(__file__).resolve().parent.parent
        cls.args = None
        _m, objects, bpm, offset = parse_osu(root / e["osu"])
        cls.call = dict(objects=objects, bpm=bpm, offset=offset,
                        audio_path=str(root / e["audio"]), diff="ExpertPlus",
                        replay_mode="off", thin=True, calibrate=True)
        cls.base_recs = []
        cls.base_out = convert_groomed(collect=cls.base_recs, **cls.call)

    def _decode(self, profile, recs):
        from convert import convert_groomed
        return convert_groomed(collect=recs, quality_profile=profile,
                               **self.call)

    def test_profile_repairs_selected_record_only_bank_untouched(self):
        recs = []
        out = self._decode({}, recs)
        # candidate bank identical to the no-profile decode (raw + scores)
        self.assertEqual([r["notes"] for r in recs],
                         [r["notes"] for r in self.base_recs])
        self.assertEqual([r["score"] for r in recs],
                         [r["score"] for r in self.base_recs])
        self.assertEqual([r["selected"] for r in recs],
                         [r["selected"] for r in self.base_recs])
        sel = next(r for r in recs if r["selected"])
        tr = sel.get("output_transform")
        self.assertIsNotNone(tr, "expected a repair on the burdened winner")
        self.assertEqual(sum(1 for r in recs if "output_transform" in r), 1)
        # emitted export equals the transformed raw, not the bank's raw
        self.assertNotEqual(out, self.base_out)
        self.assertEqual(len(out[0]), len(self.base_out[0]))  # same events
        # the metric adapter reads output_transform raw; the bank stays raw
        self.assertNotEqual(tr["raw"], sel["notes"])
        self.assertLess(sum(tr["after"]["flags"].values()),
                        sum(tr["before"]["flags"].values()))

    def test_norepair_and_failure_fall_back_to_identical_bytes(self):
        from quality_repair import RepairResult

        def _unchanged(raw, walls, grid, bpm, cap, config=None):
            from quality_repair import risk_vector
            rv = risk_vector(raw, walls, grid)
            return RepairResult(raw=list(raw), status="unchanged",
                                changed_ids=[], before=rv, after=rv,
                                trials=0, elapsed_s=0.0, sweeps_run=0)
        with mock.patch("quality_repair.repair_winner", _unchanged):
            recs = []
            out = self._decode({}, recs)
        self.assertEqual(out, self.base_out)
        self.assertNotIn("output_transform",
                         next(r for r in recs if r["selected"]))

        def _boom(*a, **k):
            raise RuntimeError("forced repair failure")
        with mock.patch("quality_repair.repair_winner", _boom):
            out2 = self._decode({}, [])
        self.assertEqual(out2, self.base_out)

    def test_all_invalid_generation_still_fails(self):
        from convert import GenerationError
        with mock.patch("convert.check", lambda *a, **k: ["forced"]):
            with self.assertRaises(GenerationError):
                from convert import convert_groomed
                convert_groomed(quality_profile={}, **self.call)


if __name__ == "__main__":
    unittest.main()
