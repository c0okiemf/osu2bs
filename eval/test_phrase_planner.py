"""Q3 planner/scheduler fixtures (cheap, synthetic) + a real-song
integration check of the bounded scheduler contract.

  .venv/bin/python -m unittest eval.test_phrase_planner -v
  .venv/bin/python -m eval.test_phrase_planner --song numb   (integration)
"""
import unittest

import torch

from phrase_planner import (WIN, _emitted_occ, _mirror_window, _offset_at,
                            _pick_offset, _traj_err, beam_schedule,
                            constant_trajectory, onset_counts,
                            song_window_inputs, window_inputs, window_targets)
from groom import MIN_GAP


def _arrays(T, pres_val=0.9, thr_val=0.5, merged=None, rest=None,
            sprinkle=None):
    return ([[pres_val, pres_val]] * T, [thr_val] * T, [thr_val] * T,
            merged or [True] * T, sprinkle or [False] * T,
            rest or [False] * T)


class TestOnsetGrid(unittest.TestCase):
    def test_exact_and_nearest_mapping(self):
        st = [0.0, 100.0, 200.0, 300.0]
        c = onset_counts([{"t": 100.0}, {"t": 149.0}, {"t": 151.0},
                          {"t": 299.0}], st)
        self.assertEqual(c.tolist(), [0.0, 2.0, 1.0, 1.0])

    def test_offset_shift_invariance(self):
        """Time-offset re-encoding: shifting onsets AND grid together maps
        every onset to the same step."""
        st = [i * 125.0 for i in range(64)]
        ts = [130.0, 511.0, 3000.0]
        a = onset_counts([{"t": t} for t in ts], st)
        b = onset_counts([{"t": t + 777.0} for t in ts],
                         [s + 777.0 for s in st])
        self.assertEqual(a.tolist(), b.tolist())

    def test_bpm_reencoding_same_real_times(self):
        """BPM re-encoding that preserves real step times (finer osu beat
        values, same ms grid) leaves grid mapping unchanged."""
        st = [i * 125.0 for i in range(32)]
        onsets = [{"t": 250.0}, {"t": 1375.0}]
        self.assertEqual(onset_counts(onsets, st).tolist(),
                         onset_counts(list(onsets), list(st)).tolist())


class TestWindowAlignment(unittest.TestCase):
    def test_boundary_event_belongs_to_next_window(self):
        """Target/source alignment: a note at step w1 is outside [w0,w1) for
        both targets and inputs."""
        T = 2 * WIN
        pres = torch.zeros(T, 2, dtype=torch.bool)
        pres[WIN, 0] = True                     # exactly on the boundary
        events = [(WIN, 0, 1, 0, 0), (WIN + 2, 0, 1, 1, 0)]
        st = [i * 125.0 for i in range(T)]
        tg0 = window_targets(pres, events, st, 0, WIN)
        tg1 = window_targets(pres, events, st, WIN, 2 * WIN)
        self.assertEqual(tg0["occ_rate"], 0.0)
        self.assertGreater(tg1["occ_rate"], 0.0)

    def test_partial_tail_excluded_from_inputs(self):
        T = 2 * WIN + 7
        afeat = torch.rand(T, 4)
        mi = torch.zeros(T)
        st = [i * 125.0 for i in range(T)]
        x = song_window_inputs(afeat, mi, st, 120.0)
        self.assertEqual(len(x), 2)             # tail window has no features


class TestMirror(unittest.TestCase):
    def test_min_gap_carries_across_windows(self):
        arr = _arrays(2 * WIN)
        n0, last = _mirror_window(*arr, 0, WIN, (-MIN_GAP, -MIN_GAP))
        self.assertEqual(last[0] % MIN_GAP, 0)
        n1, _ = _mirror_window(*arr, WIN, 2 * WIN, last)
        n1_fresh, _ = _mirror_window(*arr, WIN, 2 * WIN, (-MIN_GAP, -MIN_GAP))
        self.assertLessEqual(n1, n1_fresh)      # carried recency can only cost

    def test_rest_steps_never_scheduled(self):
        rest = [True] * WIN
        arr = _arrays(WIN, rest=rest)
        n, _ = _mirror_window(*arr, 0, WIN, (-MIN_GAP, -MIN_GAP))
        self.assertEqual(n, 0)

    def test_monotone_in_offset(self):
        arr = _arrays(WIN, pres_val=0.6, thr_val=0.5)
        counts = []
        for c in (-0.2, 0.0, 0.2):
            n, _ = _mirror_window(*arr, 0, WIN, (-MIN_GAP, -MIN_GAP), c=c)
            counts.append(n)
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])

    def test_lowering_needs_mi_support(self):
        """c<0 adds nothing on steps without evidence/sprinkle."""
        merged = [False] * WIN
        arr = _arrays(WIN, pres_val=0.45, thr_val=0.5, merged=merged)
        n, _ = _mirror_window(*arr, 0, WIN, (-MIN_GAP, -MIN_GAP), c=-0.2)
        self.assertEqual(n, 0)
        arr2 = _arrays(WIN, pres_val=0.45, thr_val=0.5)     # evidence present
        n2, _ = _mirror_window(*arr2, 0, WIN, (-MIN_GAP, -MIN_GAP), c=-0.2)
        self.assertGreater(n2, 0)

    def test_protected_accent_never_raised(self):
        self.assertEqual(_offset_at(5, 0.2, [True] * 8, [False] * 8, {5}), 0.0)
        self.assertEqual(_offset_at(4, 0.2, [True] * 8, [False] * 8, {5}), 0.2)
        # lowering is not blocked by protection (it can only keep the cell)
        self.assertEqual(_offset_at(5, -0.2, [True] * 8, [False] * 8, {5}),
                         -0.2)


class TestBeam(unittest.TestCase):
    def test_offset_search_moves_toward_desired(self):
        # graded conviction: rising thr sheds steps one band at a time
        pres = [[0.5 + 0.4 * (i / WIN)] * 2 for i in range(WIN)]
        arr = (pres, [0.45] * WIN, [0.45] * WIN, [True] * WIN,
               [False] * WIN, [False] * WIN)
        base, _ = _mirror_window(*arr, 0, WIN, (-MIN_GAP, -MIN_GAP))
        c, n, _ev = _pick_offset(arr, 0, WIN, (-MIN_GAP, -MIN_GAP),
                                 base // 2, frozenset())
        self.assertGreater(c, 0.0)
        self.assertLess(n, base)

    def test_beam_width_and_tail_frozen(self):
        T = 2 * WIN + 5
        arr = _arrays(T, pres_val=0.6, thr_val=0.5)
        windows = [(0, WIN), (WIN, 2 * WIN), (2 * WIN, T)]
        # request much lower occupancy in window 0, keep window 1
        base, _ = _mirror_window(*arr, 0, WIN, (-MIN_GAP, -MIN_GAP))
        occ_targets = [0.5 * base / WIN, base / WIN]
        scheds, _ev = beam_schedule(arr, windows, occ_targets,
                                    [base, base, 0], frozenset())
        self.assertLessEqual(len(scheds), 4)
        best = scheds[0]
        self.assertEqual(len(best["multipliers"]), 3)
        self.assertEqual(best["multipliers"][2], 1.0)       # partial tail
        self.assertEqual(best["offsets"][2], 0.0)
        self.assertEqual(best["multipliers"][0], 0.9)       # follows request
        self.assertLessEqual(best["est_err"], scheds[-1]["est_err"])

    def test_traj_err_ignores_untargeted_windows(self):
        windows = [(0, WIN), (WIN, WIN + 10)]
        self.assertEqual(_traj_err([WIN // 2, 3], windows, [0.5, None]), 0.0)
        self.assertAlmostEqual(
            _traj_err([WIN // 2, 3], windows, [0.25, None]), 0.25)

    def test_emitted_occ_counts_steps_not_notes(self):
        raw = [(0, 0, 1, 0, 1), (0, 1, 2, 0, 1), (5, 0, 1, 0, 1)]
        self.assertEqual(_emitted_occ(raw, [(0, WIN)]), [2])


class TestTrajectories(unittest.TestCase):
    def test_constant_trajectory_shape(self):
        req = torch.arange(10, dtype=torch.float32)
        t = constant_trajectory(7, req)
        self.assertEqual(t.shape, (7, 12))
        self.assertTrue(torch.equal(t[3, :10], req))

    def test_planner_rollout_bounded_and_deterministic(self):
        from pathlib import Path
        from phrase_planner import load_planner, planner_trajectory
        ckpt = (Path(__file__).resolve().parent.parent / "experiments" /
                "quality-v1" / "q3-planner" / "planner-seed20260921.pt")
        if not ckpt.exists():
            self.skipTest("planner checkpoint not present")
        model, norm = load_planner(ckpt)
        torch.manual_seed(0)
        x = torch.rand(5, 13)
        req = torch.full((10,), 0.4)
        a = planner_trajectory(model, norm, x, req)
        b = planner_trajectory(model, norm, x, req)
        self.assertEqual(a.shape, (5, 12))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(float(a[:, :10].min()) >= 0.0)
        self.assertTrue(float(a[:, :10].max()) <= 2.0)


def _integration(song):
    """Real-pipeline contract check on ONE panel song (both modes):
    rest identities preserved, accents occupied, <=2 reschedules, failure
    explicit. Prints a compact report; asserts the roadmap contract."""
    from pathlib import Path
    from convert import parse_osu
    from eval.quality_panel import dev_entries
    from phrase_planner import schedule_workload
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent.parent
    e = next(x for x in dev_entries() if song in x["song"].lower()
             or song in x["family"].lower())
    audio = Path(e["audio"])
    if not audio.is_absolute():
        audio = root / audio
    _m, objects, bpm, offset = parse_osu(root / e["osu"])
    ckpt = root / "experiments/quality-v1/q3-planner/planner-seed20260921.pt"
    for mode in ("planner", "constant"):
        r = schedule_workload(objects, bpm, offset, str(audio),
                              ckpt=ckpt, mode=mode)
        out = r["outcome"]
        assert out["protected_rest_ids"] == r["baseline"]["protected_rest_ids"]
        occupied = set(out["occupied_ids"])
        assert not (occupied & set(r["baseline"]["protected_rest_ids"])), \
            "notes on protected rests"
        if out["source"] == "reschedule":
            assert set(r["baseline"]["protected_accent_ids"]) <= occupied
        assert r["accounting"]["complete_reschedules"] <= 2
        assert r["achieved"] == (out["source"] == "reschedule")
        if not r["achieved"]:
            assert r["reasons"], "unachieved must carry explicit reasons"
        print(f"[{mode}] achieved={r['achieved']} "
              f"err {r['b0_err']:.4f} -> {out['err']:.4f} "
              f"resched={r['accounting']['complete_reschedules']} "
              f"mirror_evals={r['accounting']['mirror_evals']} "
              f"elapsed {r['elapsed_s']:.0f}s (b0 {r['accounting']['b0_s']:.0f}s)"
              f" reasons={r['reasons']}")
    print("integration contract OK")


if __name__ == "__main__":
    import sys
    if "--song" in sys.argv:
        _integration(sys.argv[sys.argv.index("--song") + 1])
    else:
        unittest.main()
