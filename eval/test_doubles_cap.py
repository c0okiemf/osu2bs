"""Frozen 8-song evaluation of the opt-in per-8-beat doubles cap (review spec).

Per song (production settings, replay off): baseline convert_groomed -> capture
per-attempt plans -> set doubles_cap_8b = (max baseline winner window doubles)-1
on every attempt -> replay. Accept only if, per paired attempt:
  - cap honored: final doubles per aligned 8-beat window <= cap everywhere
    except windows explicitly reported infeasible (accent-kept overflow);
  - occupied timestamps preserved (the set of note steps is unchanged);
  - rests preserved (identical rest masks);
  - density band status preserved (in-band before => in-band after, winner);
  - parity/validity gates preserved (empty check-problems stay empty).
Infeasible-request probe: cap=0 on the densest song at the groom seam — every
double the decode keeps must be reported in windows_over, never silent.
Defaults unchanged: without a plan cap nothing differs (covered by the no-op
suite in eval/test_phrase_plan.py).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "eval" / "clean_rhythm_panel.json"
W = 32  # 8 beats * STEPS_PER_BEAT


def _entries():
    return json.loads(PANEL.read_text())["entries"]


def window_doubles(notes):
    """Final doubles per aligned 8-beat window from step 5-tuples (heads only)."""
    by_step = {}
    for s, h, _c, _l, d in notes:
        if d != 8:
            by_step.setdefault(s, set()).add(h)
    win = {}
    for s, hs in by_step.items():
        if len(hs) == 2:
            win[s // W] = win.get(s // W, 0) + 1
    return win


def occupied(notes):
    return {s for s, *_ in notes}


def _rate_in_band(notes, grid, band):
    ts = sorted(grid.time(s) for s, _h, _c, _l, d in notes if d != 8)
    span = (ts[-1] - ts[0]) / 1000 if len(ts) > 1 else 0
    if span < 30:
        return None                                   # convert skips the check
    actual = len(ts) / span
    return band[0] * 0.98 <= actual <= band[1] * 1.02


def _synth_extra_plans(plans):
    """Capping can gate more candidates, so the replay may draw hotter seeds the
    baseline never ran. Within a pass the frozen decisions are seed-independent
    (asserted), so extend each pass's plan to seeds up to 2*N_DECODES-1."""
    from groom import N_DECODES
    by_pass = {}
    for key, p in plans.items():
        pi = int(key.split(":")[0])
        ref = by_pass.setdefault(pi, p)
        assert (p["rests"] == ref["rests"]
                and p["scheduling"] == ref["scheduling"]), \
            f"pass {pi} decisions differ across seeds — cannot synthesize"
    out = dict(plans)
    for pi, ref in by_pass.items():
        for seed in range(2 * N_DECODES):
            out.setdefault(f"{pi}:{seed}", ref)
    return out


def run_song(entry):
    import torch
    from convert import convert_groomed, parse_osu, grid_steps, diff_spec
    from eval.phrase_plan import plans_from_collect, inject_plans
    _m, objects, bpm, offset = parse_osu(ROOT / entry["osu"])
    audio = str(ROOT / entry["audio"])
    band = diff_spec("ExpertPlus")["band"]
    _steps, T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
    rec_a = []
    convert_groomed(objects, bpm, offset, audio_path=audio, diff="ExpertPlus",
                    replay_mode="off", collect=rec_a, thin=True, calibrate=True)
    win_a = next(r for r in rec_a if r["selected"])
    mx = max(window_doubles(win_a["notes"]).values(), default=0)
    if mx < 2:
        return {"song": entry["song"], "skipped": f"max window doubles {mx} < 2"}
    cap = mx - 1
    plans = _synth_extra_plans(plans_from_collect(rec_a))
    for p in plans.values():
        p["doubles_cap_8b"] = cap
    plans = json.loads(json.dumps(plans))
    rec_b = []
    try:
        convert_groomed(objects, bpm, offset, audio_path=audio, diff="ExpertPlus",
                        replay_mode="off", collect=rec_b, thin=True,
                        calibrate=True, plans=inject_plans(plans))
    except Exception as e:                         # band-retry etc. = infeasible
        return {"song": entry["song"], "cap": cap, "max_before": mx,
                "replay_failed": f"{type(e).__name__}: {e}"}
    a_by = {(r["pass_index"], r["seed"]): r for r in rec_a}
    b_by = {(r["pass_index"], r["seed"]): r for r in rec_b}
    res = {"song": entry["song"], "cap": cap, "max_before": mx,
           "pairs": 0, "cap_violations": 0, "timestamp_diffs": 0,
           "rest_diffs": 0, "gate_regressions": 0, "windows_over": 0,
           "demoted": 0}
    for key in sorted(set(a_by) & set(b_by)):
        ra, rb = a_by[key], b_by[key]
        res["pairs"] += 1
        dc = rb["trace"]["doubles_cap"]
        res["windows_over"] += sum(dc["windows_over"].values())
        res["demoted"] += dc["demoted"]
        wd = window_doubles(rb["notes"])
        for w, n in wd.items():
            over = dc["windows_over"].get(str(w), 0)
            if n - over > cap:
                res["cap_violations"] += 1
        # counter cross-check: the trace's per-window == recount from notes
        assert {int(k): v for k, v in dc["per_window"].items()} == wd, \
            f"{entry['song']} {key}: cap counter disagrees with decoded notes"
        if occupied(ra["notes"]) != occupied(rb["notes"]):
            res["timestamp_diffs"] += 1
        if not torch.equal(ra["trace"]["rest_mask"], rb["trace"]["rest_mask"]):
            res["rest_diffs"] += 1
        if ra["gates"]["check"] == [] and rb["gates"]["check"] != []:
            res["gate_regressions"] += 1
    win_b = next(r for r in rec_b if r["selected"])
    res["band_before"] = _rate_in_band(win_a["notes"], grid, band)
    res["band_after"] = _rate_in_band(win_b["notes"], grid, band)
    res["band_preserved"] = (res["band_before"] is None
                             or res["band_after"] == res["band_before"]
                             or res["band_after"] is True)
    # MOTION AUDIT : suppressed notes are not physical swings, so judge
    # the capped chart on its EMITTED notes with the ORIGINAL A/B gates
    # (repositioning@4 extents, <> pair exposure, hand-workload/monopoly, plus
    # validity/band/dup8/rest) versus the uncapped output. A failing gate marks
    # this cap request MOTION-INFEASIBLE — explicit, never silent.
    from eval.clean_rhythm import _arm_metrics, _winner_quiet
    from eval.clean_rhythm_eval import gate_song
    beat_ms = 60000.0 / bpm
    arms_a = _arm_metrics(rec_a, win_a["notes"], grid, beat_ms, band)
    arms_b = _arm_metrics(rec_b, win_b["notes"], grid, beat_ms, band)
    qa = _winner_quiet(win_a["notes"], grid, T)
    qb = _winner_quiet(win_b["notes"], grid, T)
    arms_b["quiet"] = {"a_ids": qa["quiet_ids"], "b_ids": qb["quiet_ids"],
                       "n_bins": qb["n_bins"]}
    res["motion_gates_failed"] = gate_song(arms_a, arms_b)
    res["motion_infeasible"] = bool(res["motion_gates_failed"])
    res["ok"] = (res["cap_violations"] == 0 and res["timestamp_diffs"] == 0
                 and res["rest_diffs"] == 0 and res["gate_regressions"] == 0
                 and res["band_preserved"])
    return res


def infeasible_probe():
    """cap=0 at the groom seam on the densest song: every kept double must be
    explicitly reported in windows_over (accent-kept), never silent."""
    from convert import parse_osu
    from eval.phrase_plan import capture_plan, decode_with_plan
    entry = next(e for e in _entries() if e["song"] == "still_waiting")
    _m, objects, bpm, offset = parse_osu(ROOT / entry["osu"])
    audio = str(ROOT / entry["audio"])
    plan, raw0, _w0 = capture_plan(objects, bpm, offset, audio, seed=0, temp=0.85)
    plan = json.loads(json.dumps(plan))
    plan["doubles_cap_8b"] = 0
    tr = {}
    raw1, _w1 = decode_with_plan(objects, bpm, offset, audio, plan,
                                 seed=0, temp=0.85, trace=tr)
    dc = tr["doubles_cap"]
    wd = window_doubles(raw1)
    silent = sum(1 for w, n in wd.items()
                 if n > dc["windows_over"].get(str(w), 0))
    return {"kept_doubles": sum(wd.values()),
            "reported_over": sum(dc["windows_over"].values()),
            "demoted": dc["demoted"], "silent_violations": silent,
            "timestamps_preserved": occupied(raw0) == occupied(raw1),
            "ok": silent == 0 and sum(wd.values()) == sum(
                dc["windows_over"].values())}


def admission_cases():
    """review: candidate admission — verify a MIXED-feasibility case (winner
    must come from the admitted subset) and an ALL-INFEASIBLE case (explicit
    result, no capped chart), within the existing attempt budgets."""
    from convert import parse_osu
    from eval.phrase_plan import apply_doubles_cap

    def _apply(entry, cap):
        _m, objects, bpm, offset = parse_osu(ROOT / entry["osu"])
        return apply_doubles_cap(objects, bpm, offset,
                                 str(ROOT / entry["audio"]), cap)

    mixed = None
    for e in _entries():
        r = _apply(e, "max-1")
        n_adm, n_all = len(r["admitted"]), r["attempts"]
        print(f"  {e['song']:<18} cap={r['cap']} admitted {n_adm}/{n_all} "
              f"feasible={r['feasible']} winner={r.get('winner_key')}")
        assert r["attempts"] == r["baseline_attempts"], "extra decodes used"
        if 0 < n_adm < n_all:
            mixed = (e["song"], r)
            break                          # first mixed case is the exhibit
    assert mixed, "no mixed-feasibility song found in the panel"
    song, r = mixed
    assert r["feasible"] and r["winner_key"] in r["admitted"]
    assert all(r["per_candidate"][k] == [] for k in r["admitted"])
    print(f"  MIXED exhibit: {song} — winner {r['winner_key']} from "
          f"{len(r['admitted'])} admitted of {r['attempts']}")
    infeasible = None
    for e in _entries():                   # cap=0: hunt an all-infeasible case
        r0 = _apply(e, 0)
        print(f"  {e['song']:<18} cap=0 admitted {len(r0['admitted'])}/"
              f"{r0['attempts']} feasible={r0['feasible']}")
        if not r0["feasible"]:
            infeasible = (e["song"], r0)
            break
    assert infeasible, "no all-infeasible song found at cap=0"
    song0, r0 = infeasible
    assert "notes" not in r0 and r0["admitted"] == []
    assert all(fails for fails in r0["per_candidate"].values()), \
        "infeasible result must carry explicit per-candidate reasons"
    print(f"  ALL-INFEASIBLE exhibit: {song0} cap=0 — no chart exported, "
          f"every candidate carries explicit reasons")
    print("PASS doubles-cap candidate admission (mixed + all-infeasible)")


def main():
    import sys
    import torch
    torch.set_num_threads(4)
    if "--admission" in sys.argv:
        admission_cases()
        return
    bad, skipped, motion_clean, audited = 0, 0, 0, 0
    for e in _entries():
        r = run_song(e)
        if "skipped" in r:
            skipped += 1
            print(f"  {r['song']:<18} SKIPPED ({r['skipped']})")
            continue
        if "replay_failed" in r:
            bad += 1
            print(f"  {r['song']:<18} cap={r['cap']} REPLAY FAILED: "
                  f"{r['replay_failed']}")
            continue
        flag = "OK" if r["ok"] else "FAIL"
        motion = ("clean" if not r["motion_infeasible"]
                  else "INFEASIBLE:" + ",".join(r["motion_gates_failed"]))
        print(f"  {r['song']:<18} cap={r['cap']} (max was {r['max_before']}) "
              f"pairs={r['pairs']} demoted={r['demoted']} "
              f"over={r['windows_over']} viol={r['cap_violations']} "
              f"ts_diff={r['timestamp_diffs']} rest_diff={r['rest_diffs']} "
              f"gate_reg={r['gate_regressions']} band={r['band_before']}->"
              f"{r['band_after']} motion={motion}  [{flag}]")
        bad += not r["ok"]
        audited += 1
        motion_clean += not r["motion_infeasible"]
    p = infeasible_probe()
    print(f"  cap=0 probe: kept={p['kept_doubles']} reported={p['reported_over']} "
          f"demoted={p['demoted']} silent={p['silent_violations']} "
          f"ts_preserved={p['timestamps_preserved']}  "
          f"[{'OK' if p['ok'] else 'FAIL'}]")
    print(f"DOUBLES-CAP: {8 - skipped - bad}/{8 - skipped} songs pass, "
          f"{skipped} skipped, probe {'OK' if p['ok'] else 'FAIL'}")
    print(f"MOTION AUDIT: {motion_clean}/{audited} caps motion-clean on emitted "
          f"notes (original gates); failing constraints marked infeasible above")
    assert bad == 0 and p["ok"], "doubles-cap acceptance failed"
    print("PASS opt-in per-8-beat doubles cap")


if __name__ == "__main__":
    main()
