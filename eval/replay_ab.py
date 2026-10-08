"""Replay off-vs-legacy paired experiment (plan phase 1, work packet).

  .venv/bin/python -m eval.replay_ab --osu src.osu --audio song.mp3 \
      --out experiments/flow-v1/replay-ab/<song> [--seeds 6]

Two comparisons on ONE frozen source and identical candidate seeds:
- frozen arm: direct groom_notes decodes with calibrate=False — no density
  calibration and no learned breathers, so the ONLY difference between
  modes is the literal copying (replay-only question).
- e2e arm: full convert_groomed per mode (calibration, gates, critic,
  closed loop) with every candidate collected (end-to-end question).

Writes every candidate as a v2 .dat (diagnostic artifact, NOT a game
export) + results.json with metrics, traces, and source hashes. The
audio-feature cache is redirected into --out; production caches are
never touched. Maps are never regenerated or overwritten.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# the 192-dim flow model's GEMMs are tiny: default all-core threading makes
# OpenMP spin-wait overhead dominate (measured ~100x slowdown with two
# concurrent full-thread decodes). Cap low; override via OSU2BS_THREADS.
torch.set_num_threads(int(os.environ.get("OSU2BS_THREADS", "4")))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_state(afeat=None, grid=None):
    """Every actual input digest: models, executing source, feature tensor
    and grid, git rev + dirty status (review R4)."""
    root = Path(__file__).parent.parent
    run = lambda *c: subprocess.run(c, text=True, capture_output=True,
                                    cwd=root).stdout.strip()
    st = {"code_rev": run("git", "rev-parse", "HEAD") or "unknown",
          "git_dirty": bool(run("git", "status", "--porcelain")),
          "code_sha256": {p: sha(root / p) for p in
                          ("groom.py", "convert.py", "parity.py",
                           "eval/replay_ab.py", "eval/map_reader.py")},
          "model_sha256": {p: sha(root / p) for p in
                           ("groom.pt", "flow.pt", "critic.pt")
                           if (root / p).exists()}}
    if afeat is not None:
        st["feature_sha256"] = hashlib.sha256(
            afeat.numpy().tobytes()).hexdigest()
        st["grid"] = grid
    return st


def remeasure(out):
    """Re-derive metrics + full convert.check results for every saved
    candidate in an existing experiment dir, WITHOUT regenerating anything.
    Writes remeasure-v2.json (old metric kept beside the new one)."""
    import convert
    import motion
    from eval.map_reader import read_dat, metrics, METRIC_VERSION
    r = json.loads((out / "results.json").read_text())
    beat_ms = 60000.0 / r["bpm"]
    rows = []
    for arm in ("frozen", "e2e"):
        for e in r[arm]:
            dat = read_dat(out / e["dat"])
            m = metrics(dat["notes"], beat_ms)
            notes = [{"t": b * beat_ms, "hand": h, "col": c, "layer": l,
                      "dir": d} for b, h, c, l, d in dat["notes"]]
            walls = [{"t": w["_time"] * beat_ms, "dur": w["_duration"] * beat_ms,
                      "col": w["_lineIndex"]} for w in dat["walls_raw"]]
            problems = convert.check(notes, r["bpm"], walls)
            mrep = motion.report([(b * beat_ms, h, c, l, d)
                                  for b, h, c, l, d in dat["notes"]])
            rows.append({"dat": e["dat"], "arm": arm, "mode": e["mode"],
                         "seed": e["seed"], "selected": e.get("selected"),
                         "dup8_v1": e["metrics"]["raw_dup8_pct"],
                         "metrics_v2": m, "motion": mrep,
                         "check_problems": problems})
    (out / "remeasure-v2.json").write_text(json.dumps(
        {"metric_version": METRIC_VERSION, **code_state(), "candidates": rows},
        indent=1))
    for arm in ("frozen", "e2e"):
        for mode in ("legacy", "off"):
            d = [c["metrics_v2"]["raw_dup8_pct"] for c in rows
                 if c["arm"] == arm and c["mode"] == mode]
            v1 = [c["dup8_v1"] for c in rows
                  if c["arm"] == arm and c["mode"] == mode]
            print(f"{arm:6s} {mode:6s} dup8 v1 mean {sum(v1)/len(v1):5.1f}% "
                  f"-> v2 mean {sum(d)/len(d):5.1f}%  "
                  f"(min {min(d):.1f} max {max(d):.1f}, n={len(d)})")
    for c in rows:
        if c["check_problems"]:
            print(f"CHECK FAILED {c['dat']}: {c['check_problems']}")
        if c.get("selected"):
            print(f"selected {c['dat']}: dup8 v2 "
                  f"{c['metrics_v2']['raw_dup8_pct']}%")
    print(f"-> {out}/remeasure-v2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osu")
    ap.add_argument("--audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--remeasure", action="store_true",
                    help="recompute metrics + checker for saved candidates")
    a = ap.parse_args()
    out = Path(a.out)
    if a.remeasure:
        return remeasure(out)
    out.mkdir(parents=True, exist_ok=True)

    import groom
    groom._FEAT_CACHE = out / "feats_cache.pt"  # keep production cache clean
    groom._FEATS.clear()
    import convert
    from eval.map_reader import metrics

    meta, objects, bpm, offset0 = convert.parse_osu(a.osu)
    steps, T, step_ms, offset, grid = convert.grid_steps(
        objects, bpm, offset0, meta.get("_timing"))
    afeat = groom.cached_audio_features(a.audio, [grid.time(s) for s in range(T)])
    beat_ms = 60000.0 / bpm
    spec = convert.diff_spec("ExpertPlus")

    def measure(raw, off_ms):
        import motion
        notes = sorted((round(grid.time(s) / beat_ms, 5), h, c, l, d)
                       for s, h, c, l, d in raw)
        mrep = motion.report([(b * beat_ms, h, c, l, d)
                              for b, h, c, l, d in notes])
        return notes, metrics(notes, beat_ms), mrep

    def save(tag, notes_beats, walls, off_ms):
        dat = {"_version": "2.0.0", "_events": [], "_notes": [
            {"_time": b, "_lineIndex": c, "_lineLayer": l, "_type": h,
             "_cutDirection": d} for b, h, c, l, d in notes_beats],
            "_obstacles": [
            {"_time": round(grid.time(s0) / beat_ms, 5),
             "_lineIndex": col, "_type": 0,
             "_duration": round((grid.time(s0 + ln) - grid.time(s0)) / beat_ms, 5),
             "_width": 1}
            for s0, ln, col in walls]}
        (out / f"{tag}.dat").write_text(json.dumps(dat))

    results = {"osu": a.osu, "osu_sha256": sha(a.osu),
               "audio": a.audio, "audio_sha256": sha(a.audio),
               "bpm": bpm, "seeds": a.seeds,
               **code_state(afeat, {"bpm": bpm, "step_ms": step_ms,
                                    "T": T, "offset_ms": offset}),
               "frozen": [], "e2e": []}
    if a.seeds != 6:
        # R5: convert_groomed's internal budget is N_DECODES-driven; --seeds
        # only sizes the frozen arm. Refuse silent budget mismatch.
        print(f"NOTE: --seeds={a.seeds} applies to the frozen arm only; "
              "the e2e arm budget is convert_groomed's own (recorded below)")

    print("== frozen arm (calibrate=False: replay-only difference)")
    for mode in ("legacy", "off"):
        for seed in range(a.seeds):
            tr = {}
            raw, walls = groom.groom_notes(
                steps, T, step_ms, offset, afeat=afeat, seed=seed,
                temp=groom.TEMPS[seed % len(groom.TEMPS)],
                drate=spec["cond"], replay_mode=mode, calibrate=False,
                trace=tr, grid=grid)
            nb, m, mrep = measure(raw, offset)
            tag = f"frozen-{mode}-s{seed}"
            save(tag, nb, walls, offset)
            results["frozen"].append(
                {"mode": mode, "seed": seed, "dat": f"{tag}.dat",
                 "metrics": m, "motion": mrep, "trace": tr})
            print(f"  {tag}: dup8 {m['raw_dup8_pct']:5.1f}%  "
                  f"sps {m['raw_sps']:.2f}  double {m['raw_double_pct']:.1f}%")

    print("== e2e arm (full convert_groomed selection)")
    for mode in ("legacy", "off"):
        coll = []
        notes, walls = convert.convert_groomed(
            objects, bpm, offset0, a.audio, "ExpertPlus",
            replay_mode=mode, collect=coll, timing=meta.get("_timing"))
        for i, e in enumerate(coll):
            nb, m, mrep = measure(e["notes"], offset)
            # candidate index keeps IDs collision-free across passes (R5);
            # rate_scale stored as exact repr, pass derivable from it
            tag = f"e2e-{mode}-c{i:02d}-s{e['seed']}"
            save(tag, nb, e["walls"], offset)
            results["e2e"].append(
                {"mode": mode, "cand": i, "seed": e["seed"], "temp": e["temp"],
                 "rate_scale": repr(e["rate_scale"]), "gates_ok": e["ok"],
                 "gates": e.get("gates"), "motion_cost": e.get("motion_cost"),
                 "score": round(e["score"], 3), "selected": e["selected"],
                 "dat": f"{tag}.dat", "metrics": m, "motion": mrep,
                 "trace": e["trace"]})
        sel = next(r for r in results["e2e"]
                   if r["mode"] == mode and r["selected"])
        print(f"  {mode} selected: seed {sel['seed']}  "
              f"dup8 {sel['metrics']['raw_dup8_pct']:.1f}%  "
              f"sps {sel['metrics']['raw_sps']:.2f}")

    for arm in ("frozen", "e2e"):
        for mode in ("legacy", "off"):
            d = [r["metrics"]["raw_dup8_pct"] for r in results[arm]
                 if r["mode"] == mode]
            print(f"{arm:6s} {mode:6s} dup8 mean {sum(d)/len(d):5.1f}%  "
                  f"min {min(d):5.1f}%  max {max(d):5.1f}%  n={len(d)}")
    # explicit candidate budget per arm/mode (R5); unequal = not paired
    results["budget"] = {f"{arm}_{mode}": sum(1 for r in results[arm]
                                              if r["mode"] == mode)
                         for arm in ("frozen", "e2e")
                         for mode in ("legacy", "off")}
    b = results["budget"]
    if b["e2e_legacy"] != b["e2e_off"]:
        print(f"WARNING: unequal e2e budgets {b} — treat as unpaired")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"-> {out}/results.json")


if __name__ == "__main__":
    main()
