"""Causal diagnosis: does rate calibration or rest policy drive the loss?
(Packet D, review-approved, 51 candidates, no training / no new MI.)

Four sentinels (The Prince, Dreamland, PLAY DICE!, Picture Perfect control).
Per song, ONE grid and audio-feature set (from thinned evidence) are held
fixed; the baseline rest mask and both threshold-vector sets (frozen calibrated
/ base unadjusted) are derived ONCE from thinned evidence at rate_scale 1.

Grid: 4 cells = evidence{thinned,dense} x density{calibrated,base} with the
SAME baseline rest mask, 3 paired seeds each -> 48 candidates. +3 Prince cells
(dense, base thresholds, rest mask REMOVED) -> 51. Direct groom_notes decodes,
replay off, NO best-of-N / rate re-decode. Every candidate saved (incl.
out-of-band). Step-addressable: each new-only dense cell is admitted, rejected
by a named rule, or a thinned cell was replaced.

  .venv/bin/python -m eval.pilot_causal
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import convert
import groom
import motion
from eval.map_reader import metrics
from timing import STEPS_PER_BEAT, TimeGrid

MI = Path(__file__).parent.parent / "experiments" / "pilot-mi"
OUT = Path(__file__).parent.parent / "experiments" / "pilot-causal"
SENTINELS = ["The Prince", "Dreamland", "PLAY DICE!", "Picture Perfect"]
SEEDS = (0, 1, 2)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _prov(song):
    for d in MI.iterdir():
        p = d / "provenance.json"
        if p.exists() and json.loads(p.read_text())["song"] == song:
            return d, json.loads(p.read_text())
    raise KeyError(song)


def _steps_on(grid, hits, step_ms, off):
    return {max(0, round((h[0] - off) / step_ms)) for h in hits}


def analyze_cells(emitted_steps, thin_steps, dense_steps, rest_mask, pres,
                  thr, keep, grid):
    """Account for every new-only dense cell : admitted, or rejected by
    a named rule (rest / below-keep threshold / THIN adjacency), plus how many
    thinned cells were displaced."""
    new_only = dense_steps - thin_steps
    em = emitted_steps
    out = {"new_only": len(new_only), "new_admitted": 0,
           "new_rej_rest": 0, "new_rej_threshold": 0, "new_rej_thin": 0,
           "new_rej_other": 0,
           "thin_cells_emitted": len(thin_steps & em),
           "thin_cells_dropped": len(thin_steps - em)}
    for s in new_only:
        if s in em:
            out["new_admitted"] += 1
        elif s < len(rest_mask) and bool(rest_mask[s]):
            out["new_rej_rest"] += 1
        elif max(float(pres[s, 0]), float(pres[s, 1])) < float(keep[s]):
            out["new_rej_threshold"] += 1
        elif any((s - 1) in em or (s + 1) in em for _ in [0]) and \
                grid.time(s) - grid.time(max(s - 1, 0)) < groom.THIN_MS:
            out["new_rej_thin"] += 1
        else:
            out["new_rej_other"] += 1
    return out


def summarize(raw, tr, grid, beat_ms):
    notes = sorted((round(grid.time(s) / beat_ms, 5), h, c, l, d)
                   for s, h, c, l, d in raw)
    m = metrics(notes, beat_ms)
    rep = motion.report([(grid.time(s), h, c, l, d) for s, h, c, l, d in raw])
    ok = not convert.check([{"t": grid.time(s), "hand": h, "col": c,
                             "layer": l, "dir": d} for s, h, c, l, d in raw],
                           60000.0 / beat_ms)
    return {"notes": len(raw), "sps": m["raw_sps"],
            "doubles_pct": m["raw_double_pct"], "rest2s_pct": m["raw_rest2s_pct"],
            "hand_ratio": m["raw_hand_ratio"], "dup8_pct": m["raw_dup8_pct"],
            "narrow_lr": m["lr_outward_row_doubles"],
            "flags_per_1k": rep["flags"]["flags_per_1000"],
            "longest_run": rep["workload"]["longest_run"]["count"],
            "valid": ok, "emitted_steps": sorted({s for s, *_ in raw})}


def run_song(song):
    d, prov = _prov(song)
    meta, objects, bpm, offset = convert.parse_osu(d / "gen.osu")
    beat_ms = 60000.0 / bpm
    step_ms = beat_ms / STEPS_PER_BEAT
    audio = str(next(q for q in Path(prov["corpus_dir"]).iterdir()
                     if q.suffix.lower() in (".egg", ".ogg")))
    # ONE fixed grid from thinned evidence, both variants snap to it
    lead = int(offset / (4 * step_ms)) * 4
    off = offset - lead * step_ms
    thin_hits = convert._hits(objects, thin=True)
    dense_hits = convert._hits(objects, thin=False)
    thin_steps = {max(0, round((h[0] - off) / step_ms)) for h in thin_hits}
    dense_steps = {max(0, round((h[0] - off) / step_ms)) for h in dense_hits}
    T = max(max(thin_steps), max(dense_steps)) + 8
    grid = TimeGrid.uniform(step_ms, off, T)
    afeat = groom.cached_audio_features(audio, [grid.time(s) for s in range(T)])
    spec = convert.diff_spec("ExpertPlus")
    # baseline: thinned evidence, seed 0, defaults -> mask + calibrated vectors
    base_tr = {}
    groom.groom_notes(thin_steps, T, step_ms, off, afeat=afeat, seed=0,
                      drate=spec["cond"], band_scale=spec["scale"],
                      band=spec["band"], replay_mode="off", grid=grid,
                      trace=base_tr)
    rest_mask = base_tr["rest_mask"]
    calib_vec = (base_tr["thr"], base_tr["keep"])
    # base unadjusted vectors: same thinned evidence, density_adjust off
    base2 = {}
    groom.groom_notes(thin_steps, T, step_ms, off, afeat=afeat, seed=0,
                      drate=spec["cond"], band_scale=spec["scale"],
                      band=spec["band"], replay_mode="off", grid=grid,
                      density_adjust=False, rest_mask_in=rest_mask, trace=base2)
    base_vec = (base2["thr"], base2["keep"])

    cands = []
    cells = [("thinned", "calibrated", thin_steps, calib_vec, rest_mask),
             ("thinned", "base", thin_steps, base_vec, rest_mask),
             ("dense", "calibrated", dense_steps, calib_vec, rest_mask),
             ("dense", "base", dense_steps, base_vec, rest_mask)]
    if song == "The Prince":  # +3: dense + base + mask REMOVED
        cells.append(("dense", "base_norest", dense_steps, base_vec, None))
    for ev, dens, steps, vec, mask in cells:
        for seed in SEEDS:
            tr = {}
            raw, walls = groom.groom_notes(
                steps, T, step_ms, off, afeat=afeat, seed=seed,
                drate=spec["cond"], band_scale=spec["scale"], band=spec["band"],
                replay_mode="off", grid=grid,
                rest_mask_in=mask if mask is not None else torch_zeros(T),
                rest_policy="off" if mask is None else None,
                thr_vectors=vec, trace=tr)
            s = summarize(raw, tr, grid, beat_ms)
            cell = analyze_cells(set(s["emitted_steps"]), thin_steps,
                                 dense_steps, rest_mask, tr["pres"],
                                 vec[0], vec[1], grid)
            cands.append({"evidence": ev, "density": dens, "seed": seed,
                          **{k: v for k, v in s.items() if k != "emitted_steps"},
                          "cells": cell})
    return {"song": song, "bpm": bpm, "osu_sha": sha(d / "gen.osu"),
            "audio_sha": sha(audio),
            "thin_steps": len(thin_steps), "dense_steps": len(dense_steps),
            "new_only_cells": len(dense_steps - thin_steps),
            "rest_mask_cells": int(rest_mask.sum()),
            "calib_thr_offset": base_tr["calibration"]["thr_offset"]
            if "calibration" in base_tr else 0.0,
            "candidates": cands}


def torch_zeros(T):
    import torch
    return torch.zeros(T, dtype=torch.bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fp = {"code": sha(Path(__file__).parent.parent / "groom.py"),
          "convert": sha(Path(__file__).parent.parent / "convert.py"),
          "models": {n: sha(Path(__file__).parent.parent / n)
                     for n in ("groom.pt", "flow.pt")}}
    results = []
    for song in SENTINELS:
        cache = OUT / (song.replace(" ", "_").replace("!", "") + ".json")
        r = run_song(song)
        r["fingerprint"] = fp
        cache.write_text(json.dumps(r, indent=1))
        results.append(r)
        # per-cell winner-agnostic mean (3 seeds) for the key metrics
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(list))
        for c in r["candidates"]:
            for k in ("sps", "doubles_pct", "rest2s_pct", "flags_per_1k",
                      "narrow_lr"):
                agg[(c["evidence"], c["density"])][k].append(c[k])
        print(f"== {song} (offset {r['calib_thr_offset']}, mask "
              f"{r['rest_mask_cells']}, new-only {r['new_only_cells']})")
        for key in sorted(agg):
            a = agg[key]
            print(f"   {key[0]:7s}/{key[1]:12s} sps {sum(a['sps'])/3:.2f} "
                  f"dbl {sum(a['doubles_pct'])/3:.1f} rest {sum(a['rest2s_pct'])/3:.1f} "
                  f"flags {sum(a['flags_per_1k'])/3:.1f} <> {sum(a['narrow_lr'])/3:.1f}")
    (OUT / "summary.json").write_text(json.dumps(
        {"fingerprint": fp, "songs": [r["song"] for r in results],
         "n_candidates": sum(len(r["candidates"]) for r in results)}, indent=1))
    print(f"\ntotal candidates: {sum(len(r['candidates']) for r in results)} "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
