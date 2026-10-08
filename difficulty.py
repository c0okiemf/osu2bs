"""Q3 operational workload calibration (spec 2026-09-21 §6, line "Fit
operational difficulty calibration").

A nonnegative pairwise logistic ranker over seconds-based cadence, 2s burst,
coincidence, travel and recovery descriptors, fit ONLY on authenticated
distinct Standard ranks WITHIN THE SAME SONG (same map dir / same Info.dat).
Train split fits, val split selects; no song, title, artist or genre input.
Duplicate custom rank-9 labels are never ordered; unauthenticated charts are
excluded, and a family without an authenticated distinct-rank pair yields NO
ordering claim (and never an equality claim).

This is workload calibration, NOT validated perceived difficulty.

Descriptors reuse eval.difficulty_vector's frozen grouping/gap primitives and
eval.visibility's resolve_authored/load_scene fail-closed adapters. The 2s
burst is recomputed with a bisect sliding window whose values are asserted
equal to the frozen O(n^2) measure_scene implementation in tests (the frozen
version is kept for the 45-chart audit; the corpus has ~2500 charts).

Artifacts (experiment-only, no production/shipped file):
  experiments/quality-v1/q3-ranker/desc/<sha16>.json   per-dir descriptors
  experiments/quality-v1/q3-ranker/workload_calibration.json
"""
import argparse
import bisect
import hashlib
import json
from math import hypot
from pathlib import Path

from eval.difficulty_vector import (TOL, grouped_events, _hand_gaps_ms,
                                    _occupied_gaps, nearest_rank)
from eval.visibility import load_scene, resolve_authored, _sha256, _num

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / "experiments" / "quality-v1" / "q3-ranker"
SEED = 20260921
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1)

# oriented so that larger plausibly = harder; nonnegative weights make every
# component monotone by construction (the anti-gaming property under test)
FEATURES = ["cad_l_p50_hz", "cad_r_p50_hz", "cad_peak_hz",
            "burst2_comb", "burst2_hand", "coincidence",
            "travel_rate", "travel_step",
            "recovery_deficit", "recov_p90_inv"]
RANK_NAMES = {1: "Easy", 3: "Normal", 5: "Hard", 7: "Expert", 9: "ExpertPlus"}


def burst_rates(events, instants, w=2.0):
    """Max grouped count in (t-w, t] per hand and combined, /w — value-equal
    to measure_scene's _burst rates (asserted in tests), O(n log n)."""
    out = {}
    for key, hand in (("left", 0), ("right", 1), ("combined", None)):
        ts = [e["t_s"] for e in events if hand is None or e["hand"] == hand]
        best = 0
        for inst in instants:
            t = inst["t_s"]
            n = (bisect.bisect_right(ts, t + TOL)
                 - bisect.bisect_right(ts, t - w + TOL))
            if n > best:
                best = n
        out[key] = best / w
    return out


def chart_descriptors(scene):
    """The 10 oriented workload components for one supported scene, or
    (None, reason). Fail-closed: unknown/empty/degenerate charts are excluded,
    never given fabricated values."""
    if scene.get("chart_unknown"):
        return None, scene.get("scope") or scene.get("settings_reason", "unknown")
    notes = scene["notes"]
    if not notes:
        return None, "no_colored_notes"
    g = grouped_events(notes)
    events, instants = g["events"], g["instants"]
    if len(events) < 2:
        return None, "too_few_events"
    D = events[-1]["t_s"] - events[0]["t_s"]
    if D <= 0:
        return None, "zero_span"
    gl, gr = _hand_gaps_ms(events, 0), _hand_gaps_ms(events, 1)
    if not isinstance(gl, dict) or not isinstance(gr, dict) \
            or "failed" in gl or "failed" in gr:
        return None, "insufficient_hand_events"
    br = burst_rates(events, instants)
    both = sum(1 for i in instants if len(i["hands"]) == 2)
    og = _occupied_gaps(instants, D)
    if og["p90"] is None or og["long_gap_share"] is None:
        return None, "insufficient_gaps"
    pos = {n.id: (n.col, n.row) for n in notes}
    travel, ntr = 0.0, 0
    for hand in (0, 1):
        prev = None
        for e in (ev for ev in events if ev["hand"] == hand):
            cx = sum(pos[i][0] for i in e["ids"]) / len(e["ids"])
            cy = sum(pos[i][1] for i in e["ids"]) / len(e["ids"])
            if prev is not None:
                travel += hypot(cx - prev[0], cy - prev[1])
                ntr += 1
            prev = (cx, cy)
    return {"cad_l_p50_hz": 1000.0 / gl["p50"],
            "cad_r_p50_hz": 1000.0 / gr["p50"],
            "cad_peak_hz": 1000.0 / min(gl["p10"], gr["p10"]),
            "burst2_comb": br["combined"],
            "burst2_hand": max(br["left"], br["right"]),
            "coincidence": both / len(instants),
            "travel_rate": travel / D,
            "travel_step": travel / ntr if ntr else 0.0,
            "recovery_deficit": 1.0 - og["long_gap_share"],
            "recov_p90_inv": 1.0 / max(og["p90"], 1e-6)}, None


def measure_dir(dir_path):
    """Authenticated Standard charts of one map dir -> descriptor records.
    Authentication is resolve_authored's (filename+hash+Standard binding);
    anything unverified is skipped with its reason."""
    d = Path(dir_path)
    info_p = next((f for f in d.iterdir() if f.name.lower() == "info.dat"), None)
    if info_p is None:
        return {"dir": str(d), "charts": [], "skipped": [{"reason": "no_info_dat"}]}
    try:
        info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as e:
        return {"dir": str(d), "charts": [],
                "skipped": [{"reason": f"info_unreadable: {type(e).__name__}"}]}
    bpm = _num(info.get("_beatsPerMinute"))
    if bpm is None or bpm <= 0:
        return {"dir": str(d), "charts": [], "skipped": [{"reason": "invalid_bpm"}]}
    charts, skipped = [], []
    seen = set()
    for s in info.get("_difficultyBeatmapSets", []):
        if s.get("_beatmapCharacteristicName") != "Standard":
            continue
        for b in s.get("_difficultyBeatmaps", []):
            fn = b.get("_beatmapFilename")
            if not isinstance(fn, str) or fn in seen:
                continue
            seen.add(fn)
            chart = d / fn
            if not chart.exists():
                skipped.append({"file": fn, "reason": "missing_file"})
                continue
            sha = _sha256(chart)
            auth = resolve_authored(chart, sha)
            if auth["status"] != "verified" or auth.get("rank") is None:
                skipped.append({"file": fn, "reason":
                                auth.get("reason", "no_authenticated_rank")})
                continue
            try:
                scene = load_scene({"chart": str(chart), "bpm": bpm,
                                    "sha256": sha}, "standard")
            except (ValueError, OSError) as e:
                skipped.append({"file": fn,
                                "reason": f"unreadable: {type(e).__name__}"})
                continue
            feats, why = chart_descriptors(scene)
            if feats is None:
                skipped.append({"file": fn, "reason": why})
                continue
            charts.append({"file": fn, "sha256": sha, "rank": auth["rank"],
                           "difficulty": auth.get("difficulty"),
                           "label": auth.get("label"),
                           "features": feats})
    return {"dir": str(d), "charts": charts, "skipped": skipped}


def measure_split(split, limit=10 ** 6):
    """Chunk-safe descriptor pass over the split's family-representative dirs
    (one dir per family — re-encodes never double-count a song). Idempotent,
    atomic per-dir writes; rerun to continue."""
    from eval import corpus
    out_dir = RDIR / "desc"
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = corpus.train_families_rep(split=split)
    n_new = 0
    for dp in sorted(dirs):
        key = hashlib.sha256(dp.encode()).hexdigest()[:16]
        out = out_dir / f"{key}.json"
        if out.exists():
            continue
        if n_new >= limit:
            break
        rec = measure_dir(dp)
        rec["split"] = split
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(out)
        n_new += 1
        print(f"  [{len(rec['charts'])} charts] {Path(dp).name[:60]}", flush=True)
    return n_new


def load_descriptors():
    recs = [json.loads(p.read_text()) for p in sorted((RDIR / "desc").glob("*.json"))]
    return ([r for r in recs if r["split"] == "train"],
            [r for r in recs if r["split"] == "val"])


def build_pairs(records):
    """Within-song ordered pairs (harder, easier) over DISTINCT authenticated
    ranks. Same-rank charts (incl. duplicate custom rank-9 labels) are never
    ordered — and never claimed equal. `adjacent`: no third distinct rank in
    the same song lies strictly between."""
    pairs = []
    for rec in records:
        by_rank = {}
        for c in rec["charts"]:
            by_rank.setdefault(c["rank"], []).append(c)
        ranks = sorted(by_rank)
        for i, hi in enumerate(ranks):
            for lo in ranks[:i]:
                adjacent = not any(lo < m < hi for m in ranks)
                for a in by_rank[hi]:
                    for b in by_rank[lo]:
                        pairs.append({"dir": rec["dir"], "hi": a, "lo": b,
                                      "hi_rank": hi, "lo_rank": lo,
                                      "adjacent": adjacent})
    return pairs


def _vec(chart):
    return [float(chart["features"][k]) for k in FEATURES]


def fit_weights(deltas, lam, steps=500):
    """Projected full-batch gradient on -log sigmoid(w . delta) + lam*|w|^2,
    w >= 0. Deterministic (zero init, no minibatching)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(SEED)
    d = torch.tensor(deltas, dtype=torch.float64)
    w = torch.zeros(d.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(steps):
        loss = F.softplus(-(d @ w)).mean() + lam * (w ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            w.clamp_(min=0)
    return [float(x) for x in w.detach()]


def score(features, calib):
    """Workload score for one feature dict under a serialized calibration."""
    z = [(float(features[k]) - m) / s for k, m, s in
         zip(calib["features"], calib["train_mu"], calib["train_sd"])]
    return sum(w * v for w, v in zip(calib["weights"], z))


def _accuracy(pairs, calib):
    ok = sum(1 for p in pairs
             if score(p["hi"]["features"], calib)
             > score(p["lo"]["features"], calib))
    return ok / len(pairs) if pairs else None


def _envelopes(train_records):
    """Raw component p10/p50/p90 per authenticated rank, train only."""
    by_rank = {}
    for rec in train_records:
        for c in rec["charts"]:
            by_rank.setdefault(c["rank"], []).append(c["features"])
    env = {}
    for r, feats in sorted(by_rank.items()):
        env[str(r)] = {"name": RANK_NAMES.get(r, f"rank{r}"), "n": len(feats)}
        for k in FEATURES:
            vals = [f[k] for f in feats]
            env[str(r)][k] = {"p10": nearest_rank(vals, 0.10),
                              "p50": nearest_rank(vals, 0.50),
                              "p90": nearest_rank(vals, 0.90)}
    return env


def fit():
    tr_recs, va_recs = load_descriptors()
    assert tr_recs and va_recs, "run `measure --split train/val` first"
    tr_pairs, va_pairs = build_pairs(tr_recs), build_pairs(va_recs)
    va_adj = [p for p in va_pairs if p["adjacent"]]
    charts = [c for r in tr_recs for c in r["charts"]
              if any(p["hi"] is c or p["lo"] is c for p in tr_pairs)]
    # standardization from train charts that participate in ordering
    mu = [sum(_vec(c)[i] for c in charts) / len(charts)
          for i in range(len(FEATURES))]
    sd = [max((sum((_vec(c)[i] - mu[i]) ** 2 for c in charts)
               / len(charts)) ** 0.5, 1e-9) for i in range(len(FEATURES))]

    def _z(c):
        return [(v - m) / s for v, m, s in zip(_vec(c), mu, sd)]

    deltas = [[a - b for a, b in zip(_z(p["hi"]), _z(p["lo"]))]
              for p in tr_pairs]
    results = []
    for lam in LAMBDAS:
        w = fit_weights(deltas, lam)
        calib = {"features": FEATURES, "train_mu": mu, "train_sd": sd,
                 "weights": w, "lambda": lam}
        results.append({"lambda": lam, "weights": w,
                        "val_adjacent_acc": _accuracy(va_adj, calib),
                        "val_all_acc": _accuracy(va_pairs, calib),
                        "train_all_acc": _accuracy(tr_pairs, calib)})
    best = max(results, key=lambda r: (r["val_adjacent_acc"] or 0,
                                       r["val_all_acc"] or 0, r["lambda"]))
    # Hard (rank 5) support requires >=25 val families with an authenticated
    # adjacent pair touching it; Expert/ExpertPlus are the initial targets
    hard_fams = {p["dir"] for p in va_adj if 5 in (p["hi_rank"], p["lo_rank"])}
    supported = [7, 9] + ([5] if len(hard_fams) >= 25 else [])
    rank_counts = {}
    for split, recs in (("train", tr_recs), ("val", va_recs)):
        for rec in recs:
            for c in rec["charts"]:
                key = f"{split}:{c['rank']}"
                rank_counts[key] = rank_counts.get(key, 0) + 1
    gate = (best["val_adjacent_acc"] or 0) >= 0.90
    out = {"version": 1, "seed": SEED, "kind": "workload_calibration",
           "note": "operational workload calibration, NOT validated "
                   "perceived difficulty; no song/genre input",
           "features": FEATURES, "train_mu": mu, "train_sd": sd,
           "weights": best["weights"], "lambda": best["lambda"],
           "selection": results,
           "n_train_pairs": len(tr_pairs), "n_val_pairs": len(va_pairs),
           "n_val_adjacent_pairs": len(va_adj),
           "val_adjacent_acc": best["val_adjacent_acc"],
           "val_all_acc": best["val_all_acc"],
           "train_all_acc": best["train_all_acc"],
           "gate_90pct_val_adjacent": gate,
           "supported_tiers": supported,
           "hard_adjacent_val_families": len(hard_fams),
           "rank_chart_counts": rank_counts,
           "tier_envelopes": _envelopes(tr_recs)}
    RDIR.mkdir(parents=True, exist_ok=True)
    (RDIR / "workload_calibration.json").write_text(json.dumps(out, indent=1))
    print(f"pairs: train {len(tr_pairs)} / val {len(va_pairs)} "
          f"(adjacent {len(va_adj)})")
    for r in results:
        print(f"  lambda {r['lambda']:g}: val_adj "
              f"{r['val_adjacent_acc']:.4f} val_all {r['val_all_acc']:.4f}")
    print(f"selected lambda {best['lambda']:g} -> val adjacent "
          f"{best['val_adjacent_acc']:.4f} "
          f"{'PASS (>=0.90)' if gate else 'FAIL'}; supported tiers "
          f"{[RANK_NAMES[r] for r in supported]}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--split", required=True, choices=["train", "val"])
    m.add_argument("--limit", type=int, default=10 ** 6)
    sub.add_parser("fit")
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        n = measure_split(a.split, a.limit)
        print(f"measured {n} new dirs ({a.split})")
    else:
        fit()


if __name__ == "__main__":
    main()
