"""E1 Task 2: descriptive expression profiles (spec 2026-09-23 §Measurement).

Expression is a PROFILE, not a maximum: four descriptor families over
emitted heads (followers counted separately), per hand and per
nonoverlapping 8-second window, family-pooled. No `quality` or `fun` field
exists anywhere in the output — shuffled garbage can score high entropy and
the report never labels it better.

Input: notes as (hand, col, layer, dir) tuples + a parallel seconds list
(callers convert decoder step-tuples at the boundary). Distances are grid
units — center-path proxies, never measured arm swing. Arc-like runs are a
center-path turning proxy (4 same-hand heads, 3 nonzero displacement
vectors, same-sign adjacent heading changes each 20–120°, cumulative ≥60°,
gaps >1 s and dots break runs), NOT a mandate to spin.
"""
import math

TOL = 1e-6                    # anchored simultaneity tolerance (seconds µs)
ARC_MIN_DEG, ARC_MAX_DEG, ARC_TOTAL_DEG = 20.0, 120.0, 60.0
WINDOW_S = 8.0


def _heads(notes, times):
    return [(t, h, c, l, d) for (h, c, l, d), t in zip(notes, times)
            if d != 8]


def _spatial(heads):
    per_hand = {0: [], 1: []}
    for t, h, c, l, d in heads:
        per_hand[h].append((t, c, l, d))
    disps = []
    for h, seq in per_hand.items():
        for a, b in zip(seq, seq[1:]):
            disps.append(math.hypot(b[1] - a[1], b[2] - a[2]))
    disps.sort()

    def q(p):
        return disps[min(len(disps) - 1, int(p * len(disps)))] if disps \
            else None
    rows = {0: 0, 1: 0, 2: 0}
    lowvert = 0
    for t, h, c, l, d in heads:
        rows[min(2, max(0, int(round(l))))] += 1
        if l < 1.5 and d in (0, 1):
            lowvert += 1
    n = len(heads)
    cols = [c for _t, _h, c, _l, _d in heads]
    lays = [l for _t, _h, _c, l, _d in heads]
    return {"disp": {"p50": q(0.50), "p90": q(0.90), "n": len(disps)},
            "row_occupancy": {str(r): rows[r] / n for r in rows},
            "low_vertical_share": lowvert / n,
            "bbox": {"width": max(cols) - min(cols),
                     "height": max(lays) - min(lays)}}


def _arcs(heads):
    per_hand = {0: [], 1: []}
    for t, h, c, l, d in heads:
        per_hand[h].append((t, c, l))
    runs, turns_all = 0, []
    for seq in per_hand.values():
        for i in range(len(seq) - 3):
            quad = seq[i:i + 4]
            if any(b[0] - a[0] > 1.0 for a, b in zip(quad, quad[1:])):
                continue                       # gap breaks the run
            vecs = [(b[1] - a[1], b[2] - a[2])
                    for a, b in zip(quad, quad[1:])]
            if any(abs(v[0]) < 1e-9 and abs(v[1]) < 1e-9 for v in vecs):
                continue                       # zero-length rejected
            dts = []
            ok = True
            for (x1, y1), (x2, y2) in zip(vecs, vecs[1:]):
                ang = math.degrees(math.atan2(x1 * y2 - y1 * x2,
                                              x1 * x2 + y1 * y2))
                if not (ARC_MIN_DEG <= abs(ang) <= ARC_MAX_DEG):
                    ok = False
                    break
                dts.append(ang)
            if not ok or len(dts) < 2:
                continue
            if not (all(a > 0 for a in dts) or all(a < 0 for a in dts)):
                continue                       # same-sign heading changes
            if sum(abs(a) for a in dts) < ARC_TOTAL_DEG:
                continue
            runs += 1
            turns_all += [abs(a) for a in dts]
    n = max(1, len(heads))
    return {"runs_per_100_heads": 100.0 * runs / n,
            "mean_abs_turn_deg": (sum(turns_all) / len(turns_all))
            if turns_all else 0.0, "n_runs": runs}


def _doubles(heads):
    by_t = {}
    for t, h, c, l, d in heads:
        by_t.setdefault(round(t, 6), {})[h] = (c, l, d)
    pairs = []
    for t in sorted(by_t):
        hs = by_t[t]
        if 0 in hs and 1 in hs:
            (c0, l0, d0), (c1, l1, d1) = hs[0], hs[1]
            pairs.append((d0, d1, int(round(l0)), int(round(l1)),
                          round(c1 - c0, 3)))
    if not pairs:
        return {"supported": False, "n_pairs": 0, "distinct_tokens": None,
                "concentration": None, "entropy": None, "transitions": None}
    counts = {}
    for p in pairs:
        counts[p] = counts.get(p, 0) + 1
    n = len(pairs)
    ent = -sum((c / n) * math.log(c / n) for c in counts.values())
    trans = {}
    for a, b in zip(pairs, pairs[1:]):
        trans[(a, b)] = trans.get((a, b), 0) + 1
    return {"supported": True, "n_pairs": n,
            "distinct_tokens": len(counts),
            "concentration": max(counts.values()) / n,
            "entropy": ent,
            "transitions": {"distinct": len(trans),
                            "max_share": (max(trans.values())
                                          / max(1, sum(trans.values())))
                            if trans else None}}


def _patterns(heads):
    toks = [(h, d, int(round(c)), int(round(l)))
            for _t, h, c, l, d in heads]
    n4 = len(toks) - 3
    grams = {}
    for i in range(max(0, n4)):
        g = tuple(toks[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    run_len, best_run = 1, 1
    for a, b in zip(toks, toks[1:]):
        run_len = run_len + 1 if a == b else 1
        best_run = max(best_run, run_len)
    # lagged recurrence: share of tokens equal to the token 8 steps back
    lag = 8
    rec = sum(1 for i in range(lag, len(toks)) if toks[i] == toks[i - lag])
    return {"max_4gram_share": (max(grams.values()) / n4) if n4 > 0 else None,
            "distinct_4grams": len(grams) if n4 > 0 else None,
            "longest_identical_run": best_run,
            "lag8_recurrence": rec / max(1, len(toks) - lag)}


def profile(notes, times_seconds, scene_status="supported"):
    """Descriptive profile of one emitted chart (or window)."""
    if scene_status != "supported":
        return {"status": "unknown", "reason": scene_status,
                "features": None, "counts": None}
    heads = _heads(notes, times_seconds)
    followers = len(notes) - len(heads)
    if len(heads) < 4:
        return {"status": "unsupported", "reason": "fewer_than_4_heads",
                "features": None,
                "counts": {"heads": len(heads), "followers": followers}}
    t0 = heads[0][0]
    heads = [(t - t0, h, c, l, d) for t, h, c, l, d in heads]
    windows = {}
    for t, h, c, l, d in heads:
        windows.setdefault(int(t // WINDOW_S), []).append((t, h, c, l, d))
    win_out = {}
    for w, seq in sorted(windows.items()):
        if len(seq) >= 4:
            win_out[str(w)] = {"n_heads": len(seq),
                               "spatial": _spatial(seq),
                               "arcs": _arcs(seq)}
        else:
            win_out[str(w)] = {"n_heads": len(seq), "supported": False}
    return {"status": "ok",
            "counts": {"heads": len(heads), "followers": followers,
                       "windows": len(win_out)},
            "features": {"spatial": _spatial(heads), "arcs": _arcs(heads),
                         "patterns": _patterns(heads)},
            "double_vocabulary": _doubles(heads),
            "windows": win_out}


_REF_FIELDS = (("spatial", "disp", "p50"), ("spatial", "disp", "p90"),
               ("spatial", "low_vertical_share"), ("spatial", "bbox", "width"),
               ("spatial", "bbox", "height"), ("arcs", "runs_per_100_heads"),
               ("arcs", "mean_abs_turn_deg"), ("patterns", "max_4gram_share"),
               ("patterns", "lag8_recurrence"))


def _get(d, path):
    for k in path:
        if d is None:
            return None
        d = d.get(k)
    return d


def reference(profiles_by_family):
    """Family-balanced reference distributions over approved TRAIN charts:
    per-field family means -> quantiles + IQR + support. Explanatory ranges,
    never compulsory choreography; no quality scalar exists."""
    per_field = {}
    dbl_ent = []
    for fam, profs in sorted(profiles_by_family.items()):
        oks = [p for p in profs if p.get("status") == "ok"]
        if not oks:
            continue
        for path in _REF_FIELDS:
            vals = [v for v in (_get(p["features"], path) for p in oks)
                    if v is not None]
            if vals:
                per_field.setdefault("/".join(path), []).append(
                    sum(vals) / len(vals))
        ents = [p["double_vocabulary"]["entropy"] for p in oks
                if p["double_vocabulary"]["supported"]]
        if ents:
            dbl_ent.append(sum(ents) / len(ents))

    def stats(vals):
        v = sorted(vals)

        def q(p):
            return v[min(len(v) - 1, int(p * len(v)))]
        return {"p10": q(0.10), "p25": q(0.25), "p50": q(0.50),
                "p75": q(0.75), "p90": q(0.90),
                "iqr": max(q(0.75) - q(0.25), 1e-9), "n_families": len(v)}
    fields = {k: stats(v) for k, v in per_field.items()}
    if dbl_ent:
        fields["double_vocabulary/entropy"] = stats(dbl_ent)
    return {"n_families": len({f for f, p in profiles_by_family.items()
                               if any(x.get("status") == "ok" for x in p)}),
            "fields": fields,
            "note": "explanatory ranges from approved TRAIN charts; "
                    "not compulsory choreography; no fun scalar"}


def build_train_reference():
    """Fit the reference on APPROVED TRAIN families only (all supported
    charts per family, family-balanced inside reference()). Per-family
    idempotent cache; persists field names/units + source identity."""
    import hashlib
    import json
    from pathlib import Path
    import groom
    from eval import corpus
    from eval.expressive_manifest import family_strata
    ROOT = Path(__file__).resolve().parent.parent
    out_dir = ROOT / "experiments" / "expressive-v1" / "e1"
    cache = out_dir / "ref-cache"
    cache.mkdir(parents=True, exist_ok=True)
    m = corpus._load_validated()
    approved = set(family_strata(m)["approved"])
    fam_dirs = {}
    for r in m["maps"]:
        if r["split"] == "train" and r["family"] in approved \
                and r["eligible"] == "ok":
            fam_dirs.setdefault(r["family"], []).append(r["dir"])
    profs = {}
    for fam, dirs in sorted(fam_dirs.items()):
        cp = cache / (hashlib.sha256(fam.encode()).hexdigest()[:16] + ".json")
        if cp.exists():
            profs[fam] = json.loads(cp.read_text())
            continue
        plist = []
        for dp in sorted(dirs)[:1]:          # one rep dir per family
            samples = groom.load_map_all(Path(dp))
            info_p = next((p for p in Path(dp).iterdir()
                           if p.name.lower() == "info.dat"), None)
            if not samples or info_p is None:
                continue
            bpm = json.loads(info_p.read_text(encoding="utf-8-sig")) \
                .get("_beatsPerMinute")
            if not bpm:
                continue
            step_s = 60.0 / float(bpm) / 4
            for name, s in sorted(samples.items()):
                events = s[2]
                notes = [(e[1], e[3], e[4], e[2]) for e in events]
                times = [e[0] * step_s for e in events]
                plist.append(profile(notes, times))
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps(plist))
        tmp.replace(cp)
        profs[fam] = plist
    ref = reference(profs)
    ref["source"] = {"split": "train", "stratum": "approved",
                     "n_families_input": len(fam_dirs)}
    (out_dir / "train_reference.json").write_text(
        json.dumps(ref, indent=1, sort_keys=True))
    print(f"reference: {ref['n_families']}/{len(fam_dirs)} approved train "
          f"families, {len(ref['fields'])} fields")
    return ref


if __name__ == "__main__":
    build_train_reference()
