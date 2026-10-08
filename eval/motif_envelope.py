"""Human train envelope of per-window motif concentration — v2 MATCHED
(review ): per-window max 4-gram share quantiles
CONDITIONED on window workload — event-count (n_grams), double share and
cadence bins — because sparse windows have coarser shares and a global p90
conflates workload with repetitiveness.

Bins: n_grams {<12,<24,<48,>=48} x double_share {<.05,<.2,>=.2} x cadence
{<2,<3.5,>=3.5}/s. p90 per bin with the frozen fallback chain: full bin ->
drop cadence -> drop double -> global. Same computation applies to generated
charts (eval side) via `window_features` + `matched_p90`.

Per-dir idempotent cache (v2 keys); no generation decodes.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "quality-v1" / "q4-style"

NG_EDGES = (12, 24, 48)
DBL_EDGES = (0.05, 0.2)
CAD_EDGES = (2.0, 3.5)


def _bucket(v, edges):
    for i, e in enumerate(edges):
        if v < e:
            return i
    return len(edges)


def bin_key(n_grams, dbl, cad):
    return f"{_bucket(n_grams, NG_EDGES)}:{_bucket(dbl, DBL_EDGES)}:" \
           f"{_bucket(cad, CAD_EDGES)}"


def window_features(notes_ms, beat_ms):
    """Per-32-beat-window workload features from ms notes: distinct-instant
    double share and cadence (instants/s). Window ids match
    quality_metrics.four_gram_stats' windows."""
    span_ms = 32 * beat_ms
    by_w = {}
    for t, h, *_ in notes_ms:
        w = int(t // span_ms)
        by_w.setdefault(w, {}).setdefault(round(t, 6), set()).add(h)
    out = {}
    for w, inst in by_w.items():
        n = len(inst)
        dbl = sum(1 for hs in inst.values() if len(hs) == 2) / n if n else 0.0
        out[w] = {"dbl": dbl, "cad": n / (span_ms / 1000.0)}
    return out


def matched_p90(env, n_grams, dbl, cad):
    """The matched bin's p90 with the frozen fallback chain."""
    bins = env["bins"]
    ng, db, cd = (_bucket(n_grams, NG_EDGES), _bucket(dbl, DBL_EDGES),
                  _bucket(cad, CAD_EDGES))
    for key in (f"{ng}:{db}:{cd}", f"{ng}:{db}:*", f"{ng}:*:*"):
        b = bins.get(key)
        if b is not None:
            return b["p90"]
    return env["global"]["p90"]


def excess_of(motif_windows, env):
    """Matched-envelope excess: mean over supported windows of
    max(0, share - matched p90). Windows must carry share/n_grams/dbl/cad."""
    if not motif_windows:
        return None
    tot = 0.0
    for w in motif_windows:
        tot += max(0.0, w["share"]
                   - matched_p90(env, w["n_grams"], w["dbl"], w["cad"]))
    return tot / len(motif_windows)


def _q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def build(limit=10 ** 6):
    import groom
    from eval import corpus
    from eval.quality_metrics import four_gram_stats
    from pref import UniformGrid
    cache = OUT / "motif-desc-v2"
    cache.mkdir(parents=True, exist_ok=True)
    wins, n_charts = [], 0
    for dp in sorted(corpus.train_families_rep("train")):
        cp = cache / (hashlib.sha256(dp.encode()).hexdigest()[:16] + ".json")
        if cp.exists():
            rec = json.loads(cp.read_text())
        else:
            if n_charts >= limit:
                continue
            rec = _one(dp, groom, four_gram_stats, UniformGrid)
            tmp = cp.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(cp)
        if rec["status"] != "ok":
            continue
        wins += rec["windows"]
        n_charts += 1
    bins = {}
    for w in wins:
        for key in (bin_key(w["n_grams"], w["dbl"], w["cad"]),
                    f"{_bucket(w['n_grams'], NG_EDGES)}:"
                    f"{_bucket(w['dbl'], DBL_EDGES)}:*",
                    f"{_bucket(w['n_grams'], NG_EDGES)}:*:*"):
            bins.setdefault(key, []).append(w["share"])
    env_bins = {k: {"n": len(v), "p50": _q(v, 0.50), "p90": _q(v, 0.90)}
                for k, v in bins.items() if len(v) >= 50}
    env = {"version": 2, "n_charts": n_charts, "n_windows": len(wins),
           "edges": {"n_grams": NG_EDGES, "dbl": DBL_EDGES,
                     "cad": CAD_EDGES},
           "bins": env_bins,
           "global": {"p50": _q([w["share"] for w in wins], 0.50),
                      "p90": _q([w["share"] for w in wins], 0.90),
                      "p95": _q([w["share"] for w in wins], 0.95)},
           "vocab_p50": _q([w["vocab"] for w in wins], 0.50),
           "note": "matched per-window max 4-gram share envelope, train "
                   "human charts; fallback chain full->no-cad->no-dbl->global"}
    (OUT / "human_motif_envelope.json").write_text(json.dumps(env, indent=1))
    print(f"envelope v2: {n_charts} charts {len(wins)} windows "
          f"{len(env_bins)} bins; global p90 {env['global']['p90']:.4f}")
    return env


def _one(dp, groom, four_gram_stats, UniformGrid):
    d = Path(dp)
    samples = groom.load_map_all(d)
    if not samples:
        return {"status": "no_human_chart"}
    name = sorted(samples, key=lambda n: len(samples[n][2]))[-1]
    _inp, _pres, events, _wl, _wr = samples[name][:5]
    info_p = next((p for p in d.iterdir() if p.name.lower() == "info.dat"),
                  None)
    if info_p is None:
        return {"status": "no_info"}
    bpm = json.loads(info_p.read_text(encoding="utf-8-sig")) \
        .get("_beatsPerMinute")
    if not bpm:
        return {"status": "no_bpm"}
    grid = UniformGrid(60000.0 / float(bpm) / 4)
    beat_ms = 60000.0 / float(bpm)
    notes_ms = [(grid.time(e[0]), e[1], e[3], e[4], e[2]) for e in events]
    fg = four_gram_stats(notes_ms, beat_ms)
    wf = window_features(notes_ms, beat_ms)
    out = []
    for w, rec in fg["windows"].items():
        if rec["n_grams"] < 4 or w not in wf:
            continue
        out.append({"share": rec["max_4gram_share"],
                    "vocab": rec["effective_vocab"],
                    "n_grams": rec["n_grams"],
                    "dbl": wf[w]["dbl"], "cad": wf[w]["cad"]})
    if not out:
        return {"status": "too_short"}
    return {"status": "ok", "windows": out}


if __name__ == "__main__":
    build()
