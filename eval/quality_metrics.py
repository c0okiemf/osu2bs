"""Emitted-note quality metrics for the next-level-generation roadmap (Q1.1).

Everything here is computed from EMITTED notes in physical milliseconds. The
historical rounded evaluators (motion.flag_stats / motion.lr_doubles) stay the
frozen acceptance metrics; this module ADDS, under new names:

- unrounded transition diagnostics (optimization must not exploit rounding);
- a position-independent opposite-horizontal pair count (changing rows, color
  order or equal-column placement cannot manufacture a pair improvement);
- a 3x3 threshold-neighborhood flag count (a 2.01->1.99 dodge is visible);
- 4-gram geometry-token concentration in 32-beat windows (anti-gaming);
- the Q1 immutable event signature (identities/times/hands/roles/groups/walls
  — everything a geometry repair must NOT change).

Note layouts: raw production tuples are (step, hand, col, layer, dir); motion
consumes (t_ms, hand, col, layer, dir). Convert explicitly via grid.time.
"""
import hashlib
import json
import math

import motion
from parity import DIR_VEC

# frozen 3x3 robustness neighborhood around the flag thresholds (spec Q1)
NEIGHBOR_DISTS = (1.9, 2.0, 2.1)
NEIGHBOR_MSS = (180, 200, 220)


def to_ms(raw, grid):
    """(step, hand, col, layer, dir) -> (t_ms, hand, col, layer, dir)."""
    return [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw]


def to_ms_walls(wall_runs, grid):
    """(s0, len, col) step runs -> {t, dur, col} dicts (convert.check shape)."""
    return [{"t": grid.time(s0), "dur": grid.time(s0 + ln) - grid.time(s0),
             "col": col} for s0, ln, col in wall_runs]


def immutable_signature(raw, wall_runs, grid):
    """Bytes over everything a repair must preserve: ordered note identities,
    absolute hit times, hand, head-vs-dot role counts per (step, hand) group,
    and byte-identical walls. Columns/layers/directions are deliberately
    EXCLUDED — they are what a repair may change."""
    groups = {}
    for s, h, _c, _l, d in raw:
        g = groups.setdefault((int(s), int(h)), [0, 0])
        g[0 if d != 8 else 1] += 1
    events = [[s, repr(grid.time(s)), h, heads, dots]
              for (s, h), (heads, dots) in sorted(groups.items())]
    payload = {"events": events,
               "walls": sorted([int(a), int(b), int(c)]
                               for a, b, c in wall_runs)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).digest()


def _heads_ms(notes_ms):
    return sorted(n for n in notes_ms if n[4] != 8)


def unrounded_transitions(notes_ms, ext=motion.CUT_EXT):
    """motion.transitions math without any rounding: [{t, hand, ms, dist,
    speed}] with full float precision, for optimization and diagnostics."""
    out, last = [], {}
    for t, h, c, l, d in _heads_ms(notes_ms):
        p = last.get(h)
        if p:
            pt, pc, pl, pd = p
            ux, uy = motion._unit(pd)
            vx, vy = motion._unit(d)
            dist = math.hypot((c - ext * vx) - (pc + ext * ux),
                              (l - ext * vy) - (pl + ext * uy))
            out.append({"t": t, "hand": h, "ms": t - pt, "dist": dist,
                        "speed": dist / max((t - pt) / 1000, 1e-9)})
        last[h] = (t, c, l, d)
    return out


def threshold_neighborhood(notes_ms, ext=motion.CUT_EXT):
    """Total flagged-transition count over the frozen 3x3 (dist, ms) threshold
    grid, plus the summed continuous distance and time demand of every
    transition flagged ANYWHERE in the neighborhood. A chart tuned to sit just
    under one cutoff still shows its burden here."""
    tr = unrounded_transitions(notes_ms, ext)
    total = 0
    burden_dist = burden_ms = 0.0
    seen = set()
    for dist_thr in NEIGHBOR_DISTS:
        for ms_thr in NEIGHBOR_MSS:
            for i, x in enumerate(tr):
                if x["ms"] <= ms_thr and x["dist"] >= dist_thr:
                    total += 1
                    if i not in seen:
                        seen.add(i)
                        burden_dist += x["dist"]
                        burden_ms += x["ms"]
    return {"total_flags": total, "distinct_transitions": len(seen),
            "burden_dist": burden_dist, "burden_ms": burden_ms}


def opposite_horizontal(notes_ms):
    """Position-independent count of simultaneous different-hand head pairs
    whose directions are {left, right} = {2, 3}, in ANY row/column/color
    arrangement (unlike motion.lr_doubles, equal columns are NOT skipped)."""
    by_ts = {}
    for t, h, c, l, d in _heads_ms(notes_ms):
        by_ts.setdefault(round(t, 1), []).append((h, d))
    n = 0
    for v in by_ts.values():
        if len(v) == 2 and v[0][0] != v[1][0] \
                and {v[0][1], v[1][1]} == {2, 3}:
            n += 1
    return n


def _dir_family(d):
    if d == 8:
        return "dot"
    vx, vy = DIR_VEC[d]
    if vx == 0:
        return "vert"
    if vy == 0:
        return "horiz"
    return "diag"


def geometry_tokens(notes_ms, beat_ms):
    """Spec anti-gaming tokens: (hand, direction-family, relative-column,
    relative-layer, simultaneous-hand-bit) per head, grouped into 32-beat
    windows. Relative deltas are against the previous same-hand head (clamped
    to +-3/+-2), independent of absolute lane translation and BPM encoding."""
    heads = _heads_ms(notes_ms)
    times = {round(t, 1) for t, *_ in heads}
    counts = {}
    for t, *_ in heads:
        counts[round(t, 1)] = counts.get(round(t, 1), 0) + 1
    toks, last = [], {}
    for t, h, c, l, d in heads:
        p = last.get(h)
        dc = max(-3, min(3, c - p[0])) if p else 0
        dl = max(-2, min(2, l - p[1])) if p else 0
        simul = counts[round(t, 1)] > 1
        toks.append((int(t // (32 * beat_ms)),
                     (h, _dir_family(d), dc, dl, simul)))
        last[h] = (c, l)
    return toks


def four_gram_stats(notes_ms, beat_ms):
    """Per 32-beat window: maximum 4-gram share and effective vocabulary
    (exp of Shannon entropy) over geometry tokens; plus the whole-chart max."""
    toks = geometry_tokens(notes_ms, beat_ms)
    by_win = {}
    for w, tok in toks:
        by_win.setdefault(w, []).append(tok)
    windows = {}
    worst = 0.0
    for w, seq in sorted(by_win.items()):
        grams = {}
        for i in range(len(seq) - 3):
            g = tuple(seq[i:i + 4])
            grams[g] = grams.get(g, 0) + 1
        n = max(1, len(seq) - 3)
        mx = max(grams.values(), default=0) / n
        ent = -sum((c / n) * math.log(c / n) for c in grams.values()) \
            if grams else 0.0
        windows[w] = {"max_4gram_share": mx, "n_grams": n,
                      "effective_vocab": math.exp(ent)}
        worst = max(worst, mx)
    return {"windows": windows, "max_4gram_share": worst}


def longest_token_run(notes_ms, beat_ms):
    """Longest run of one repeated geometry token (repair guard: original+1)."""
    seq = [tok for _w, tok in geometry_tokens(notes_ms, beat_ms)]
    best = cur = 0
    prev = None
    for t in seq:
        cur = cur + 1 if t == prev else 1
        prev = t
        best = max(best, cur)
    return best


def j_score(m):
    """The spec's repair/selection objective over a quality_metrics dict:
    J = mean_e(flags_per_1000[e])/100 + (narrow+converging)/5 + broad/10."""
    flags = m["flags_by_ext"]
    return (sum(flags[str(e)] for e in motion.EXT_VARIANTS) / len(
        motion.EXT_VARIANTS) / 100
        + (m["narrow"] + m["converging"]) / 5 + m["broad"] / 10)


def unknown_dot_burden(notes_ms):
    """Dots in groups motion.dot_stacks cannot classify (no single directional
    head in the group). Unknown/unsupported geometry is a BURDEN, never a
    zero-risk score."""
    groups = {}
    for t, h, _c, _l, d in notes_ms:
        groups.setdefault((round(t, 1), h), []).append(d)
    n = 0
    for g in groups.values():
        dirs = sum(1 for d in g if d != 8)
        dots = sum(1 for d in g if d == 8)
        if dots and dirs != 1:
            n += dots
    return n


def quality_metrics(raw, wall_runs, grid, bpm):
    """The Q1 metric bundle from emitted notes. Frozen historical evaluators
    (rounded) keep their exact values; the additions live under new names."""
    from groom import run_stats
    notes_ms = to_ms(raw, grid)
    beat_ms = 60000.0 / bpm
    rep = motion.report(notes_ms)              # frozen: rounded flags, pairs...
    heads = [n for n in raw if n[4] != 8]
    lat = (sum(1 for *_, d in heads if d in (2, 3)) / max(1, len(heads)))
    vert, longrun = run_stats(raw)
    lr = motion.lr_doubles(notes_ms)
    speeds = sorted(x["speed"] for x in unrounded_transitions(notes_ms))
    return {
        "motion": rep,
        "flags_by_ext": rep["flags_by_ext"],                 # frozen values
        "narrow": len(lr["narrow"]),
        "broad": len(lr["broad"]),
        "converging": len(lr["converging"]),
        "opposite_horizontal": opposite_horizontal(notes_ms),
        "dot_classes": motion.dot_stacks(notes_ms),
        "unknown_dot_burden": unknown_dot_burden(notes_ms),
        "style": {"vert": vert, "longrun": longrun, "lat": lat},
        "neighborhood": {str(e): threshold_neighborhood(notes_ms, e)
                         for e in motion.EXT_VARIANTS},
        "p95_speed_unrounded": speeds[int(0.95 * (len(speeds) - 1))]
                               if speeds else 0.0,
        "four_gram": four_gram_stats(notes_ms, beat_ms),
        "longest_token_run": longest_token_run(notes_ms, beat_ms),
        "signature": immutable_signature(raw, wall_runs, grid).hex(),
        "n_heads": len(heads), "n_notes": len(raw),
    }
