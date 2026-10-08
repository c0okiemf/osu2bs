"""Approximate motion report + decode-time cost helpers (review Packet B).

All timing is physical (milliseconds), all distance in grid spacings — a
half/double-BPM re-encoding of the same absolute motion scores identically
because beats never enter this module. The model is deliberately crude
(catalog §"deliberately approximate physical model"): each cut is a segment
of extent CUT_EXT along the arrow through the note center; a transition
connects the previous cut's exit to the next cut's entry over the actual
time gap. Flags are comparative regression gates, not human limits.

Report families implemented here (rest of the catalog is phase-3 work):
- B02/B04 rapid large repositioning (the review's flag diagnostic, with
  multi-extent sensitivity variants)
- C08/C09 rolling per-hand workload, max-imbalance window, longest
  one-hand run (the 37:0 case)
- C01/C02/F05 outward `<>` (narrow same-row red-left subtype + broad
  family) and converging `><`, with recent/song exposure costs
- A05/B09 dot-stack classification: collinear-forward / collinear-reverse
  (playable v2 stack, flagged not banned) / off-axis (needs path analysis)

Self-check: .venv/bin/python motion.py  (fixtures use the review's real
Still Waiting timestamps)
"""
import math

from parity import DIR_VEC

# knobs (comparative diagnostics; calibrate against paired candidates)
REPO_MS = 200        # flag: transition elapsed <= this ...
REPO_DIST = 2.0      # ... and exit->entry distance >= this (grid spacings)
CUT_EXT = 0.5        # default assumed cut extent along the arrow
EXT_VARIANTS = (0.0, 0.25, 0.5, 0.75)  # sensitivity sweep for reports
WORK_WIN_MS = 8000   # rolling workload window (review's 8s monopoly case)
# decode-time repositioning penalty (per grid step beyond 1, per axis)
REPO_PEN_MS = 250
REPO_PEN = 1.5       # linear per-axis logit cost; refit if the
                     # flag-rate gate stalls
# <> exposure (catalog §4): small always-on base cost, escalating with
# recent (8s/16s) and whole-song exposure past a 3-use preference budget
LR_BASE, LR_RECENT_W, LR_SONG_W, LR_SONG_FREE = 0.8, 0.5, 0.35, 3
LR_IN_SCALE = 0.5    # converging >< gets a milder version (C02)
# candidate-selection soft cost weights (critic logit scale, spread ~1-2)
W_FLAG, W_MONO, W_LR = 0.3, 0.3, 0.3
MONO_OK_RUN = 12     # one-hand runs up to this length are free (C09 allows
                     # bounded one-hand passages)
MONO_DIFF = 8        # decode: rolling 8s per-hand count gap that makes the
                     # scheduler prefer the idle hand for single-hand events
DBL_MAX = 0.6        # C08: rolling share of two-hand events above which a
                     # non-accent double is demoted to the fresher hand


def _unit(d):
    vx, vy = DIR_VEC[d]
    n = math.hypot(vx, vy) or 1.0
    return vx / n, vy / n


def _heads(notes):
    return sorted(n for n in notes if n[4] != 8)


def transitions(notes, ext=CUT_EXT):
    """Per-hand adjacent directional transitions of [(t_ms, hand, col,
    layer, dir)] -> [{t, hand, ms, dist, speed}] (grid spacings, seconds)."""
    out, last = [], {}
    for t, h, c, l, d in _heads(notes):
        p = last.get(h)
        if p:
            pt, pc, pl, pd = p
            ux, uy = _unit(pd)
            vx, vy = _unit(d)
            dist = math.hypot((c - ext * vx) - (pc + ext * ux),
                              (l - ext * vy) - (pl + ext * uy))
            out.append({"t": t, "hand": h, "ms": t - pt,
                        "dist": round(dist, 2),
                        "speed": round(dist / max((t - pt) / 1000, 1e-9), 1)})
        last[h] = (t, c, l, d)
    return out


def flag_stats(notes, ext=CUT_EXT):
    tr = transitions(notes, ext)
    fl = [x for x in tr if x["ms"] <= REPO_MS and x["dist"] >= REPO_DIST]
    speeds = sorted(x["speed"] for x in tr)
    return {"ext": ext, "transitions": len(tr), "flags": len(fl),
            "flags_per_1000": round(1000 * len(fl) / max(1, len(tr)), 1),
            "p95_speed": speeds[int(0.95 * (len(speeds) - 1))] if speeds else 0,
            "worst": sorted(fl, key=lambda x: -x["speed"])[:5]}


def workload(notes, win=WORK_WIN_MS):
    """Sliding-window per-hand counts + longest uninterrupted one-hand run.
    Window origin slides with the events, so shifting origins can't hide
    imbalance (review gate)."""
    hs = _heads(notes)
    best = {"t": 0, "l": 0, "r": 0, "diff": 0}
    j = 0
    for i, (t, *_ ) in enumerate(hs):
        while hs[j][0] < t - win:
            j += 1
        l = sum(1 for x in hs[j:i + 1] if x[1] == 0)
        r = i + 1 - j - l
        if abs(l - r) > best["diff"]:
            best = {"t": hs[j][0], "l": l, "r": r, "diff": abs(l - r)}
    run = {"hand": None, "count": 0, "start": 0, "span_ms": 0}
    cur_h, cur_n, cur_t0, prev_t = None, 0, 0, 0
    for t, h, *_ in hs:
        if h == cur_h:
            cur_n += 1
        else:
            cur_h, cur_n, cur_t0 = h, 1, t
        if cur_n > run["count"]:
            run = {"hand": cur_h, "count": cur_n, "start": cur_t0,
                   "span_ms": t - cur_t0}
        prev_t = t
    best.pop("diff")
    return {"max_window": best, "longest_run": run}


def lr_doubles(notes):
    """Outward `<>` occurrences: narrow = same layer, red(0) left of
    blue(1); broad = any two-head outward-horizontal pair. Plus `><`."""
    by_ts = {}
    for t, h, c, l, d in _heads(notes):
        by_ts.setdefault(round(t, 1), []).append((h, c, l, d))
    narrow, broad, conv = [], [], []
    for ts, v in sorted(by_ts.items()):
        if len(v) != 2:
            continue
        a, b = sorted(v, key=lambda n: n[1])
        if a[1] == b[1]:
            continue
        if a[3] == 2 and b[3] == 3:
            broad.append(ts)
            if a[2] == b[2] and a[0] == 0 and b[0] == 1:
                narrow.append(ts)
        elif a[3] == 3 and b[3] == 2:
            conv.append(ts)
    return {"narrow": narrow, "broad": broad, "converging": conv}


def dot_stacks(notes):
    """Classify dots against their group's directional note: collinear
    forward / collinear reverse (may be a playable dot-first v2 stack —
    flagged, not banned) / off-axis (needs feasible-path analysis)."""
    groups = {}
    for t, h, c, l, d in notes:
        groups.setdefault((round(t, 1), h), []).append((c, l, d))
    out = {"forward": 0, "reverse": 0, "offaxis": 0, "events": []}
    for (ts, h), g in sorted(groups.items()):
        dirs = [n for n in g if n[2] != 8]
        dots = [n for n in g if n[2] == 8]
        if len(dirs) != 1 or not dots:
            continue
        hc, hl, hd = dirs[0]
        ux, uy = DIR_VEC[hd]
        for dc, dl, _ in dots:
            rx, ry = dc - hc, dl - hl
            if rx * uy - ry * ux != 0:
                kind = "offaxis"
            elif rx * ux + ry * uy > 0:
                kind = "forward"
            else:
                kind = "reverse"
            out[kind] += 1
            if kind != "forward":
                out["events"].append({"t": ts, "hand": h, "kind": kind})
    return out


def doubles_share(notes):
    ts_hands = {}
    for t, h, *_ in _heads(notes):
        ts_hands.setdefault(round(t, 1), set()).add(h)
    return round(100 * sum(1 for v in ts_hands.values() if len(v) == 2)
                 / max(1, len(ts_hands)), 1)


def report(notes):
    """Full motion report for [(t_ms, hand, col, layer, dir)] notes,
    computed AFTER all repairs (catalog §5.6)."""
    return {"flags": flag_stats(notes),
            "flags_by_ext": {str(e): flag_stats(notes, e)["flags_per_1000"]
                             for e in EXT_VARIANTS},
            "workload": workload(notes),
            "lr_doubles": {k: (v if k == "narrow" else len(v))
                           for k, v in lr_doubles(notes).items()},
            "dot_stacks": dot_stacks(notes),
            "doubles_share_pct": doubles_share(notes)}


def select_cost(rep):
    """Soft candidate-ranking cost on the critic-logit scale. Gates stay
    hard elsewhere; this only orders otherwise-acceptable candidates."""
    return (W_FLAG * rep["flags"]["flags_per_1000"] / 100
            + W_MONO * max(0, rep["workload"]["longest_run"]["count"]
                           - MONO_OK_RUN) / 10
            + W_LR * len(rep["lr_doubles"]["narrow"]) / 5)


class LrExposure:
    """Decode-time `<>` exposure state (catalog §4): base cost always,
    escalation with recent and whole-song use. Reuse/copying cannot reset
    it — the counter lives on the destination timeline."""

    def __init__(self):
        self.times = []

    def add(self, t_ms):
        self.times.append(t_ms)

    def cost(self, t_ms):
        r8 = sum(1 for x in self.times if t_ms - x <= 8000)
        r16 = sum(1 for x in self.times if t_ms - x <= 16000)
        return (LR_BASE + LR_RECENT_W * (r8 + 0.5 * r16)
                + LR_SONG_W * max(0, len(self.times) - LR_SONG_FREE))


def repo_penalty(gap_ms, delta):
    """Per-axis decode-time logit penalty for repositioning `delta` grid
    steps with `gap_ms` available. Zero for slow or short moves."""
    if gap_ms > REPO_PEN_MS:
        return 0.0
    return REPO_PEN * max(0, delta - 1)


def _selfcheck():
    # review's real Still Waiting off-candidate passage: col0/top upLeft at
    # 65.285s -> col2/bottom down at 65.441 -> col0/top up at 65.598;
    # 156ms each, 2.83 grid centers — both transitions must flag at every
    # extent variant (the review verified direction-stability over exts)
    sw = [(65285, 0, 0, 2, 4), (65441, 0, 2, 0, 1), (65598, 0, 0, 2, 0)]
    for e in EXT_VARIANTS:
        st = flag_stats(sw, e)
        assert st["flags"] == 2, (e, st)
    # counterexample: the same cells as a slow wide sweep are fine
    slow = [(0, 0, 0, 2, 4), (800, 0, 2, 0, 1), (1600, 0, 0, 2, 0)]
    assert flag_stats(slow)["flags"] == 0
    # 37:0 monopoly (review case): longest run and window imbalance
    mono = [(114285 + i * 216, 0, 1, 0, (0, 1)[i % 2]) for i in range(37)]
    w = workload(mono)
    assert w["longest_run"]["count"] == 37 and w["max_window"]["r"] == 0, w
    both = sorted(mono + [(114285 + 18 * 216 + 50, 1, 2, 0, 1)])
    assert workload(both)["longest_run"]["count"] < 37
    # <> narrow/broad/converging classification
    lr = lr_doubles([(0, 0, 1, 0, 2), (0, 1, 2, 0, 3),
                     (1000, 0, 1, 0, 3), (1000, 1, 2, 0, 2),
                     (2000, 0, 0, 2, 2), (2000, 1, 3, 0, 3)])
    assert lr["narrow"] == [0.0] and len(lr["broad"]) == 2 \
        and lr["converging"] == [1000.0], lr
    # exposure: first use cheap, tenth much more expensive; spread uses
    # cheaper than clustered (catalog regression suite)
    x = LrExposure()
    first = x.cost(0)
    x.add(0)
    clustered = x.cost(1000)
    for i in range(9):
        x.add(2000 + i * 500)
    tenth = x.cost(7000)
    y = LrExposure()
    for i in range(3):
        y.add(i * 60000)
    spread = y.cost(200000)
    assert first < clustered < tenth and spread < tenth, \
        (first, clustered, spread, tenth)
    # dot stacks: forward, reverse (playable stack — flagged only), off-axis
    ds = dot_stacks([(0, 0, 1, 0, 0), (0, 0, 1, 1, 8),
                     (1000, 0, 1, 1, 0), (1000, 0, 1, 0, 8),
                     (2000, 0, 1, 0, 3), (2000, 0, 1, 1, 8)])
    assert (ds["forward"], ds["reverse"], ds["offaxis"]) == (1, 1, 1), ds
    # decode penalty: slow moves free, big fast jumps charged per step
    assert repo_penalty(400, 3) == 0 and repo_penalty(156, 1) == 0
    assert repo_penalty(156, 3) == 2 * REPO_PEN
    print("motion selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
