"""Q1: deterministic, schedule-preserving repair of the emitted winner.

Runs AFTER production selection and rate retries, on the fully emitted winner.
The immutable signature (note identities, absolute times, hands, head/dot
roles, per-(step,hand) grouping, walls) is preserved exactly: V1 moves only
positions (single unambiguous heads to any legal cell; head+dot groups by a
rigid one-cell translation), never deletes, adds, re-times, re-hands or
changes roles. Directions are untouched in V1, so the parity stream cannot
change; `convert.check` and `parity.violations` remain the authoritative
guards on every accepted edit.

Search: severity-ordered sites (endpoints of <=200 ms flagged transitions at
any extent; narrow/converging pairs beyond the free-use budget plus their
immediate same-hand neighbors), at most 2 sweeps x 64 sites x 128 trials per
site, edits to <=10% of directional heads. No RNG; deterministic tuple order;
inputs never mutated. Budget exhaustion returns the best accepted chart.

Acceptance per edit (all against the CURRENT chart, guards against the
ORIGINAL): spec J strictly improves; every extent's flag count, every pair
count (narrow/broad/converging/opposite-horizontal) and off-axis/reverse-dot
counts non-increasing; p95 unrounded speed <=1.05x original; existing style
share/run gates; longest geometry-token run <= original+1; 4-gram
concentration <= original + 0.02; and, when a flag reduction drives the
acceptance, the 3x3 threshold-neighborhood total must drop with >=5% less
continuous demand on the affected transitions (a bare threshold crossing is
rejected). Ties prefer fewer changed notes, then shorter displacement.
"""
import math
import time
from dataclasses import dataclass, field

import motion
import parity
from eval.quality_metrics import (immutable_signature, to_ms, to_ms_walls,
                                  four_gram_stats, longest_token_run)

BANNED_CELLS = {(1, 1), (2, 1)}          # centre ban (row 1, cols 1-2)
FREE_PAIRS = motion.LR_SONG_FREE         # narrow/converging free-use budget
EPS = 1e-9


@dataclass
class RepairConfig:
    variant: str = "position"
    sweeps: int = 2
    max_sites: int = 64
    max_trials_per_site: int = 128
    max_edit_frac: float = 0.10


@dataclass
class RepairResult:
    raw: list
    status: str
    changed_ids: list
    before: dict
    after: dict
    trials: int
    elapsed_s: float
    sweeps_run: int
    witnesses: list = field(default_factory=list)


def _heads_with_ids(raw):
    """Sorted directional heads as (t-order key) with their raw indices."""
    return sorted((s, h, c, l, d, i) for i, (s, h, c, l, d) in enumerate(raw)
                  if d != 8)


def _groups(raw):
    g = {}
    for i, (s, h, _c, _l, d) in enumerate(raw):
        e = g.setdefault((s, h), {"heads": [], "dots": []})
        e["heads" if d != 8 else "dots"].append(i)
    return g


def _wall_steps(wall_runs):
    occ = {0: set(), 3: set()}
    for s0, ln, col in wall_runs:
        occ[col].update(range(s0, s0 + ln))
    return occ


def _id_transitions(raw, grid, ext):
    """Unrounded per-hand transitions carrying the raw ids of both endpoints."""
    out, last = [], {}
    for s, h, c, l, d, i in _heads_with_ids(raw):
        t = grid.time(s)
        p = last.get(h)
        if p:
            pt, pc, pl, pd, pi = p
            ux, uy = motion._unit(pd)
            vx, vy = motion._unit(d)
            dist = math.hypot((c - ext * vx) - (pc + ext * ux),
                              (l - ext * vy) - (pl + ext * uy))
            out.append({"ids": (pi, i), "ms": t - pt, "dist": dist,
                        "speed": dist / max((t - pt) / 1000, 1e-9),
                        "t": t, "hand": h})
        last[h] = (t, c, l, d, i)
    return out


class _TransCache:
    """Per-chart lazy cache of unrounded id-carrying transitions per extent.
    Everything downstream (rounded flag counts, neighborhood, p95, demand)
    derives from ONE build per extent, preserving the frozen evaluators'
    exact values (flags use round(dist, 2) like motion.transitions)."""

    def __init__(self, raw, grid):
        self.raw, self.grid, self._by_ext = raw, grid, {}

    def trs(self, ext):
        if ext not in self._by_ext:
            self._by_ext[ext] = _id_transitions(self.raw, self.grid, ext)
        return self._by_ext[ext]

    def flags(self, ext):
        return sum(1 for x in self.trs(ext)
                   if x["ms"] <= motion.REPO_MS
                   and round(x["dist"], 2) >= motion.REPO_DIST)

    def flags_p1000(self, ext):
        tr = self.trs(ext)
        return round(1000 * self.flags(ext) / max(1, len(tr)), 1)

    def neighborhood_total(self):
        tot = 0
        for e in motion.EXT_VARIANTS:
            tr = self.trs(e)
            for dt in (1.9, 2.0, 2.1):
                for mt in (180, 200, 220):
                    tot += sum(1 for x in tr
                               if x["ms"] <= mt and x["dist"] >= dt)
        return tot

    def p95(self):
        speeds = sorted(x["speed"] for x in self.trs(motion.CUT_EXT))
        return speeds[int(0.95 * (len(speeds) - 1))] if speeds else 0.0

    def demand(self, ids):
        idset = set(ids)
        return sum(x["speed"] for e in motion.EXT_VARIANTS
                   for x in self.trs(e) if idset & set(x["ids"]))


def risk_vector(raw, wall_runs, grid, cache=None):
    """Compact componentwise risk counts for edit acceptance."""
    tc = cache or _TransCache(raw, grid)
    notes_ms = to_ms(raw, grid)
    lr = motion.lr_doubles(notes_ms)
    ds = motion.dot_stacks(notes_ms)
    from eval.quality_metrics import opposite_horizontal, unknown_dot_burden
    from groom import run_stats
    vert, longrun = run_stats(raw)
    heads = [n for n in raw if n[4] != 8]
    return {
        "flags": {e: tc.flags(e) for e in motion.EXT_VARIANTS},
        "flags_p1000": {str(e): tc.flags_p1000(e)
                        for e in motion.EXT_VARIANTS},
        "narrow": len(lr["narrow"]), "broad": len(lr["broad"]),
        "converging": len(lr["converging"]),
        "opposite": opposite_horizontal(notes_ms),
        "offaxis": ds["offaxis"], "reverse": ds["reverse"],
        "unknown_dots": unknown_dot_burden(notes_ms),
        "p95": tc.p95(),
        "style": {"vert": vert, "longrun": longrun,
                  "lat": sum(1 for *_, d in heads if d in (2, 3))
                         / max(1, len(heads))},
        "neighborhood_total": tc.neighborhood_total(),
    }


def _j(rv):
    return (sum(rv["flags_p1000"][str(e)] for e in motion.EXT_VARIANTS)
            / len(motion.EXT_VARIANTS) / 100
            + (rv["narrow"] + rv["converging"]) / 5 + rv["broad"] / 10)


def _sites(raw, grid, cfg):
    """Severity-ordered repair sites; deterministic."""
    flagged = {}
    for e in motion.EXT_VARIANTS:
        for tr in _id_transitions(raw, grid, e):
            if tr["ms"] <= motion.REPO_MS and tr["dist"] >= motion.REPO_DIST:
                k = tr["ids"]
                if k not in flagged or tr["speed"] > flagged[k]["speed"]:
                    flagged[k] = tr
    sites = [{"kind": "flag", "ids": list(k), "severity": v["speed"],
              "t": v["t"], "hand": v["hand"]} for k, v in flagged.items()]
    # narrow/converging pairs beyond the free-use budget (+ same-hand neighbors)
    notes_ms = to_ms(raw, grid)
    lr = motion.lr_doubles(notes_ms)
    pair_times = sorted(set(lr["narrow"]) | set(lr["converging"]))
    heads = _heads_with_ids(raw)
    by_hand = {}
    for s, h, c, l, d, i in heads:
        by_hand.setdefault(h, []).append((grid.time(s), i))
    for n_used, pt in enumerate(pair_times):
        if n_used < FREE_PAIRS:
            continue
        core, nbrs = set(), set()
        for s, h, c, l, d, i in heads:
            if round(grid.time(s), 1) == pt:
                core.add(i)
                seq = by_hand[h]
                j = next(k for k, (_t, ii) in enumerate(seq) if ii == i)
                for jj in (j - 1, j + 1):
                    if 0 <= jj < len(seq):
                        nbrs.add(seq[jj][1])
        # the pair heads FIRST: the per-site trial budget must reach them
        sites.append({"kind": "pair",
                      "ids": sorted(core) + sorted(nbrs - core),
                      "severity": 0.0, "t": pt, "hand": -1})
    sites.sort(key=lambda x: (-x["severity"], x["t"], x["hand"],
                              tuple(x["ids"])))
    return sites[:cfg.max_sites]


def _legal_cells(step, exclude_id, raw, wall_occ):
    """Cells legal at `step` given other notes' occupancy and walls."""
    used = {(c, l) for i, (s, _h, c, l, _d) in enumerate(raw)
            if s == step and i != exclude_id}
    out = []
    for c in range(4):
        for l in range(3):
            if (c, l) in BANNED_CELLS or (c, l) in used:
                continue
            if (c == 0 and step in wall_occ[0]) or (c == 3 and step in wall_occ[3]):
                continue
            out.append((c, l))
    return out


def _same_family(d):
    """Directions sharing d's parity family (safe, provably parity-preserving:
    the fore/back/lateral sequence is untouched)."""
    for fam in (parity.FOREHAND, parity.BACKHAND, parity.LATERAL):
        if d in fam:
            return sorted(fam)
    return [d]


def _trials_for(nid, raw, groups, wall_occ, variant="position", hand_seq=None):
    """Deterministic candidate edits for one note id. Moves are full-note
    replacements [(id, c, l, d)]. V1: positions only (single unambiguous heads
    to any legal cell; head+dot groups translate rigidly). paired-direction
    adds family-preserving direction changes on a dot-free same-hand two-head
    window (the note alone, or the note plus its adjacent same-hand head)."""
    s, h, c, l, d = raw[nid]
    if d == 8:
        return []                                  # dots move with their head
    g = groups[(s, h)]
    if len(g["heads"]) != 1:
        return []                                  # ambiguous group: pinned
    out = []
    if not g["dots"]:
        if variant == "paired-direction":
            # the spec's full single-head move set: legal positions x cuts 0-7
            # (parity preserved through the suffix — guarded per trial)
            cells = [(c, l)] + _legal_cells(s, nid, raw, wall_occ)
            out += [([nid], [(nid, nc, nl, nd)])
                    for nc, nl in cells for nd in range(8)
                    if (nc, nl, nd) != (c, l, d)]
        else:
            out += [([nid], [(nid, nc, nl, d)])
                    for nc, nl in _legal_cells(s, nid, raw, wall_occ)
                    if (nc, nl) != (c, l)]
    else:
        # head + dots: rigid one-cell translation of the whole group
        members = [nid] + g["dots"]
        for dc in (-1, 0, 1):
            for dl in (-1, 0, 1):
                if (dc, dl) == (0, 0):
                    continue
                moves, ok = [], True
                for m in members:
                    mc, ml = raw[m][2] + dc, raw[m][3] + dl
                    if not (0 <= mc <= 3 and 0 <= ml <= 2) \
                            or (mc, ml) in BANNED_CELLS:
                        ok = False
                        break
                    if (mc == 0 and s in wall_occ[0]) \
                            or (mc == 3 and s in wall_occ[3]):
                        ok = False
                        break
                    moves.append((m, mc, ml, raw[m][4]))
                if not ok:
                    continue
                tgt = {(mc, ml) for _m, mc, ml, _d in moves}
                others = {(cc, ll) for i, (ss, _hh, cc, ll, _dd)
                          in enumerate(raw) if ss == s and i not in members}
                if tgt & others:
                    continue
                out.append((list(members), moves))
    if variant == "paired-direction" and not g["dots"]:
        for nd in _same_family(d):
            if nd != d:
                out.append(([nid], [(nid, c, l, nd)]))
        seq = (hand_seq or {}).get(h, [])
        try:
            j = seq.index(nid)
        except ValueError:
            j = -1
        for jj in ((j - 1, j + 1) if j >= 0 else ()):
            if not (0 <= jj < len(seq)):
                continue
            mid = seq[jj]
            ms_, mh_, mc_, ml_, md_ = raw[mid]
            mg = groups[(ms_, mh_)]
            if len(mg["heads"]) != 1 or mg["dots"]:
                continue                           # window must be dot-free
            for nd in _same_family(d):
                for nmd in _same_family(md_):
                    if nd == d and nmd == md_:
                        continue
                    out.append(([nid, mid],
                                [(nid, c, l, nd), (mid, mc_, ml_, nmd)]))
    return out


def _cheap_gain(cur_tc, cand_tc, grid, members):
    """Cheap local prune before the full risk vector: the continuous demand on
    the transitions touching the edited notes must strictly decrease, OR the
    local pair burden (narrow+broad+converging+opposite at the edited
    timestamps) must drop (direction edits fix pairs without moving demand).
    Under V1 a layer shuffle that merely reclassifies a pair changes neither,
    so it is pruned here — and the dodge guard rejects it regardless."""
    if cand_tc.demand(members) < cur_tc.demand(members) - EPS:
        return True
    steps = {cur_tc.raw[m][0] for m in members}

    def _pair_burden(raw):
        notes = [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw
                 if s in steps]
        from eval.quality_metrics import opposite_horizontal
        lr = motion.lr_doubles(notes)
        return (len(lr["narrow"]) + len(lr["converging"]) + len(lr["broad"])
                + opposite_horizontal(notes))
    return _pair_burden(cand_tc.raw) < _pair_burden(cur_tc.raw)


def _hammer_count(raw, steps):
    """Same-step equal-direction in-line head pairs ('>>') at the given steps —
    a V2 direction change must never mint one."""
    by_step = {}
    for s, h, c, l, d in raw:
        if d != 8 and s in steps:
            by_step.setdefault(s, []).append((c, l, d))
    n = 0
    for v in by_step.values():
        for i in range(len(v)):
            for k in range(i + 1, len(v)):
                a, b = v[i], v[k]
                if a[2] == b[2] and parity.is_hammer(a[0], a[1], b[0], b[1],
                                                     a[2]):
                    n += 1
    return n




def repair_winner(raw, wall_runs, grid, bpm, cap, config=None):
    """Deterministic V1 repair. Never mutates inputs; returns RepairResult."""
    from convert import check
    cfg = config or RepairConfig()
    t0 = time.perf_counter()
    orig = [tuple(n) for n in raw]
    walls = [tuple(w) for w in wall_runs]
    sig0 = immutable_signature(orig, walls, grid)
    beat_ms = 60000.0 / bpm
    notes_ms0 = to_ms(orig, grid)
    orig_rv = risk_vector(orig, walls, grid)
    orig_run = longest_token_run(notes_ms0, beat_ms)
    orig_4g = four_gram_stats(notes_ms0, beat_ms)["max_4gram_share"]
    orig_par = parity.violations([(t, h, d) for t, h, _c, _l, d in notes_ms0])
    orig_probs = set(check(
        [{"t": t, "hand": h, "col": c, "layer": l, "dir": d}
         for t, h, c, l, d in notes_ms0], bpm, to_ms_walls(walls, grid), cap))
    wall_occ = _wall_steps(walls)
    groups = _groups(orig)
    heads_total = sum(1 for n in orig if n[4] != 8)
    edit_cap = int(cfg.max_edit_frac * heads_total)

    cur = list(orig)
    cur_rv = orig_rv
    cur_tc = _TransCache(cur, grid)
    changed, witnesses, trials = set(), [], 0
    sweeps_run = 0
    for _sweep in range(cfg.sweeps):
        sweeps_run += 1
        edited_this_sweep = False
        hand_seq = {}
        for s_, h_, _c2, _l2, _d2, i_ in _heads_with_ids(cur):
            hand_seq.setdefault(h_, []).append(i_)
        for site in _sites(cur, grid, cfg):
            if len(changed) >= edit_cap:
                break
            best = None
            site_trials = 0
            for nid in site["ids"]:
                for members, moves in _trials_for(nid, cur, groups, wall_occ,
                                                  cfg.variant, hand_seq):
                    if site_trials >= cfg.max_trials_per_site:
                        break
                    site_trials += 1
                    trials += 1
                    cand = list(cur)
                    for m, nc, nl, nd in moves:
                        s_, h_, _c, _l, _d = cand[m]
                        cand[m] = (s_, h_, nc, nl, nd)
                    if immutable_signature(cand, walls, grid) != sig0:
                        continue                    # never: guarded anyway
                    cand_tc = _TransCache(cand, grid)
                    if not _cheap_gain(cur_tc, cand_tc, grid, members):
                        continue
                    esteps = {cur[m][0] for m in members}
                    if _hammer_count(cand, esteps) > _hammer_count(cur, esteps):
                        continue                    # never mint a '>>' hammer
                    if any(mv[3] != cur[mv[0]][4] for mv in moves):
                        # direction edit: the full parity suffix stays legal
                        if parity.violations(
                                [(grid.time(s2), h2, d2)
                                 for s2, h2, _c3, _l3, d2 in cand]) > orig_par:
                            continue
                    rv = risk_vector(cand, walls, grid, cache=cand_tc)
                    # position-only dodge (spec): a pair reduction may drive
                    # acceptance only if the position-independent opposite-
                    # horizontal count ALSO drops — V1 cannot, so pair
                    # reclassification earns no credit
                    if (rv["narrow"] + rv["converging"]
                            < cur_rv["narrow"] + cur_rv["converging"]) \
                            and rv["opposite"] >= cur_rv["opposite"]:
                        continue
                    # componentwise non-increase + strict J improvement
                    if any(rv["flags"][e] > cur_rv["flags"][e]
                           for e in motion.EXT_VARIANTS):
                        continue
                    if any(rv[k] > cur_rv[k] for k in
                           ("narrow", "broad", "converging", "opposite",
                            "offaxis", "reverse", "unknown_dots")):
                        continue
                    if _j(rv) >= _j(cur_rv) - EPS:
                        continue
                    if rv["p95"] > 1.05 * orig_rv["p95"] + EPS:
                        continue
                    # style share/run gates: keep passing, or at least never
                    # worsen a component the original already exceeded
                    st, so = rv["style"], orig_rv["style"]
                    if st["vert"] > max(0.80, so["vert"] + EPS) \
                            or st["longrun"] > max(0.55, so["longrun"] + EPS) \
                            or st["lat"] > max(0.30, so["lat"] + EPS):
                        continue
                    nm = to_ms(cand, grid)
                    if longest_token_run(nm, beat_ms) > orig_run + 1:
                        continue
                    # "excessive concentration" = +0.02 over original
                    if four_gram_stats(nm, beat_ms)["max_4gram_share"] \
                            > orig_4g + 0.02:
                        continue
                    if any(rv["flags"][e] < cur_rv["flags"][e]
                           for e in motion.EXT_VARIANTS):
                        # motion claim: robust in the 3x3 neighborhood, with
                        # >=5% less continuous demand on affected transitions
                        if rv["neighborhood_total"] \
                                >= cur_rv["neighborhood_total"]:
                            continue
                        if cand_tc.demand(members) \
                                > 0.95 * cur_tc.demand(members):
                            continue
                    disp = sum(abs(cand[m][2] - cur[m][2])
                               + abs(cand[m][3] - cur[m][3])
                               + (cand[m][4] != cur[m][4])
                               for m, _nc, _nl, _nd in moves)
                    key = (_j(rv), len(members), disp,
                           tuple(moves))
                    if best is None or key < best[0]:
                        best = (key, cand, rv, members, moves)
            if best is None:
                continue
            _key, cand, rv, members, moves = best
            nm = to_ms(cand, grid)
            probs = check([{"t": t, "hand": h, "col": c, "layer": l, "dir": d}
                           for t, h, c, l, d in nm], bpm,
                          to_ms_walls(walls, grid), cap)
            if set(probs) - orig_probs:
                continue          # authoritative veto: NEW hard failure
            if parity.violations([(t, h, d) for t, h, _c, _l, d in nm]) \
                    > orig_par:
                continue
            witnesses.append({
                "site": {k: site[k] for k in ("kind", "t", "hand")},
                "moves": [{"id": m, "from": list(cur[m][2:5]),
                           "to": [nc, nl, nd]} for m, nc, nl, nd in moves],
                "context": {"before": [cur[i] for i in
                                       sorted(set(site["ids"]) | set(members))],
                            "after": [cand[i] for i in
                                      sorted(set(site["ids"]) | set(members))]},
                "j": {"before": _j(cur_rv), "after": _j(rv)}})
            cur, cur_rv = cand, rv
            cur_tc = _TransCache(cur, grid)
            changed.update(members)
            edited_this_sweep = True
        if not edited_this_sweep:
            break
    status = "unchanged" if not changed else (
        "budget_exhausted" if len(changed) >= edit_cap else "repaired")
    return RepairResult(
        raw=cur, status=status, changed_ids=sorted(changed),
        before=orig_rv, after=cur_rv, trials=trials,
        elapsed_s=time.perf_counter() - t0, sweeps_run=sweeps_run,
        witnesses=witnesses)
