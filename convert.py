"""osu! standard -> Beat Saber v2 converter.

Usage: python convert.py map.osu audio.(mp3|ogg|egg) outdir/
Emits outdir/{Info.dat, ExpertPlus.dat, song.egg} and runs sanity checks.
"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from parity import HandParity, is_hammer, violations

LEFT, RIGHT = 0, 1


class GenerationError(Exception):
    """Generation or export failed validation; do not publish an invalid map."""


def pick_best(cands):
    """cands: [((ok: bool, score: float), *payload)]. Return the winner.
    A valid candidate outranks ANY invalid one (ok compared first), ties
    break on score — so a higher-scoring invalid candidate never wins over
    a valid one. Winner's ok flag gates the GenerationError."""
    return max(cands, key=lambda c: c[0])

# fixed thresholds; tune after play-tests (Phase 3)
STREAM_MS = 90        # consecutive gaps below this get thinned 2:1
FORCE_ALT_MS = 250    # below this gap, hands must alternate
DOUBLE_MIN_GAP_MS = 350   # new-combo jumps at least this far apart become doubles
DOUBLE_MIN_DIST = 120     # osu px travel needed for a double
SLIDER_TAIL_MIN_MS = 300  # sliders shorter than this emit only a head note
NJS = 18
NPS_CAP = 11        # peak sustained swings/sec over a 3.5s window (Expert+)
MAX_ADOPT_SEG = 4   # piecewise grid ceiling: a real song has a handful of
                    # genuine tempo/offset changes; more = MI auto-timing
                    # noise, so keep uniform even if it lowers snap error
# difficulty ladder, measured over 800 multi-difficulty corpus maps:
# (game slot rank, NJS median, conditioning rate = per-tier swings/s median,
# band/cap scale vs the pinned Expert+ reference)
DIFFS = {
    "Easy":       (1, 10, 1.4, 0.29),
    "Normal":     (3, 12, 2.0, 0.42),
    "Hard":       (5, 13, 3.0, 0.61),
    "Expert":     (7, 16, 3.9, 0.82),
    "ExpertPlus": (9, 18, 4.9, 1.0),
}
TIER_UP = 1.22  # corpus EP/Expert rate ratio — each extra '+' multiplies by it
MAX_DIFFICULTIES = len(DIFFS)  # five unique native slots per characteristic


def diff_spec(name):
    """Difficulty name -> spec dict, scales from the LEARNED ladder
    (ladder.py fit over ~2000 corpus tier pairs: the per-tier gap shrinks as
    difficulty rises, so Expert++/+++/... extrapolate indefinitely instead of
    compounding a constant). ExpertPlus2, ExpertPlus3, ... are Expert++,
    Expert+++ — custom labels, ladder rank 11, 13, ...; export assigns unique slots.
    Their band FLOOR is the previous tier's ceiling: an Expert++ map must
    actually play harder than Expert+, not just be allowed to."""
    import ladder
    from groom import TARGET_BAND
    lo, hi = TARGET_BAND
    if name in DIFFS:
        rank = DIFFS[name][0]
        scale = ladder.scale(rank)
        return {"name": name, "slot": name, "label": name, "rank": rank,
                "njs": ladder.njs(rank), "cond": 4.9 * scale, "scale": scale,
                "band": (lo * scale, hi * scale)}
    if name.startswith("ExpertPlus") and name[len("ExpertPlus"):].isdigit():
        n = int(name[len("ExpertPlus"):])
        if n < 2:
            raise ValueError(f"unknown difficulty: {name}")
        rank = 9 + 2 * (n - 1)  # ladder rank; export assigns unique game slots
        scale = ladder.scale(rank)
        return {"name": name, "slot": "ExpertPlus",
                "label": "Expert" + "+" * n, "rank": 9,
                "njs": ladder.njs(rank), "cond": 4.9 * scale, "scale": scale,
                "band": (hi * ladder.scale(rank - 2), hi * scale)}
    raise SystemExit(f"unknown difficulty: {name}")


def selected_difficulties(diffs=None):
    names = [n.strip() for n in ("ExpertPlus" if diffs is None else diffs).split(",") if n.strip()]
    if not 1 <= len(names) <= MAX_DIFFICULTIES or len(set(names)) != len(names):
        raise GenerationError("Choose one to five distinct difficulties per map")
    for name in names:
        diff_spec(name)
    return sorted(names, key=lambda n: (diff_spec(n)["rank"], diff_spec(n)["scale"]))


def difficulty_slots(names):
    """Pack extended tiers into unique native slots, ordered from easiest to hardest."""
    names = selected_difficulties(",".join(names))
    slots = list(DIFFS)
    available = MAX_DIFFICULTIES - 1
    assigned = {}
    for name in reversed(names):
        available = min(available, slots.index(diff_spec(name)["slot"]))
        assigned[name] = (slots[available], DIFFS[slots[available]][0])
        available -= 1
    return {name: assigned[name] for name in names}
LEAD_MS = 5000      # target lead-in: longer note-free intros get trimmed,
MIN_LEAD_MS = 3000  # hotter starts get padded, both to exactly LEAD_MS
TAIL_MS = 3000      # audio kept past the last note; the rest is cut + faded

# osu screen angle (0=right, 90=down) -> BS cut direction
_DIRS = [(0, 3), (45, 7), (90, 1), (135, 6), (180, 2), (225, 4), (270, 0), (315, 5)]
_MIRROR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}
GROOM_PT = Path(__file__).parent / "groom.pt"


def parse_osu(path):
    """Return (meta dict, hit objects list, first BPM).
    Hit object: dict(t, x, y, kind, new_combo, end_t, end_x, end_y)."""
    section = None
    meta = {"Title": "Unknown", "Artist": "Unknown", "AudioFilename": "",
            "SliderMultiplier": 1.4}
    timing = []      # (time, beat_len, uninherited)
    objects = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("["):
            section = line.strip("[]")
            continue
        if section in ("General", "Metadata", "Difficulty") and ":" in line:
            k, v = line.split(":", 1)
            if k in meta:
                meta[k] = v.strip()
        elif section == "TimingPoints":
            p = line.split(",")
            timing.append((float(p[0]), float(p[1]),
                           int(p[6]) if len(p) > 6 else 1))
        elif section == "HitObjects":
            p = line.split(",")
            x, y, t, typ = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            o = {"t": t, "x": x, "y": y, "new_combo": bool(typ & 4),
                 "end_t": t, "end_x": x, "end_y": y}
            if typ & 1:
                o["kind"] = "circle"
            elif typ & 2:
                o["kind"] = "slider"
                curve, slides, length = p[5], int(p[6]), float(p[7])
                beat_len, sv = _timing_at(timing, t)
                sm = float(meta["SliderMultiplier"])
                dur = length / (sm * 100 * sv) * beat_len * slides
                o["end_t"] = t + dur
                if slides % 2 == 1:  # odd repeats end at the tail anchor
                    last = curve.split("|")[-1]
                    if ":" in last:
                        ex, ey = last.split(":")
                        o["end_x"], o["end_y"] = int(float(ex)), int(float(ey))
            elif typ & 8:
                o["kind"] = "spinner"
                o["end_t"] = int(p[5])
            else:
                continue
            objects.append(o)
    bpm_points = [(t, 60000.0 / bl) for t, bl, uninh in timing if uninh == 1 and bl > 0]
    bpm = bpm_points[0][1] if bpm_points else 120.0
    offset = bpm_points[0][0] if bpm_points else 0.0
    # retain the full uninherited tempo map (absolute ms + beat length) so a
    # tempo-changing song can be gridded per segment (Packet C). SV/inherited
    # points stay with _timing_at for slider semantics. Stashed in meta to
    # keep the 4-tuple signature every caller unpacks.
    meta["_timing"] = sorted((t, bl) for t, bl, uninh in timing
                             if uninh == 1 and bl > 0)
    return meta, objects, bpm, offset


def _timing_at(timing, t):
    """(beat_length_ms, slider_velocity_multiplier) active at time t."""
    beat_len, sv = 500.0, 1.0
    for pt, bl, uninh in timing:
        if pt > t:
            break
        if uninh == 1:
            beat_len, sv = bl, 1.0
        elif bl < 0:
            sv = -100.0 / bl
    return beat_len, sv


def travel_dir(dx, dy):
    """Cursor travel vector (osu coords, y down) -> BS cut direction."""
    if dx == 0 and dy == 0:
        return 1
    ang = math.degrees(math.atan2(dy, dx)) % 360
    return min(_DIRS, key=lambda a: min(abs(ang - a[0]), 360 - abs(ang - a[0])))[1]


def grid_pos(x, y, hand):
    col = min(3, max(0, x * 4 // 512))
    layer = min(2, max(0, 2 - y * 3 // 384))
    if layer == 1 and col in (1, 2):
        layer = 0                      # no vision-blocking middles
    if hand == LEFT and col == 3:
        col = 2                        # no full cross-body
    if hand == RIGHT and col == 0:
        col = 1
    return col, layer


def _hits(objects, thin=True):
    """Explode sliders into timed hits (spinners = rest). thin=True applies
    the legacy 2:1 stream thinning at evidence-extraction time.
    Packet C note: moving thinning fully into the schedule layer (thin=False)
    is wired and ready, but on the CURRENT rhythm model — trained on
    thinned-density evidence — dense input raises doubles and <> (numb 1->5,
    FENT 0->3 in the five-song suite). So the move is DEFERRED to Packet D,
    where the rhythm model is retrained on dense evidence and can control
    density itself. Until then the groomed path keeps thin=True."""
    hits = []
    for o in objects:
        if o["kind"] == "spinner":
            continue
        hits.append((o["t"], o["x"], o["y"], o["new_combo"]))
        if o["kind"] == "slider" and o["end_t"] - o["t"] >= SLIDER_TAIL_MIN_MS:
            hits.append((o["end_t"], o["end_x"], o["end_y"], False))
    hits.sort()
    if not thin:
        return hits
    thinned, drop = [], False
    for i, h in enumerate(hits):
        if i and h[0] - hits[i - 1][0] < STREAM_MS:
            drop = not drop
            if drop:
                continue
        else:
            drop = False
        thinned.append(h)
    return thinned


def convert(objects, bpm):
    """osu objects -> list of BS note dicts (time in ms kept for checks)."""
    hits = _hits(objects)
    machines = {LEFT: HandParity(), RIGHT: HandParity()}
    notes = []
    prev_hand, prev_t, prev_x, prev_y = RIGHT, None, 256, 192
    for t, x, y, new_combo in hits:
        gap = None if prev_t is None else t - prev_t
        if gap is not None and gap < FORCE_ALT_MS:
            hand = LEFT if prev_hand == RIGHT else RIGHT
        elif x < 512 / 3:
            hand = LEFT
        elif x > 1024 / 3:
            hand = RIGHT
        else:
            hand = LEFT if prev_hand == RIGHT else RIGHT

        desired = travel_dir(x - prev_x, y - prev_y)
        d = machines[hand].next_direction(desired, t)
        col, layer = grid_pos(x, y, hand)
        notes.append({"t": t, "hand": hand, "col": col, "layer": layer, "dir": d})

        dist = math.hypot(x - prev_x, y - prev_y)
        if (new_combo and gap is not None and gap >= DOUBLE_MIN_GAP_MS
                and dist >= DOUBLE_MIN_DIST):
            other = LEFT if hand == RIGHT else RIGHT
            md = machines[other].next_direction(_MIRROR[d], t)
            mcol, mlayer = grid_pos(512 - x, y, other)
            if (mcol, mlayer) == (col, layer):
                mcol = max(0, mcol - 1) if other == LEFT else min(3, mcol + 1)
            notes.append({"t": t, "hand": other, "col": mcol, "layer": mlayer, "dir": md})

        prev_hand, prev_t, prev_x, prev_y = hand, t, x, y
    from swing_clearance import repair_approaches
    raw, _ = repair_approaches([(n['t'], n['hand'], n['col'], n['layer'], n['dir']) for n in notes])
    return [dict(t=t, hand=h, col=c, layer=l, dir=d) for t, h, c, l, d in raw]


def grid_steps(objects, bpm, offset, timing=None, thin=True):
    """osu objects -> (steps, T, step_ms, adjusted offset, grid).
    Extends the grid over audio before the osu offset (whole beats), so a
    long unmapped intro is a sprinkle-able dead zone instead of off-grid.
    timing: uninherited [(ms, beat_len_ms)]; with >1 segment the grid is
    piecewise (per-segment 1/4 spacing) so a tempo-changing song grids on
    its LOCAL tempo. Single/none -> the legacy uniform grid, byte-identical.
    Returns the TimeGrid; step_ms/offset are the first-segment scalars kept
    for back-compat. Snap error is reported to stderr, not silently eaten."""
    from timing import TimeGrid, STEPS_PER_BEAT, merge_tempo
    step_ms = 60000.0 / bpm / STEPS_PER_BEAT
    lead = int(offset / (4 * step_ms)) * 4
    offset -= lead * step_ms
    hits = _hits(objects, thin=thin)  # thin=False = dense evidence (experiments)
    if not hits:
        return set(), 0, step_ms, offset, TimeGrid.uniform(step_ms, offset, 0)
    merged = merge_tempo(timing) if timing else []
    steps = {max(0, round((h[0] - offset) / step_ms)) for h in hits}
    T = max(steps) + 8 if steps else 0
    grid = TimeGrid.uniform(step_ms, offset, T)

    def p95(errs):
        return sorted(errs)[int(0.95 * (len(errs) - 1))] if errs else 0.0
    uerr = [abs(h[0] - grid.time(round((h[0] - offset) / step_ms))) for h in hits]
    uni_p95 = p95(uerr)
    grid.decision = {"grid": "uniform", "segments": 1, "bpm": bpm,
                     "uniform_p95_ms": round(uni_p95, 1),
                     "candidate_segments": len(merged)}
    # ADOPT the piecewise grid only when it measurably lowers snap error AND
    # the change is a plausible handful of segments. A real change (RC
    # 130->260, an offset reset) grids notes far better with few segments;
    # MI auto-timing noise (rather_be's 54-122 flailing, 12 segments) can also
    # lower global snap error because notes sit on MI's own wobbly lines, so a
    # segment cap rejects it — global p95 alone can't tell signal from noise
    # (review follow-up 3). Steady/noisy songs keep uniform, byte-identical.
    if 1 < len(merged) <= MAX_ADOPT_SEG:
        end_ms = max(h[0] for h in hits)
        pg = TimeGrid.covering(merged, offset, end_ms + step_ms * 8, step_ms)
        perr = [pg.snap(h[0])[1] for h in hits]
        pg_p95 = p95(perr)
        # per-segment tail error, so a short bad passage can't hide behind p95
        seg_starts = [s for s, _ in pg.segments]
        seg_worst = 0.0
        for lo, hi in zip(seg_starts, seg_starts[1:] + [float("inf")]):
            e = [pg.snap(h[0])[1] for h in hits if lo <= h[0] < hi]
            if e:
                seg_worst = max(seg_worst, p95(e))
        adopt = pg_p95 < uni_p95 - 5.0 and pg_p95 < 0.75 * uni_p95
        if adopt:
            steps = {pg.snap(h[0])[0] for h in hits}
            T = max(steps) + 8 if steps else 0
            if T > pg.n:
                pg = TimeGrid.covering(merged, offset,
                                       pg.time(pg.n - 1) + step_ms * 8, step_ms)
            grid = pg
            grid.decision = {"grid": "piecewise", "segments": len(merged),
                             "bpms": [round(60000.0 / bl, 2) for _, bl in merged],
                             "uniform_p95_ms": round(uni_p95, 1),
                             "piecewise_p95_ms": round(pg_p95, 1),
                             "piecewise_max_ms": round(max(perr), 1),
                             "worst_segment_p95_ms": round(seg_worst, 1)}
            print(f"  piecewise grid: {len(merged)} segments, snap p95 "
                  f"{pg_p95:.1f}ms (uniform {uni_p95:.1f}ms), worst-seg "
                  f"{seg_worst:.1f}ms, max {max(perr):.1f}ms", file=sys.stderr)
        else:
            grid.decision["rejected_piecewise_p95_ms"] = round(pg_p95, 1)
            if uni_p95 > 5.0:
                print(f"  kept uniform grid (piecewise p95 {pg_p95:.1f}ms did "
                      f"not beat uniform {uni_p95:.1f}ms)", file=sys.stderr)
    elif len(merged) > MAX_ADOPT_SEG:
        grid.decision["rejected_reason"] = f"{len(merged)} segments > cap"
        print(f"  kept uniform grid ({len(merged)} candidate segments > "
              f"{MAX_ADOPT_SEG} cap; auto-timing noise)", file=sys.stderr)
    return steps, T, step_ms, offset, grid


def convert_groomed(objects, bpm, offset, audio_path=None, diff="ExpertPlus",
                    replay_mode="off", collect=None, timing=None,
                    thin=True, calibrate=True, rhythm_model=None,
                    flow_model=None, plans=None, quality_profile=None):
    """Like convert(), but the groomer model re-times the stream onto per-hand
    swing rhythm AND chooses geometry (direction/column/layer) learned from
    hand-mapped maps; parity legality is enforced inside groom_notes. Audio
    energy conditions the rhythm model, so section dynamics (sparse verses,
    dense drops) transfer from the human maps.
    Times snap to the 1/4-beat grid by design — that's the BS idiom.
    replay_mode: 'off' is the production default (2026-09-18) — literal
    section copying is the dominant cause of exact repetition and off drives
    it to ~0 with no motion regression in the paired motion evaluation. Pass 'legacy' for reproducibility/A-B only.
    collect, if a list, receives EVERY decoded candidate (notes/walls/score/
    gates/trace), not just the winner — experiment diagnostics, no effect on
    the result.
    plans, if a dict, replays per-ATTEMPT frozen decode decisions keyed
    "pass:seed" (values: rest_mask/thr/keep tensors + the bound rate_scale) —
    the phrase-plan no-op replay seam (eval/phrase_plan). Missing or
    rate-mismatched plans fail closed. None (production) changes nothing.
    """
    from groom import groom_notes, cached_audio_features
    steps, T, step_ms, offset, grid = grid_steps(objects, bpm, offset, timing,
                                                 thin=thin)
    if not steps:
        return []
    afeat = None
    if audio_path:
        try:
            afeat = cached_audio_features(
                audio_path, [grid.time(s) for s in range(T)])
        except Exception as e:
            print(f"audio features unavailable ({e}); converting without dynamics")
    from groom import score_notes, rot_share, N_DECODES, ROT_BAND
    import critic
    use_critic = critic.CRITIC_PT.exists()
    from groom import TEMPS, run_stats
    spec = diff_spec(diff)
    cap = NPS_CAP * spec["scale"]
    cands = []

    def try_seed(seed, temp, rate_scale=1.0, pass_index=0):
        tr = {} if collect is not None else None
        # plans: per-ATTEMPT frozen decode decisions (phrase-plan replay).
        # Each attempt (pass_index, seed) must have its own plan — never shared —
        # and a missing or mismatched plan fails closed rather than recomputing.
        inject = {}
        if plans is not None:
            p = plans[f"{pass_index}:{seed}"]         # KeyError = fail closed
            if abs(p.get("rate_scale", rate_scale) - rate_scale) > 1e-12:
                raise GenerationError(
                    f"plan {pass_index}:{seed} bound to rate_scale "
                    f"{p['rate_scale']} but attempt runs at {rate_scale}")
            inject = dict(rest_mask_in=p["rest_mask"],
                          thr_vectors=(p["thr"], p["keep"]),
                          rest_policy="off", density_adjust=False)
            if p.get("doubles_cap_8b") is not None:   # opt-in plan request
                inject["doubles_cap_8b"] = int(p["doubles_cap_8b"])
        # rhythm_model / flow_model inject explicit (e.g. experimental)
        # checkpoints for A/B; None keeps the shipped models groom_notes loads
        raw, wall_runs = groom_notes(steps, T, step_ms, offset,
                                     model=rhythm_model, flow=flow_model,
                                     afeat=afeat,
                                     seed=seed, temp=temp,
                                     rate_scale=rate_scale,
                                     drate=spec["cond"],
                                     band_scale=spec["scale"],
                                     band=spec["band"],
                                     replay_mode=replay_mode, trace=tr,
                                     grid=grid, calibrate=calibrate, **inject)
        # candidates over the density cap lose the vote (check would fail
        # them); so do style-collapsed decodes (axis-run/vert/lateral shares
        # past any corpus map) — gates select, the sampler stays free
        vert, longrun = run_stats(raw)
        lat = (sum(1 for *_, d in raw if d in (2, 3))
               / max(1, sum(1 for *_, d in raw if d != 8)))
        ok = (peak_nps(sorted(grid.time(s)
                              for s, _, _, _, d in raw if d != 8)) <= cap
              and vert <= 0.80 and longrun <= 0.55 and lat <= 0.30)
        # full post-repair validity check gates the vote too — a candidate
        # main() would report as CHECKS FAILED must not win silently
        problems = check([{"t": grid.time(s_), "hand": h_, "col": c_,
                           "layer": l_, "dir": d_}
                          for s_, h_, c_, l_, d_ in raw], bpm,
                         [{"t": grid.time(s0),
                           "dur": grid.time(s0 + ln) - grid.time(s0),
                           "col": col} for s0, ln, col in wall_runs], cap)
        ok = ok and not problems
        # motion soft cost (post-repair notes): fast repositioning, hand
        # monopoly, <> exposure order otherwise-acceptable candidates
        import motion
        rep = motion.report([(grid.time(s_), h_, c_, l_, d_)
                             for s_, h_, c_, l_, d_ in raw])
        mcost = motion.select_cost(rep)
        if use_critic:  # learned judge: human-likeness logit
            key = (ok, critic.score(raw, wall_runs, T, afeat) - mcost)
        else:           # fallback: flow likelihood inside a sanity band
            key = (ok and ROT_BAND[0] <= rot_share(raw) <= ROT_BAND[1],
                   score_notes(raw, wall_runs, T, afeat) - mcost)
        cands.append((key, raw, wall_runs, seed))
        if collect is not None:
            collect.append({"seed": seed, "temp": temp, "pass_index": pass_index,
                            "rate_scale": rate_scale, "ok": ok,
                            "gates": {"vert": round(vert, 3),
                                      "longrun": round(longrun, 3),
                                      "lat": round(lat, 3),
                                      "check": problems},
                            "score": key[1], "motion_cost": round(mcost, 3),
                            "motion": rep, "notes": raw,
                            "walls": wall_runs, "trace": tr})
        print(f"  seed {seed} (t{temp}): score {key[1]:.2f} "
              f"(motion -{mcost:.2f}, flags/1k "
              f"{rep['flags']['flags_per_1000']})"
              + ("" if ok else " (gated)"))
        return ok

    from groom import TARGET_BAND
    rate_scale = 1.0
    for _pass in (0, 1):
        cands.clear()
        for seed in range(N_DECODES):
            try_seed(seed, TEMPS[seed % len(TEMPS)], rate_scale, pass_index=_pass)
        extra = N_DECODES
        while not any(c[0][0] for c in cands) and extra < 2 * N_DECODES:
            # every candidate failed a gate: draw hotter seeds until one passes
            try_seed(extra, TEMPS[-1], rate_scale, pass_index=_pass)
            extra += 1
        best = pick_best(cands)
        # closed loop: the calibration proxy can't see everything (THIN,
        # budget, replay skips) — if the decoded rate still tops the band
        # ceiling, redo once with the target scaled by the overshoot
        # rate over the DECODED note span — the input onset span can be
        # longer (dropped edges), which understated FENT's 6.4 as in-band
        ts = [s for s, *_, d in best[1] if d != 8]
        span = (grid.time(max(ts)) - grid.time(min(ts))) / 1000 \
            if len(ts) > 1 else 0
        actual = len(ts) / max(1e-9, span)
        blo, bhi = spec["band"]
        if span < 30 or _pass or blo * 0.98 <= actual <= bhi * 1.02:
            break
        rate_scale = (bhi if actual > bhi else blo) / actual
        print(f"  off band ({actual:.1f} swings/s vs {blo:.1f}-{bhi:.1f}): "
              f"re-decoding at x{rate_scale:.2f}")
    (ok, sc), raw, wall_runs, seed = best
    if collect is not None:
        for e in collect:
            e["selected"] = e["notes"] is raw
    print(f"best-of-{N_DECODES} ({'critic' if use_critic else 'likelihood'}): "
          f"seed {seed}, score {sc:.2f}" + ("" if ok else " (ALL GATED)"))
    # quality_profile (opt-in, Q1): repair the EMITTED winner's geometry after
    # the original selection and rate retries are fully resolved. The candidate
    # bank, scores, selected seed and retry decisions above are untouched; the
    # transformed winner is an output-stage derivative recorded on the selected
    # collect record, and a failed repair falls back to the original winner.
    if quality_profile is not None and ok:
        try:
            from quality_repair import RepairConfig, repair_winner
            qcfg = (RepairConfig(**quality_profile)
                    if isinstance(quality_profile, dict) else quality_profile)
            rr = repair_winner(raw, wall_runs, grid, bpm, cap, qcfg)
        except Exception as e:
            print(f"  quality repair failed ({e}); exporting original winner")
            rr = None
        if rr is not None and rr.changed_ids:
            if collect is not None:
                for c_rec in collect:
                    if c_rec["selected"]:
                        c_rec["output_transform"] = {
                            "raw": rr.raw, "status": rr.status,
                            "changed_ids": rr.changed_ids,
                            "before": rr.before, "after": rr.after,
                            "trials": rr.trials, "elapsed_s": rr.elapsed_s,
                            "witnesses": rr.witnesses}
            print(f"  quality repair: {len(rr.changed_ids)} notes adjusted "
                  f"({rr.status}, {rr.trials} trials, {rr.elapsed_s:.1f}s)")
            raw = rr.raw
    notes = [{"t": grid.time(s), "hand": h, "col": col, "layer": lay,
              "dir": d} for s, h, col, lay, d in raw]
    walls = [{"t": grid.time(s0),
              "dur": grid.time(s0 + ln) - grid.time(s0), "col": col}
             for s0, ln, col in wall_runs]
    # every candidate failed a hard gate: never return an invalid map as if
    # selected. collect mode (experiment harness) still gets the artifact —
    # it records per-candidate ok/reasons and exports nothing to the game.
    if not ok and collect is None:
        reason = "; ".join(check(notes, bpm, walls, cap)) or "style/rate gate"
        raise GenerationError(f"{diff}: no valid candidate after {extra} "
                              f"decodes; best (seed {seed}) fails: {reason}")
    return notes, walls


def peak_nps(times):
    """Sustained swings/sec over a 3.5s rolling window (times sorted, ms)."""
    peak, j = 0.0, 0
    for i, t in enumerate(times):
        j = max(j, i)
        while j < len(times) and times[j] - t <= 3500:
            j += 1
        peak = max(peak, (j - i) / 3.5)
    return peak


WALL_MIN_BEATS, WALL_MAX_BEATS = 2, 8
WALL_GAP_BEATS = 8      # spacing between walls
WALL_MARGIN = 0.5       # beats clear of same-column notes at both ends


def lean_walls(notes, bpm):
    """Full-height width-1 walls on side columns wherever notes leave that
    column free — lean left/right only; crouch walls (_type 1) never emitted."""
    if len(notes) < 2:
        return []
    beat = 60000.0 / bpm
    cands = []
    for col in (0, 3):
        used = sorted(n["t"] for n in notes if n["col"] == col)
        edges = [notes[0]["t"]] + used + [notes[-1]["t"]]
        for a, b in zip(edges, edges[1:]):
            start, gap_end = a + WALL_MARGIN * beat, b - WALL_MARGIN * beat
            while start + WALL_MIN_BEATS * beat <= gap_end:
                end = min(start + WALL_MAX_BEATS * beat, gap_end)
                if sum(1 for n in notes if start <= n["t"] <= end) >= 2:
                    cands.append((start, end, col))
                start = end + WALL_GAP_BEATS * beat
    cands.sort()
    walls, last_end = [], float("-inf")
    for start, end, col in cands:
        if start - last_end < WALL_GAP_BEATS * beat:
            continue
        walls.append({"t": start, "dur": end - start, "col": col})
        last_end = end
    return walls


def check(notes, bpm, walls=(), cap=NPS_CAP):
    """Automated Phase-2 checks. Returns list of problem strings (empty = pass)."""
    problems = []
    from swing_clearance import blocked_approaches
    # Invalid directions are diagnosed below; do not let them break validation.
    blocked = blocked_approaches([(n['t'], n['hand'], n['col'], n['layer'], n['dir'])
                                 for n in notes if n['dir'] in range(9)])
    if blocked:
        problems.append(f"{len({t for t, _, _ in blocked})} wrong-color cut-approach obstructions")
    v = violations([(n["t"], n["hand"], n["dir"]) for n in notes])
    if v:
        problems.append(f"{v} parity violations")
    for n in notes:
        if not (0 <= n["col"] <= 3 and 0 <= n["layer"] <= 2
                and 0 <= n["dir"] <= 8 and n["hand"] in (0, 1)):
            problems.append(f"out-of-range note at {n['t']}ms: {n}")
            break
    times = [n["t"] for n in notes if n["dir"] != 8]  # dots aren't swings
    if [n["t"] for n in notes] != sorted(n["t"] for n in notes):
        problems.append("notes not sorted by time")
    by_time = {}
    for n in notes:
        by_time.setdefault((round(n["t"]), n["hand"]), []).append(n)
    dupes = sum(1 for v_ in by_time.values()
                if sum(1 for n in v_ if n["dir"] != 8) > 1)  # chains use dots
    if dupes:
        problems.append(f"{dupes} same-hand simultaneous swings")
    pairs = {}
    for n in notes:
        pairs.setdefault(round(n["t"]), []).append(n)
    hams = sum(1 for g in pairs.values() if len(g) == 2
               and g[0]["dir"] == g[1]["dir"]
               and is_hammer(g[0]["col"], g[0]["layer"],
                             g[1]["col"], g[1]["layer"], g[0]["dir"]))
    if hams:
        problems.append(f"{hams} hammer doubles (same-dir in line, like '>>')")
    peak = peak_nps(times)
    if peak > cap:
        problems.append(f"peak sustained nps {peak:.1f} > {cap:.1f}")
    for w in walls:
        if w["col"] not in (0, 3) or w["dur"] <= 0:
            problems.append(f"bad wall {w}")
            break
        if any(n["col"] == w["col"] and w["t"] <= n["t"] < w["t"] + w["dur"] - 1
               for n in notes):  # end-exclusive: a note may land as the wall ends
            problems.append(f"wall through notes at {w['t']:.0f}ms col {w['col']}")
            break
    return problems


def export_charts(maps, bpm, meta, outdir):
    """The difficulty .dat files + Info.dat of write_map (no audio)."""
    from swing_clearance import blocked_approaches
    slots = difficulty_slots(maps)
    # Validate the exact rounded timestamps BEFORE writing any difficulty.
    # This also catches bypass callers and pairs merged by export precision.
    beats = 60000.0 / round(bpm, 3)
    for name, (notes, _, _) in maps.items():
        if blocked_approaches([(round(n['t'] / beats, 5), n['hand'], n['col'], n['layer'], n['dir'])
                               for n in notes]):
            raise GenerationError(f"{name}: exported chart has wrong-color cut-approach obstructions")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # notes MUST convert with the same BPM the Info file exposes, or a player
    # reading the rounded Info BPM plays every note off its intended absolute
    # time (drift grows with duration for fractional BPM) — review follow-up 4
    bpm = round(bpm, 3)
    beats = 60000.0 / bpm
    beatmaps, written = [], []
    for name, (slot, rank) in slots.items():
        notes, walls, spec = maps[name]
        bs_notes = [{"_time": round(n["t"] / beats, 5), "_lineIndex": n["col"],
                     "_lineLayer": n["layer"], "_type": n["hand"],
                     "_cutDirection": n["dir"]} for n in notes]
        bs_walls = [{"_time": round(w["t"] / beats, 5), "_lineIndex": w["col"],
                     "_type": 0, "_duration": round(w["dur"] / beats, 5),
                     "_width": 1} for w in walls]
        fname = f"{name}.dat"
        (outdir / fname).write_text(json.dumps(
            {"_version": "2.0.0", "_notes": bs_notes, "_obstacles": bs_walls,
             "_events": []}))
        written.append(fname)
        bm = {"_difficulty": slot, "_difficultyRank": rank,
              "_beatmapFilename": fname,
              "_noteJumpMovementSpeed": spec["njs"],
              "_noteJumpStartBeatOffset": 0}
        if spec["label"] != spec["slot"] or slot != spec["slot"]:
            label = "Expert+" if name == "ExpertPlus" else spec["label"]
            bm["_customData"] = {"_difficultyLabel": label}
        beatmaps.append(bm)
    info = {
        "_version": "2.0.0",
        "_songName": meta["Title"], "_songSubName": "",
        "_songAuthorName": meta["Artist"], "_levelAuthorName": "osu2bs",
        "_beatsPerMinute": round(bpm, 3),
        "_shuffle": 0, "_shufflePeriod": 0.5,
        "_previewStartTime": 12, "_previewDuration": 10,
        "_songFilename": "song.egg", "_coverImageFilename": "",
        "_environmentName": "DefaultEnvironment", "_songTimeOffset": 0,
        "_difficultyBeatmapSets": [{
            "_beatmapCharacteristicName": "Standard",
            "_difficultyBeatmaps": beatmaps}]}
    (outdir / "Info.dat").write_text(json.dumps(info, indent=2))
    return written


def _audio_info(path):
    """Inspect bytes, never infer the codec from .egg/.ogg filenames."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=format_name,duration:stream=codec_type,codec_name,sample_rate,channels",
             "-of", "json", str(path)], capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("audio duration must be finite and positive")
        if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
            raise ValueError("no audio stream")
        return info, duration
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError) as e:
        raise GenerationError(f"Cannot read audio {path}: {e}") from e


def _game_audio(info):
    streams = info.get("streams", [])
    return (info["format"]["format_name"] == "ogg" and len(streams) == 1
            and streams[0].get("codec_name") == "vorbis"
            and streams[0].get("sample_rate") in ("44100", "48000")
            and streams[0].get("channels") in (1, 2))


def validate_audio(path):
    """Require real Ogg Vorbis, positive duration, and a successful full decode."""
    info, duration = _audio_info(path)
    if not _game_audio(info):
        raise GenerationError(f"{path}: song.egg must contain mono/stereo Ogg Vorbis at 44.1/48 kHz")
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
             "-map", "0:a:0", "-f", "null", "-"],
            capture_output=True, text=True, check=True)
        if result.stderr.strip():
            raise ValueError(result.stderr.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as e:
        raise GenerationError(f"Cannot decode exported audio {path}: {e}") from e
    return duration


def write_audio(audio_in, outdir, shift_ms=0, end_ms=None):
    """Normalize audio for the game, preserving compatible unedited Vorbis bytes."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audio_in = Path(audio_in)
    egg = outdir / "song.egg"
    # edits always start from a kept pristine copy so reconverting never
    # double-shifts or double-cuts
    orig = outdir / "song_orig.egg"
    src = orig if orig.exists() and (shift_ms or end_ms is not None) else audio_in
    info, duration = _audio_info(src)
    dur_ms = 1000 * duration
    cut = end_ms is not None and dur_ms + shift_ms > end_ms + 1000
    if shift_ms or cut:
        if not orig.exists():
            shutil.copyfile(audio_in, orig)
        src = orig
    # Never replace an existing playable file with an incomplete transcode.
    with tempfile.NamedTemporaryFile(dir=outdir, suffix=".egg", delete=False) as f:
        tmp = Path(f.name)
    try:
        if not (shift_ms or cut) and _game_audio(info):
            shutil.copyfile(src, tmp)
        else:
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-xerror"]
            if shift_ms < 0:
                cmd += ["-ss", f"{-shift_ms / 1000:.3f}"]
            cmd += ["-i", str(src), "-map", "0:a:0"]
            af = [f"adelay={shift_ms}:all=1"] if shift_ms > 0 else []
            if cut:
                cmd += ["-t", f"{end_ms / 1000:.3f}"]
                af.append(f"afade=t=out:st={max(0, end_ms - 2000) / 1000:.3f}:d=2")
            if af:
                cmd += ["-af", ",".join(af)]
            cmd += ["-f", "ogg", "-vn", "-c:a", "libvorbis", "-q:a", "6",
                    "-ar", "44100", "-ac", "2", str(tmp)]
            subprocess.run(cmd, check=True, capture_output=True)
        validate_audio(tmp)
        tmp.replace(egg)
    except (OSError, subprocess.CalledProcessError) as e:
        raise GenerationError(f"Audio export failed for {audio_in}: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)
    return egg


def package_map(outdir, zip_path=None):
    """Shared release boundary, including review maps assembled from saved charts."""
    outdir = Path(outdir)
    info = json.loads((outdir / "Info.dat").read_text())
    for group in info["_difficultyBeatmapSets"]:
        difficulties = group["_difficultyBeatmaps"]
        slots = [d["_difficulty"] for d in difficulties]
        if (not 1 <= len(slots) <= MAX_DIFFICULTIES or len(set(slots)) != len(slots)
                or any(s not in DIFFS for s in slots)
                or any(d["_difficultyRank"] != DIFFS[d["_difficulty"]][0] for d in difficulties)):
            raise GenerationError("Each game mode needs one to five unique native difficulty slots with matching ranks")
    names = ["Info.dat", info["_songFilename"]]
    names += [d["_beatmapFilename"] for bs in info["_difficultyBeatmapSets"]
              for d in bs["_difficultyBeatmaps"]]
    if info.get("_coverImageFilename"):
        names.append(info["_coverImageFilename"])
    for name in names:
        if Path(name).name != name or not (outdir / name).is_file():
            raise GenerationError(f"Missing or non-local map asset: {name}")
    validate_audio(outdir / info["_songFilename"])
    zip_path = Path(zip_path) if zip_path is not None else outdir.with_suffix(".zip")
    with tempfile.NamedTemporaryFile(dir=zip_path.parent, suffix=".zip", delete=False) as f:
        tmp = Path(f.name)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in dict.fromkeys(names):
                z.write(outdir / name, name)
        tmp.replace(zip_path)
    finally:
        tmp.unlink(missing_ok=True)
    return zip_path


def write_map(maps, bpm, meta, audio_in, outdir, shift_ms=0, end_ms=None):
    """Export charts and validated game audio, then publish a root-level map ZIP."""
    write_audio(audio_in, outdir, shift_ms, end_ms)
    export_charts(maps, bpm, meta, outdir)
    package_map(outdir)


def main_onsets(osu_path, audio_path, outdir, diffs, checkpoint):
    """One immutable upstream osu timeline per selected Beat Saber difficulty."""
    from hashlib import sha256
    from onset_flow import load_model, source_onsets, decode, SCHEMA
    import torch
    names = selected_difficulties(diffs)
    if isinstance(osu_path, dict):
        sources = osu_path
        if set(sources) != set(names):
            raise GenerationError("each selected difficulty needs its own upstream osu map")
    elif len(names) == 1:
        sources = {names[0]: osu_path}
    else:
        raise GenerationError("exact-onset mode needs one output difficulty per upstream osu map")
    torch.set_num_threads(2)
    try:
        model = load_model(checkpoint)
    except (ValueError, OSError, RuntimeError) as exc:
        raise GenerationError(f"onset-flow generation failed: {exc}") from exc
    maps, ledgers, records = {}, {}, {}
    output_meta = output_bpm = None
    for name in names:
        spec = diff_spec(name)
        meta, objects, bpm, _ = parse_osu(sources[name])
        onsets = source_onsets(objects, meta.get("_timing"), bpm)
        if not onsets:
            raise GenerationError(f"{name}: upstream osu map contains no playable onsets")
        try:
            notes = decode(onsets, model)
        except (ValueError, OSError, RuntimeError) as exc:
            raise GenerationError(f"{name}: onset-flow generation failed: {exc}") from exc
        problems = check(notes, bpm, cap=float("inf"))
        if len({(n['t'], n['col'], n['layer']) for n in notes}) != len(notes):
            problems.append("duplicate block occupancy")
        if problems:
            raise GenerationError(f"{name}: " + "; ".join(problems))
        if output_meta is None:
            output_meta, output_bpm = meta, bpm
        maps[name] = (notes, [], spec)
        ledgers[name] = onsets
        records[name] = dict(
            source_osu=str(sources[name]),
            source_osu_sha256=sha256(Path(sources[name]).read_bytes()).hexdigest(),
            source_onsets=len(onsets),
            peak_nps=peak_nps([n['t'] for n in notes if n['dir'] != 8]),
            density_policy_diagnostics=check(notes, bpm, cap=NPS_CAP * spec["scale"]))
        print(f"{spec['label']}: {len(onsets)} exact osu onsets, {len(notes)} learned BS blocks")
    # All charts use milliseconds against the same unshifted audio. Export maps
    # them to one common Info BPM even when upstream timing points differ.
    write_map(maps, output_bpm, output_meta, audio_path, outdir)
    outdir = Path(outdir)
    (outdir / "source-onsets.json").write_text(json.dumps(
        ledgers[names[0]] if len(names) == 1 else ledgers, indent=1))
    metadata = dict(
        mode="osu-onsets", schema=SCHEMA,
        checkpoint_sha256=sha256(Path(checkpoint).read_bytes()).hexdigest(),
        energy_source="osu unique-onset density, centered 2s window",
        lead_shift_ms=0, outro_end_ms=None,
        grid_decision="exact upstream timestamps; no snapping",
        difficulties=names, per_difficulty=records)
    if len(names) == 1:
        metadata.update(records[names[0]])
    (outdir / "generation.json").write_text(json.dumps(metadata, indent=1))
    print(f"Exported {len(names)} difficulties -> {outdir}")
    return 0


def main(osu_path, audio_path, outdir, diffs=None, *, onset_checkpoint=None):
    if onset_checkpoint is not None:
        return main_onsets(osu_path, audio_path, outdir, diffs, onset_checkpoint)
    meta, objects, bpm, offset = parse_osu(osu_path)
    outdir = Path(outdir)
    if diffs is None:
        # regen: preserve whatever difficulty set the folder already has
        # (file stems, so ExpertPlus2 etc. round-trip through _customData)
        diffs = "ExpertPlus"
        if (outdir / "Info.dat").exists():
            try:
                info = json.loads((outdir / "Info.dat").read_text())
                names = [Path(dm["_beatmapFilename"]).stem
                         for s in info["_difficultyBeatmapSets"]
                         for dm in s["_difficultyBeatmaps"]]
                diffs = ",".join(names) or diffs
            except Exception:
                pass
    names = selected_difficulties(diffs)
    if meta["Title"].startswith("Unknown"):
        meta["Title"] = outdir.name.replace("_", " ").title()
    groomed = GROOM_PT.exists() and GROOM_PT.with_name("flow.pt").exists()
    maps, failed = {}, []
    for name in names:
        spec = diff_spec(name)
        if groomed:
            print(f"-- {spec['label']}")
            try:
                # Q5 scoped release: the installed quality policy routes
                # generation through the evaluated bundle (motion+pairs
                # claims only); "b0"/absent policy = unchanged production
                from quality_policy import active_policy, bundle_song
                if active_policy() == "bundle-v1":
                    notes, walls, _info = bundle_song(
                        objects, bpm, offset, audio_path, name,
                        timing=meta.get("_timing"))
                else:
                    notes, walls = convert_groomed(objects, bpm, offset,
                                                   audio_path, name,
                                                   timing=meta.get("_timing"))
            except GenerationError as e:
                # hard-invalid: report and DO NOT export this difficulty
                print(f"GENERATION FAILED: {e}")
                failed.append(name)
                continue
        elif name == "ExpertPlus":  # legacy path knows one difficulty
            notes = convert(objects, bpm)
            walls = lean_walls(notes, bpm)
        else:
            print(f"skip {name}: models missing, legacy is ExpertPlus-only")
            continue
        maps[name] = (notes, walls, spec)
    if not maps:  # nothing valid to write — do not emit an empty/partial map
        raise GenerationError(
            f"no difficulty produced a valid chart ({', '.join(failed)})")
    # one lead shift / outro cut across all difficulties: same audio file
    t0 = min((n[0]["t"] for n, _, _ in maps.values() if n), default=0)
    shift = LEAD_MS - t0 if (t0 < MIN_LEAD_MS or t0 > LEAD_MS) else 0
    end, rc, mode = 0, 0, "groomed" if groomed else "legacy"
    for name, (notes, walls, spec) in maps.items():
        if shift:
            for x in list(notes) + list(walls):
                x["t"] += shift
            walls = [w for w in walls if w["t"] + w["dur"] > 0]
            for w in walls:  # intro walls clipped by a lead trim
                if w["t"] < 0:
                    w["dur"] += w["t"]
                    w["t"] = 0.0
            maps[name] = (notes, walls, spec)
        end = max(end, max([n["t"] for n in notes]
                           + [w["t"] + w["dur"] for w in walls], default=0))
        problems = check(notes, bpm, walls, cap=NPS_CAP * spec["scale"])
        dur = (notes[-1]["t"] - notes[0]["t"]) / 1000 if len(notes) > 1 else 1
        print(f"{spec['label']}: {len(notes)} notes, {len(walls)} lean walls "
              f"({mode}" + (f", lead {shift / 1000:+.1f}s" if shift else "")
              + f"), bpm {bpm:.1f}, avg nps {len(notes) / dur:.1f}")
        if problems:
            # FAIL CLOSED (QA architecture 2026-09-24): a difficulty that
            # fails the post-shift whole-map check is never exported — no
            # human is the safety net downstream anymore
            print("CHECKS FAILED:\n  " + "\n  ".join(problems))
            failed.append(name)
            rc = 1
    for name in failed:
        maps.pop(name, None)
    if failed:  # some diffs were dropped; the folder is a partial set
        rc = 1
        print(f"dropped invalid difficulties: {', '.join(failed)}")
    if not maps:
        raise GenerationError(
            "no difficulty passed final checks — nothing exported "
            f"({', '.join(failed)})")
    write_map(maps, bpm, meta, audio_path, outdir, shift, end + TAIL_MS)
    if groomed and maps:  # persist the timing/grid decision beside the job
        try:
            _, _, _, _, grid = grid_steps(objects, bpm, offset,
                                          meta.get("_timing"))
            (outdir / "generation.json").write_text(json.dumps({
                "source_osu": str(osu_path),
                "raw_timing": meta.get("_timing"),
                "grid_decision": getattr(grid, "decision", None),
                "export_bpm": round(bpm, 3),
                "lead_shift_ms": shift, "outro_end_ms": end + TAIL_MS,
                "difficulties": list(maps), "dropped": failed,
            }, indent=1))
        except Exception as e:
            print(f"(generation.json not written: {e})")
    print(("all checks passed -> " if rc == 0 else "-> ") + str(outdir))
    return rc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osu_path")
    parser.add_argument("audio_path")
    parser.add_argument("outdir")
    parser.add_argument("diffs", nargs="?")
    parser.add_argument("--onset-flow", dest="onset_checkpoint", type=Path)
    args = parser.parse_args()
    sys.exit(main(args.osu_path, args.audio_path, args.outdir, args.diffs,
                  onset_checkpoint=args.onset_checkpoint))
