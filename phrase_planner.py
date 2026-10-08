"""Q3: learned local musical planning — window dataset construction.

Fixed 8-beat planning windows (32 grid steps) with a 32-beat (4-window)
context. INPUTS per window come from audio + MI evidence + local tempo +
the requested workload + the previous window's plan + a neutral style vector.
TARGETS come from the paired HUMAN chart on the same audio (clock alignment by
construction — same file): measurable window choices only, no verse/chorus or
melodic-focus labels (those slots stay empty).

Everything is grid-aligned by reusing groom.load_map_all's tensors:
pres[T,2] per-hand occupancy, events geometry, afeat[T,4] audio channels.
MI gen.osu onsets are mapped onto the SAME grid via grid times.

Dataset build is chunked and checkpointed per family (<1h rule): each family
writes experiments/quality-v1/q3-planner/windows/<fam>.pt atomically.
"""
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
MI_ROOT = ROOT / "experiments" / "quality-v1" / "q3-mi"
OUT = ROOT / "experiments" / "quality-v1" / "q3-planner" / "windows"
WIN = 32                 # 8 beats * 4 steps
CTX_WINDOWS = 4          # 32-beat context
SEED = 20260921

# target vector layout (bounded scalars unless noted)
TARGETS = ["occ_rate",          # occupied-step share in the window
           "coincidence",       # both-hand share of occupied steps
           "cad_l", "cad_r",    # per-hand swings/s
           "burst",             # peak 2s swing rate inside the window
           "travel",            # mean per-hand transition cell distance
           "axis_vert", "axis_horiz", "axis_diag",   # head direction mix
           "motif_conc"]        # max 4-gram share (window tokens)
BINARY = ["quiet",              # window is (near) silent in the human chart
          "accent_occupied"]    # top audio-accent step carries a note


def _dir_family(d):
    from parity import DIR_VEC
    if d == 8:
        return None
    vx, vy = DIR_VEC[d]
    if vx == 0:
        return "vert"
    if vy == 0:
        return "horiz"
    return "diag"


def window_targets(pres, events, step_times, w0, w1):
    """Measurable human choices for steps [w0, w1)."""
    span_s = max((step_times[min(w1, len(step_times) - 1)]
                  - step_times[w0]) / 1000.0, 1e-6)
    occ = pres[w0:w1].any(dim=1)
    occ_rate = float(occ.float().mean())
    both = pres[w0:w1].all(dim=1)
    coincidence = float(both.sum()) / max(1, int(occ.sum()))
    ev = [e for e in events if w0 <= e[0] < w1]
    cad = {0: 0, 1: 0}
    for e in ev:
        cad[e[1]] += 1
    times = sorted(step_times[e[0]] for e in ev)
    burst = 0
    for i, t in enumerate(times):
        j = i
        while j < len(times) and times[j] <= t + 2000:
            j += 1
        burst = max(burst, j - i)
    lastc = {}
    travel, ntr = 0.0, 0
    for e in sorted(ev):
        s, h, d, c, l = e[0], e[1], e[2], e[3], e[4]
        p = lastc.get(h)
        if p is not None:
            travel += abs(c - p[0]) + abs(l - p[1])
            ntr += 1
        lastc[h] = (c, l)
    fams = [f for f in (_dir_family(e[2]) for e in ev) if f]
    n = max(1, len(fams))
    toks = [(e[1], _dir_family(e[2])) for e in sorted(ev)]
    grams = {}
    for i in range(len(toks) - 3):
        g = tuple(toks[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    motif = max(grams.values(), default=0) / max(1, len(toks) - 3)
    return {"occ_rate": occ_rate, "coincidence": coincidence,
            "cad_l": cad[0] / span_s / 2, "cad_r": cad[1] / span_s / 2,
            "burst": burst / 24.0,
            "travel": min(1.0, (travel / max(1, ntr)) / 5.0),
            "axis_vert": sum(f == "vert" for f in fams) / n,
            "axis_horiz": sum(f == "horiz" for f in fams) / n,
            "axis_diag": sum(f == "diag" for f in fams) / n,
            "quiet": 1.0 if occ_rate < 0.02 else 0.0,
            "motif_conc": motif}


def accent_target(afeat, pres, w0, w1):
    """Does the loudest-onset step in the window carry a note?"""
    seg = afeat[w0:w1, 0]
    if len(seg) == 0:
        return 0.0
    top = int(torch.argmax(seg))
    return float(pres[w0 + top].any())


def onset_counts(objects, step_times):
    """Evidence onset count per grid step (nearest-step mapping)."""
    counts = torch.zeros(len(step_times))
    st = list(step_times)
    import bisect
    for o in objects:
        t = o["t"] if isinstance(o, dict) else (
            o[0] if isinstance(o, (list, tuple)) else o)
        i = bisect.bisect_left(st, t)
        if i > 0 and (i == len(st) or abs(st[i - 1] - t) <= abs(st[i] - t)):
            i -= 1
        if 0 <= i < len(counts):
            counts[i] += 1
    return counts


def mi_onsets_on_grid(mi_osu, step_times):
    """MI evidence onset count per grid step (nearest-step mapping)."""
    from convert import parse_osu
    _m, objects, _bpm, _off = parse_osu(mi_osu)
    return onset_counts(objects, step_times)


def window_inputs(afeat, mi_counts, step_times, w0, w1, bpm_local):
    seg_a = afeat[w0:w1]
    seg_m = mi_counts[w0:w1]
    span_s = max((step_times[min(w1, len(step_times) - 1)]
                  - step_times[w0]) / 1000.0, 1e-6)
    occ = (seg_m > 0).float()
    gaps = torch.diff(torch.nonzero(occ).flatten().float()) \
        if int(occ.sum()) > 1 else torch.zeros(1)
    return torch.tensor([
        *(float(seg_a[:, k].mean()) for k in range(seg_a.shape[1])),
        *(float(seg_a[:, k].max()) for k in range(seg_a.shape[1])),
        float(occ.mean()),                       # MI occupancy share
        float(seg_m.sum()) / span_s / 10.0,      # MI onset rate (scaled)
        float(gaps.mean()) / WIN,                # MI spacing
        float(gaps.std()) / WIN if len(gaps) > 1 else 0.0,
        bpm_local / 300.0,
    ])


def human_window_targets(corpus_dir):
    """Human window-target matrix y[W, 12] + bpm for one corpus dir — the
    target half of build_family, no MI evidence needed (style descriptors,
    trajectory references). Returns (y, bpm) or (None, reason)."""
    import groom
    corpus_dir = Path(corpus_dir)
    samples = groom.load_map_all(corpus_dir)
    if not samples:
        return None, "no_human_chart"
    name = sorted(samples, key=lambda n: len(samples[n][2]))[-1]
    inp, pres, events, _wl, _wr = samples[name][:5]
    T = len(pres)
    afeat = inp[:, :4]
    info_p = next((p for p in corpus_dir.iterdir()
                   if p.name.lower() == "info.dat"), None)
    if info_p is None:
        return None, "no_info_dat"
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    bpm = info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm")
    if not bpm:
        return None, "no_bpm"
    step_ms = 60000.0 / float(bpm) / 4
    step_times = [s * step_ms for s in range(T)]
    ys = []
    for w0 in range(0, T - WIN + 1, WIN):
        w1 = w0 + WIN
        tg = window_targets(pres, events, step_times, w0, w1)
        tg["accent_occupied"] = accent_target(afeat, pres, w0, w1)
        ys.append(torch.tensor([tg[k] for k in TARGETS]
                               + [tg[k] for k in BINARY]))
    if len(ys) < CTX_WINDOWS + 1:
        return None, "too_short"
    return torch.stack(ys), float(bpm)


def build_family(fam_dir, out_path):
    """One family -> {x[W,F], y[W,K], meta}. Atomic write. Returns status."""
    import groom
    prov = json.loads((fam_dir / "provenance.json").read_text())
    corpus_dir = Path(prov["corpus_dir"])
    samples = groom.load_map_all(corpus_dir)
    if not samples:
        return "no_human_chart"
    # highest available Standard tier = densest reference
    name = sorted(samples, key=lambda n: len(samples[n][2]))[-1]
    inp, pres, events, _wl, _wr = samples[name][:5]
    T = len(pres)
    afeat = inp[:, :4]                       # the four audio channels
    # grid times: load_map_all built features on the longest diff's grid at
    # uniform beat spacing; reconstruct from the corpus chart's bpm
    bpm = prov.get("bpm") or 120.0
    info = json.loads(next(p for p in corpus_dir.iterdir()
                           if p.name.lower() == "info.dat")
                      .read_text(encoding="utf-8-sig"))
    bpm = info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm", bpm)
    step_ms = 60000.0 / bpm / 4
    step_times = [s * step_ms for s in range(T)]
    mi = mi_onsets_on_grid(fam_dir / "gen.osu", step_times)
    xs, ys = [], []
    for w0 in range(0, T - WIN + 1, WIN):
        w1 = w0 + WIN
        tg = window_targets(pres, events, step_times, w0, w1)
        tg["accent_occupied"] = accent_target(afeat, pres, w0, w1)
        xs.append(window_inputs(afeat, mi, step_times, w0, w1, bpm))
        ys.append(torch.tensor([tg[k] for k in TARGETS]
                               + [tg[k] for k in BINARY]))
    if len(xs) < CTX_WINDOWS + 1:
        return "too_short"
    payload = {"x": torch.stack(xs), "y": torch.stack(ys),
               "meta": {"family": prov["family"], "split": prov["split"],
                        "tier": name, "bpm": float(bpm), "T": T,
                        "n_windows": len(xs), "targets": TARGETS,
                        "binary": BINARY}}
    tmp = out_path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(out_path)
    return "ok"


def build_all(limit=10 ** 6):
    """Chunk-safe dataset build: idempotent per family, atomic writes."""
    OUT.mkdir(parents=True, exist_ok=True)
    log, n = [], 0
    for fam_dir in sorted(MI_ROOT.iterdir()):
        if not (fam_dir / "gen.osu").exists():
            continue
        out = OUT / (fam_dir.name + ".pt")
        if out.exists():
            continue
        if n >= limit:
            break
        try:
            status = build_family(fam_dir, out)
        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"
        log.append({"family": fam_dir.name, "status": status})
        n += 1
        print(f"  [{status[:24]:24s}] {fam_dir.name}", flush=True)
    (OUT.parent / "build_log.json").write_text(json.dumps(log, indent=1))
    return log


# --- shared trajectory descriptors (human vs emitted, grid-free) ------------
# The Q3 D trajectory gate compares generated output to the PAIRED HUMAN
# chart per 8-beat human window using seconds-based components only
# (cadence L/R, 2s burst, coincidence) — occ_rate is grid-step-based and
# incommensurable across MI/human BPM encodings, so it is excluded (disclosed).

TRAJ_COMPONENTS = ["cad_l", "cad_r", "burst", "coincidence"]


def traj_descriptors(notes_th, w0_ms, w1_ms):
    """The 4 seconds-based window components from (t_ms, hand) pairs —
    definitions mirror window_targets' cadence/burst/coincidence exactly."""
    span_s = max((w1_ms - w0_ms) / 1000.0, 1e-6)
    win = [(t, h) for t, h in notes_th if w0_ms <= t < w1_ms]
    cad = {0: 0, 1: 0}
    for _t, h in win:
        cad[h] += 1
    times = sorted(t for t, _h in win)
    burst = 0
    for i, t in enumerate(times):
        j = i
        while j < len(times) and times[j] <= t + 2000:
            j += 1
        burst = max(burst, j - i)
    by_t = {}
    for t, h in win:
        by_t.setdefault(t, set()).add(h)
    coinc = (sum(1 for hs in by_t.values() if len(hs) == 2) / len(by_t)
             if by_t else 0.0)
    # occ_per_s: grid-independent occupancy DIAGNOSTIC (review R2 ruling) —
    # distinct occupied timestamps per second; not part of the gate error
    return {"cad_l": cad[0] / span_s / 2, "cad_r": cad[1] / span_s / 2,
            "burst": burst / 24.0, "coincidence": coinc,
            "occ_per_s": len(by_t) / span_s}


def human_trajectory(corpus_dir):
    """Per-8-beat-window components of the densest Standard human chart in a
    corpus dir, plus the window bounds in ms (the alignment reference).
    Returns (windows_ms, [desc per window]) or (None, reason)."""
    import groom
    from pathlib import Path
    d = Path(corpus_dir)
    samples = groom.load_map_all(d)
    if not samples:
        return None, "no_human_chart"
    name = sorted(samples, key=lambda n: len(samples[n][2]))[-1]
    _inp, pres, events, _wl, _wr = samples[name][:5]
    T = len(pres)
    info_p = next((p for p in d.iterdir() if p.name.lower() == "info.dat"), None)
    if info_p is None:
        return None, "no_info_dat"
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    bpm = info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm")
    if not bpm:
        return None, "no_bpm"
    step_ms = 60000.0 / float(bpm) / 4
    notes_th = [(e[0] * step_ms, e[1]) for e in events]
    windows = [(w0 * step_ms, (w0 + WIN) * step_ms)
               for w0 in range(0, T - WIN + 1, WIN)]
    if not windows:
        return None, "too_short"
    return windows, [traj_descriptors(notes_th, a, b) for a, b in windows]


def trajectory_error(notes_th, windows_ms, human_desc):
    """Mean |generated - human| over windows and TRAJ_COMPONENTS, with the
    generated notes binned into the HUMAN window bounds."""
    errs = []
    for (a, b), hd in zip(windows_ms, human_desc):
        gd = traj_descriptors(notes_th, a, b)
        errs += [abs(gd[k] - hd[k]) for k in TRAJ_COMPONENTS]
    return sum(errs) / len(errs)


# --- Q3 inference: planner trajectory + bounded workload scheduler ----------
# Spec §6: fixed three-choice occupancy multiplier {0.9,1.0,1.1} per window,
# beam width 4, at most two complete reschedules per song, starting from the
# selected attempt. B0 quiet/rest identities protected; original occupied
# accent cells kept; secondary cells only where MI evidence supports them.
# Cheap beam estimates (the geometry-free scheduling mirror) are never final
# gates — the achieved workload is measured on the actual emitted output.

MULTS = (0.9, 1.0, 1.1)
BEAM_WIDTH = 4
MAX_RESCHEDULES = 2
# candidate per-window threshold offsets probed per multiplier (sign-limited)
OFFSET_LADDER = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35)


def train_mean_targets():
    """The per-tier constant plan: mean human window-target vector over the
    TRAIN split of the windows dataset (also the planner's request vector)."""
    ys = []
    for p in sorted(OUT.glob("*.pt")):
        d = torch.load(p)
        if d["meta"]["split"] == "train":
            ys.append(d["y"])
    assert ys, "windows dataset missing — run build_all first"
    return torch.cat(ys).mean(dim=0)


def load_planner(ckpt_path):
    from eval.quality_train import Planner
    d = torch.load(ckpt_path)
    model = Planner(d["n_in"])
    model.load_state_dict(d["state"])
    model.eval()
    return model, d["norm"]


def song_window_inputs(afeat, mi_counts, step_times, bpm):
    """Per-FULL-window input features on the decode grid; the partial tail
    window (if any) gets no planner features and is never rescheduled."""
    T = len(step_times)
    xs = []
    for w0 in range(0, T - WIN + 1, WIN):
        xs.append(window_inputs(afeat, mi_counts, step_times, w0, w0 + WIN, bpm))
    return torch.stack(xs) if xs else torch.zeros(0, 13)


def planner_trajectory(model, norm, x, workload_req):
    """Autoregressive rollout: the previous-window plan input is the model's
    OWN prediction (fit 1's train/inference mismatch — teacher-forced targets
    at train; fit 2's curriculum is the declared branch if rollout fails)."""
    from eval.quality_train import N_SCALAR, N_BIN, STYLE_DIM
    mu, sd = norm
    prev = torch.zeros(N_SCALAR + N_BIN)
    hist, outs = [], []
    for w in range(len(x)):
        f = torch.cat([x[w], workload_req, prev, torch.zeros(STYLE_DIM)])
        hist.append((f - mu) / sd)
        with torch.no_grad():
            sc, bl = model(torch.stack(hist)[None])
        prev = torch.cat([sc[0, -1], (bl[0, -1] > 0).float()])
        outs.append(prev.clone())
    return torch.stack(outs) if outs else torch.zeros(0, N_SCALAR + N_BIN)


def constant_trajectory(n_windows, workload_req):
    """Equal-budget control: the constant per-tier plan through the SAME
    scheduler, so beam search alone cannot claim musical learning."""
    from eval.quality_train import N_BIN
    row = torch.cat([workload_req, torch.zeros(N_BIN)])
    return row.expand(n_windows, -1).clone()


def _offset_at(s, c, merg, spr, protect):
    """Per-step threshold offset rule: never raise thr on a protected accent
    step; lower (add notes) only where MI evidence or a sprinkle supports it."""
    if c > 0.0 and s in protect:
        return 0.0
    if c < 0.0 and not (merg[s] or spr[s]):
        return 0.0
    return c


def _mirror_window(presl, thrl, keepl, mergl, sprl, restl, w0, w1, last,
                   c=0.0, protect=frozenset()):
    """Occupied-step count the decode loop would schedule in [w0,w1) at
    window offset c, carrying per-hand recency — the cheap beam estimate
    (mirrors groom._sched_windows' hands logic; never a final gate)."""
    from groom import MIN_GAP
    l = list(last)
    occ = 0
    for s in range(w0, w1):
        if restl[s]:
            continue
        cs = _offset_at(s, c, mergl, sprl, protect)
        t, k = thrl[s] + cs, keepl[s] + cs
        p = presl[s]
        hands = [h for h in (0, 1) if p[h] > t and s - l[h] >= MIN_GAP]
        if mergl[s] and not hands:
            h = int(p[1] > p[0])
            if s - l[h] < MIN_GAP:
                h = 1 - h
            if (p[h] >= k or sprl[s]) and s - l[h] >= MIN_GAP:
                hands = [h]
        if hands:
            occ += 1
            for h in hands:
                l[h] = s
    return occ, tuple(l)


def _pick_offset(arrays, w0, w1, last, desired, protect):
    """Smallest-|c| ladder offset whose mirrored count lands closest to
    `desired`. Deterministic; returns (c, count, evals)."""
    base, _ = _mirror_window(*arrays, w0, w1, last)
    best = (abs(base - desired), 0.0, 0.0, base)      # (gap, |c|, c, count)
    evals = 1
    if base != desired:
        sign = 1.0 if desired < base else -1.0        # raise thr to shrink
        for mag in OFFSET_LADDER:
            c = sign * mag
            n, _ = _mirror_window(*arrays, w0, w1, last, c=c, protect=protect)
            evals += 1
            cand = (abs(n - desired), mag, c, n)
            if cand[:2] < best[:2]:
                best = cand
    return best[2], best[3], evals


def beam_schedule(arrays, windows, occ_targets, b0_occ, protect):
    """Beam over per-window multipliers {0.9,1.0,1.1} (width 4) minimizing
    |mirrored occupancy share - planner target|. Partial tail windows and
    windows without a target stay at 1.0. Returns ranked schedules
    [{multipliers, offsets, est_err}] and the mirror-eval count."""
    from groom import MIN_GAP
    states = [(0.0, (-MIN_GAP, -MIN_GAP), (), ())]
    evals = 0
    for wi, (w0, w1) in enumerate(windows):
        tgt = occ_targets[wi] if wi < len(occ_targets) else None
        full = (w1 - w0) == WIN
        nxt = []
        for err, last, ms, cs in states:
            for m in (MULTS if (full and tgt is not None) else (1.0,)):
                if m == 1.0:
                    c = 0.0
                    n, last2 = _mirror_window(*arrays, w0, w1, last)
                    evals += 1
                else:
                    desired = max(0, min(w1 - w0, round(m * b0_occ[wi])))
                    c, n, ev = _pick_offset(arrays, w0, w1, last, desired,
                                            protect)
                    evals += ev
                    _, last2 = _mirror_window(*arrays, w0, w1, last, c=c,
                                              protect=protect)
                    evals += 1
                e = abs(n / (w1 - w0) - float(tgt)) if tgt is not None else 0.0
                nxt.append((err + e, last2, ms + (m,), cs + (c,)))
        states = sorted(nxt, key=lambda t: (t[0], t[2]))[:BEAM_WIDTH]
    return [{"multipliers": list(ms), "offsets": list(cs), "est_err": err}
            for err, _l, ms, cs in states], evals


def _emitted_occ(raw, windows):
    """ACTUAL per-window occupied-step counts of an emitted chart — the only
    admissible achieved-workload measure (never scheduler shadow state)."""
    steps = {s for s, *_ in raw}
    return [sum(1 for s in steps if w0 <= s < w1) for w0, w1 in windows]


def _traj_err(occ_counts, windows, occ_targets):
    errs = [abs(occ_counts[i] / (w1 - w0) - float(occ_targets[i]))
            for i, (w0, w1) in enumerate(windows)
            if i < len(occ_targets) and occ_targets[i] is not None]
    return sum(errs) / len(errs) if errs else 0.0


def schedule_workload(objects, bpm, offset, audio_path, *, ckpt=None,
                      mode="planner", request=None, diff="ExpertPlus",
                      thin=True, b0=None):
    """Full bounded-scheduler driver for one song.

    1. B0 production decode (convert_groomed, replay off) -> selected attempt.
    2. Target trajectory: learned planner rollout (mode="planner") or the
       constant per-tier plan (mode="constant", the equal-budget control).
    3. Beam over window multipliers; up to MAX_RESCHEDULES complete decodes
       through the plan-injection seam (same seed/temp/rate_scale as the
       selected attempt; frozen B0 rest mask; modified thr/keep).
    4. Verify protections + trajectory error on the EMITTED output; then the
       accepted geometry policy (Q1-V2 repair) + whole-chart admission vs B0.
    A failed request is an explicit unachieved result with B0 retained —
    never a success label.
    """
    import time as _time
    from convert import (NPS_CAP, check, convert_groomed, diff_spec,
                         grid_steps)
    import groom
    from quality_repair import RepairConfig, repair_winner

    spec = diff_spec(diff)
    band = spec["band"]
    cap = NPS_CAP * spec["scale"]
    steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset, thin=thin)
    step_times = [grid.time(s) for s in range(T)]
    afeat = groom.cached_audio_features(audio_path, step_times) \
        if audio_path else None
    t0 = _time.perf_counter()
    if b0 is not None:                       # share one B0 decode across arms
        recs, b0_s = b0
    else:
        recs = []
        convert_groomed(objects, bpm, offset, audio_path=audio_path,
                        diff=diff, replay_mode="off", collect=recs,
                        thin=thin, calibrate=True)
        b0_s = _time.perf_counter() - t0
    sel = next(r for r in recs if r["selected"])
    raw0, walls0, tr = sel["notes"], sel["walls"], sel["trace"]
    windows = [(w0, min(w0 + WIN, T)) for w0 in range(0, T, WIN)]

    # targets
    if request is None:
        request = train_mean_targets()[:10]
    if mode == "planner":
        assert afeat is not None, "planner mode needs audio features"
        model, norm = load_planner(ckpt)
        mi = onset_counts(objects, step_times)
        x = song_window_inputs(afeat, mi, step_times, bpm)
        traj = planner_trajectory(model, norm, x, request)
    elif mode == "constant":
        n_full = sum(1 for w0, w1 in windows if w1 - w0 == WIN)
        traj = constant_trajectory(n_full, request)
    else:
        raise ValueError(mode)
    occ_targets = [float(traj[i, 0]) if i < len(traj) else None
                   for i in range(len(windows))]

    # protections from B0 (rest identities frozen; occupied accent cells kept)
    restl = tr["rest_mask"].tolist()
    protected_rest = [s for s in range(T) if restl[s]]
    occ0 = {s for s, *_ in raw0}
    protect_accent = set()
    if afeat is not None:
        for w0, w1 in windows:
            seg = afeat[w0:w1, 0]
            if len(seg):
                a = w0 + int(torch.argmax(seg))
                if a in occ0:
                    protect_accent.add(a)

    arrays = (tr["pres"].tolist(), tr["thr"].tolist(), tr["keep"].tolist(),
              [bool(v) for v in tr["merged"].tolist()],
              [bool(v) for v in tr["sprinkle"].tolist()], restl)
    b0_occ = _emitted_occ(raw0, windows)
    b0_err = _traj_err(b0_occ, windows, occ_targets)
    schedules, mirror_evals = beam_schedule(arrays, windows, occ_targets,
                                            b0_occ, protect_accent)

    thr0, keep0 = tr["thr"], tr["keep"]
    log = []
    chosen = None
    n_resched = 0
    for cand in schedules[:MAX_RESCHEDULES]:
        n_resched += 1
        thr1, keep1 = thr0.clone(), keep0.clone()
        for wi, (w0, w1) in enumerate(windows):
            c = cand["offsets"][wi]
            if c:
                for s in range(w0, w1):
                    cs = _offset_at(s, c, arrays[3], arrays[4], protect_accent)
                    thr1[s] += cs
                    keep1[s] += cs
        raw1, walls1 = groom.groom_notes(
            steps, T, step_ms, off2, afeat=afeat, seed=sel["seed"],
            temp=sel["temp"], rate_scale=sel["rate_scale"], drate=spec["cond"],
            band_scale=spec["scale"], band=band, replay_mode="off", grid=grid,
            rest_mask_in=tr["rest_mask"], thr_vectors=(thr1, keep1),
            rest_policy="off", density_adjust=False)
        occ1s = {s for s, *_ in raw1}
        rest_viol = sorted(occ1s & set(protected_rest))
        accent_miss = sorted(protect_accent - occ1s)
        occ1 = _emitted_occ(raw1, windows)
        err1 = _traj_err(occ1, windows, occ_targets)
        entry = {"multipliers": cand["multipliers"], "est_err": cand["est_err"],
                 "emitted_err": err1, "rest_violations": rest_viol,
                 "accent_missing": accent_miss}
        log.append(entry)
        if not rest_viol and not accent_miss and err1 <= b0_err + 1e-9:
            chosen = (cand, raw1, walls1, occ1, err1)
            break

    result = {
        "mode": mode, "b0_err": b0_err, "b0_occ": b0_occ,
        "occ_targets": occ_targets,
        "baseline": {"protected_rest_ids": protected_rest,
                     "protected_accent_ids": sorted(protect_accent),
                     "notes": raw0, "walls": walls0},
        "accounting": {"complete_reschedules": n_resched,
                       "mirror_evals": mirror_evals, "geometry_calls": 0,
                       "b0_s": b0_s},
        "reschedule_log": log, "achieved": False, "reasons": [],
    }
    if chosen is None:
        result["reasons"].append("no reschedule met protections and error bar")
        result["outcome"] = {"notes": raw0, "walls": walls0, "source": "b0",
                             "occupied_ids": sorted(occ0),
                             "occ_by_window": b0_occ, "err": b0_err,
                             "protected_rest_ids": protected_rest}
        result["elapsed_s"] = _time.perf_counter() - t0
        return result

    cand, raw1, walls1, occ1, err1 = chosen
    # accepted geometry policy (Q1-V2 repair), then whole-chart admission
    rr = repair_winner(raw1, walls1, grid, bpm, cap,
                       RepairConfig(variant="paired-direction"))
    result["accounting"]["geometry_calls"] = 1
    raw2 = rr.raw if rr is not None else raw1
    notes_d = [{"t": grid.time(s), "hand": h, "col": col, "layer": lay,
                "dir": d} for s, h, col, lay, d in raw2]
    walls_d = [{"t": grid.time(s0), "dur": grid.time(s0 + ln) - grid.time(s0),
                "col": col} for s0, ln, col in walls1]
    problems = check(notes_d, bpm, walls_d, cap)
    gate_fails = list(problems)
    if not gate_fails:
        import motion as motion_mod
        from eval.clean_rhythm import _cand_metrics
        from eval.phrase_plan import _cand_gates
        beat_ms = 60000.0 / bpm
        ma = _cand_metrics(raw0, motion_mod.report(
            [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw0]),
            grid, beat_ms)
        mb = _cand_metrics(raw2, motion_mod.report(
            [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw2]),
            grid, beat_ms)
        gate_fails = _cand_gates(ma, mb, band)
    if gate_fails:
        result["reasons"].append(f"admission failed: {gate_fails}")
        result["outcome"] = {"notes": raw0, "walls": walls0, "source": "b0",
                             "occupied_ids": sorted(occ0),
                             "occ_by_window": b0_occ, "err": b0_err,
                             "protected_rest_ids": protected_rest}
    else:
        result["achieved"] = True
        result["schedule"] = cand
        result["outcome"] = {"notes": raw2, "walls": walls1,
                             "source": "reschedule",
                             "occupied_ids": sorted({s for s, *_ in raw2}),
                             "occ_by_window": occ1, "err": err1,
                             "protected_rest_ids": protected_rest,
                             "repaired": len(rr.changed_ids) if rr else 0}
    result["elapsed_s"] = _time.perf_counter() - t0
    return result


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 6
    log = build_all(lim)
    ok = sum(1 for r in log if r["status"] == "ok")
    print(f"windows built: {ok}/{len(log)} new families ok")
