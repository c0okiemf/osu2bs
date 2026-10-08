"""No-op decode-plan adapter (phrase-plan phase).

Extends the diagnostic phrase-draft schema (eval/audio_structure.phrase_draft —
per-section, seconds, semantic slots left empty) with a concrete DECODE PLAN:
the baseline TIMING (the TimeGrid), RESTS (the frozen rest mask) and SCHEDULING
decisions (the per-step density thr/keep vectors) that groom_notes derives during
a normal decode. The plan is then fed straight back into flow decoding via the
existing rest_mask_in / thr_vectors injection seam.

It is a pure data path, not a behavior change: decoding FROM a captured plan is
bit-identical to the baseline decode (same grid, same seed, frozen rest/thr/keep
skip the recompute but hold the same values). This establishes the plan->decode
plumbing a future phrase planner would use, with zero effect on generation. No
training; no musical-structure claims (the phrase-draft's semantic slots stay
empty and untouched).

Baseline decisions are read from groom_notes' `trace` (rest_mask/thr/keep, the
"baseline derivation" outputs); see docs and groom.groom_notes.
"""
import json

PLAN_VERSION = 1


def _setup(objects, bpm, offset, audio_path, diff, thin, timing):
    """Reproduce convert_groomed's per-song decode setup (grid + audio features
    + difficulty spec) so capture and replay see identical inputs."""
    from convert import grid_steps, diff_spec
    from groom import cached_audio_features
    steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset, timing,
                                               thin=thin)
    afeat = None
    if audio_path:
        afeat = cached_audio_features(audio_path, [grid.time(s) for s in range(T)])
    return steps, T, step_ms, off2, grid, afeat, diff_spec(diff)


def _groom_kwargs(spec, seed, temp, rate_scale):
    import groom
    return dict(afeat=None, seed=seed,
                temp=groom.TEMP if temp is None else temp,
                rate_scale=rate_scale, drate=spec["cond"],
                band_scale=spec["scale"], band=spec["band"], replay_mode="off")


def capture_plan(objects, bpm, offset, audio_path, *, seed=0, temp=None,
                 diff="ExpertPlus", thin=True, timing=None, rate_scale=1.0):
    """Baseline decode with a trace; return (plan_dict, raw_notes, walls).
    plan_dict is JSON-serializable and self-contained."""
    import groom
    steps, T, step_ms, off2, grid, afeat, spec = _setup(
        objects, bpm, offset, audio_path, diff, thin, timing)
    tr = {}
    kw = _groom_kwargs(spec, seed, temp, rate_scale)
    kw["afeat"] = afeat
    raw, walls = groom.groom_notes(steps, T, step_ms, off2, trace=tr, grid=grid,
                                   calibrate=True, **kw)
    plan = {
        "version": PLAN_VERSION, "seconds_authoritative": False, "T": int(T),
        "timing": {"step_ms": float(step_ms), "offset": float(off2),
                   "times": [float(x) for x in grid.times],
                   "pos": [int(x) for x in grid.pos],
                   "bar": [int(x) for x in grid.bar],
                   "segments": [[float(a), float(b)] for a, b in grid.segments]},
        "rests": [bool(x) for x in tr["rest_mask"].tolist()],
        "scheduling": {"thr": [float(x) for x in tr["thr"].tolist()],
                       "keep": [float(x) for x in tr["keep"].tolist()]},
    }
    return plan, raw, walls


def grid_from_plan(plan):
    """Rebuild the exact baseline TimeGrid from the plan's serialized arrays."""
    from timing import TimeGrid
    t = plan["timing"]
    return TimeGrid(list(t["times"]), list(t["pos"]), list(t["bar"]),
                    [tuple(s) for s in t["segments"]])


def decode_with_plan(objects, bpm, offset, audio_path, plan, *, seed=0, temp=None,
                     diff="ExpertPlus", thin=True, timing=None, rate_scale=1.0,
                     trace=None):
    """Decode driven by a captured plan: reuse its grid, freeze its rests and
    thr/keep. Bit-identical to the baseline capture under the same seed —
    unless the plan carries a request (e.g. doubles_cap_8b), which is the
    opt-in behavior-affecting path."""
    import torch
    import groom
    steps, T, step_ms, off2, _grid, afeat, spec = _setup(
        objects, bpm, offset, audio_path, diff, thin, timing)
    grid = grid_from_plan(plan)
    rest = torch.tensor(plan["rests"], dtype=torch.bool)
    thr = torch.tensor(plan["scheduling"]["thr"], dtype=torch.float32)
    keep = torch.tensor(plan["scheduling"]["keep"], dtype=torch.float32)
    kw = _groom_kwargs(spec, seed, temp, rate_scale)
    kw["afeat"] = afeat
    if plan.get("doubles_cap_8b") is not None:
        kw["doubles_cap_8b"] = int(plan["doubles_cap_8b"])
    return groom.groom_notes(steps, T, step_ms, off2, grid=grid,
                             rest_mask_in=rest, thr_vectors=(thr, keep),
                             rest_policy="off", density_adjust=False,
                             trace=trace, **kw)


def extend_draft(draft, plan):
    """Attach the decode plan to a phrase-draft dict under a new schema slot,
    leaving every diagnostic phrase field untouched."""
    out = dict(draft)
    out["decode_plan"] = plan
    return out


# --- convert_groomed-level per-attempt capture/replay -----------------------

def plans_from_collect(recs):
    """Serializable per-ATTEMPT plans from a capture run's collect records.
    Each attempt (pass_index, seed) gets its OWN plan — never shared — bound to
    the rate_scale it ran at. Keys are "pass:seed" strings (JSON-friendly)."""
    plans = {}
    for r in recs:
        key = f"{r['pass_index']}:{r['seed']}"
        assert key not in plans, f"duplicate attempt {key}"
        tr = r["trace"]
        plans[key] = {
            "version": PLAN_VERSION,
            "rate_scale": float(r["rate_scale"]),
            "rests": [bool(x) for x in tr["rest_mask"].tolist()],
            "scheduling": {"thr": [float(x) for x in tr["thr"].tolist()],
                           "keep": [float(x) for x in tr["keep"].tolist()]},
        }
    return plans


def inject_plans(plans_ser):
    """Deserialize "pass:seed"-keyed plans into convert_groomed's `plans`
    input (tensors + bound rate_scale)."""
    import torch
    out = {}
    for key, p in plans_ser.items():
        out[key] = {
            "rate_scale": float(p["rate_scale"]),
            "rest_mask": torch.tensor(p["rests"], dtype=torch.bool),
            "thr": torch.tensor(p["scheduling"]["thr"], dtype=torch.float32),
            "keep": torch.tensor(p["scheduling"]["keep"], dtype=torch.float32),
        }
        if p.get("doubles_cap_8b") is not None:      # opt-in plan request
            out[key]["doubles_cap_8b"] = int(p["doubles_cap_8b"])
    return out


def _attempt_fields(recs):
    """The per-attempt equivalence fields: identity, gates, scores, artifacts."""
    return [{"seed": r["seed"], "pass_index": r["pass_index"],
             "rate_scale": r["rate_scale"], "ok": r["ok"], "score": r["score"],
             "gates": r["gates"], "notes": r["notes"], "walls": r["walls"],
             "selected": r["selected"]} for r in recs]


def convert_roundtrip_identical(objects, bpm, offset, audio_path, **kw):
    """Full-production-path no-op: baseline convert_groomed (capture) vs a
    replay driven by per-attempt plans (after a JSON round-trip). Returns
    (ok, detail): every candidate's notes/walls/scores/gates, retry decisions
    (pass_index/rate_scale), attempt count, winner and final output must match."""
    from convert import convert_groomed
    rec_a, rec_b = [], []
    out_a = convert_groomed(objects, bpm, offset, audio_path=audio_path,
                            collect=rec_a, **kw)
    plans = json.loads(json.dumps(plans_from_collect(rec_a)))
    out_b = convert_groomed(objects, bpm, offset, audio_path=audio_path,
                            collect=rec_b, plans=inject_plans(plans), **kw)
    fa, fb = _attempt_fields(rec_a), _attempt_fields(rec_b)
    detail = {"attempts_a": len(rec_a), "attempts_b": len(rec_b),
              "retry": any(r["pass_index"] > 0 for r in rec_a),
              "attempts_equal": fa == fb, "output_equal": out_a == out_b}
    return (len(rec_a) == len(rec_b) and fa == fb and out_a == out_b), detail


def _cand_gates(a, b, band):
    """Per-candidate emitted-motion gates, mirroring gate_song's winner-level
    thresholds: a = uncapped reference metrics, b = capped candidate metrics
    (both from eval.clean_rhythm._cand_metrics). Returns failing gate names."""
    from eval.clean_rhythm_eval import _dup_ok, _flags_ok
    fails = []
    if not (band[0] * 0.98 <= b["raw_sps"] <= band[1] * 1.02):
        fails.append("rate_band")
    if not _dup_ok(a["dup8"], b["dup8"]):
        fails.append("dup8")
    for ext, av in a["flags_by_ext"].items():
        if not _flags_ok(av, b["flags_by_ext"].get(ext, 0.0)):
            fails.append(f"repositioning@{ext}")
    if not (b["narrow"] <= max(a["narrow"], 1) + 1e-12
            and b["conv"] <= max(a["conv"], 1) + 1e-12):
        fails.append("pair_exposure")
    if not (b["longest_run"] <= max(a["longest_run"], 12)):
        fails.append("hand_monopoly")
    return fails


def apply_doubles_cap(objects, bpm, offset, audio_path, cap, *,
                      diff="ExpertPlus", replay_mode="off", thin=True,
                      calibrate=True):
    """Planner APPLY path for a doubles-cap request, with candidate admission:
    run the normal uncapped decode (the reference), replay every attempt with
    the cap, then ADMIT only capped candidates that (1) hard-validate, (2) meet
    the cap with zero accent overflow, and (3) pass the emitted-motion gates
    against their own uncapped twin. The winner is the best-scoring admitted
    candidate; if none qualify the result is EXPLICITLY infeasible and no
    capped chart is returned. Uses the existing attempt budget — no decodes
    beyond convert_groomed's own loop."""
    from convert import convert_groomed, grid_steps, diff_spec
    from eval.clean_rhythm import _cand_metrics
    band = diff_spec(diff)["band"]
    _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=thin)
    beat_ms = 60000.0 / bpm
    rec_a, rec_b = [], []
    convert_groomed(objects, bpm, offset, audio_path=audio_path, diff=diff,
                    replay_mode=replay_mode, collect=rec_a, thin=thin,
                    calibrate=calibrate)
    if cap == "max-1":     # exercise-the-cap convenience: winner's max window - 1
        win_a = next(r for r in rec_a if r["selected"])
        by_step = {}
        for s, h, _c, _l, d in win_a["notes"]:
            if d != 8:
                by_step.setdefault(s, set()).add(h)
        wins = {}
        for s, hs in by_step.items():
            if len(hs) == 2:
                wins[s // 32] = wins.get(s // 32, 0) + 1
        cap = max(1, max(wins.values(), default=1) - 1)
    plans = _plans_with_extras(rec_a)
    for p in plans.values():
        p["doubles_cap_8b"] = int(cap)
    convert_groomed(objects, bpm, offset, audio_path=audio_path, diff=diff,
                    replay_mode=replay_mode, collect=rec_b, thin=thin,
                    calibrate=calibrate, plans=inject_plans(plans))
    a_by = {f"{r['pass_index']}:{r['seed']}": r for r in rec_a}
    per_cand, admitted = {}, []
    for r in rec_b:
        key = f"{r['pass_index']}:{r['seed']}"
        fails = []
        if not r["ok"]:
            fails.append("validity")
        if sum(r["trace"]["doubles_cap"]["windows_over"].values()):
            fails.append("cap_overflow")            # accents alone exceed cap
        ref = a_by.get(key)
        if ref is None:
            fails.append("no_uncapped_reference")   # hotter seed baseline lacked
        else:
            ma = _cand_metrics(ref["notes"], ref["motion"], grid, beat_ms)
            mb = _cand_metrics(r["notes"], r["motion"], grid, beat_ms)
            fails += _cand_gates(ma, mb, band)
        per_cand[key] = fails
        if not fails:
            admitted.append((r["score"], key, r))
    out = {"cap": int(cap), "attempts": len(rec_b),
           "baseline_attempts": len(rec_a), "per_candidate": per_cand,
           "admitted": sorted(k for _sc, k, _r in admitted),
           "feasible": bool(admitted)}
    if admitted:
        _sc, key, r = max(admitted, key=lambda t: t[0])
        out.update(winner_key=key, notes=r["notes"], walls=r["walls"])
    return out


def _plans_with_extras(recs):
    """plans_from_collect + per-pass extension to the hotter-seed budget (the
    frozen decisions are seed-independent within a pass, asserted)."""
    from groom import N_DECODES
    plans = plans_from_collect(recs)
    by_pass = {}
    for key, p in plans.items():
        pi = int(key.split(":")[0])
        ref = by_pass.setdefault(pi, p)
        assert (p["rests"] == ref["rests"]
                and p["scheduling"] == ref["scheduling"])
    for pi, ref in by_pass.items():
        for seed in range(2 * N_DECODES):
            plans.setdefault(f"{pi}:{seed}", dict(ref))
    return plans


def roundtrip_identical(objects, bpm, offset, audio_path, *, seed=0, **kw):
    """Capture then replay; return (ok, plan). ok == bit-identical notes+walls."""
    plan, raw0, walls0 = capture_plan(objects, bpm, offset, audio_path,
                                      seed=seed, **kw)
    # the plan must survive a JSON round-trip and still drive an identical decode
    plan = json.loads(json.dumps(plan))
    raw1, walls1 = decode_with_plan(objects, bpm, offset, audio_path, plan,
                                    seed=seed, **kw)
    return (raw0 == raw1 and walls0 == walls1), plan


if __name__ == "__main__":
    # The real no-op acceptance (8 panel songs x production seeds, bit-identical)
    # lives in eval/test_phrase_plan.py.
    print("phrase_plan OK; PLAN_VERSION", PLAN_VERSION,
          "- run eval/test_phrase_plan.py for the panel acceptance")
