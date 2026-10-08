"""E1 Task 4: actual B0-policy checkpoint comparison (spec §E1 + safety
floors section).

Candidates are FULL production geometry rollouts (flow_decode.decode_geometry
on the B0 winner's frozen EventSchedule) judged on emitted notes. Ceilings
are admission bounds frozen in the run config — never rank scores; ranking
in E1 is the UNCHANGED B0 production key (validity, critic − motion cost).
`critic_floor` exists for E2's policy branch only. Every attempt persists
before the next starts; a completed key is never recomputed; a fallback is
always visible and never a checkpoint win.
"""
import json
import hashlib
from pathlib import Path

import motion

ROOT = Path(__file__).resolve().parent.parent
E1 = ROOT / "experiments" / "expressive-v1" / "e1"
WINDOW_MS = 4000.0


def window_flags(notes_ms, ext):
    """Flagged transitions per fixed 4-second window from audio origin,
    binned by the transition's END time. Frozen flag rule = motion's
    REPO_MS/REPO_DIST at the given extent."""
    from eval.quality_metrics import unrounded_transitions
    out = {}
    for tr in unrounded_transitions(notes_ms, ext=ext):
        if tr["ms"] <= motion.REPO_MS and round(tr["dist"], 2) \
                >= motion.REPO_DIST:
            w = str(int(tr["t_end"] // WINDOW_MS)) if "t_end" in tr else \
                str(int(tr["t"] // WINDOW_MS)) if "t" in tr else None
            if w is None:
                raise KeyError("transition record lacks an end-time field")
            out[w] = out.get(w, 0) + 1
    return out


def chart_admission_metrics(raw, grid, bpm):
    """The exact metric bundle the ceilings compare: per-extent flag counts,
    narrow/conv (separate), broad/opposite (reported only), 4s windows."""
    from eval.quality_metrics import (opposite_horizontal, to_ms,
                                      unrounded_transitions)
    notes_ms = to_ms(raw, grid)
    flags = {}
    for e in motion.EXT_VARIANTS:
        flags[str(e)] = sum(
            1 for tr in unrounded_transitions(notes_ms, ext=e)
            if tr["ms"] <= motion.REPO_MS
            and round(tr["dist"], 2) >= motion.REPO_DIST)
    lr = motion.lr_doubles(notes_ms)
    wins = {}
    for e in motion.EXT_VARIANTS:
        for w, n in window_flags(notes_ms, e).items():
            wins[w] = max(wins.get(w, 0), n)
    return {"flags_by_ext": flags, "narrow": len(lr["narrow"]),
            "conv": len(lr["converging"]), "broad": len(lr["broad"]),
            "opposite": opposite_horizontal(notes_ms), "windows_4s": wins}


def admit(candidate, baseline, contract):
    """Frozen experiment ceilings vs the paired B0 chart. Boundary values
    admit; reasons name every violation; broad/opposite never reject."""
    reasons = []
    for e, b in baseline["flags_by_ext"].items():
        lim = max(1.25 * b, b + 10)
        c = candidate["flags_by_ext"].get(e, 0)
        if c > lim + 1e-9:
            reasons.append(f"flags@{e}: {c} > {lim:g}")
    slack = contract.get("pair_slack", 2)
    for k in ("narrow", "conv"):
        if candidate.get(k, 0) > baseline.get(k, 0) + slack:
            reasons.append(f"{k}: {candidate.get(k, 0)} > "
                           f"{baseline.get(k, 0)}+{slack}")
    wslack = contract.get("window_slack", 2)
    for w, c in (candidate.get("windows_4s") or {}).items():
        b = (baseline.get("windows_4s") or {}).get(w, 0)
        if c > b + wslack:
            reasons.append(f"window {w}: {c} > {b}+{wslack}")
    return {"admitted": not reasons, "reasons": reasons}


def select_arm(records, selector="b0"):
    """E1: unchanged B0 production key. E2 policy branch: critic_floor =
    highest shipped critic among ELIGIBLE candidates, ties by canonical
    attempt id. Ceiling slack never ranks. No eligible candidate is an
    explicit b0 fallback, never a win."""
    if selector == "b0":
        pool = records
        best = max(pool, key=lambda r: (r["production_key"], r["id"])) \
            if pool else None
    elif selector == "critic_floor":
        pool = [r for r in records if r.get("eligible")]
        best = min(pool, key=lambda r: (-r["critic"], r["id"])) \
            if pool else None
    else:
        raise ValueError(selector)
    if best is None:
        return {"id": None, "status": "fallback_b0", "selector": selector}
    return {"id": best["id"], "status": "selected", "selector": selector}


def packet_coverage(required, completed):
    return "COMPLETE" if completed >= required else "INCOMPLETE"


# ---------------- the E1 driver ----------------

def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_e1_eval(arm, max_songs=6, flow_factory=None, tag=None):
    """Six-attempt comparison for one arm's selected checkpoint on each B0
    winner's exact EventSchedule, chunk-safe per (song, stage). Stage b0:
    canonical six-attempt baseline via convert_groomed(collect). Stage
    inject: shipped-weight injection must replay the winner bit-exact.
    Stage ckpt: seeds 0-5 geometry rollouts, admission ceilings, B0
    selection among admitted; fallback recorded explicitly."""
    import torch
    torch.set_num_threads(4)
    from convert import convert_groomed, grid_steps, parse_osu
    import critic as critic_mod
    import groom
    from flow_decode import EventSchedule, decode_geometry
    from groom import Flow
    from eval.expression_profile import profile
    from eval.expressive_manifest import (load_run, partial_completed,
                                          save_partial)
    from eval.expressive_train import select_snapshot

    run = load_run(E1)
    cfg = run["config"]
    for f, want in cfg["protected"].items():
        if _sha(ROOT / f) != want:
            raise RuntimeError(f"protected artifact changed: {f}")
    if flow_factory is not None:
        # E2/control path: flow_factory(schedule, step_ms) builds the
        # per-song decode flow; `arm` is the partial-key tag (e.g. "E2")
        sel, ck_sha, model = {"update": tag}, tag, None
    else:
        arm_dir = E1 / f"arm-{arm}"
        hist = json.loads((arm_dir / "history.json").read_text())
        sel = hist.get("selected") or select_snapshot(
            hist["updates"], 5, cfg["approved_val_families_n"])
        if sel["step_zero"]:
            print(f"[e1-eval {arm}] step-zero winner: no new checkpoint; "
                  "nothing to evaluate")
            return {"arm": arm, "status": "STEP_ZERO"}
        ck_p = arm_dir / f"snapshot-{sel['update']}.pt"
        ck_sha = _sha(ck_p)
        model = Flow()
        model.load_state_dict(torch.load(ck_p, map_location="cpu"))
        model.eval()
    spec = None
    done = 0
    for song in cfg["songs"]:
        if done >= max_songs:
            break
        sid = song["song"]
        key_b0 = f"{sid}:b0"
        key_arm = f"{sid}:arm{arm}"
        if partial_completed(E1, key_arm):
            continue
        osu = ROOT / song["osu"]
        audio = Path(song["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        assert _sha(osu) == song["osu_sha256"], f"{sid}: osu drifted"
        from convert import diff_spec
        _m, objects, bpm, offset = parse_osu(osu)
        steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset,
                                                   thin=True)
        step_times = [grid.time(s) for s in range(T)]
        afeat = groom.cached_audio_features(str(audio), step_times)
        spec = diff_spec("ExpertPlus")

        # stage 1: canonical B0 baseline (cached in the run partial)
        b0_p = E1 / "partial" / (key_b0.replace("/", "_") + ".json")
        if partial_completed(E1, key_b0):
            b0 = json.loads(b0_p.read_text())
        else:
            recs = []
            convert_groomed(objects, bpm, offset, audio_path=str(audio),
                            diff="ExpertPlus", replay_mode="off",
                            collect=recs, thin=True, calibrate=True)
            selr = next(r for r in recs if r["selected"])
            b0 = save_partial(E1, key_b0, {
                "completed": True, "winner_seed": selr["seed"],
                "winner_temp": selr["temp"],
                "winner_rate_scale": selr["rate_scale"],
                "notes": selr["notes"], "walls": selr["walls"],
                "metrics": chart_admission_metrics(selr["notes"], grid, bpm),
                "attempts": [{"seed": r["seed"], "temp": r["temp"],
                              "ok": r["ok"], "score": r["score"]}
                             for r in recs]})
        raw0 = [tuple(int(v) for v in n) for n in b0["notes"]]
        walls0 = [tuple(int(v) for v in w) for w in b0["walls"]]
        sched = EventSchedule.from_raw(raw0, walls0, grid, T, step_ms)

        # stage 2: shipped-weight injection replay (bit-exact requirement)
        # — at the winner's EXACT production temperature, not the default
        p = decode_geometry(sched, steps, T, step_ms, off2, grid, afeat,
                            spec, flow_model=None, seed=b0["winner_seed"],
                            temp=b0["winner_temp"],
                            checkpoint="shipped-inject")
        if p.raw != raw0 or not p.honored:
            raise RuntimeError(f"{sid}: shipped-weight injection did not "
                               "replay the B0 winner bit-exactly")

        # stage 3: six checkpoint attempts, persisted individually
        cands = []
        for sd in range(6):
            akey = f"{sid}:arm{arm}:s{sd}"
            ap = E1 / "partial" / (akey.replace("/", "_") + ".json")
            if partial_completed(E1, akey):
                cands.append(json.loads(ap.read_text()))
                continue
            fm = flow_factory(sched, step_ms) if flow_factory else model
            g = decode_geometry(sched, steps, T, step_ms, off2, grid,
                                afeat, spec, flow_model=fm, seed=sd,
                                temp=groom.TEMPS[sd % len(groom.TEMPS)],
                                checkpoint=f"arm{arm}@{sel['update']}")
            if not g.honored or g.infeasible:
                rec = {"completed": True, "seed": sd, "ok": False,
                       "reason": "not_honored" if not g.honored
                       else "infeasible"}
            else:
                mets = chart_admission_metrics(g.raw, grid, bpm)
                adm = admit(mets, b0["metrics"],
                            {"pair_slack": 2, "window_slack": 2})
                notes_s = [(grid.time(s) / 1000.0, h, c, l, d)
                           for s, h, c, l, d in g.raw]
                prof = profile([(h, c, l, d)
                                for _t, h, c, l, d in notes_s],
                               [t for t, *_ in notes_s])
                mrep = motion.report([(grid.time(s), h, c, l, d)
                                      for s, h, c, l, d in g.raw])
                crit = float(critic_mod.score(g.raw, g.walls, T, afeat))
                rec = {"completed": True, "seed": sd, "ok": True,
                       "notes": g.raw, "walls": g.walls,
                       "metrics": mets, "admission": adm,
                       "critic": crit,
                       "production_key": [adm["admitted"],
                                          crit - motion.select_cost(mrep)],
                       "profile_summary": {
                           "low_vertical_share": prof["features"]["spatial"]
                           ["low_vertical_share"] if prof["status"] == "ok"
                           else None,
                           "arc_runs": prof["features"]["arcs"]
                           ["runs_per_100_heads"] if prof["status"] == "ok"
                           else None,
                           "double_entropy": prof["double_vocabulary"]
                           ["entropy"] if prof["status"] == "ok" else None},
                       "eligible": adm["admitted"]}
            save_partial(E1, akey, rec)
            cands.append(rec)
        pool = [{"id": f"s{r['seed']}", "critic": r.get("critic", -1e9),
                 "production_key": tuple(r.get("production_key",
                                               (False, -1e9))),
                 "eligible": r.get("eligible", False)}
                for r in cands if r.get("ok")]
        eligible = [r for r in pool if r["eligible"]]
        choice = select_arm(eligible, selector="b0") if eligible else \
            {"id": None, "status": "fallback_b0", "selector": "b0"}
        save_partial(E1, key_arm, {
            "completed": True, "checkpoint_sha256": ck_sha,
            "update": sel["update"], "selection": choice,
            "n_admitted": len(eligible), "n_ok": len(pool)})
        done += 1
        print(f"  [{sid:<16}] arm {arm}: admitted {len(eligible)}/6 -> "
              f"{choice['status']}:{choice['id']}", flush=True)
    n_done = sum(1 for s in cfg["songs"]
                 if partial_completed(E1, f"{s['song']}:arm{arm}"))
    status = packet_coverage(len(cfg["songs"]), n_done)
    print(f"[e1-eval {arm}] {n_done}/{len(cfg['songs'])} songs -> {status}")
    return {"arm": arm, "status": status, "checkpoint_sha256": ck_sha}


if __name__ == "__main__":
    import sys
    run_e1_eval(sys.argv[1] if len(sys.argv) > 1 else "A")
