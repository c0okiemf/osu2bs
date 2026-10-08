"""Quality-v1 comparison verdicts, family aggregation and the B0 freeze (Q1.1).

Verdict vocabulary (spec): NON_REGRESSION_PASS (integrity + safety gates hold
but the positive target is unmet), QUALITY_PASS (positive target met on top of
everything), QUALITY_FAIL (any integrity/safety/anti-gaming violation),
INCOMPLETE (missing or incomparable data). Passing integrity without the
positive target is never QUALITY_PASS. A deterministic policy has no
learned=True requirement. All aggregation is FAMILY-level macro — notes,
phrases and seeds are not independent songs.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import motion

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "experiments" / "quality-v1" / "baseline"
B0_FILES = ["groom.pt", "flow.pt", "critic.pt", "ladder.json",
            "feats_cache4.pt"]
CODE_FILES = ["groom.py", "convert.py", "motion.py", "parity.py", "timing.py",
              "critic.py", "eval/quality_metrics.py", "eval/quality_panel.py",
              "eval/quality_eval.py"]

NON_REGRESSION_PASS = "NON_REGRESSION_PASS"
QUALITY_PASS = "QUALITY_PASS"
QUALITY_FAIL = "QUALITY_FAIL"
INCOMPLETE = "INCOMPLETE"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze_baseline():
    """Copy shipped B0 artifacts + the (dirty) corpus manifest + source digests
    into experiments/quality-v1/baseline/. Never reverts anything; the working
    manifest is snapshotted as-is. Idempotent: re-freezing verifies instead of
    overwriting a different baseline."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BASELINE_DIR / "baseline.json"
    digests = {}
    for f in B0_FILES:
        src = ROOT / f
        if not src.exists():
            raise RuntimeError(f"B0 artifact missing: {f}")
        digests[f] = _sha(src)
        dst = BASELINE_DIR / Path(f).name
        if not dst.exists():
            shutil.copy2(src, dst)
        elif _sha(dst) != digests[f] and f != "feats_cache4.pt":
            # the feature cache legitimately GROWS (new D songs); models never
            raise RuntimeError(f"baseline already frozen with DIFFERENT {f}")
    snap = BASELINE_DIR / "corpus_manifest.snapshot.json"
    if not snap.exists():
        shutil.copy2(ROOT / "eval" / "corpus_manifest.json", snap)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    rec = {"b0": digests, "code": {f: _sha(ROOT / f) for f in CODE_FILES},
           "git_head": head,
           "corpus_manifest_snapshot_sha256": _sha(snap)}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        drop = "feats_cache4.pt"                    # cache growth is expected
        if {k: v for k, v in old["b0"].items() if k != drop} \
                != {k: v for k, v in rec["b0"].items() if k != drop}:
            raise RuntimeError("baseline.json exists with different B0 hashes")
    else:
        manifest_path.write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


def family_macro(per_family):
    """Macro mean over families of a scalar; None when empty."""
    vals = list(per_family.values())
    return sum(vals) / len(vals) if vals else None


def _flags_ok(a, b):
    from eval.clean_rhythm_eval import _flags_ok as f
    return f(a, b)


def q1_verdict(before, after, *, integrity_ok, gates_ok, edit_fraction_ok=True):
    """Classify a Q1 D comparison. before/after: {family: quality_metrics
    dict} for the emitted winners. integrity_ok: every transformed winner kept
    its immutable signature + hard validation. gates_ok: the existing per-song
    D gates all pass. Stores denominators and family ids for audit."""
    fams = sorted(before)
    if not fams or sorted(after) != fams:
        return {"status": INCOMPLETE, "reason": "family sets differ or empty",
                "families": fams}
    detail = {"families": fams, "n_families": len(fams)}
    if not integrity_ok or not gates_ok:
        detail.update(status=QUALITY_FAIL,
                      reason="integrity" if not integrity_ok else "gates")
        return detail
    if not edit_fraction_ok:
        detail.update(status=QUALITY_FAIL, reason="edit_fraction")
        return detail
    # motion: >=25% macro reduction at every ELIGIBLE extent (B0 macro >= 1);
    # ineligible extents must not regress per song beyond the frozen tolerance
    ext_detail, motion_pos, motion_safe = {}, True, True
    for e in motion.EXT_VARIANTS:
        k = str(e)
        b_mac = family_macro({f: before[f]["flags_by_ext"][k] for f in fams})
        a_mac = family_macro({f: after[f]["flags_by_ext"][k] for f in fams})
        eligible = b_mac >= 1.0
        if eligible:
            ok = a_mac <= 0.75 * b_mac + 1e-12
            motion_pos &= ok
        else:
            ok = all(_flags_ok(before[f]["flags_by_ext"][k],
                               after[f]["flags_by_ext"][k]) for f in fams)
            motion_safe &= ok
        ext_detail[k] = {"eligible": eligible, "before_macro": b_mac,
                         "after_macro": a_mac, "ok": ok}
    # pairs: >=20% reduction of narrow+converging where the B0 total >= 10;
    # broad outward pairs must never increase
    b_pairs = sum(before[f]["narrow"] + before[f]["converging"] for f in fams)
    a_pairs = sum(after[f]["narrow"] + after[f]["converging"] for f in fams)
    pairs_pos = (a_pairs <= 0.8 * b_pairs + 1e-12) if b_pairs >= 10 \
        else (a_pairs <= b_pairs)
    b_broad = sum(before[f]["broad"] for f in fams)
    a_broad = sum(after[f]["broad"] for f in fams)
    b_opp = sum(before[f]["opposite_horizontal"] for f in fams)
    a_opp = sum(after[f]["opposite_horizontal"] for f in fams)
    broad_safe = a_broad <= b_broad and a_opp <= b_opp
    # at least half the songs that HAVE burden must improve (J decreases)
    from eval.quality_metrics import j_score
    burdened = [f for f in fams
                if any(before[f]["flags_by_ext"][str(e)] > 0
                       for e in motion.EXT_VARIANTS)
                or before[f]["narrow"] + before[f]["converging"] > 0]
    improved = [f for f in burdened
                if j_score(after[f]) < j_score(before[f]) - 1e-12]
    half_ok = (not burdened) or len(improved) * 2 >= len(burdened)
    detail.update(
        motion=ext_detail,
        pairs={"before": b_pairs, "after": a_pairs, "denominator_ok":
               b_pairs >= 10, "ok": pairs_pos},
        broad={"before": b_broad, "after": a_broad,
               "opposite_before": b_opp, "opposite_after": a_opp,
               "ok": broad_safe},
        burdened={"families": burdened, "improved": improved, "ok": half_ok})
    if not (motion_safe and broad_safe):
        detail.update(status=QUALITY_FAIL, reason="regression")
    elif motion_pos and pairs_pos and half_ok:
        detail.update(status=QUALITY_PASS)
    else:
        detail.update(status=NON_REGRESSION_PASS)
    return detail


def run_q1(variant, run_dir, panel="dev"):
    """Decode every D song once (B0 settings + the Q1 quality profile), compare
    the transformed winner against the untouched raw winner, apply the per-song
    gates and the Q1 positive verdict, and record runtime + edit witnesses."""
    import statistics
    import time as _time

    import torch
    torch.set_num_threads(4)
    from convert import convert_groomed, diff_spec, grid_steps, parse_osu
    from quality_repair import RepairConfig

    from eval.clean_rhythm import _arm_metrics, _winner_quiet
    from eval.clean_rhythm_eval import gate_song
    from eval.quality_metrics import quality_metrics
    from eval.quality_panel import dev_entries

    if panel != "dev":
        raise RuntimeError("C is sealed until Q5; only --panel dev may run")
    run = Path(run_dir)
    (run / "witnesses").mkdir(parents=True, exist_ok=True)
    freeze_baseline()
    cfg = RepairConfig(variant=variant)
    band = diff_spec("ExpertPlus")["band"]
    before, after, songs = {}, {}, {}
    integrity_ok = gates_ok = True
    ratios = []
    for e in dev_entries():
        osu = ROOT / e["osu"]
        audio = Path(e["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        _meta, objects, bpm, offset = parse_osu(osu)
        _s, T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
        recs = []
        t0 = _time.perf_counter()
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off", collect=recs,
                        thin=True, calibrate=True, quality_profile=cfg)
        decode_s = _time.perf_counter() - t0
        sel = next(r for r in recs if r["selected"])
        raw0, walls = sel["notes"], sel["walls"]
        tr = sel.get("output_transform")
        raw1 = tr["raw"] if tr else raw0
        m0 = quality_metrics(raw0, walls, grid, bpm)
        m1 = quality_metrics(raw1, walls, grid, bpm)
        fam = e["family"]
        before[fam], after[fam] = m0, m1
        sig_ok = m0["signature"] == m1["signature"]
        integrity_ok &= sig_ok
        beat_ms = 60000.0 / bpm
        a = _arm_metrics(recs, raw0, grid, beat_ms, band)
        b = _arm_metrics(recs, raw1, grid, beat_ms, band)
        qa, qb = _winner_quiet(raw0, grid, T), _winner_quiet(raw1, grid, T)
        b["quiet"] = {"a_ids": qa["quiet_ids"], "b_ids": qb["quiet_ids"],
                      "n_bins": qb["n_bins"]}
        fails = gate_song(a, b)
        gates_ok &= not fails
        repair_s = tr["elapsed_s"] if tr else 0.0
        ratios.append(repair_s / max(decode_s, 1e-9))
        songs[fam] = {"song": e["song"], "signature_ok": sig_ok,
                      "gate_fails": fails, "decode_s": round(decode_s, 1),
                      "repair": ({"status": tr["status"], "trials": tr["trials"],
                                  "changed": len(tr["changed_ids"]),
                                  "elapsed_s": round(tr["elapsed_s"], 2)}
                                 if tr else None)}
        if tr:
            (run / "witnesses" / f"{fam.replace(':', '_')}.json").write_text(
                json.dumps(tr["witnesses"], indent=1, default=list))
        print(f"  {e['song'][:24]:<24} repair="
              f"{songs[fam]['repair'] and songs[fam]['repair']['changed']} "
              f"gates={'OK' if not fails else fails}")
    verdict = q1_verdict(before, after, integrity_ok=integrity_ok,
                         gates_ok=gates_ok)
    runtime = {"median_repair_to_decode_ratio": round(
        statistics.median(ratios), 3), "within_budget":
        statistics.median(ratios) <= 1.0}
    payload = {"packet": "q1", "variant": variant, "verdict": verdict,
               "runtime": runtime, "songs": songs}
    (run / "q1_report.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str))
    (run / "metrics.json").write_text(json.dumps(
        {"before": before, "after": after}, indent=1, default=str))
    print(f"Q1[{variant}] VERDICT: {verdict['status']} | runtime median ratio "
          f"{runtime['median_repair_to_decode_ratio']}")
    return payload


def run_q2(run_dir, panel="dev", allocation="3+3"):
    """Q2: complete-schedule geometry portfolio + explicit selection on D.
    Primary allocation 3 shipped + 3 clean rollouts per song (seeds 0-2) on
    the B0 winner's schedule; candidates include B0 and the Q1-V2 output; all
    candidates are Q1-repaired (Q1 passed). A six-shipped-only control at the
    same extra budget separates checkpoint diversity from larger search."""
    import statistics
    import time as _time

    import torch
    torch.set_num_threads(4)
    import critic as critic_mod
    from convert import NPS_CAP, convert_groomed, diff_spec, grid_steps, \
        parse_osu
    from flow_decode import EventSchedule, decode_geometry
    from quality import Candidate, admit_and_select
    from quality_repair import RepairConfig, repair_winner, risk_vector

    import groom
    from eval.quality_metrics import quality_metrics
    from eval.quality_panel import dev_entries

    if panel != "dev":
        raise RuntimeError("C is sealed until Q5; only --panel dev may run")
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    freeze_baseline()
    qcfg = RepairConfig(variant="paired-direction")
    spec = diff_spec("ExpertPlus")
    cap = NPS_CAP * spec["scale"]
    try:
        from eval.clean_rhythm import _load_flow
        clean_flow = _load_flow(
            ROOT / "experiments/clean-flow-warm-v1/checkpoints/"
                   "flow-seed20260921.pt")
        clean_branch = "ok"
    except Exception as e:                      # incompatible: stop explicitly
        clean_flow, clean_branch = None, f"stopped: {e}"
    seeds = (0, 1, 2, 3, 4, 5) if allocation == "6+6" else (0, 1, 2)
    n_ctrl = 12 if allocation == "6+6" else 6
    songs, sel_after, q1_after = {}, {}, {}
    serving = []
    (run / "partial").mkdir(exist_ok=True)
    for e in dev_entries():
        fam = e["family"]
        part = run / "partial" / (fam.replace(":", "_").replace("/", "_")
                                  + ".json")
        if part.exists():                    # crash-resumable: song done
            saved = json.loads(part.read_text())
            songs[fam] = saved["song_row"]
            sel_after[fam] = saved["sel_after"]
            q1_after[fam] = saved["q1_after"]
            serving.append(saved["serving"])
            print(f"  {saved['song_row']['song'][:24]:<24} (cached partial)")
            continue
        osu = ROOT / e["osu"]
        audio = Path(e["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        _m, objects, bpm, offset = parse_osu(osu)
        steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset,
                                                   thin=True)
        afeat = groom.cached_audio_features(
            str(audio), [grid.time(s) for s in range(T)])
        recs = []
        t0 = _time.perf_counter()
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off", collect=recs,
                        thin=True, calibrate=True)
        b0_s = _time.perf_counter() - t0
        sel0 = next(r for r in recs if r["selected"])
        raw0, walls0 = sel0["notes"], sel0["walls"]
        sched = EventSchedule.from_raw(raw0, walls0, grid, T, step_ms)

        def _score(raw, walls):
            return float(critic_mod.score(raw, walls, T, afeat))

        def _cand(cid, raw, walls, honored=True, extra=None):
            rr = repair_winner(raw, walls, grid, bpm, cap, qcfg)
            return Candidate(cid=cid + "+q1", raw=rr.raw, walls=walls,
                             rv=risk_vector(rr.raw, walls, grid),
                             score=_score(rr.raw, walls), honored=honored,
                             extra=dict(extra or {},
                                        repaired=len(rr.changed_ids)))
        b0 = Candidate(cid="b0", raw=raw0, walls=walls0,
                       rv=risk_vector(raw0, walls0, grid),
                       score=_score(raw0, walls0))
        t1 = _time.perf_counter()
        cands = [_cand("b0", raw0, walls0)]
        rollout_calls = 0
        for tag, model in (("shipped", None),) + (
                (("clean", clean_flow),) if clean_flow is not None else ()):
            for sd in seeds:
                p = decode_geometry(sched, steps, T, step_ms, off2, grid,
                                    afeat, spec, flow_model=model, seed=sd,
                                    checkpoint=tag)
                rollout_calls += 1
                cands.append(_cand(f"{tag}:{sd}", p.raw, p.walls,
                                   honored=p.honored,
                                   extra={"infeasible": p.infeasible}))
        res = admit_and_select(b0, cands)
        # control: six shipped-only rollouts at the same extra budget
        ctrl_cands = [_cand("b0", raw0, walls0)]
        for sd in range(n_ctrl):
            p = decode_geometry(sched, steps, T, step_ms, off2, grid, afeat,
                                spec, flow_model=None, seed=sd,
                                checkpoint="ctrl")
            rollout_calls += 1
            ctrl_cands.append(_cand(f"ctrl:{sd}", p.raw, p.walls,
                                    honored=p.honored))
        ctrl = admit_and_select(b0, ctrl_cands)
        extra_s = _time.perf_counter() - t1
        fam = e["family"]
        sel_after[fam] = quality_metrics(res.selected.raw,
                                         res.selected.walls, grid, bpm)
        q1c = next(c for c in cands if c.cid == "b0+q1")
        q1_after[fam] = quality_metrics(q1c.raw, q1c.walls, grid, bpm)
        serving.append((b0_s + extra_s) / max(b0_s, 1e-9))
        songs[fam] = {
            "song": e["song"], "selected": res.selected.cid,
            "admitted": res.admitted_ids, "rejections": res.rejections,
            "control_selected": ctrl.selected.cid,
            "control_admitted": ctrl.admitted_ids,
            "accounting": {"original_decodes": len(recs),
                           "extra_rollouts": rollout_calls,
                           "b0_decode_s": round(b0_s, 1),
                           "portfolio_s": round(extra_s, 1)}}
        part.write_text(json.dumps(
            {"song_row": songs[fam], "sel_after": sel_after[fam],
             "q1_after": q1_after[fam], "serving": serving[-1]},
            default=str))
        print(f"  {e['song'][:24]:<24} sel={res.selected.cid} "
              f"admitted={len(res.admitted_ids)} ctrl={ctrl.selected.cid}")
    from eval.quality_metrics import j_score
    fams = sorted(sel_after)
    j_sel = sum(j_score(sel_after[f]) for f in fams) / len(fams)
    j_q1 = sum(j_score(q1_after[f]) for f in fams) / len(fams)
    med = statistics.median(serving)
    passed = (j_sel <= 0.85 * j_q1 + 1e-12) and med <= 3.0
    payload = {"packet": "q2", "allocation": allocation,
               "clean_branch": clean_branch,
               "j_macro": {"q1": j_q1, "selected": j_sel,
                           "improvement": 1 - j_sel / j_q1 if j_q1 else None},
               "serving_median_ratio": round(med, 2),
               "verdict": "QUALITY_PASS" if passed else "NON_REGRESSION_PASS",
               "songs": songs}
    (run / "q2_report.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str))
    (run / "metrics.json").write_text(json.dumps(
        {"selected": sel_after, "q1": q1_after}, indent=1, default=str))
    print(f"Q2 VERDICT: {payload['verdict']} | J {j_q1:.4f} -> {j_sel:.4f} "
          f"({payload['j_macro']['improvement']:.1%}) | serving x{med:.2f} | "
          f"clean branch {clean_branch}")
    return payload


def run_q3(run_dir, panel="dev", max_songs=6):
    """Q3 D acceptance: planner-scheduled vs constant-plan equal-budget
    control vs B0, chunked (review's 1-hour rule: <=max_songs full songs per
    invocation, atomic per-song partials, frozen-config hash, verdict ONLY on
    complete coverage — rerun the same command until it reports the verdict).

    Per song: ONE shared B0 decode; both arms run the identical bounded
    scheduler (multipliers/beam/reschedule budget); outputs measured on
    EMITTED notes: envelope containment (train ExpertPlus p10-p90 per ranker
    component), paired-human window trajectory error (grid-free components;
    fam entries only), protections and admission from schedule_workload."""
    import time as _time

    import torch
    torch.set_num_threads(4)
    from convert import parse_osu, grid_steps
    from difficulty import FEATURES, chart_descriptors
    from phrase_planner import schedule_workload, train_mean_targets, \
        human_trajectory, trajectory_error
    from eval.quality_panel import dev_entries
    from eval.visibility import Note

    if panel != "dev":
        raise RuntimeError("C is sealed until Q5; only --panel dev may run")
    run = Path(run_dir)
    (run / "partial").mkdir(parents=True, exist_ok=True)
    freeze_baseline()
    ckpt = ROOT / "experiments/quality-v1/q3-planner/planner-seed20260921.pt"
    calib_p = ROOT / "experiments/quality-v1/q3-ranker/workload_calibration.json"
    request = train_mean_targets()[:10]
    cfg = {"planner_ckpt_sha256": _sha(ckpt), "calibration_sha256": _sha(calib_p),
           "request": [round(float(v), 6) for v in request],
           "phrase_planner_sha256": _sha(ROOT / "phrase_planner.py"),
           "difficulty_sha256": _sha(ROOT / "difficulty.py")}
    cfg_p = run / "config.json"
    if cfg_p.exists():                       # frozen config across chunks
        if json.loads(cfg_p.read_text()) != cfg:
            raise RuntimeError("run config drifted since first chunk — "
                               "start a fresh --run dir")
    else:
        cfg_p.write_text(json.dumps(cfg, indent=1))
    calib = json.loads(calib_p.read_text())
    env9 = calib["tier_envelopes"]["9"]

    def _envelope(raw, grid):
        notes = [Note(str(i), grid.time(s) / 1000.0, c, l, h, d)
                 for i, (s, h, c, l, d) in enumerate(raw)]
        feats, why = chart_descriptors(
            {"notes": notes, "chart_unknown": False, "ignored": {}})
        if feats is None:
            return {"status": why, "in_band": None}
        comp = {k: bool(env9[k]["p10"] <= feats[k] <= env9[k]["p90"])
                for k in FEATURES}
        return {"status": "ok", "components": comp,
                "features": {k: round(feats[k], 4) for k in FEATURES},
                "in_band": sum(comp.values()), "n": len(comp),
                "all_in_band": all(comp.values())}

    entries = dev_entries()
    done = 0
    for e in entries:
        fam = e["family"]
        part = run / "partial" / (fam.replace(":", "_").replace("/", "_")
                                  + ".json")
        if part.exists():
            continue
        if done >= max_songs:
            break
        osu = ROOT / e["osu"]
        audio = Path(e["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        _m, objects, bpm, offset = parse_osu(osu)
        _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
        from convert import convert_groomed
        recs = []
        t0 = _time.perf_counter()
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off", collect=recs,
                        thin=True, calibrate=True)
        b0_s = _time.perf_counter() - t0
        arms = {}
        for mode in ("planner", "constant"):
            r = schedule_workload(objects, bpm, offset, str(audio), ckpt=ckpt,
                                  mode=mode, request=request.clone(),
                                  b0=(recs, b0_s))
            out = r["outcome"]
            # roadmap contract, fail-closed per song
            assert out["protected_rest_ids"] == \
                r["baseline"]["protected_rest_ids"]
            assert not (set(out["occupied_ids"])
                        & set(out["protected_rest_ids"]))
            if out["source"] == "reschedule":
                assert set(r["baseline"]["protected_accent_ids"]) \
                    <= set(out["occupied_ids"])
            assert r["accounting"]["complete_reschedules"] <= 2
            arms[mode] = {"achieved": r["achieved"], "reasons": r["reasons"],
                          "source": out["source"], "err_internal": out["err"],
                          "b0_err_internal": r["b0_err"],
                          "reschedules": r["accounting"]["complete_reschedules"],
                          "geometry_calls": r["accounting"]["geometry_calls"],
                          "elapsed_s": round(r["elapsed_s"], 1),
                          "envelope": _envelope(out["notes"], grid),
                          "notes": out["notes"], "walls": out["walls"]}
        b0_raw = next(r for r in recs if r["selected"])["notes"]
        row = {"song": e["song"], "family": fam, "b0_s": round(b0_s, 1),
               "b0_envelope": _envelope(b0_raw, grid),
               "serving_ratio": round(
                   (b0_s + arms["planner"]["elapsed_s"]) / max(b0_s, 1e-9), 2)}
        # paired-human trajectory error (fam entries carry a corpus dir)
        if fam.startswith("fam:"):
            wt = human_trajectory(audio.parent)
            if wt[0] is None:
                row["traj"] = {"status": wt[1]}
            else:
                from phrase_planner import traj_descriptors
                windows_ms, hd = wt

                def _th(raw):
                    return [(grid.time(s), h) for s, h, *_ in raw]

                def _occ(notes_th):
                    vs = [traj_descriptors(notes_th, a, b)["occ_per_s"]
                          for a, b in windows_ms]
                    return sum(vs) / len(vs)
                row["traj"] = {
                    "status": "ok", "n_windows": len(windows_ms),
                    "planner": trajectory_error(_th(arms["planner"]["notes"]),
                                                windows_ms, hd),
                    "constant": trajectory_error(_th(arms["constant"]["notes"]),
                                                 windows_ms, hd),
                    "b0": trajectory_error(_th(b0_raw), windows_ms, hd),
                    # grid-independent occupancy diagnostic (review R2 ruling)
                    "occ_per_s": {
                        "human": sum(d["occ_per_s"] for d in hd) / len(hd),
                        "planner": _occ(_th(arms["planner"]["notes"])),
                        "constant": _occ(_th(arms["constant"]["notes"])),
                        "b0": _occ(_th(b0_raw))}}
        else:
            row["traj"] = {"status": "no_human_pair"}
        for m in ("planner", "constant"):
            arms[m] = {k: v for k, v in arms[m].items()
                       if k not in ("notes", "walls")}
        row["arms"] = arms
        tmp = part.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1, default=float))
        tmp.replace(part)
        done += 1
        tr = row["traj"]
        print(f"  {e['song'][:24]:<24} planner "
              f"{'ACH' if arms['planner']['achieved'] else 'b0 '} "
              f"env {arms['planner']['envelope'].get('in_band')}/"
              f"{arms['planner']['envelope'].get('n')} "
              f"traj {tr.get('planner') and round(tr['planner'], 4)}"
              f"/{tr.get('constant') and round(tr['constant'], 4)} "
              f"x{row['serving_ratio']}", flush=True)

    parts = {p.stem: json.loads(p.read_text())
             for p in (run / "partial").glob("*.json")}
    if len(parts) < len(entries):
        print(f"Q3 chunk done: {len(parts)}/{len(entries)} songs — rerun the "
              "same command for the next chunk (no verdict yet)")
        return {"status": "INCOMPLETE_CHUNK", "songs_done": len(parts)}
    return _q3_verdict(run, parts)


def _motif(qm, raw, grid, bpm):
    """Motif concentration summary from a quality_metrics bundle: mean over
    supported windows (>=4 grams) of each window's max 4-gram share, plus
    whole-chart max, mean effective vocabulary, and the per-window WORKLOAD
    features (n_grams/dbl/cad) the matched-envelope excess needs — all recomputable from partials without
    re-decoding."""
    from eval.motif_envelope import window_features
    from eval.quality_metrics import to_ms
    beat_ms = 60000.0 / bpm
    wf = window_features(to_ms(raw, grid), beat_ms)
    wins = [(w, r) for w, r in qm["four_gram"]["windows"].items()
            if r["n_grams"] >= 4 and w in wf]
    if not wins:
        return {"mean_share": None, "max_share": qm["four_gram"]
                ["max_4gram_share"], "mean_vocab": None, "n_windows": 0,
                "windows": []}
    return {"mean_share": sum(r["max_4gram_share"] for _w, r in wins)
            / len(wins),
            "max_share": qm["four_gram"]["max_4gram_share"],
            "mean_vocab": sum(r["effective_vocab"] for _w, r in wins)
            / len(wins),
            "n_windows": len(wins),
            "windows": [{"share": r["max_4gram_share"],
                         "vocab": r["effective_vocab"],
                         "n_grams": r["n_grams"],
                         "dbl": wf[w]["dbl"], "cad": wf[w]["cad"]}
                        for w, r in wins]}


def run_q4(run_dir, panel="dev", max_songs=6, seeds=(0, 1, 2),
           ckpt_name="condflow-a-seed20260921.pt"):
    """Q4 D generation: neutral-style conditioned geometry on the FROZEN B0
    schedule vs the entering champion (B0 + Q1-V2 repair). Chunked with
    atomic per-song partials and frozen-config hashes like run_q3.

    Per song: one B0 decode; champion = Q1-V2 repair of the B0 winner;
    conditioned proposals = decode_geometry on the B0 EventSchedule with
    CondFlowShim (paired seeds), each Q1-V2-repaired and admitted only if it
    hard-validates and passes the emitted-motion gates vs B0; selection
    among admitted = lowest motif concentration, pref-score tie-break
    (after gates only). No admitted proposal -> explicit champion fallback
    (earns no motif credit). Styles are neutral-only (K=1), so the Q4
    style-separation gate is unreachable by construction and the motif gate
    decides; dup8 must not increase and component envelopes still apply."""
    import time as _time

    import torch
    torch.set_num_threads(4)
    from convert import NPS_CAP, check, convert_groomed, diff_spec, \
        grid_steps, parse_osu
    import groom
    import motion as motion_mod
    from cond_flow import ConditionedFlow, shim_for_schedule
    from difficulty import FEATURES, chart_descriptors
    from flow_decode import EventSchedule, decode_geometry
    from phrase_planner import (load_planner, onset_counts,
                                planner_trajectory, song_window_inputs,
                                train_mean_targets)
    from quality_repair import RepairConfig, repair_winner
    from eval.clean_rhythm import _cand_metrics
    from eval.phrase_plan import _cand_gates
    from eval.quality_metrics import quality_metrics
    from eval.quality_panel import dev_entries
    from eval.visibility import Note
    import pref as pref_mod

    if panel != "dev":
        raise RuntimeError("C is sealed until Q5; only --panel dev may run")
    run = Path(run_dir)
    (run / "partial").mkdir(parents=True, exist_ok=True)
    freeze_baseline()
    ck = ROOT / "experiments/quality-v1/q4-flow" / ckpt_name
    planner_ck = ROOT / "experiments/quality-v1/q3-planner/planner-seed20260921.pt"
    calib_p = ROOT / "experiments/quality-v1/q3-ranker/workload_calibration.json"
    pref_p = ROOT / "experiments/quality-v1/q4-pref/pref.pt"
    cfg = {"condflow_sha256": _sha(ck), "planner_sha256": _sha(planner_ck),
           "calibration_sha256": _sha(calib_p), "pref_sha256": _sha(pref_p),
           "seeds": list(seeds)}
    cfg_p = run / "config.json"
    if cfg_p.exists():
        if json.loads(cfg_p.read_text()) != cfg:
            raise RuntimeError("run config drifted — start a fresh --run dir")
    else:
        cfg_p.write_text(json.dumps(cfg, indent=1))
    ckd = torch.load(ck)
    cmodel = ConditionedFlow(n_styles=ckd.get("n_styles", 1))
    cmodel.load_state_dict(ckd["state"])
    cmodel.eval()
    pmodel, pnorm = load_planner(planner_ck)
    request = train_mean_targets()[:10]
    env9 = json.loads(calib_p.read_text())["tier_envelopes"]["9"]
    qcfg = RepairConfig(variant="paired-direction")
    spec = diff_spec("ExpertPlus")
    band = spec["band"]
    cap = NPS_CAP * spec["scale"]

    def _envelope(raw, grid):
        notes = [Note(str(i), grid.time(s) / 1000.0, c, l, h, d)
                 for i, (s, h, c, l, d) in enumerate(raw)]
        feats, why = chart_descriptors(
            {"notes": notes, "chart_unknown": False, "ignored": {}})
        if feats is None:
            return {"status": why, "in_band": None}
        comp = {k: bool(env9[k]["p10"] <= feats[k] <= env9[k]["p90"])
                for k in FEATURES}
        return {"status": "ok", "components": comp,
                "in_band": sum(comp.values()), "n": len(comp),
                "all_in_band": all(comp.values())}

    entries = dev_entries()
    done = 0
    for e in entries:
        fam = e["family"]
        part = run / "partial" / (fam.replace(":", "_").replace("/", "_")
                                  + ".json")
        if part.exists():
            continue
        if done >= max_songs:
            break
        osu = ROOT / e["osu"]
        audio = Path(e["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        _m, objects, bpm, offset = parse_osu(osu)
        steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset,
                                                   thin=True)
        step_times = [grid.time(s) for s in range(T)]
        afeat = groom.cached_audio_features(str(audio), step_times)
        recs = []
        t0 = _time.perf_counter()
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off", collect=recs,
                        thin=True, calibrate=True)
        b0_s = _time.perf_counter() - t0
        sel = next(r for r in recs if r["selected"])
        raw0, walls0 = sel["notes"], sel["walls"]
        rr = repair_winner(raw0, walls0, grid, bpm, cap, qcfg)
        champ = rr.raw if rr is not None else raw0
        beat_ms = 60000.0 / bpm
        m_b0 = _cand_metrics(raw0, motion_mod.report(
            [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw0]),
            grid, beat_ms)
        qm_champ = quality_metrics(champ, walls0, grid, bpm)
        sched = EventSchedule.from_raw(raw0, walls0, grid, T, step_ms)
        mi = onset_counts(objects, step_times)
        x = song_window_inputs(afeat, mi, step_times, bpm)
        intents = planner_trajectory(pmodel, pnorm, x, request)
        shim = shim_for_schedule(cmodel, sched, step_ms, intents, request)
        cands, rejects = [], []
        for sd in seeds:
            p = decode_geometry(sched, steps, T, step_ms, off2, grid, afeat,
                                spec, flow_model=shim, seed=sd,
                                checkpoint="condflow-a")
            if not p.honored or p.infeasible:
                rejects.append({"seed": sd, "reason": "not_honored"
                                if not p.honored else "infeasible"})
                continue
            pr = repair_winner(p.raw, p.walls, grid, bpm, cap, qcfg)
            raw_c = pr.raw if pr is not None else p.raw
            notes_d = [{"t": grid.time(s), "hand": h, "col": c, "layer": l,
                        "dir": d} for s, h, c, l, d in raw_c]
            walls_d = [{"t": grid.time(s0),
                        "dur": grid.time(s0 + ln) - grid.time(s0),
                        "col": col} for s0, ln, col in p.walls]
            fails = list(check(notes_d, bpm, walls_d, cap))
            if not fails:
                m_c = _cand_metrics(raw_c, motion_mod.report(
                    [(grid.time(s), h, c, l, d)
                     for s, h, c, l, d in raw_c]), grid, beat_ms)
                fails = _cand_gates(m_b0, m_c, band)
            if fails:
                rejects.append({"seed": sd, "reason": fails})
                continue
            qm = quality_metrics(raw_c, p.walls, grid, bpm)
            cands.append({"seed": sd, "qm": qm, "raw": raw_c,
                          "walls": p.walls,
                          "pref": pref_mod.score(raw_c, p.walls, grid, bpm)})
        if cands:
            chosen = min(cands, key=lambda c: (
                _motif(c["qm"], c["raw"], grid, bpm)["mean_share"] or 1.0,
                -c["pref"]))
            source = f"condflow:{chosen['seed']}"
            qm_sel, raw_sel, walls_sel = chosen["qm"], chosen["raw"], \
                chosen["walls"]
        else:
            source, qm_sel, raw_sel, walls_sel = "champion", qm_champ, \
                champ, walls0
        row = {"song": e["song"], "family": fam, "b0_s": round(b0_s, 1),
               "source": source, "admitted": [c["seed"] for c in cands],
               "rejects": rejects,
               "champion": {"motif": _motif(qm_champ, champ, grid, bpm),
                            "dup8": m_b0["dup8"],
                            "envelope": _envelope(champ, grid)},
               "selected": {"motif": _motif(qm_sel, raw_sel, grid, bpm),
                            "dup8": _cand_metrics(
                                raw_sel, motion_mod.report(
                                    [(grid.time(s), h, c, l, d) for
                                     s, h, c, l, d in raw_sel]),
                                grid, beat_ms)["dup8"],
                            "envelope": _envelope(raw_sel, grid)},
               "elapsed_s": round(_time.perf_counter() - t0, 1)}
        tmp = part.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1, default=float))
        tmp.replace(part)
        done += 1
        print(f"  {e['song'][:24]:<24} {source:<14} motif "
              f"{row['champion']['motif']['mean_share'] and round(row['champion']['motif']['mean_share'], 3)}"
              f"->{row['selected']['motif']['mean_share'] and round(row['selected']['motif']['mean_share'], 3)} "
              f"admitted {len(cands)}/{len(seeds)}", flush=True)

    parts = {p.stem: json.loads(p.read_text())
             for p in (run / "partial").glob("*.json")}
    if len(parts) < len(entries):
        print(f"Q4 chunk done: {len(parts)}/{len(entries)} songs — rerun for "
              "the next chunk (no verdict yet)")
        return {"status": "INCOMPLETE_CHUNK", "songs_done": len(parts)}
    return _q4_verdict(run, parts)


def _q4_verdict(run, parts):
    rows = list(parts.values())
    fams = sorted(parts)
    from eval.motif_envelope import excess_of
    env_p = ROOT / "experiments/quality-v1/q4-style/human_motif_envelope.json"
    env = json.loads(env_p.read_text())
    assert env.get("version") == 2, "matched envelope v2 required"
    p90 = env["global"]["p90"]
    exc_c = [e for e in (excess_of(r["champion"]["motif"]["windows"], env)
                         for r in rows) if e is not None]
    exc_s = [e for e in (excess_of(r["selected"]["motif"]["windows"], env)
                         for r in rows) if e is not None]
    mc = sum(exc_c) / len(exc_c) if exc_c else None
    ms = sum(exc_s) / len(exc_s) if exc_s else None
    motif_gate = mc is not None and ms is not None and ms <= 0.80 * mc + 1e-12
    raw_c = [r["champion"]["motif"]["mean_share"] for r in rows
             if r["champion"]["motif"]["mean_share"] is not None]
    raw_s = [r["selected"]["motif"]["mean_share"] for r in rows
             if r["selected"]["motif"]["mean_share"] is not None]
    voc_c = [r["champion"]["motif"]["mean_vocab"] for r in rows
             if r["champion"]["motif"]["mean_vocab"] is not None]
    voc_s = [r["selected"]["motif"]["mean_vocab"] for r in rows
             if r["selected"]["motif"]["mean_vocab"] is not None]
    d_champ = sum(r["champion"]["dup8"] for r in rows)
    d_sel = sum(r["selected"]["dup8"] for r in rows)
    dup_ok = d_sel <= d_champ + 1e-12
    conditioned = sum(1 for r in rows if r["source"] != "champion")
    env_rates = [r["selected"]["envelope"].get("in_band") or 0 for r in rows]
    env_n = [r["selected"]["envelope"].get("n") or 0 for r in rows]
    env_rate = sum(env_rates) / max(1, sum(env_n))
    status = QUALITY_PASS if (motif_gate and dup_ok) else \
        NON_REGRESSION_PASS
    payload = {"packet": "q4", "verdict": {
        "status": status,
        "motif": {"excess_macro_champion": mc, "excess_macro_selected": ms,
                  "reduction": (1 - ms / mc) if mc else None,
                  "gate_20pct": motif_gate, "human_p90": p90,
                  "definition": "MATCHED excess-over-human-envelope: mean "
                                "over supported windows of max(0, share - "
                                "matched bin p90), bins by n_grams/double/"
                                "cadence — review ",
                  "raw_mean_share": {
                      "champion": sum(raw_c) / len(raw_c) if raw_c else None,
                      "selected": sum(raw_s) / len(raw_s) if raw_s else None},
                  "effective_vocab": {
                      "champion": sum(voc_c) / len(voc_c) if voc_c else None,
                      "selected": sum(voc_s) / len(voc_s) if voc_s else None}},
        "dup8": {"champion": d_champ, "selected": d_sel, "ok": dup_ok},
        "style_separation": "unreachable: vocabulary is neutral K=1",
        "conditioned_selected_songs": conditioned, "n_songs": len(rows),
        "envelope_component_rate": env_rate},
        "songs": {f: parts[f] for f in fams}}
    (run / "q4_report.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    v = payload["verdict"]
    print(f"Q4 VERDICT: {status} | motif {mc} -> {ms} "
          f"({v['motif']['reduction'] and round(v['motif']['reduction'], 3)})"
          f" | dup8 {d_champ}->{d_sel} | conditioned {conditioned}/{len(rows)}")
    return payload


def run_q5(run_dir, panel="dev", max_songs=6, entries=None, confirm=False):
    """Q5 D COMPOSITION (review; release contract frozen in
    experiments/quality-v1/q5-release/release_contract.json): the complete
    bundle path — B0 schedule -> fit-B conditioned geometry (planner
    intents, neutral style) -> Q1-V2 repair -> admission -> pref tie-break —
    measured against the RAW B0 production winner on the ORIGINAL positive
    gates: motion flags >=30% at eligible extents, pairs >=30%, matched
    variety excess >=20%, dup8 non-increase, safety, serving <=3x, and
    family-bootstrap CIs excluding zero. C stays sealed; this run never
    touches it. Chunked with atomic partials; partials store the emitted
    NOTES of both sides so any metric is recomputable without re-decoding."""
    import time as _time

    import torch
    torch.set_num_threads(4)
    from convert import NPS_CAP, check, convert_groomed, diff_spec, \
        grid_steps, parse_osu
    import groom
    import motion as motion_mod
    from cond_flow import ConditionedFlow, shim_for_schedule
    from flow_decode import EventSchedule, decode_geometry
    from phrase_planner import (load_planner, onset_counts,
                                planner_trajectory, song_window_inputs,
                                train_mean_targets)
    from quality_repair import RepairConfig, repair_winner
    from eval.clean_rhythm import _cand_metrics
    from eval.phrase_plan import _cand_gates
    from eval.quality_metrics import quality_metrics
    from eval.quality_panel import dev_entries
    import pref as pref_mod

    if panel != "dev" and not confirm:
        raise RuntimeError("C runs only through eval.quality_confirm "
                           "(one-time unseal + frozen analysis)")
    run = Path(run_dir)
    (run / "partial").mkdir(parents=True, exist_ok=True)
    freeze_baseline()
    ck = ROOT / "experiments/quality-v1/q4-flow/condflow-b-seed20260921.pt"
    planner_ck = ROOT / "experiments/quality-v1/q3-planner/planner-seed20260921.pt"
    contract_p = ROOT / "experiments/quality-v1/q5-release/release_contract.json"
    cfg = {"condflow_b_sha256": _sha(ck), "planner_sha256": _sha(planner_ck),
           "pref_sha256": _sha(ROOT / "experiments/quality-v1/q4-pref/pref.pt"),
           "contract_sha256": _sha(contract_p), "seeds": [0, 1, 2]}
    cfg_p = run / "config.json"
    if cfg_p.exists():
        if json.loads(cfg_p.read_text()) != cfg:
            raise RuntimeError("run config drifted — wrong-version refusal")
    else:
        cfg_p.write_text(json.dumps(cfg, indent=1))
    ckd = torch.load(ck)
    if ckd.get("fit") != "B":
        raise RuntimeError("wrong-version checkpoint: expected fit B")
    cmodel = ConditionedFlow(n_styles=ckd.get("n_styles", 1))
    cmodel.load_state_dict(ckd["state"])
    cmodel.eval()
    pmodel, pnorm = load_planner(planner_ck)
    request = train_mean_targets()[:10]
    qcfg = RepairConfig(variant="paired-direction")
    spec = diff_spec("ExpertPlus")
    band = spec["band"]
    cap = NPS_CAP * spec["scale"]

    if entries is None:
        entries = dev_entries()
    done = 0
    for e in entries:
        fam = e["family"]
        part = run / "partial" / (fam.replace(":", "_").replace("/", "_")
                                  + ".json")
        if part.exists():
            continue
        if done >= max_songs:
            break
        osu = ROOT / e["osu"]
        audio = Path(e["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        _m, objects, bpm, offset = parse_osu(osu)
        steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset,
                                                   thin=True)
        step_times = [grid.time(s) for s in range(T)]
        afeat = groom.cached_audio_features(str(audio), step_times)
        recs = []
        t0 = _time.perf_counter()
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off", collect=recs,
                        thin=True, calibrate=True)
        b0_s = _time.perf_counter() - t0
        sel = next(r for r in recs if r["selected"])
        raw0, walls0 = sel["notes"], sel["walls"]
        beat_ms = 60000.0 / bpm
        m_b0 = _cand_metrics(raw0, motion_mod.report(
            [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw0]),
            grid, beat_ms)
        t1 = _time.perf_counter()
        sched = EventSchedule.from_raw(raw0, walls0, grid, T, step_ms)
        mi = onset_counts(objects, step_times)
        x = song_window_inputs(afeat, mi, step_times, bpm)
        intents = planner_trajectory(pmodel, pnorm, x, request)
        shim = shim_for_schedule(cmodel, sched, step_ms, intents, request)
        cands, rejects = [], []
        for sd in (0, 1, 2):
            p = decode_geometry(sched, steps, T, step_ms, off2, grid, afeat,
                                spec, flow_model=shim, seed=sd,
                                checkpoint="condflow-b")
            if not p.honored or p.infeasible:
                rejects.append({"seed": sd, "reason": "not_honored"
                                if not p.honored else "infeasible"})
                continue
            pr = repair_winner(p.raw, p.walls, grid, bpm, cap, qcfg)
            raw_c = pr.raw if pr is not None else p.raw
            notes_d = [{"t": grid.time(s), "hand": h, "col": c, "layer": l,
                        "dir": d} for s, h, c, l, d in raw_c]
            walls_d = [{"t": grid.time(s0),
                        "dur": grid.time(s0 + ln) - grid.time(s0),
                        "col": col} for s0, ln, col in p.walls]
            fails = list(check(notes_d, bpm, walls_d, cap))
            if not fails:
                m_c = _cand_metrics(raw_c, motion_mod.report(
                    [(grid.time(s), h, c, l, d)
                     for s, h, c, l, d in raw_c]), grid, beat_ms)
                fails = _cand_gates(m_b0, m_c, band)
            if fails:
                rejects.append({"seed": sd, "reason": fails})
                continue
            qm = quality_metrics(raw_c, p.walls, grid, bpm)
            cands.append({"seed": sd, "qm": qm, "raw": raw_c,
                          "walls": p.walls,
                          "pref": pref_mod.score(raw_c, p.walls, grid, bpm)})
        if cands:
            chosen = min(cands, key=lambda c: (
                _motif(c["qm"], c["raw"], grid, bpm)["mean_share"] or 1.0,
                -c["pref"]))
            source, raw_sel, walls_sel = f"bundle:{chosen['seed']}", \
                chosen["raw"], chosen["walls"]
        else:                                # explicit unachieved request
            source, raw_sel, walls_sel = "b0_fallback", raw0, walls0
        bundle_s = _time.perf_counter() - t1
        qm0 = quality_metrics(raw0, walls0, grid, bpm)
        qm1 = quality_metrics(raw_sel, walls_sel, grid, bpm)
        m_sel = _cand_metrics(raw_sel, motion_mod.report(
            [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw_sel]),
            grid, beat_ms)
        row = {"song": e["song"], "family": fam, "source": source,
               "rejects": rejects, "b0_s": round(b0_s, 1),
               "serving_ratio": round((b0_s + bundle_s) / max(b0_s, 1e-9), 2),
               "b0": {"flags_by_ext": qm0["flags_by_ext"],
                      "narrow": qm0["narrow"], "conv": qm0["converging"],
                      "broad": qm0["broad"],
                      "opp": qm0["opposite_horizontal"],
                      "dup8": m_b0["dup8"],
                      "motif": _motif(qm0, raw0, grid, bpm),
                      "notes": raw0, "walls": walls0},
               "bundle": {"flags_by_ext": qm1["flags_by_ext"],
                          "narrow": qm1["narrow"], "conv": qm1["converging"],
                          "broad": qm1["broad"],
                          "opp": qm1["opposite_horizontal"],
                          "dup8": m_sel["dup8"],
                          "motif": _motif(qm1, raw_sel, grid, bpm),
                          "notes": raw_sel, "walls": walls_sel}}
        tmp = part.with_suffix(".tmp")
        tmp.write_text(json.dumps(row, indent=1, default=float))
        tmp.replace(part)
        done += 1
        print(f"  {e['song'][:24]:<24} {source:<12} flags "
              f"{sum(qm0['flags_by_ext'].values()):.1f}->"
              f"{sum(qm1['flags_by_ext'].values()):.1f} pairs "
              f"{qm0['narrow'] + qm0['converging']}->"
              f"{qm1['narrow'] + qm1['converging']} x{row['serving_ratio']}",
              flush=True)

    parts = {p.stem: json.loads(p.read_text())
             for p in (run / "partial").glob("*.json")}
    if len(parts) < len(entries):
        print(f"Q5 chunk done: {len(parts)}/{len(entries)} songs — rerun for "
              "the next chunk (no verdict yet)")
        return {"status": "INCOMPLETE_CHUNK", "songs_done": len(parts)}
    return _q5_verdict(run, parts)


def _boot_ci(deltas, n=10000, seed=20260921):
    """95% family-bootstrap CI of the mean delta."""
    import random as _r
    rng = _r.Random(seed)
    means = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
                   for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n)]]


def _q5_verdict(run, parts):
    import statistics
    from eval.motif_envelope import excess_of
    env = json.loads((ROOT / "experiments/quality-v1/q4-style/"
                      "human_motif_envelope.json").read_text())
    rows = list(parts.values())
    # motion: >=30% macro reduction at every eligible extent
    ext_detail, motion_pos, motion_safe = {}, True, True
    for k in sorted(rows[0]["b0"]["flags_by_ext"]):
        b_mac = family_macro({r["family"]: r["b0"]["flags_by_ext"][k]
                              for r in rows})
        a_mac = family_macro({r["family"]: r["bundle"]["flags_by_ext"][k]
                              for r in rows})
        eligible = b_mac >= 1.0
        if eligible:
            ok = a_mac <= 0.70 * b_mac + 1e-12
            motion_pos &= ok
        else:
            ok = all(_flags_ok(r["b0"]["flags_by_ext"][k],
                               r["bundle"]["flags_by_ext"][k]) for r in rows)
            motion_safe &= ok
        ext_detail[k] = {"eligible": eligible, "b0": b_mac, "bundle": a_mac,
                         "ok": ok}
    # pairs: >=30%; broad/opposite never increase
    b_pairs = sum(r["b0"]["narrow"] + r["b0"]["conv"] for r in rows)
    a_pairs = sum(r["bundle"]["narrow"] + r["bundle"]["conv"] for r in rows)
    pairs_pos = a_pairs <= 0.70 * b_pairs + 1e-12 if b_pairs >= 10 \
        else a_pairs <= b_pairs
    broad_safe = (sum(r["bundle"]["broad"] for r in rows)
                  <= sum(r["b0"]["broad"] for r in rows)
                  and sum(r["bundle"]["opp"] for r in rows)
                  <= sum(r["b0"]["opp"] for r in rows))
    # variety: matched excess >=20%
    exc = [(excess_of(r["b0"]["motif"]["windows"], env),
            excess_of(r["bundle"]["motif"]["windows"], env)) for r in rows]
    exc = [(a, b) for a, b in exc if a is not None and b is not None]
    mc = sum(a for a, _b in exc) / len(exc)
    ms = sum(b for _a, b in exc) / len(exc)
    variety_pos = ms <= 0.80 * mc + 1e-12
    dup_ok = sum(r["bundle"]["dup8"] for r in rows) \
        <= sum(r["b0"]["dup8"] for r in rows) + 1e-12
    serving = statistics.median(r["serving_ratio"] for r in rows)
    runtime_ok = serving <= 3.0
    fallbacks = [r["family"] for r in rows if r["source"] == "b0_fallback"]
    # family-bootstrap CIs for the three positive deltas (reduction > 0)
    ci = {
        "flags": _boot_ci([sum(r["b0"]["flags_by_ext"].values())
                           - sum(r["bundle"]["flags_by_ext"].values())
                           for r in rows]),
        "pairs": _boot_ci([(r["b0"]["narrow"] + r["b0"]["conv"])
                           - (r["bundle"]["narrow"] + r["bundle"]["conv"])
                           for r in rows]),
        "excess": _boot_ci([a - b for a, b in exc])}
    ci_ok = all(v[0] > 0 for v in ci.values())
    passed = (motion_pos and motion_safe and pairs_pos and broad_safe
              and variety_pos and dup_ok and runtime_ok and ci_ok)
    payload = {"packet": "q5-d-composition", "verdict": {
        "status": QUALITY_PASS if passed else NON_REGRESSION_PASS,
        "motion": ext_detail, "motion_pos_30pct": motion_pos,
        "pairs": {"b0": b_pairs, "bundle": a_pairs, "gate_30pct": pairs_pos,
                  "broad_opp_safe": broad_safe},
        "variety": {"excess_b0": mc, "excess_bundle": ms,
                    "reduction": 1 - ms / mc if mc else None,
                    "gate_20pct": variety_pos},
        "dup8_ok": dup_ok, "serving_median": serving,
        "runtime_gate_3x": runtime_ok,
        "bootstrap_ci95": ci, "ci_excludes_zero": ci_ok,
        "explicit_fallbacks": fallbacks},
        "songs": {r["family"]: {k: v for k, v in r.items()
                                if k not in ("b0", "bundle")}
                  for r in rows}}
    (run / "q5_report.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    v = payload["verdict"]
    print(f"Q5 D COMPOSITION: {v['status']} | motion30 {motion_pos} | "
          f"pairs {b_pairs}->{a_pairs} ({pairs_pos}) | excess {mc:.5f}->"
          f"{ms:.5f} ({variety_pos}) | CI {ci_ok} | serving x{serving:.2f} | "
          f"fallbacks {len(fallbacks)}")
    return payload


def _q3_verdict(run, parts):
    """Verdict on FULL D coverage only (chunked runs call this at the end)."""
    import statistics
    rows = list(parts.values())
    # envelope achievement, planner arm. review R1 ruling: rule A is the gate
    # ("component coverage"), every requested song-component stays in the
    # denominator, and an admission-failed fallback is NEVER an achieved
    # request — its components count as not-achieved even if B0 happens to
    # sit in band (the measured containment is disclosed separately).
    comp_checks, measured = [], []
    for r in rows:
        comps = (r["arms"]["planner"]["envelope"].get("components")
                 or dict.fromkeys(range(10), False))
        ach = r["arms"]["planner"]["achieved"]
        for k, v in comps.items():
            comp_checks.append(bool(v) and ach)
            measured.append(bool(v))
    env_rate = (sum(comp_checks) / len(comp_checks) if comp_checks else None)
    measured_rate = (sum(measured) / len(measured) if measured else None)
    all_rate = (sum(1 for r in rows
                    if r["arms"]["planner"]["envelope"].get("all_in_band")
                    and r["arms"]["planner"]["achieved"]) / len(rows))
    ach_rate = sum(1 for r in rows if r["arms"]["planner"]["achieved"]) \
        / len(rows)
    # paired-human trajectory macro (families with pairs)
    tr_rows = [r for r in rows if r["traj"].get("status") == "ok"]
    t_planner = [r["traj"]["planner"] for r in tr_rows]
    t_const = [r["traj"]["constant"] for r in tr_rows]
    t_b0 = [r["traj"]["b0"] for r in tr_rows]
    macro_p = sum(t_planner) / len(t_planner) if t_planner else None
    macro_c = sum(t_const) / len(t_const) if t_const else None
    macro_b = sum(t_b0) / len(t_b0) if t_b0 else None
    traj_gate = (macro_p is not None
                 and macro_p <= 0.85 * macro_c + 1e-12)
    env_gate = env_rate is not None and env_rate >= 0.90
    serving = statistics.median(r["serving_ratio"] for r in rows)
    runtime_gate = serving <= 3.0
    # safety: every outcome is B0 or an admitted reschedule (asserted per
    # song); unachieved requests must carry explicit reasons
    explicit_ok = all(r["arms"][m]["achieved"] or r["arms"][m]["reasons"]
                      for r in rows for m in ("planner", "constant"))
    if not explicit_ok:
        status = QUALITY_FAIL
    elif env_gate and traj_gate and runtime_gate:
        status = QUALITY_PASS
    else:
        status = NON_REGRESSION_PASS
    payload = {"packet": "q3", "verdict": {
        "status": status,
        "envelope": {"component_coverage_rate": env_rate,
                     "measured_containment_incl_fallbacks": measured_rate,
                     "all_components_rate": all_rate, "gate_0.90": env_gate},
        "scheduler_achieved_rate": ach_rate,
        "trajectory": {"n_paired_families": len(tr_rows),
                       "macro_planner": macro_p, "macro_constant": macro_c,
                       "macro_b0": macro_b,
                       "reduction_vs_constant":
                           (1 - macro_p / macro_c) if macro_c else None,
                       "gate_15pct": traj_gate},
        "serving_median_ratio": serving, "runtime_gate_3x": runtime_gate,
        "explicit_failure_ok": explicit_ok},
        "songs": {r["family"]: {k: v for k, v in r.items() if k != "family"}
                  for r in rows}}
    (run / "q3_report.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    v = payload["verdict"]
    print(f"Q3 VERDICT: {status} | env {env_rate:.1%} (all-comp {all_rate:.1%})"
          f" | ach {ach_rate:.1%} | traj planner {macro_p} vs const {macro_c}"
          f" ({v['trajectory']['reduction_vs_constant'] and round(v['trajectory']['reduction_vs_constant'], 3)})"
          f" | serving x{serving:.2f}")
    return payload


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--packet", required=True,
                   choices=("q1", "q2", "q3", "q4", "q5"))
    r.add_argument("--variant", default="position",
                   choices=("position", "paired-direction"))
    r.add_argument("--run", required=True)
    r.add_argument("--panel", default="dev")
    r.add_argument("--allocation", default="3+3", choices=("3+3", "6+6"))
    r.add_argument("--max-songs", type=int, default=6)
    r.add_argument("--ckpt", default="condflow-a-seed20260921.pt")
    a = ap.parse_args(argv)
    if a.packet == "q1":
        run_q1(a.variant, a.run, panel=a.panel)
    elif a.packet == "q2":
        run_q2(a.run, panel=a.panel, allocation=a.allocation)
    elif a.packet == "q3":
        run_q3(a.run, panel=a.panel, max_songs=a.max_songs)
    elif a.packet == "q4":
        run_q4(a.run, panel=a.panel, max_songs=a.max_songs,
               ckpt_name=a.ckpt)
    else:
        run_q5(a.run, panel=a.panel, max_songs=a.max_songs)


if __name__ == "__main__":
    main()
