"""E01 visibility panel audit (Task 3): frozen-panel report + witnesses.

Read-only. Verifies the panel, traces every target under the standard scenario
(and, where authored settings are verified, separately under those), selects
witness strata, runs one-factor sensitivity + a 5 ms convergence check, and
writes a canonical deterministic report plus static witness plots. Findings are
NOT failures; only integrity/fixture/sampling-gate failure exits nonzero. No
result unmasks a cell or changes generation.

  .venv/bin/python -m eval.visibility_audit --panel eval/visibility_panel.json \\
    --out experiments/visibility/run-a
"""
import argparse
import json
import sys
import time
from pathlib import Path

# reuse the wall audit's repaired panel/integrity checks (same panel schema)
from eval.wall_audit import (_check_integrity, _code_rev, _resolve, _sha256,
                             _validate_panel)
from eval.visibility import (STD_LIFETIME_S, STD_NJS, is_banned_cell, load_scene,
                             resolve_authored, trace_target)

NOMINAL = {"eye": (0.0, 1.60, -0.65), "face_side_m": 0.5, "step_s": 0.010}
SENS = {
    "eye_x-0.15": {"eye": (-0.15, 1.60, -0.65)},
    "eye_x+0.15": {"eye": (0.15, 1.60, -0.65)},
    "eye_y1.40": {"eye": (0.0, 1.40, -0.65)},
    "eye_y1.80": {"eye": (0.0, 1.80, -0.65)},
    "side0.45": {"face_side_m": 0.45},
    "side0.55": {"face_side_m": 0.55},
    "njs12": {"_njs": 12.0},
    "njs20": {"_njs": 20.0},
    "T0.40": {"_lifetime_s": 0.40},
    "T0.80": {"_lifetime_s": 0.80},
}
# the seven design metrics gated by the 5 ms convergence check
COV_METRIC = "area_time_integral"
DUR_METRICS = ("time_ge_0.5_s", "longest_ge_0.5_s", "time_ge_0.8_s",
               "longest_ge_0.8_s", "early_clear_s", "terminal_clear_s")
COV_TOL = 0.02
DUR_TOL = 0.020


def converged(base, fine):
    """True iff the 5 ms trace matches the 10 ms trace within the stated bounds
    on ALL seven metrics: mean coverage <=0.02, every duration <=20 ms."""
    if abs(fine[COV_METRIC] - base[COV_METRIC]) > COV_TOL:
        return False
    return all(abs(fine[k] - base[k]) <= DUR_TOL for k in DUR_METRICS)


def _check_authored(entries):
    """Re-resolve every verified-authored binding LIVE from Info and compare it
    to the panel's recorded record. A stale/altered verified record is an
    integrity failure, not a silent fallback (review R2). Explicit unknown
    authored records are fine (standard scenario still applies)."""
    failures = []
    for e in entries:
        a = e.get("authored") or {}
        if a.get("status") != "verified":
            continue
        live = resolve_authored(_resolve(e["chart"]), e.get("sha256"))
        if (live["status"] != "verified" or _fnum(a.get("njs")) != live["njs"]
                or _fnum(a.get("offset_beats")) != live["offset_beats"]
                or a.get("info_sha256") != live["info_sha256"]):
            failures.append({"id": e["id"], "reason": "authored_record_altered"})
    return failures


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _check_scenario(panel):
    """The declared standard scenario must equal the executed constants."""
    sc = panel.get("standard_scenario") or {}
    if (_fnum(sc.get("njs")) != STD_NJS
            or _fnum(sc.get("lifetime_s")) != STD_LIFETIME_S):
        return [{"id": "<panel>", "reason": "standard_scenario_mismatch"}]
    return []


def _scene_variant(scene, ov):
    s = dict(scene)
    if "_njs" in ov:
        s["njs"] = ov["_njs"]          # hold T, change NJS
    if "_lifetime_s" in ov:
        s["lifetime_s"] = ov["_lifetime_s"]   # hold NJS, change T
    return s


def _config_variant(ov):
    c = dict(NOMINAL)
    for k in ("eye", "face_side_m", "step_s"):
        if k in ov:
            c[k] = ov[k]
    return c


def _summarize(metrics):
    """Per-chart proxy counts from the nominal full-chart traces."""
    hidden = [m for m in metrics if m["max_coverage"] >= 0.5]
    by_banned = [m for m in hidden
                 if any(c["banned"] for c in m["hidden_occluder_cells"])]
    only_nb = [m for m in hidden if m["hidden_occluder_cells"]
               and not any(c["banned"] for c in m["hidden_occluder_cells"])]
    return {"n_targets": len(metrics), "n_hidden_ge_0.5": len(hidden),
            "n_hidden_by_banned_occluder": len(by_banned),
            "n_hidden_only_nonbanned": len(only_nb),
            "max_coverage": max((m["max_coverage"] for m in metrics), default=0.0)}


def _witnesses(scene, metrics):
    """Up to 20 witness target/occluder ids across four strata, deterministic."""
    notes = {n.id: n for n in scene["notes"]}
    contributing = set()
    for m in metrics:
        contributing.update(m["contributing_ids"])
    by_hit = sorted(metrics, key=lambda m: (m["hit_s"], m["target_id"]))
    cat1 = [m["target_id"] for m in by_hit if m["max_coverage"] >= 0.5
            and any(c["banned"] for c in m["hidden_occluder_cells"])][:5]
    # banned-cell notes that hide no target (any positive overlap counts)
    cat2 = sorted((n.id for n in scene["notes"]
                   if is_banned_cell(n.col, n.row) and n.id not in contributing),
                  key=lambda i: (notes[i].hit_s, i))[:5]
    cat3 = [m["target_id"] for m in by_hit if m["max_coverage"] >= 0.5
            and m["hidden_occluder_cells"]
            and not any(c["banned"] for c in m["hidden_occluder_cells"])][:5]
    # uniformly spaced valid targets across the chart
    cat4 = []
    if by_hit:
        step = max(1, len(by_hit) // 5)
        cat4 = [by_hit[i]["target_id"] for i in range(0, len(by_hit), step)][:5]
    strata = {"hidden_by_banned": cat1, "banned_hides_nothing": cat2,
              "hidden_only_nonbanned": cat3, "uniform_valid": cat4}
    ordered, seen = [], set()
    for group in (cat1, cat3, cat4, cat2):
        for tid in group:
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
    return strata, ordered[:20]


def _m7(m):
    return {COV_METRIC: m[COV_METRIC], **{k: m[k] for k in DUR_METRICS}}


def _sensitivity(scene, tid):
    """One-factor viewpoint/size/NJS/T variants (reported) + a 5 ms convergence
    gate over all seven metrics (enforced)."""
    base = trace_target(scene, tid, NOMINAL, detail=False)
    out = {"base": _m7(base), "variants": {}}
    for name, ov in SENS.items():
        m = trace_target(_scene_variant(scene, ov), tid, _config_variant(ov),
                         detail=False)
        if m.get("status") != "ok":
            out["variants"][name] = {"status": m.get("status")}
            continue
        out["variants"][name] = {
            "d_area_time_integral": m["area_time_integral"] - base["area_time_integral"],
            "d_time_ge_0.5_s": m["time_ge_0.5_s"] - base["time_ge_0.5_s"]}
    fine = trace_target({**scene}, tid, {**NOMINAL, "step_s": 0.005}, detail=False)
    out["fine_5ms"] = _m7(fine)
    out["converged"] = converged(_m7(base), _m7(fine))
    return out


def _trace_all(scene):
    return [trace_target(scene, n.id, NOMINAL, detail=False)
            for n in scene["notes"]]


def audit(panel_path, out_dir, make_plots=True):
    t0 = time.time()
    panel = json.loads(_resolve(panel_path).read_text())
    pfail, entries = _validate_panel(panel)
    failures = pfail + _check_integrity(entries)
    if not pfail:            # only bind authored/scenario once entries are sane
        failures += _check_authored(entries) + _check_scenario(panel)
    integrity_ok = not failures
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, scenes, witness_ids = [], {}, {}
    conv_flags = []          # per-witness 5ms convergence; tri-state overall
    if integrity_ok:
        for e in sorted(entries, key=lambda x: x["id"]):
            std = load_scene(e, "standard")
            entry_out = {"id": e["id"], "role": e["role"], "bpm": float(e["bpm"]),
                         "authored_status": (e.get("authored") or {}).get("status"),
                         "ignored": std.get("ignored", {}),
                         "n_invalid_notes": len(std.get("invalid", [])),
                         "n_unsupported_notes": len(std.get("unsupported", []))}
            if std["chart_unknown"]:
                entry_out["standard"] = {"status": "chart_unknown",
                                         "scope": std.get("scope")}
                maps.append(entry_out)
                continue
            metrics = _trace_all(std)
            strata, wids = _witnesses(std, metrics)
            sens = {tid: _sensitivity(std, tid) for tid in wids}
            conv_flags += [s["converged"] for s in sens.values()]
            wdetail = [trace_target(std, tid, NOMINAL, detail=True) for tid in wids]
            entry_out["standard"] = {"status": "ok", **_summarize(metrics),
                                     "witness_strata": strata,
                                     "witnesses": wdetail, "sensitivity": sens}
            scenes[e["id"]] = std
            witness_ids[e["id"]] = wids
            # authored mode, separately, where verified
            if (e.get("authored") or {}).get("status") == "verified":
                auth = load_scene(e, "authored")
                if not auth["chart_unknown"]:
                    entry_out["authored"] = {"status": "ok",
                                             "njs": auth["njs"],
                                             "lifetime_s": auth["lifetime_s"],
                                             **_summarize(_trace_all(auth))}
            maps.append(entry_out)

    # tri-state: no witnesses is not_applicable, never a vacuous pass
    if not integrity_ok:
        sampling = "not_run"
    elif not conv_flags:
        sampling = "not_applicable"
    else:
        sampling = "converged" if all(conv_flags) else "not_converged"
    sampling_ok = sampling != "not_converged"

    agg = _aggregate(maps)
    report = {
        "schema_version": 1,
        "certification": "CERTIFIED" if integrity_ok else "UNCERTIFIED",
        "sampling": sampling,
        "sampling_witnesses": len(conv_flags),
        "config": {"model": "e01_projected_face_occlusion", "nominal": {
            "eye": list(NOMINAL["eye"]), "face_side_m": NOMINAL["face_side_m"],
            "step_s": NOMINAL["step_s"], "row_y": [0.60, 1.15, 1.65]},
            "standard_scenario_declared": panel.get("standard_scenario"),
            "standard_scenario_executed": {"njs": STD_NJS,
                                           "lifetime_s": STD_LIFETIME_S},
            "code_rev": _code_rev(), "panel_sha256": _sha256(_resolve(panel_path)),
            "panel_frozen": panel.get("frozen")},
        "integrity": {"ok": integrity_ok, "checked": len(entries),
                      "failures": sorted(failures, key=lambda f: (f["id"], f["reason"]))},
        "panel_incomplete": bool(panel.get("missing")) or not integrity_ok,
        "aggregate": agg,
        "maps": maps,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")

    plots = []
    if integrity_ok and make_plots:
        plots = _plots(maps, scenes, out_dir)
    (out_dir / "resources.log").write_text(json.dumps({
        "wall_clock_s": round(time.time() - t0, 3), "n_maps": len(maps),
        "plots": plots}, indent=1) + "\n")
    # findings never fail the run; integrity/sampling gates do
    code = 0 if (integrity_ok and sampling_ok) else 1
    return code, report


def _aggregate(maps):
    ok = [m for m in maps if m.get("standard", {}).get("status") == "ok"]
    def tot(role, key):
        return sum(m["standard"][key] for m in ok if m["role"] == role)
    agg = {}
    for role in ("generated", "reference"):
        rms = [m for m in ok if m["role"] == role]
        agg[role] = {
            "n_maps": len(rms),
            "n_targets": tot(role, "n_targets"),
            "n_hidden_ge_0.5": tot(role, "n_hidden_ge_0.5"),
            "n_hidden_by_banned_occluder": tot(role, "n_hidden_by_banned_occluder"),
            "n_hidden_only_nonbanned": tot(role, "n_hidden_only_nonbanned"),
        }
    # the proxy-vs-ban comparison headline (occluder cell, not target cell)
    agg["hidden_by_banned_occluder_total"] = sum(
        m["standard"]["n_hidden_by_banned_occluder"] for m in ok)
    agg["hidden_total"] = sum(m["standard"]["n_hidden_ge_0.5"] for m in ok)
    agg["charts_chart_unknown"] = sum(
        1 for m in maps if m.get("standard", {}).get("status") == "chart_unknown")
    return agg


# each figure category, the stratum it MUST draw from, and its role label. The
# benign category is an OUTGOING-occlusion claim (a centre note that hides no
# other note) and must come from banned_hides_nothing, never a uniform target.
PLOT_CATS = [
    ("benign_centre", "banned_hides_nothing", "outgoing_occluder_hides_nothing",
     "plot_benign_centre.png"),
    ("centre_caused_hiding", "hidden_by_banned", "target_hidden_by_banned_occluder",
     "plot_centre_hiding.png"),
    ("noncentre_hiding", "hidden_only_nonbanned", "target_hidden_by_nonbanned",
     "plot_noncentre_hiding.png"),
    ("recovery", "recovery", "target_recovers_before_hit", "plot_recovery.png"),
]


def _plot_selection(maps):
    """Choose one witness per figure category from the REQUIRED stratum, in
    deterministic map order, recording IDs/time/camera/settings/role. A uniform
    target can never satisfy the benign-centre category. Absent categories are
    marked for a labeled construction fixture rather than a wrong witness."""
    recs = []
    for cat, stratum, role, fname in PLOT_CATS:
        rec = None
        for m in maps:
            st = m.get("standard", {})
            if st.get("status") != "ok":
                continue
            if stratum == "recovery":
                w = next((x for x in st["witnesses"]
                          if x.get("max_coverage", 0) >= 0.5
                          and x.get("terminal_clear_s", 0) >= 0.05), None)
                tid = w["target_id"] if w else None
            else:
                ids = st["witness_strata"].get(stratum) or []
                tid = ids[0] if ids else None
            if tid is None:
                continue
            w = next((x for x in st["witnesses"] if x["target_id"] == tid), None)
            # peak-coverage sample time, else spawn+T/2 for a never-hidden note
            t_snap = None
            if w and w.get("samples"):
                peak = max(w["samples"], key=lambda s: s["coverage"])
                t_snap = (peak["start_s"] + peak["end_s"]) / 2
            rec = {"category": cat, "role": role, "fname": fname,
                   "entry_id": m["id"], "target_id": tid,
                   "snapshot_time_s": t_snap,
                   "camera_eye": list(NOMINAL["eye"]),
                   "settings": {"njs": STD_NJS, "lifetime_s": STD_LIFETIME_S,
                                "face_side_m": NOMINAL["face_side_m"]},
                   "outgoing_targets_hidden": 0 if cat == "benign_centre" else None,
                   "absent": False}
            break
        if rec is None:
            rec = {"category": cat, "role": role, "fname": fname,
                   "entry_id": None, "target_id": None, "absent": True}
        recs.append(rec)
    return recs


def _plots(maps, scenes, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from eval.visibility import note_face, project_face

    made = []
    for rec in _plot_selection(maps):
        if rec["absent"]:
            continue        # a labeled construction fixture would go here
        entry = _entry(maps, rec["entry_id"])
        w = next((x for x in entry["standard"]["witnesses"]
                  if x["target_id"] == rec["target_id"]), None)
        scene = scenes.get(rec["entry_id"])
        if w is None or "samples" not in w or scene is None:
            continue
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.4))
        # left: projected snapshot at the peak time (target + nearer occluders)
        t = rec["snapshot_time_s"]
        njs, T = scene["njs"], scene["lifetime_s"]
        side, eye = NOMINAL["face_side_m"], NOMINAL["eye"]
        notes = {n.id: n for n in scene["notes"]}
        tgt = notes[rec["target_id"]]
        def _rect(n, color, lw, fill):
            f = note_face(n, t, njs, side)
            if f.z - eye[2] <= 0:
                return
            u0, u1, v0, v1, _ = project_face(f, eye)
            axL.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, facecolor=color,
                          edgecolor=color, alpha=(0.35 if fill else 1.0),
                          fill=fill, lw=lw))
        for n in scene["notes"]:
            if n.id != tgt.id and n.hit_s - T <= t < n.hit_s and n.hit_s < tgt.hit_s:
                banned = (n.row == 1 and n.col in (1, 2))
                _rect(n, "#8e44ad" if banned else "#c0392b", 1.0, True)
        _rect(tgt, "#27ae60", 1.6, False)
        axL.set_xlim(-0.5, 0.5)
        axL.set_ylim(-0.5, 0.5)
        axL.set_aspect("equal")
        axL.set_xlabel("projected u")
        axL.set_ylabel("projected v")
        axL.set_title(f"snapshot t={t:.3f}s (green=target, red=occluder,\n"
                      "purple=banned-cell occluder)", fontsize=7)
        # right: coverage vs time
        rel0 = w["samples"][0]["start_s"]
        ts = [(s["start_s"] + s["end_s"]) / 2 - rel0 for s in w["samples"]]
        axR.plot(ts, [s["coverage"] for s in w["samples"]], color="#c0392b",
                 label="face coverage")
        axR.plot(ts, [s.get("marker_coverage", 0.0) for s in w["samples"]],
                 color="#2980b9", ls="--", label="central-marker proxy")
        axR.axhline(0.5, color="#7f8c8d", lw=0.5, ls=":")
        axR.set_ylim(-0.02, 1.02)
        axR.set_xlabel("s since target spawn")
        axR.set_ylabel("covered fraction")
        axR.legend(fontsize=6, loc="upper left")
        fig.suptitle(f"{rec['category']} [{rec['role']}] {rec['entry_id']} "
                     f"{rec['target_id']}", fontsize=8)
        fig.text(0.01, 0.01, "projected-face occlusion proxy; NJS16/T0.6, "
                 "eye(0,1.60,-0.65); omits jump arc/rotation/FOV. Not readability.",
                 fontsize=5)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(out_dir / rec["fname"], dpi=110, metadata={"Software": None})
        plt.close(fig)
        made.append(rec["fname"])
    return made


def _entry(maps, eid):
    return next(m for m in maps if m["id"] == eid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="eval/visibility_panel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    code, rep = audit(a.panel, a.out, make_plots=not a.no_plots)
    print(f"{rep['certification']}: {rep['integrity']['checked']} charts, "
          f"sampling={rep['sampling']}")
    if rep["maps"]:
        ag = rep["aggregate"]
        print(f"hidden(>=.5) total {ag['hidden_total']}, "
              f"by banned-cell occluder {ag['hidden_by_banned_occluder_total']}, "
              f"charts unknown {ag['charts_chart_unknown']}")
    print(f"-> {a.out}/report.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
