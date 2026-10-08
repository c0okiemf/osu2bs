"""Cadence / coordination / recovery difficulty-vector diagnostic (runner).

Read-only. Measures per-hand cadence, two-hand coincidence and inter-event gaps
on 45 saved charts to test whether aggregate rate hides per-hand differences.
Descriptive evidence only — NOT a difficulty score, a learned ranking, or a
claim of equal perceived difficulty across genres. No generation/training/
scoring/viewer changes. See
docs/specs/2026-09-20-difficulty-vector-diagnostic-design.md.

Task 1 here: freeze + validate the bounded panel. Measurement (difficulty_vector)
and reporting (audit) follow.
"""
import argparse
import json
import sys
import time
from math import isfinite
from pathlib import Path

from eval.wall_audit import _check_integrity, _code_rev, _resolve, _sha256
from eval.visibility import load_scene, resolve_authored
from eval.difficulty_vector import measure_scene

COHORT_COUNTS = {"replay_off": 30, "human": 11, "historical_export": 4}
REQUIRED = ("id", "role", "cohort", "chart", "sha256", "bpm", "origin_s",
            "timing_source", "timing_source_sha256", "family", "label",
            "label_status", "selected", "seed", "source_kind", "source_id")


def _finite_pos(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) and v > 0


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)


def _check_historical(entries):
    """Bind each historical-export chart's BPM and hash to its named benchmark
    song record directly (NOT the generated results branch); altered recorded
    authority values/bytes are integrity failures."""
    failures, bench = [], None
    for e in entries:
        ts = _resolve(e["timing_source"])
        chart = _resolve(e["chart"])
        if not ts.exists():
            failures.append({"id": e["id"], "reason": "timing_source_missing"})
            continue
        if _sha256(ts) != e["timing_source_sha256"]:
            failures.append({"id": e["id"], "reason": "timing_source_sha256_mismatch"})
        if not chart.exists():
            failures.append({"id": e["id"], "reason": "chart_missing"})
            continue
        if _sha256(chart) != e["sha256"]:
            failures.append({"id": e["id"], "reason": "chart_sha256_mismatch"})
        if bench is None:
            bench = {s["id"]: s for s in json.loads(ts.read_text())["songs"]}
        song = bench.get(e["source_id"])
        if song is None:
            failures.append({"id": e["id"], "reason": "benchmark_song_missing"})
            continue
        if abs(float(song["bpm"]) - float(e["bpm"])) > 1e-9:
            failures.append({"id": e["id"], "reason": "bpm_not_bound_to_source"})
        want = (song.get("difficulties") or {}).get(Path(e["chart"]).name)
        if want != e["sha256"]:
            failures.append({"id": e["id"], "reason": "chart_hash_not_in_benchmark"})
    return failures


def authored_label(chart_path, chart_sha256, cohort):
    """Live authored rank/label for an entry: replay-off is a generated target
    (no human tier); otherwise re-resolve Info by exact Standard filename+hash.
    Returns {label_status, authored_rank}."""
    if cohort == "replay_off":
        return {"label_status": "generated_target", "authored_rank": None}
    r = resolve_authored(_resolve(chart_path), chart_sha256)
    if r["status"] == "verified":
        # the authoritative display label is the Info custom label, else its
        # difficulty name; info_sha256 is the frozen Info provenance
        return {"label_status": "verified", "authored_rank": r.get("rank"),
                "authored_njs": r.get("njs"),
                "authored_label": r.get("label") or r.get("difficulty"),
                "info_sha256": r.get("info_sha256")}
    return {"label_status": "benchmark_label_unverified", "authored_rank": None,
            "authored_label": None, "info_sha256": None}


def _check_labels(entries):
    """A stored label record must match the live Info binding (review R2 pattern):
    a duplicate rank-9 custom label or an unbound alias cannot become an
    authenticated ordering."""
    failures = []
    for e in entries:
        live = authored_label(e["chart"], e["sha256"], e["cohort"])
        if e.get("label_status") != live["label_status"]:
            failures.append({"id": e["id"], "reason": "label_status_altered"})
        elif live["label_status"] == "verified":
            if e.get("authored_rank") != live["authored_rank"]:
                failures.append({"id": e["id"], "reason": "authored_rank_altered"})
            elif e.get("label") != live["authored_label"]:
                failures.append({"id": e["id"], "reason": "label_text_altered"})
            # frozen Info.dat provenance: the on-disk Info hash must match the
            # value frozen in the panel (catches any Info byte change, not only
            # ones that alter the resolved rank/label)
            elif not e.get("info_sha256"):
                failures.append({"id": e["id"], "reason": "missing_info_sha256"})
            elif e["info_sha256"] != live["info_sha256"]:
                failures.append({"id": e["id"], "reason": "info_hash_altered"})
    return failures


def _check_selected(entries):
    """A replay-off entry's selected flag and seed must match its results.json
    e2e off record — a candidate cannot be relabeled selected in the panel."""
    failures, cache = [], {}
    for e in entries:
        if e["cohort"] != "replay_off":
            continue
        ts = _resolve(e["timing_source"])
        key = str(ts)
        if key not in cache:
            try:
                cache[key] = json.loads(ts.read_text())
            except (OSError, json.JSONDecodeError):
                cache[key] = None
        doc = cache[key]
        if not doc:
            continue                # hash check already flagged a bad source
        name = Path(e["chart"]).name
        rec = next((r for r in doc.get("e2e", [])
                    if r.get("mode") == "off" and r.get("dat") == name), None)
        if rec is None:
            failures.append({"id": e["id"], "reason": "candidate_record_missing"})
            continue
        if bool(rec.get("selected")) != bool(e["selected"]):
            failures.append({"id": e["id"], "reason": "selected_flag_mismatch"})
        if rec.get("seed") != e["seed"]:
            failures.append({"id": e["id"], "reason": "seed_mismatch"})
    return failures


def validate_panel(panel):
    """Return (structurally-usable entries, failures). Validates schema, frozen
    expected membership and cohort counts, required fields/types, unique IDs,
    finite BPM/origin, per-cohort integrity (inherited wall entries via the wall
    check; historical vs the benchmark record) and live label binding. Callers
    do not measure after any failure."""
    failures = []
    if not isinstance(panel, dict) or panel.get("schema_version") != 1:
        failures.append({"id": "<panel>", "reason": "bad_or_missing_schema_version"})
    if isinstance(panel, dict) and panel.get("missing"):
        failures.append({"id": "<panel>", "reason": "declared_missing_entries"})
    entries = panel.get("entries") if isinstance(panel, dict) else None
    if not isinstance(entries, list) or not entries:
        failures.append({"id": "<panel>", "reason": "empty_or_missing_entries"})
        return [], failures
    good, seen, ids = [], set(), []
    for e in entries:
        eid = e.get("id", "<no-id>") if isinstance(e, dict) else "<not-object>"
        if not isinstance(e, dict) or any(f not in e for f in REQUIRED):
            failures.append({"id": eid, "reason": "missing_required_field"})
            continue
        if e["cohort"] not in COHORT_COUNTS:
            failures.append({"id": eid, "reason": "bad_cohort"})
            continue
        ok = True
        if not _finite_pos(e["bpm"]):
            failures.append({"id": eid, "reason": "nonfinite_bpm"})
            ok = False
        if not _finite(e["origin_s"]):
            failures.append({"id": eid, "reason": "nonfinite_origin"})
            ok = False
        if eid in seen:
            failures.append({"id": eid, "reason": "duplicate_id"})
            ok = False
        seen.add(eid)
        ids.append(eid)
        if ok:
            good.append(e)
    exp = panel.get("expected_ids")
    if not isinstance(exp, list):
        failures.append({"id": "<panel>", "reason": "missing_expected_ids"})
    else:
        for m in sorted(set(exp) - set(ids)):
            failures.append({"id": m, "reason": "missing_expected_entry"})
        for x in sorted(set(ids) - set(exp)):
            failures.append({"id": x, "reason": "unexpected_entry"})
    ec = panel.get("expected_counts")
    if not isinstance(ec, dict):
        failures.append({"id": "<panel>", "reason": "missing_expected_counts"})
    else:
        for coh, want in ec.items():
            got = sum(1 for e in good if e.get("cohort") == coh)
            if got != want:
                failures.append({"id": f"<{coh}>",
                                 "reason": f"cohort_count_mismatch:{got}!={want}"})
    inherited = [e for e in good if e["cohort"] in ("replay_off", "human")]
    historical = [e for e in good if e["cohort"] == "historical_export"]
    failures += _check_integrity(inherited)
    failures += _check_historical(historical)
    failures += _check_selected(good)
    failures += _check_labels(good)
    return good, failures


# --- Task 3: report the existing selections and close the packet ---------

def rate_close(a, b, tol=0.10):
    return a is not None and b is not None and a > 0 and b > 0 \
        and abs(a - b) / ((a + b) / 2) <= tol


def _axis(v):
    """The comparison axes surfaced from a supported measure vector."""
    if v.get("status") != "ok":
        return None
    r = v["rates"] or {}
    lg = v["hand_gaps_ms"]["left"] or {}
    rg = v["hand_gaps_ms"]["right"] or {}
    return {
        "grouped_per_s": (r or {}).get("grouped_per_s"),
        "coincidence_share": v["coincidence_share"],
        "burst_2s_combined": v["burst_2s"]["combined"]["rate"],
        "burst_2s_peak_hand": max([x for x in (v["burst_2s"]["left"]["rate"],
                                   v["burst_2s"]["right"]["rate"]) if x is not None],
                                  default=None),
        "cadence_p50_ms_L": lg.get("p50"), "cadence_p50_ms_R": rg.get("p50"),
        "imbalance_8s": (v["imbalance_8s"] or {}).get("value"),
        "long_gap_share": v["occupied_gaps"]["long_gap_share"],
    }


def _quant(values):
    from eval.difficulty_vector import nearest_rank
    vs = [x for x in values if x is not None]
    if not vs:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": nearest_rank(vs, 0.50), "min": min(vs), "max": max(vs),
            "n": len(vs)}


def _summaries(charts):
    """Per generated song: axis median/min/max over its 6 candidates + the
    selected candidate separately (never reselected). Human grouped by family
    (RC collapses); historical listed separately, never pooled."""
    axes = ["grouped_per_s", "coincidence_share", "burst_2s_combined",
            "imbalance_8s", "long_gap_share"]
    gen = {}
    for c in charts:
        if c["cohort"] != "replay_off":
            continue
        gen.setdefault(c["family"], []).append(c)
    gen_out = {}
    for fam, cs in sorted(gen.items()):
        vecs = [_axis(c["measure"]) for c in cs]
        sel = next((c for c in cs if c["selected"]), None)
        gen_out[fam] = {
            "n_candidates": len(cs),
            "by_axis": {a: _quant([v[a] for v in vecs if v]) for a in axes},
            "selected_id": sel["id"] if sel else None,
            "selected_axis": _axis(sel["measure"]) if sel else None,
        }
    human = {}
    for c in charts:
        if c["cohort"] != "human":
            continue
        human.setdefault(c["family"], []).append(
            {"id": c["id"], "label": c["label"], "label_status": c["label_status"],
             "status": c["measure"].get("status"), "axis": _axis(c["measure"])})
    historical = [{"id": c["id"], "label": c["label"], "authored_rank":
                   c.get("authored_rank"), "status": c["measure"].get("status"),
                   "axis": _axis(c["measure"])}
                  for c in charts if c["cohort"] == "historical_export"]
    return {"generated_by_song": gen_out,
            "human_by_family": {k: v for k, v in sorted(human.items())},
            "historical_export": sorted(historical, key=lambda x: x["id"])}


def _selected_pairs(charts):
    """All eligible rate-close pairs across DIFFERENT selected songs, with the
    component differences (not only the biggest contrast)."""
    sel = sorted((c for c in charts if c["cohort"] == "replay_off"
                  and c["selected"] and c["measure"].get("status") == "ok"),
                 key=lambda c: c["id"])
    out = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            a, b = sel[i], sel[j]
            if a["family"] == b["family"]:
                continue
            ax, bx = _axis(a["measure"]), _axis(b["measure"])
            if not rate_close(ax["grouped_per_s"], bx["grouped_per_s"]):
                continue
            out.append({"a": a["id"], "b": b["id"],
                        "grouped_per_s": [ax["grouped_per_s"], bx["grouped_per_s"]],
                        "diffs": {k: (None if ax[k] is None or bx[k] is None
                                      else ax[k] - bx[k])
                                  for k in ("coincidence_share", "burst_2s_peak_hand",
                                            "cadence_p50_ms_L", "imbalance_8s",
                                            "long_gap_share")}})
    return out


def _witnesses(charts, pairs):
    """Three deterministic witnesses; missing category -> not_applicable."""
    byid = {c["id"]: c for c in charts}
    sel = [c for c in charts if c["cohort"] == "replay_off" and c["selected"]
           and c["measure"].get("status") == "ok"]
    w = {}
    # 1: eligible selected pair with the largest abs 2s per-hand peak-rate diff
    best = None
    for p in pairs:
        pa = _axis(byid[p["a"]]["measure"])["burst_2s_peak_hand"]
        pb = _axis(byid[p["b"]]["measure"])["burst_2s_peak_hand"]
        if pa is None or pb is None:
            continue
        d = abs(pa - pb)
        if best is None or d > best[0] + 1e-12:
            hi = p["a"] if pa >= pb else p["b"]
            burst = byid[hi]["measure"]["burst_2s"]
            # window of the PEAK HAND (higher of L/R), not the combined peak
            peak_key = "left" if (burst["left"]["rate"] or -1) >= \
                (burst["right"]["rate"] or -1) else "right"
            best = (d, {"kind": "selected_pair_peak_rate", "pair": [p["a"], p["b"]],
                        "raster_id": hi, "peak_rate_diff": d, "peak_hand": peak_key,
                        "window_end_s": burst[peak_key]["window_end_s"],
                        "peak_hand_rate": burst[peak_key]["rate"]})
    w["peak_rate_pair"] = best[1] if best else "not_applicable"
    # 2: selected chart with the largest eligible imbalance
    imb = [(c["measure"]["imbalance_8s"]["value"], c) for c in sel
           if c["measure"]["imbalance_8s"]]
    if imb:
        val, c = max(imb, key=lambda x: (x[0], -0))  # ties -> earliest id below
        val = max(v for v, _ in imb)
        c = min((cc for v, cc in imb if v == val), key=lambda cc: cc["id"])
        w["max_imbalance"] = {"kind": "selected_imbalance", "raster_id": c["id"],
                              "value": val,
                              "window_end_s": c["measure"]["imbalance_8s"]["window_end_s"]}
    else:
        w["max_imbalance"] = "not_applicable"
    # 3: selected chart with the longest occupied-event gap
    gaps = [(c["measure"]["occupied_gaps"]["max"], c) for c in sel
            if c["measure"]["occupied_gaps"]["max"] is not None]
    if gaps:
        val = max(v for v, _ in gaps)
        c = min((cc for v, cc in gaps if v == val), key=lambda cc: cc["id"])
        og = c["measure"]["occupied_gaps"]
        w["longest_gap"] = {"kind": "selected_long_gap", "raster_id": c["id"],
                            "gap_s": val, "gap_start_s": og.get("max_gap_start_s"),
                            "gap_end_s": og.get("max_gap_end_s")}
    else:
        w["longest_gap"] = "not_applicable"
    return w


def audit(panel_path, out_dir, make_plots=True):
    t0 = time.time()
    panel = json.loads(_resolve(panel_path).read_text())
    good, failures = validate_panel(panel)
    integrity_ok = not failures
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    charts = []
    if integrity_ok:
        for e in sorted(good, key=lambda x: x["id"]):
            v = measure_scene(load_scene(e, "standard"))
            charts.append({"id": e["id"], "cohort": e["cohort"],
                           "family": e["family"], "label": e["label"],
                           "label_status": e["label_status"],
                           "authored_rank": e.get("authored_rank"),
                           "selected": bool(e["selected"]), "measure": v})
    entries_by_id = {e["id"]: e for e in good}
    pairs = _selected_pairs(charts) if integrity_ok else []
    witnesses = _witnesses(charts, pairs) if integrity_ok else {}
    n_supported = sum(1 for c in charts if c["measure"].get("status") == "ok")
    report = {
        "schema_version": 1,
        "certification": "CERTIFIED" if integrity_ok else "UNCERTIFIED",
        "config": {"model": "grouped_event_time_vector",
                   "code_rev": _code_rev(),
                   "panel_sha256": _sha256(_resolve(panel_path)),
                   "panel_frozen": panel.get("frozen"),
                   "rate_close_tol": 0.10},
        "integrity": {"ok": integrity_ok, "checked": len(panel.get("entries", [])),
                      "failures": sorted(failures, key=lambda f: (f["id"], f["reason"]))},
        "coverage": {"total": len(charts), "supported": n_supported,
                     "unknown": len(charts) - n_supported},
        "charts": [{"id": c["id"], "cohort": c["cohort"], "family": c["family"],
                    "label": c["label"], "label_status": c["label_status"],
                    "authored_rank": c["authored_rank"], "selected": c["selected"],
                    "measure": c["measure"]} for c in charts],
        "summaries": _summaries(charts) if integrity_ok else {},
        "selected_song_pairs": pairs if pairs else "not_applicable",
        "witnesses": witnesses,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")
    plots = []
    if integrity_ok and make_plots:
        plots = _plots(entries_by_id, witnesses, out_dir)
    (out_dir / "resources.log").write_text(json.dumps({
        "wall_clock_s": round(time.time() - t0, 3), "n_charts": len(charts),
        "plots": plots}, indent=1) + "\n")
    return (0 if integrity_ok else 1), report


def _plots(entries_by_id, witnesses, out_dir):
    """At most three static timing rasters (hand rows, dot vs directional
    markers, measurement window). Only the <=3 witness charts are reloaded. A
    grouped-event timing raster, NOT a physical swing visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from eval.difficulty_vector import grouped_events

    made = []
    specs = [("peak_rate_pair", "plot_peak_rate.png", 2.0),
             ("max_imbalance", "plot_imbalance.png", 8.0),
             ("longest_gap", "plot_longest_gap.png", None)]
    for key, fname, wsize in specs:
        wit = witnesses.get(key)
        if not isinstance(wit, dict):
            continue
        cid = wit["raster_id"]
        e = entries_by_id.get(cid)
        if e is None:
            continue
        scene = load_scene(e, "standard")
        if scene.get("chart_unknown"):
            continue
        g = grouped_events(scene["notes"])
        if wsize is not None:
            end = wit.get("window_end_s")
            win = (end - wsize, end) if end is not None else None
        else:
            win = (wit.get("gap_start_s"), wit.get("gap_end_s"))
        lo, hi = (win if win and None not in win else
                  (g["events"][0]["t_s"], g["events"][-1]["t_s"]))
        pad = 0.5
        fig, ax = plt.subplots(figsize=(8, 2.6))
        for ev in g["events"]:
            if lo - pad <= ev["t_s"] <= hi + pad:
                row = 0 if ev["hand"] == 0 else 1
                marker = "o" if ev["n_directional"] == 0 else "|"
                ax.plot([ev["t_s"]], [row], marker, color="#c0392b" if row == 0
                        else "#2980b9", ms=7, mew=2)
        if win and None not in win:
            ax.axvspan(win[0], win[1], color="#f1c40f", alpha=0.25,
                       label="measurement window")
            ax.legend(fontsize=6, loc="upper right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["L", "R"])
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel("seconds")
        ax.set_title(f"{wit['kind']}: {cid}", fontsize=8)
        fig.text(0.01, 0.01, "grouped-event timing raster (| = directional, o = "
                 "dot-only group); NOT a physical swing visualization.", fontsize=5)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=110, metadata={"Software": None})
        plt.close(fig)
        made.append(fname)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="eval/difficulty_panel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    code, rep = audit(a.panel, a.out, make_plots=not a.no_plots)
    print(f"{rep['certification']}: {rep['coverage']['supported']}/"
          f"{rep['coverage']['total']} supported")
    if rep["integrity"]["failures"]:
        print("failures:", rep["integrity"]["failures"][:5])
    pairs = rep["selected_song_pairs"]
    print("rate-close selected pairs:",
          len(pairs) if isinstance(pairs, list) else pairs)
    print(f"-> {a.out}/report.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
