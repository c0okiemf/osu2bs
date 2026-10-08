"""Read-only wall-corridor panel audit (Task 3).

Verifies the frozen panel's hashes and timing sources, runs the model-conditional
A06/D01 diagnostics at three head radii, and emits a canonical deterministic
report plus static witness plots. It changes nothing in generation. A diagnostic
finding is NOT an error; only an integrity failure makes the CLI exit nonzero
and mark the run UNCERTIFIED. Partial charts never receive a whole-chart
clearance verdict.

  .venv/bin/python -m eval.wall_audit --panel eval/wall_panel.json \\
    --out experiments/wall-corridor/run-a
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from math import isfinite
from pathlib import Path

from eval.wall_geometry import analyze_walls, normalize_v2

RADII = [0.15, 0.25, 0.35]
NOMINAL = 0.25
ROOT = Path(__file__).resolve().parent.parent


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _code_rev():
    try:
        rev = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
        return {"rev": rev, "dirty": dirty}
    except Exception:
        return {"rev": None, "dirty": None}


REQUIRED_FIELDS = ("id", "role", "chart", "sha256", "bpm", "origin_s",
                   "timing_source", "timing_source_sha256")


def _validate_panel(panel):
    """Validate the panel itself before any hashing/loading (review review R2):
    schema, required fields/types, unique IDs, finite positive BPM and origin,
    valid role, declared frozen membership/counts, and no declared-missing
    entries. Returns (failures, structurally-usable entries) — malformed entries
    are dropped so downstream code cannot hit an incidental KeyError. Structured
    reasons only, never a traceback."""
    failures = []
    if not isinstance(panel, dict) or panel.get("schema_version") != 1:
        failures.append({"id": "<panel>", "reason": "bad_or_missing_schema_version"})
    if isinstance(panel, dict) and panel.get("missing"):
        failures.append({"id": "<panel>", "reason": "declared_missing_entries"})
    entries = panel.get("entries") if isinstance(panel, dict) else None
    if not isinstance(entries, list) or not entries:
        failures.append({"id": "<panel>", "reason": "empty_or_missing_entries"})
        return failures, []
    good, seen, ids = [], set(), []
    for e in entries:
        eid = e.get("id", "<no-id>") if isinstance(e, dict) else "<not-object>"
        if not isinstance(e, dict) or any(f not in e for f in REQUIRED_FIELDS):
            failures.append({"id": eid, "reason": "missing_required_field"})
            continue
        if e.get("role") not in ("generated", "reference"):
            failures.append({"id": eid, "reason": "bad_role"})
            continue
        bpm, org = e["bpm"], e["origin_s"]
        ok = True
        if not isinstance(bpm, (int, float)) or isinstance(bpm, bool) \
                or not (isfinite(bpm) and bpm > 0):
            failures.append({"id": eid, "reason": "nonfinite_or_nonpositive_bpm"})
            ok = False
        if not isinstance(org, (int, float)) or isinstance(org, bool) \
                or not isfinite(org):
            failures.append({"id": eid, "reason": "nonfinite_origin"})
            ok = False
        if eid in seen:
            failures.append({"id": eid, "reason": "duplicate_id"})
            ok = False
        seen.add(eid)
        ids.append(eid)
        if ok:
            good.append(e)
    # frozen self-declared membership catches dropped/added entries
    exp_ids = panel.get("expected_ids")
    if isinstance(exp_ids, list) and set(ids) != set(exp_ids):
        for m in sorted(set(exp_ids) - set(ids)):
            failures.append({"id": m, "reason": "missing_expected_entry"})
        for x in sorted(set(ids) - set(exp_ids)):
            failures.append({"id": x, "reason": "unexpected_entry"})
    exp_counts = panel.get("expected_counts")
    if isinstance(exp_counts, dict):
        for role, want in exp_counts.items():
            got = sum(1 for e in good if e.get("role") == role)
            if got != want:
                failures.append({"id": f"<{role}>", "reason":
                                 f"role_count_mismatch:{got}!={want}"})
    return failures, good


def _source_bpm(entry, cache):
    """The BPM recorded by the entry's referenced source (results.json for
    generated, benchmark.json for reference) — the VALUE, not just its hash.
    Source docs are cached by their path so multiple distinct timing sources do
    not collide in one global slot."""
    ts = _resolve(entry["timing_source"])
    key = str(ts)
    if key not in cache:
        cache[key] = json.loads(ts.read_text())
    doc = cache[key]
    if entry["role"] == "generated":
        return float(doc["bpm"])
    # reference: benchmark song id is the entry id minus its difficulty segment
    song_id = entry["id"].rsplit("/", 1)[0]
    songs = {s["id"]: s for s in doc["songs"]}
    return float(songs[song_id]["bpm"])


def _check_integrity(entries):
    """Verify each chart + timing source exists, matches its recorded hash, and
    that the entry BPM is bound to the source record's value. A hash mismatch,
    missing file, or BPM that does not match its source is an integrity failure;
    the panel is not silently shrunk or regenerated."""
    failures = []
    bench_cache = {}
    for e in entries:
        chart = _resolve(e["chart"])
        if not chart.exists():
            failures.append({"id": e["id"], "reason": "chart_missing"})
            continue
        if _sha256(chart) != e["sha256"]:
            failures.append({"id": e["id"], "reason": "chart_sha256_mismatch"})
        ts = _resolve(e["timing_source"])
        if not ts.exists():
            failures.append({"id": e["id"], "reason": "timing_source_missing"})
            continue
        if _sha256(ts) != e["timing_source_sha256"]:
            failures.append({"id": e["id"], "reason": "timing_source_sha256_mismatch"})
        try:
            if abs(_source_bpm(e, bench_cache) - float(e["bpm"])) > 1e-9:
                failures.append({"id": e["id"], "reason": "bpm_not_bound_to_source"})
        except (KeyError, ValueError, json.JSONDecodeError):
            failures.append({"id": e["id"], "reason": "bpm_source_unresolvable"})
    return failures


def _analyze_entry(e):
    """Per-map findings across the three radii. Findings retain witness IDs and
    timestamps. Never a whole-chart clearance when coverage is partial."""
    raw = json.loads(_resolve(e["chart"]).read_text(encoding="utf-8-sig"))
    spb = 60.0 / float(e["bpm"])
    normalized = normalize_v2(raw, spb, e.get("origin_s", 0))
    out = {"id": e["id"], "role": e["role"], "bpm": float(e["bpm"]),
           "scope": normalized["scope"],
           "chart_unknown": normalized["chart_unknown"],
           "n_invalid": len(normalized["invalid"]),
           "n_unknown_objects": len(normalized["unknown"]),
           "radii": {}}
    for r in RADII:
        res = analyze_walls(normalized, r)
        empties = [{"start_s": s["start_s"], "end_s": s["end_s"],
                    "wall_ids": s["wall_ids"]}
                   for s in res["segments"] if s["status"] == "empty_model"]
        pos = [{"from_ids": t["from_ids"], "to_ids": t["to_ids"],
                "from_end_s": t["from_end_s"], "to_start_s": t["to_start_s"],
                "gap_s": t["gap_s"], "distance_lane": t["distance_lane"],
                "speed_lane_s": t["speed_lane_s"]}
               for t in res["transitions"]
               if t["distance_lane"] not in (None, 0.0)]
        disc = [{"from_ids": t["from_ids"], "to_ids": t["to_ids"],
                 "at_s": t["to_start_s"], "distance_lane": t["distance_lane"]}
                for t in res["transitions"]
                if t["reason"] == "discontinuous_model_constraint"]
        speeds = sorted(p["speed_lane_s"] for p in pos
                        if p["speed_lane_s"] is not None)
        # partial coverage: any unknown/invalid objects or unknown segments ->
        # no whole-chart clearance
        partial = (out["n_unknown_objects"] > 0 or out["n_invalid"] > 0
                   or normalized["chart_unknown"]
                   or res["counts"]["unknown_s"] > 0)
        out["radii"][f"{r:.2f}"] = {
            "counts": res["counts"],
            "empty_corridors": empties,
            "positive_travel": pos,
            "discontinuities": disc,
            "speed_lane_s_max": speeds[-1] if speeds else None,
            "coverage": "partial" if partial else "full",
        }
    return out, normalized


def audit(panel_path, out_dir, make_plots=True):
    """Run the audit. Returns (exit_code, report). exit_code is nonzero only on
    an integrity failure; diagnostic findings never fail the run."""
    t0 = time.time()
    panel = json.loads(_resolve(panel_path).read_text())
    pfail, entries = _validate_panel(panel)
    failures = pfail + _check_integrity(entries)
    integrity_ok = not failures
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    maps, norms = [], {}
    if integrity_ok:
        for e in sorted(entries, key=lambda x: x["id"]):
            m, n = _analyze_entry(e)
            maps.append(m)
            norms[e["id"]] = n            # retained for witness plots

    agg = _aggregate(maps)
    report = {
        "schema_version": 1,
        "certification": "CERTIFIED" if integrity_ok else "UNCERTIFIED",
        "config": {
            "model": "1d_head_center_lane_[0,4]",
            "tangency": "allowed_by_model",
            "radii": RADII,
            "nominal_radius": NOMINAL,
            "code_rev": _code_rev(),
            "panel_path": str(Path(panel_path)),
            "panel_sha256": _sha256(_resolve(panel_path)),
            "panel_frozen": panel.get("frozen"),
        },
        "integrity": {"ok": integrity_ok, "checked": len(entries),
                      "failures": sorted(failures, key=lambda f: (f["id"], f["reason"]))},
        "panel_incomplete": bool(panel.get("missing")) or not integrity_ok,
        "aggregate": agg,
        "maps": maps,
    }
    # canonical, deterministic; no wall-clock runtime inside
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n")

    plots = []
    if integrity_ok and make_plots:
        plots = _plots(maps, norms, out_dir)
    # resources go in a SEPARATE non-canonical log
    (out_dir / "resources.log").write_text(json.dumps({
        "wall_clock_s": round(time.time() - t0, 3),
        "n_maps": len(maps), "plots": plots}, indent=1) + "\n")

    return (0 if integrity_ok else 1), report


def _aggregate(maps):
    """Per-role, per-radius rollup. Zero applicable pairs is not_applicable, not
    a perfect pass. Explicitly states whether generated walls force any travel."""
    agg = {}
    for role in ("generated", "reference"):
        rms = [m for m in maps if m["role"] == role]
        per_r = {}
        for r in RADII:
            k = f"{r:.2f}"
            walls = sum(m["radii"][k]["counts"]["supported_walls"] for m in rms)
            unk = sum(m["radii"][k]["counts"]["unknown_objects"] for m in rms)
            pos = sum(len(m["radii"][k]["positive_travel"]) for m in rms)
            disc = sum(len(m["radii"][k]["discontinuities"]) for m in rms)
            empty_s = sum(m["radii"][k]["counts"]["empty_model_s"] for m in rms)
            trans = sum(m["radii"][k]["counts"]["transitions"] for m in rms)
            speeds = [m["radii"][k]["speed_lane_s_max"] for m in rms
                      if m["radii"][k]["speed_lane_s_max"] is not None]
            per_r[k] = {
                "supported_walls": walls, "unknown_objects": unk,
                "evaluated_transitions": trans,
                "positive_travel_pairs": pos,
                "discontinuities": disc,
                "empty_model_s": empty_s,
                "max_speed_lane_s": max(speeds) if speeds else None,
                # zero evaluated transitions is not_applicable (denominator
                # zero), NOT a clean pass
                "forced_travel": ("not_applicable" if trans == 0
                                  else ("present" if pos else "none")),
            }
        agg[role] = {"n_maps": len(rms), "by_radius": per_r}
    gen_nom = agg["generated"]["by_radius"][f"{NOMINAL:.2f}"]
    agg["generated_zero_forced_travel_nominal"] = (
        gen_nom["positive_travel_pairs"] == 0 and gen_nom["discontinuities"] == 0)
    agg["generated_supported_walls_nominal"] = gen_nom["supported_walls"]
    return agg


def _pick(maps, pred):
    for m in maps:            # maps already sorted by id -> deterministic pick
        if pred(m):
            return m
    return None


def _plots(maps, norms, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    nk = f"{NOMINAL:.2f}"

    def draw(entry_id, title, fname, window=None):
        n = norms[entry_id]
        res = analyze_walls(n, NOMINAL)
        fig, ax = plt.subplots(figsize=(6, 4))
        segs = res["segments"]
        tspan = (min((s["start_s"] for s in segs), default=0.0),
                 max((s["end_s"] for s in segs), default=1.0))
        if window:
            tspan = window
        for w in n["walls"]:
            if w.end_s < tspan[0] or w.start_s > tspan[1]:
                continue
            ax.add_patch(Rectangle((w.x0, w.start_s), w.x1 - w.x0,
                                   w.end_s - w.start_s, color="#c0392b",
                                   alpha=0.55, lw=0))
        for u in n["unknown"]:
            if u["start_s"] is None:
                continue
            if u["end_s"] < tspan[0] or u["start_s"] > tspan[1]:
                continue
            ax.add_patch(Rectangle((0, u["start_s"]), 4, u["end_s"] - u["start_s"],
                                   facecolor="none", edgecolor="#7f8c8d",
                                   hatch="///", lw=0.5))
        for s in segs:
            if s["end_s"] < tspan[0] or s["start_s"] > tspan[1]:
                continue
            for lo, hi in s["safe_intervals"]:
                ax.add_patch(Rectangle((lo, s["start_s"]), max(hi - lo, 0.02),
                                       s["end_s"] - s["start_s"],
                                       color="#27ae60", alpha=0.30, lw=0))
        ax.set_xlim(0, 4)
        ax.set_ylim(tspan[1], tspan[0])
        ax.set_xlabel("lane units (head center, model r=0.25)")
        ax.set_ylabel("seconds")
        ax.set_title(title, fontsize=9)
        fig.text(0.01, 0.01, "1-D model corridor; red=wall, green=safe center, "
                 "hatch=unknown geometry. Not a human avatar.", fontsize=6)
        fig.tight_layout()
        p = out_dir / fname
        fig.savefig(p, dpi=110, metadata={"Software": None})
        plt.close(fig)
        return fname

    made = []
    gen = _pick(maps, lambda m: m["role"] == "generated"
                and m["radii"][nk]["counts"]["supported_walls"] > 0)
    if gen:
        made.append(draw(gen["id"], f"generated counterexample: {gen['id']}\n"
                         "width-1 outer walls -> zero wall-required travel",
                         "plot_generated_counterexample.png",
                         window=(0.0, 60.0)))
    ref = _pick(maps, lambda m: m["role"] == "reference"
                and m["radii"][nk]["positive_travel"])
    if ref:
        t = ref["radii"][nk]["positive_travel"][0]
        w0 = max(0.0, t["from_end_s"] - 5)
        made.append(draw(ref["id"], f"reference wider-wall shift: {ref['id']}\n"
                         f"distance {t['distance_lane']:.2f} lane over "
                         f"{t['gap_s']:.2f}s", "plot_reference_transition.png",
                         window=(w0, t["to_start_s"] + 5)))
    unkm = _pick(maps, lambda m: m["n_unknown_objects"] > 0)
    if unkm:
        made.append(draw(unkm["id"], f"unknown geometry: {unkm['id']}\n"
                         "crouch/custom walls -> unknown coverage, no clearance",
                         "plot_unknown_geometry.png", window=(0.0, 40.0)))
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="eval/wall_panel.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    code, report = audit(a.panel, a.out, make_plots=not a.no_plots)
    cert = report["certification"]
    ag = report["aggregate"]
    print(f"{cert}: {report['integrity']['checked']} charts, "
          f"{len(report['maps'])} analyzed")
    if report["integrity"]["failures"]:
        print("integrity failures:", report["integrity"]["failures"][:5])
    if report["maps"]:
        print("generated zero forced travel (nominal):",
              ag["generated_zero_forced_travel_nominal"],
              f"({ag['generated_supported_walls_nominal']} supported walls)")
    print(f"-> {a.out}/report.json")
    return code


if __name__ == "__main__":
    sys.exit(main())
