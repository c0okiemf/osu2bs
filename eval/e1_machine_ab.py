"""E1 vs B0 machine A/B, exactly as preregistered in
docs/specs/e1-machine-ab-prereg.md (commit 2bbb40d). Descriptive claim
only: measurable expressive-property change on exposed dev songs.

Resumable per chart (experiments/expressive-v1/e1/machine_ab/).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E1 = ROOT / "experiments" / "expressive-v1" / "e1"
OUT = E1 / "machine_ab"
PROPS = {"P1": (("features", "arcs", "runs_per_100_heads"), "up"),
         "P2": (("features", "spatial", "low_vertical_share"), "down"),
         "P3": (("double_vocabulary", "entropy"), "up")}
GUARD = (("features", "patterns", "max_4gram_share"), 0.02)


def _get(d, path):
    for k in path:
        d = None if d is None else d.get(k)
    return d


def _grid(song):
    from convert import grid_steps, parse_osu
    _m, objects, bpm, offset = parse_osu(ROOT / song["osu"])
    _s, _t, _ms, _o, grid = grid_steps(objects, bpm, offset, thin=True)
    return grid


def measure(sid, tag, raw, grid, bank, threshold, warn):
    """profile + machine protections for one cached chart."""
    p = OUT / f"{sid}.{tag}.json"
    if p.exists():
        return json.loads(p.read_text())
    from eval.expression_profile import profile
    from qa.certificates import contradictions
    from qa.comparator_support import measure_support
    from qa.comparator_v2 import chart_verdict
    secs = [grid.time(s) / 1000.0 for s, *_ in raw]
    prof = profile([(h, c, l, d) for _s, h, c, l, d in raw], secs)
    from convert import diff_spec
    # the exported Info's NJS: the support descriptor encodes it, and an
    # absent value reads as an off-distribution approach setting
    scene = {"scope": None, "chart_sha256": None, "walls": [], "bombs": [],
             "settings": {"njs": diff_spec("ExpertPlus")["njs"]},
             "notes": sorted((t, c, l, h, d)
                             for t, (_s, h, c, l, d) in zip(secs, raw))}
    contr = contradictions(scene, {"speed_warning": warn})
    sup = measure_support(scene, bank, threshold, None)
    rec = {"song": sid, "chart": tag, "profile_status": prof["status"],
           "props": {k: _get(prof, path) for k, (path, _d) in PROPS.items()},
           "guard": _get(prof, GUARD[0]),
           "contradiction_status": contr["status"],
           "share_supported": sup["share_supported"],
           "verdict": chart_verdict(contr, sup), "completed": True}
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1))
    print(f"  {sid}.{tag}: {rec['props']} {rec['verdict']}", flush=True)
    return rec


def decide(b0, arm):
    """Per-song outcome per the preregistered rule."""
    same = arm is None                       # fallback_b0 = no change
    out = {}
    for k, (_p, d) in PROPS.items():
        if same or arm["props"][k] is None or b0["props"][k] is None:
            out[k] = False
        else:
            delta = arm["props"][k] - b0["props"][k]
            out[k] = delta > 0 if d == "up" else delta < 0
    out["guard"] = same or (arm["guard"] is not None and b0["guard"] is not None
                            and arm["guard"] <= b0["guard"] + GUARD[1])
    bad = ("HARD_FAIL", "REGENERATE_MODEL_CONTRADICTION")
    out["protect"] = same or (
        not (arm["verdict"] in bad and b0["verdict"] not in bad)
        and (arm["share_supported"] or 0)
        >= (b0["share_supported"] or 0) - 0.05)
    return out


def run(tags=("A", "B"), pairs=None, report_name="report.json"):
    """pairs: [(new, baseline)] tag comparisons; default = each arm vs b0."""
    named = pairs is not None
    pairs = pairs or [(t, "b0") for t in tags]
    from qa.certificates import load_speed_warning
    from qa.neighbours import bank_from_role
    cfg = json.loads((E1 / "run.json").read_text())["config"]
    thr = json.loads((ROOT / "experiments/qa-v4/comparator-v2/development/"
                             "threshold.json").read_text())
    bank, _r = bank_from_role(identity="qa-train-v2")
    warn = load_speed_warning()
    rows = {}
    for song in cfg["songs"]:
        sid = song["song"]
        grid = _grid(song)
        part = E1 / "partial"
        b0 = measure(sid, "b0", json.loads(
            (part / f"{sid}:b0.json").read_text())["notes"], grid, bank,
            thr, warn)
        rows[sid] = {"b0": b0}
        for arm in tags:
            sel = json.loads((part / f"{sid}:arm{arm}.json").read_text())[
                "selection"]
            if sel["status"] != "selected":
                rows[sid][arm] = None
                continue
            raw = json.loads((part / f"{sid}:arm{arm}:{sel['id']}.json")
                             .read_text())["notes"]
            rows[sid][arm] = measure(sid, f"arm{arm}", raw, grid, bank,
                                     thr, warn)
    report = {"prereg": "docs/specs/e1-machine-ab-prereg.md", "arms": {}}
    for new, basetag in pairs:
        arm = f"{new}_vs_{basetag}" if named else new
        per = {s: (decide(r[basetag], r[new]) if r.get(basetag) is not None
                   else decide(r["b0"], r[new]))
               for s, r in rows.items()}
        counts = {k: sum(v[k] for v in per.values())
                  for k in ("P1", "P2", "P3", "guard", "protect")}
        shown = (all(counts[k] >= 5 for k in ("P1", "P2", "P3", "guard"))
                 and counts["protect"] == len(per))
        report["arms"][arm] = {
            "verdict": "IMPROVEMENT_SHOWN" if shown else "NOT_SHOWN",
            "counts_of_6": counts, "per_song": per,
            "deltas": {s: None if r[new] is None or r.get(basetag) is None
                       else {k: (None if r[new]["props"][k] is None
                                 or r[basetag]["props"][k] is None
                                 else round(r[new]["props"][k]
                                            - r[basetag]["props"][k], 4))
                             for k in PROPS} for s, r in rows.items()}}
    (OUT / report_name).write_text(json.dumps(report, indent=1))
    for arm, r in report["arms"].items():
        print(f"arm {arm}: {r['verdict']} {r['counts_of_6']}")
    return report


if __name__ == "__main__":
    run()
