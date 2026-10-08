"""comparator-v2 one-shot seal confirmation runner (spec §8).

Kept OUTSIDE the frozen evaluator code so the evaluator freeze stays
byte-exact: the frozen `qa.comparator_v2.machine_record` and `score` are
reused unchanged; this module only (1) refuses to run before FRESH_PASS,
(2) materializes sanitized seal records from restricted originals, and
(3) serves them to machine_record through the provider interface.
Lower-tier seal families test scope abstention only and never pad the
ExpertPlus count; fewer than 4 in-scope ExpertPlus families =>
CONFIRMATION_INSUFFICIENT. One run; identical-hash resume only.
"""
import json
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from qa import comparator_v2 as v2
from qa.comparator_contract import atomic_json, require_stage

ROOT = Path(__file__).resolve().parent.parent
ACQ = ROOT / "experiments" / "qa-v1-acquisition"
SEAL_DIR = v2.V2 / "seal"
MIN_EXPERTPLUS = 4


def seal_families():
    """(family, chart_rec) for every sealed family, both state files."""
    out = []
    for name, side in (("collect2_state.json", "seal"),
                       ("pilot_state.json", "qa")):
        st = json.loads((ROOT / "experiments/qa-v1" / name).read_text())
        for fam, c in sorted(st["charts"].items()):
            if c.get("side") == side and c.get("replays"):
                out.append((fam, c))
    return out


def materialize(fam, chart, scene):
    """Parse + sanitize seal originals; derive window records."""
    import torch
    from qa.bsor import parse_bsor
    from qa.contract import sanitized_record
    from qa.scene import align
    from qa.telemetry_v2 import window_record
    mig = json.loads((ACQ / "migration_map.json").read_text())
    recs = []
    for entry in chart["replays"]:
        m = mig.get(entry["file"]) or {}
        if m.get("status") != "seal_deferred":
            continue
        blob = (ACQ / "originals" / (m["original_sha256"] + ".bsor")) \
            .read_bytes()
        san = sanitized_record(parse_bsor(blob),
                               player_token=entry["player_token"],
                               source_sha256=m["original_sha256"])
        sp = ROOT / entry["file"].replace(".bsor", ".sanitized.pt")
        if not sp.exists():
            torch.save(san, sp)
        lh = bool(san["info"].get("leftHanded"))
        al = align(scene, san["notes"], left_handed=lh)
        view = v2_mirror(scene) if lh else scene
        profile = {"height": entry.get("height"),
                   "height_known": (entry.get("height") or 0) > 0.2,
                   "left_handed": lh}
        for wi in range(len(al["aligned"])):
            w = window_record(san, al, wi, mirrored=lh)
            if not w.get("supported") or w.get("outcome") != "good":
                continue
            recs.append({"window_id": f"{fam}:{entry['sha256'][:16]}:"
                                      f"{w['chart_index']}:{wi}",
                         "family": fam, "player_token": entry["player_token"],
                         "chart_index": w["chart_index"], "window": w,
                         "scene": view, "profile": profile})
    return recs


def v2_mirror(scene):
    from qa.train import mirror_scene
    return mirror_scene(scene)


@contextmanager
def serve(records_by_family):
    """Serve seal records to the frozen machine_record (which asks the
    provider for role qa_validate_comparator/validate)."""
    def fake(role, purpose, families=None):
        for f in sorted(families or records_by_family):
            yield from records_by_family.get(f, [])
    with mock.patch("qa.features.records_for_role", fake):
        yield


def confirm():
    state = v2.load_state()
    require_stage(state, "FRESH_PASS")
    if state["stage"] != "FRESH_PASS":
        raise PermissionError(f"seal needs FRESH_PASS, at {state['stage']}")
    fz = v2._check_frozen_code()
    from qa.coverage_sample import select_windows
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    _s, manifest, bank, warn = v2._common()
    SEAL_DIR.mkdir(parents=True, exist_ok=True)
    records, tiers = {}, {}
    for fam, chart in seal_families():
        tiers[fam] = chart.get("difficulty")
        dat_p, info_p = _family_chart(manifest, fam, chart["difficulty"])
        if dat_p is None:
            continue
        scene = read_scene(dat_p, info_p)
        out = SEAL_DIR / (fam.replace(":", "_") + ".machine.json")
        if out.exists():
            records[fam] = json.loads(out.read_text())
            continue
        if scene["scope"] is not None or chart["difficulty"] != "ExpertPlus":
            records[fam] = {"family": fam, "tier": chart["difficulty"],
                            "scope": scene["scope"], "in_scope": False,
                            "completed": True}
            atomic_json(out, records[fam])
            continue
        recs = materialize(fam, chart, scene)
        players = {r["player_token"] for r in recs}
        ids = select_windows([{"player_token": r["player_token"],
                               "window_id": r["window_id"]} for r in recs],
                             cap=2000)
        if len(recs) < 100 or len(players) < 3:
            rec = {"family": fam, "in_scope": False,
                   "reason": "insufficient_telemetry", "completed": True}
        else:
            excl = {"families": {fam},
                    "players": {e["player_token"]
                                for e in chart["replays"]}}
            with serve({fam: recs}):
                rec = v2.machine_record(
                    fam, "qa_validate_comparator", scene, ids, bank,
                    {"T_support": fz["threshold"]}, warn, excl,
                    SEAL_DIR / fam.replace(":", "_"))
            rec["in_scope"] = True
        atomic_json(out, rec)
        records[fam] = rec
        print(f"  [seal] {fam}: {rec.get('chart_verdict')}", flush=True)
    in_scope = {f: r for f, r in records.items() if r.get("in_scope")}
    if len(in_scope) < MIN_EXPERTPLUS:
        rep = {"stage": "seal", "status": "FAILED",
               "first_failing": {"stage": "seal",
                                 "dimension": "CONFIRMATION_INSUFFICIENT"},
               "tiers": tiers, "completed": True}
    else:
        rep = v2.score(in_scope, "seal")
        rep["out_of_scope"] = {f: r for f, r in records.items()
                               if not r.get("in_scope")}
    atomic_json(v2.V2 / "seal_report.json", rep)
    if rep["status"] == "PASS":
        v2._save(v2.advance(state, "CONFIRMED", {
            "seal_report_sha256": v2._sha(v2.V2 / "seal_report.json")}))
        print("AUTOMATED_QA_VALIDATED_WITHIN_SCOPE (machine validity + "
              "support; quality not evaluated)")
    else:
        v2._save(v2.advance(state, "FAILED", rep["first_failing"]))
        print(f"seal: {rep['first_failing']}")
    return rep


if __name__ == "__main__":
    confirm()
