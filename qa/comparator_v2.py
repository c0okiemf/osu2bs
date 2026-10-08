"""comparator-v2: machine-only replacement contract.

Authority = structural validity, independently verified conditional
geometric contradictions and raw-reference execution support. NOT quality
and NOT proven playability: every former judge row is recorded as
QUALITY_NOT_EVALUATED, never as a pass. Zero review judgments at any stage.
comparator-v1 stays FAILED and its code/artifacts untouched.

D1 support rule: >=90% supported heads is retained; every clustered
unsupported run (>2 s span, >=4 heads) is reported as SUPPORT_UNKNOWN and
denies that chart an unqualified support pass (chart_verdict), but it is
not an evaluator-retention failure by itself.

Stages (strict, via qa.comparator_contract.advance): BUILDING ->
RECIPE_FROZEN -> DEVELOPMENT_PASS -> EVALUATOR_FROZEN -> FRESH_PASS ->
CONFIRMED; any gate failure or insufficient coverage -> FAILED, stop.
"""
import hashlib
import json
from pathlib import Path

from qa.comparator_contract import (RESERVED_FAMILIES, advance, atomic_json,
                                    require_stage)

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "experiments" / "qa-v4" / "comparator-v2"
STATE_P = V2 / "state.json"
RECIPE_P = V2 / "recipe.json"
FREEZE_P = V2 / "evaluator_freeze.json"
LABEL = "comparator-v2"
MACHINE_GATES = ("provenance", "structural_scope", "retention",
                 "certified_negatives", "benign_machine")
NOT_EVALUATED = ("quality_sensitivity", "benign_judge", "judge_consistency",
                 "evidence_honesty")
CODE_FILES = ("qa/comparator_v2.py", "qa/certificates.py",
              "qa/comparator_support.py", "qa/comparator_controls.py",
              "qa/comparator_contract.py", "qa/scene.py", "qa/physics.py",
              "qa/neighbours.py", "qa/features.py", "qa/coverage_sample.py",
              "qa/contract.py", "qa/telemetry_v2.py")
MAX_FRESH_PER_FAMILY = 6


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_state():
    if STATE_P.exists():
        return json.loads(STATE_P.read_text())
    return {"stage": "BUILDING", "label": LABEL, "history": []}


def _save(state):
    atomic_json(STATE_P, state)
    return state


# ---------------- pure scoring ----------------

def chart_verdict(contr, support):
    """Serving-time machine verdict for ONE chart (quality not evaluated)."""
    if contr["status"] == "STRUCTURAL_CONTRADICTION":
        return "HARD_FAIL"
    if contr["status"] == "MODEL_CONTRADICTION":
        return "REGENERATE_MODEL_CONTRADICTION"
    if contr["status"] == "UNKNOWN" or contr.get("unknowns"):
        return "SCOPE_OR_EVIDENCE_UNKNOWN"
    if not support.get("share_supported") \
            or support["share_supported"] < 0.90:
        return "SUPPORT_UNKNOWN"
    if not support.get("run_gate", False):
        return "SUPPORT_UNKNOWN"            # D1: clustered gap
    return "MACHINE_PASS_QUALITY_NOT_EVALUATED"


def _rate(xs):
    return sum(xs) / len(xs) if xs else None


def score(records, stage):
    """Machine gates only; zero-denominator rows fail, never pass."""
    fams = sorted(records)
    g = {}
    g["provenance"] = {"pass": bool(fams) and all(
        records[f].get("provenance_ok") for f in fams)}
    g["structural_scope"] = {"pass": bool(fams) and all(
        _rate(records[f]["structural"]) == 1.0
        and not records[f]["human_hard_fail"] for f in fams),
        "per_family": {f: [sum(records[f]["structural"]),
                           len(records[f]["structural"]),
                           records[f]["human_hard_fail"]] for f in fams}}
    ret = {}
    for f in fams:
        r = records[f]["retention"]
        ret[f] = {**r, "pass": r["pooled"] >= 0.90
                  and r["equal_player"] >= 0.90
                  and (r["share_supported"] or 0) >= 0.90}
    g["retention"] = {"pass": bool(fams) and all(v["pass"]
                                                 for v in ret.values()),
                      "per_family": ret,
                      "support_unknown_runs": {
                          f: records[f]["retention"]["clustered_runs"]
                          for f in fams}}
    g["certified_negatives"] = {"pass": bool(fams) and all(
        (_rate(records[f]["wall"]) or 0) >= 0.90
        and records[f]["certificates_verified"] for f in fams),
        "per_family": {f: [sum(records[f]["wall"]), len(records[f]["wall"])]
                       for f in fams}}
    g["benign_machine"] = {"pass": bool(fams) and all(
        all(records[f]["benign"].values()) for f in fams),
        "per_family": {f: records[f]["benign"] for f in fams}}
    for name in NOT_EVALUATED:
        g[name] = {"pass": None, "status": "QUALITY_NOT_EVALUATED"}
    failing = [n for n in MACHINE_GATES if not g[n]["pass"]]
    return {"stage": stage, "gates": g, "families": fams,
            "status": "FAILED" if failing else "PASS",
            "first_failing": ({"stage": stage, "dimension": failing[0]}
                              if failing else None),
            "claim": "machine validity + execution support only; quality "
                     "not evaluated",
            "completed": True}


# ---------------- per-family machine record ----------------

def _same_outcome(a, b):
    return (a["status"] == b["status"]
            and [len(a[k]) for k in ("structural", "model", "unknowns")]
            == [len(b[k]) for k in ("structural", "model", "unknowns")])


def machine_record(fam, role, scene, sample_ids, bank, threshold, warn,
                   exclusions, progress_dir):
    from qa.certificates import contradictions, verify_certificate
    from qa.comparator_controls import (make_cases, roundtrip_scene,
                                        score_case, time_origin_shift)
    from qa.comparator_support import measure_support
    from qa.features import records_for_role
    from qa.train import mirror_scene
    k = {"speed_warning": warn}
    contr = contradictions(scene, k)
    support = measure_support(scene, bank, threshold, exclusions,
                              progress_dir=progress_dir)
    cases = make_cases(fam, scene, None, {"label": LABEL})
    struct = [score_case(c["expected"], contradictions(c["scene"], k),
                         [])["detected"] for c in cases["structural"]]
    wall = [score_case(c["expected"], contradictions(c["scene"], k),
                       [])["detected"] for c in cases["wall"]]
    certs_ok = bool(cases["wall"]) and all(
        verify_certificate(c["scene"], c["certificate"])["valid"]
        for c in cases["wall"])
    shifted, _ = time_origin_shift(scene, None, 2.0)
    benign = {"serialization_exact":
              contr == contradictions(roundtrip_scene(scene), k),
              "time_origin": _same_outcome(contr,
                                           contradictions(shifted, k)),
              "mirror": _same_outcome(contr,
                                      contradictions(mirror_scene(scene), k))}
    purpose = {"qa_calib": "calibrate", "qa_calib2": "calibrate",
               "qa_validate_comparator": "validate"}[role]
    recs = {r["window_id"]: r for r in records_for_role(role, purpose,
                                                        families={fam})}
    by_player, n_ok = {}, 0
    for wid in sample_ids:
        r = recs.get(wid)
        ok = bool(r and r["window"].get("supported")
                  and scene.get("scope") is None
                  and contr["status"] == "NO_CONTRADICTION_FOUND")
        n_ok += ok
        by_player.setdefault(r["player_token"] if r else "?",
                             []).append(ok)
    means = [sum(v) / len(v) for v in by_player.values()]
    return {"family": fam, "role": role, "completed": True,
            "provenance_ok": all(w in recs for w in sample_ids)
            and bool(sample_ids),
            "contradiction_status": contr["status"],
            "human_hard_fail": contr["status"] == "STRUCTURAL_CONTRADICTION",
            "structural": struct, "wall": wall,
            "certificates_verified": certs_ok,
            "benign": benign,
            "retention": {"pooled": round(n_ok / max(1, len(sample_ids)), 4),
                          "equal_player": round(sum(means) / len(means), 4)
                          if means else 0.0,
                          "share_supported": support["share_supported"],
                          "clustered_runs": sum(
                              r["violating"]
                              for r in support["unsupported_runs"])},
            "chart_verdict": chart_verdict(contr, support)}


# ---------------- stages ----------------

def freeze_recipe():
    state = load_state()
    if state["stage"] != "BUILDING":
        return json.loads(RECIPE_P.read_text())
    missing = [c for c in CODE_FILES if not (ROOT / c).exists()]
    if missing:
        raise ValueError(f"missing code files: {missing}")
    rec = {"label": LABEL, "ruling": "review (D1-D3)",
           "supersedes": "comparator-v1 (FAILED; preserved)",
           "code_sha256": {c: _sha(ROOT / c) for c in CODE_FILES},
           "speed_threshold_sha256": _sha(
               ROOT / "experiments/qa-v1/speed_threshold.json"),
           "reserved_families": list(RESERVED_FAMILIES),
           "bank_identity": "qa-train-v2",
           "support_cutoff": "T_support = max over the 8 development "
                             "families of p95 valid diverse support "
                             "distance on their frozen equal-player "
                             "windows; >=90% valid diverse retrieval per "
                             "family else DATA_SUPPORT_INSUFFICIENT",
           "gates": {"structural_scope": "100% structural negatives "
                     "HARD_FAIL; zero HARD_FAIL on human controls",
                     "retention": ">=0.90 window pooled AND equal-player "
                     "AND full-chart supported-head share per family; "
                     "clustered runs reported as SUPPORT_UNKNOWN, not "
                     "gating (D1)",
                     "certified_negatives": ">=90% per family detected "
                     "with interval overlap; 100% independent "
                     "certificate verification",
                     "benign_machine": "serialization exact; time-origin "
                     "and mirror contradiction outcomes preserved "
                     "(support under transforms not re-measured)",
                     "quality": "QUALITY_NOT_EVALUATED (D2)"},
           "stages": "development (8 dev families) -> evaluator freeze "
                     "-> 4 reserved fresh families -> 6-family seal; "
                     "every required family must pass; failure or "
                     "insufficiency stops",
           "astra_judgments": 0, "completed": True}
    rec["recipe_sha256"] = hashlib.sha256(json.dumps(
        rec, sort_keys=True).encode()).hexdigest()
    atomic_json(RECIPE_P, rec)
    _save(advance(state, "RECIPE_FROZEN",
                  {"recipe_sha256": rec["recipe_sha256"]}))
    print(f"v2 recipe frozen: {rec['recipe_sha256'][:12]}")
    return rec


def _common():
    from qa.certificates import load_speed_warning
    from qa.neighbours import bank_from_role
    state_j = json.loads((ROOT / "experiments/qa-v1/collect2_state.json")
                         .read_text())
    manifest = json.loads((ROOT / "eval/corpus_manifest.json").read_text())
    bank, _r = bank_from_role(identity="qa-train-v2")
    return state_j, manifest, bank, load_speed_warning()


def _run_families(stage_dir, specs, threshold):
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    state_j, manifest, bank, warn = _common()
    records = {}
    for fam, sp in sorted(specs.items()):
        out = stage_dir / (fam.replace(":", "_") + ".machine.json")
        if out.exists():
            records[fam] = json.loads(out.read_text())
            continue
        dat_p, info_p = _family_chart(
            manifest, fam, state_j["charts"][fam]["difficulty"])
        scene = read_scene(dat_p, info_p)
        excl = {"families": {fam},
                "players": {e["player_token"]
                            for e in state_j["charts"][fam]["replays"]}}
        rec = machine_record(fam, sp["role"], scene, sp["ids"], bank,
                             threshold, warn, excl,
                             stage_dir / fam.replace(":", "_"))
        atomic_json(out, rec)
        records[fam] = rec
        print(f"  [{stage_dir.name}] {fam}: {rec['chart_verdict']}, "
              f"retention {rec['retention']}", flush=True)
    return records


def develop():
    state = load_state()
    require_stage(state, "RECIPE_FROZEN")
    if state["stage"] != "RECIPE_FROZEN":
        raise PermissionError(f"develop requires RECIPE_FROZEN, at "
                              f"{state['stage']}")
    from qa.comparator_support import freeze_support
    from qa.expand import EVAL_FREEZE_P
    from qa.features import records_for_role
    from qa.neighbours import descriptor, retrieve
    from qa.parity_recheck import verify_sample
    d = V2 / "development"
    d.mkdir(parents=True, exist_ok=True)
    samples = json.loads(EVAL_FREEZE_P.read_text())["samples"]
    thr_p = d / "threshold.json"
    if thr_p.exists():
        threshold = json.loads(thr_p.read_text())
    else:
        _s, _m, bank, _w = _common()
        rows = []
        for fam, sm in sorted(samples.items()):
            verify_sample(sm)
            recs = {r["window_id"]: r for r in records_for_role(
                sm["role"], "calibrate", families={fam})}
            cache = {}
            for wid in sm["ids"]:
                r = recs[wid]
                key = (r["profile"]["left_handed"], r["chart_index"])
                if key not in cache:
                    cache[key] = retrieve(bank, descriptor(
                        r["scene"], r["chart_index"]), exclude=None)
                rows.append({"family": fam, "window_id": wid,
                             "distance": cache[key]["support_distance"],
                             "diverse": cache[key]["status"] == "ok"})
        threshold = {**freeze_support(rows, "qa-train-v2"),
                     "completed": True}
        atomic_json(thr_p, threshold)
    specs = {f: {"role": sm["role"], "ids": sm["ids"]}
             for f, sm in samples.items()}
    rep = score(_run_families(d, specs, threshold), "development")
    atomic_json(V2 / "development_report.json", rep)
    _finish(state, rep, "DEVELOPMENT_PASS", "development_report_sha256",
            V2 / "development_report.json")
    return rep


def _finish(state, rep, target, key, path):
    if rep["status"] == "PASS":
        _save(advance(state, target, {key: _sha(path)}))
    else:
        _save(advance(state, "FAILED", rep["first_failing"]))
    print(f"{rep['stage']}: {rep['status']} {rep['first_failing'] or ''}")


def freeze_evaluator():
    state = load_state()
    if state["stage"] != "DEVELOPMENT_PASS":
        raise PermissionError(f"needs DEVELOPMENT_PASS, at {state['stage']}")
    rec = {"recipe_sha256": json.loads(RECIPE_P.read_text())["recipe_sha256"],
           "threshold": json.loads((V2 / "development/threshold.json")
                                   .read_text())["T_support"],
           "development_report_sha256": _sha(V2 /
                                             "development_report.json"),
           "code_sha256": {c: _sha(ROOT / c) for c in CODE_FILES},
           "completed": True}
    atomic_json(FREEZE_P, rec)
    _save(advance(state, "EVALUATOR_FROZEN",
                  {"evaluator_freeze_sha256": _sha(FREEZE_P)}))
    print("evaluator frozen")


def _check_frozen_code():
    fz = json.loads(FREEZE_P.read_text())
    changed = [c for c, h in fz["code_sha256"].items()
               if _sha(ROOT / c) != h]
    if changed:
        raise RuntimeError(f"code changed after evaluator freeze: {changed}")
    return fz


def fetch_fresh():
    """Reserved families only, <=6 replays each; no substitution."""
    state = load_state()
    require_stage(state, "EVALUATOR_FROZEN")
    _check_frozen_code()
    from qa.collect2 import STATE_P as COLLECT_P, _fetch_family
    from qa.expand import thirds_picks
    from qa.replays import _save as save_collect, _secret_key
    res = json.loads((ROOT / "experiments/qa-v3/chart-latent/"
                             "reservation.json").read_text())
    st = json.loads(COLLECT_P.read_text())
    key = _secret_key()
    for entry in res["families"]:
        if entry["family"] in st["charts"]:
            continue
        rec = _fetch_family(entry, "qa_validate_comparator", st, key,
                            picks_fn=thirds_picks,
                            max_picks=MAX_FRESH_PER_FAMILY,
                            max_replays=MAX_FRESH_PER_FAMILY)
        st["charts"][entry["family"]] = rec
        save_collect(COLLECT_P, st)
        print(f"  [fresh] {entry['family']}: {len(rec['replays'])} replays",
              flush=True)
    from qa.contract import migrate_originals
    from qa.telemetry_v2 import derive_role
    migrate_originals()
    derive_role("qa_validate_comparator")


def validate_fresh():
    state = load_state()
    require_stage(state, "EVALUATOR_FROZEN")
    if state["stage"] != "EVALUATOR_FROZEN":
        raise PermissionError(f"needs EVALUATOR_FROZEN, at {state['stage']}")
    fz = _check_frozen_code()
    from qa.coverage_sample import select_windows
    from qa.features import records_for_role
    d = V2 / "fresh"
    d.mkdir(parents=True, exist_ok=True)
    man_p = d / "sample_manifest.json"
    if not man_p.exists():
        specs, short = {}, []
        for fam in RESERVED_FAMILIES:
            recs = [{"player_token": r["player_token"],
                     "window_id": r["window_id"]}
                    for r in records_for_role("qa_validate_comparator",
                                              "validate", families={fam})]
            players = {r["player_token"] for r in recs}
            if len(recs) < 100 or len(players) < 3:
                short.append(fam)
            specs[fam] = {"role": "qa_validate_comparator",
                          "ids": select_windows(recs, cap=2000)}
        atomic_json(man_p, {"specs": specs, "insufficient": short,
                            "completed": True})
    man = json.loads(man_p.read_text())
    if man["insufficient"]:
        rep = {"stage": "fresh", "status": "FAILED",
               "first_failing": {"stage": "fresh",
                                 "dimension": "VALIDATION_INSUFFICIENT",
                                 "families": man["insufficient"]},
               "completed": True}
    else:
        rep = score(_run_families(d, man["specs"],
                                  {"T_support": fz["threshold"]}), "fresh")
    atomic_json(V2 / "fresh_report.json", rep)
    _finish(state, rep, "FRESH_PASS", "fresh_report_sha256",
            V2 / "fresh_report.json")
    return rep


if __name__ == "__main__":
    import sys
    {"freeze-recipe": freeze_recipe, "develop": develop,
     "freeze-evaluator": freeze_evaluator, "fetch-fresh": fetch_fresh,
     "validate-fresh": validate_fresh}[sys.argv[1]]()
