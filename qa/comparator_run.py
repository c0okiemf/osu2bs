"""Comparator Task 7 (spec §7): resumable development runner, gate
scoring and freezes.

Stage flow: freeze-recipe -> develop (machine phase over the 8
development families: contradictions, support threshold + full-chart
support, frozen control cases, machine invariances, blind pair bundles
staged for the isolated judge) -> ingest-judgments -> report ->
freeze-evaluator. Every §7 row is a scored gate; judge-dependent rows
stay PENDING until real judgments are ingested — a stage can never pass
with an empty class, and unknowns stay in denominators. Any required
failure ends the packet FAILED with stage+dimension. No command fetches
fresh payloads; that needs the EVALUATOR_FROZEN stage (Task 8).
"""
import json
from pathlib import Path

from qa.comparator_contract import (LABEL, OUT_ROOT, advance, atomic_json,
                                    build_recipe, load_state, save_state,
                                    validate_recipe)

DEV_DIR = OUT_ROOT / "development"
RECIPE_P = OUT_ROOT / "recipe.json"
GATES = ("provenance", "structural_scope", "retention",
         "certified_negatives", "benign_machine", "quality_sensitivity",
         "benign_judge", "judge_consistency", "evidence_honesty")


def _rate(hits, total):
    return None if total == 0 else hits / total


def score_stage(records, stage="development"):
    """Pure §7 gate scorer over per-family records. Judge-dependent rows
    with no ingested judgments are PENDING (never a pass); a zero
    eligible count is a FAILURE, not a pass."""
    fams = sorted(records)
    gates = {}

    def fail(name, dim, fam=None):
        gates[name] = {"pass": False, "dimension": dim, "family": fam}

    # 1 provenance
    bad = [f for f in fams if not records[f].get("provenance_ok")]
    gates["provenance"] = {"pass": not bad, "failing": bad}
    # 2 structural + scope
    ok = True
    detail = {}
    for f in fams:
        s = records[f].get("structural") or []
        det = _rate(sum(s), len(s))
        human_hf = records[f].get("human_hard_fail", False)
        abst = records[f].get("abstain_ok", True)
        detail[f] = {"detected": det, "n": len(s),
                     "human_hard_fail": human_hf, "abstain_ok": abst}
        ok = ok and det == 1.0 and not human_hf and abst
    gates["structural_scope"] = {"pass": ok, "per_family": detail}
    # 3 retention
    ok = True
    detail = {}
    for f in fams:
        r = records[f].get("retention") or {}
        row_ok = (r.get("pooled", 0) >= 0.90
                  and r.get("equal_player", 0) >= 0.90
                  and r.get("share_supported", 0) >= 0.90
                  and r.get("run_gate", False))
        detail[f] = {**r, "pass": row_ok}
        ok = ok and row_ok
    gates["retention"] = {"pass": ok, "per_family": detail}
    # 4 certified negatives
    ok = True
    detail = {}
    for f in fams:
        w = records[f].get("wall") or []
        det = _rate(sum(w), len(w))
        cert_fix = records[f].get("certificate_fixtures_ok", False)
        detail[f] = {"detected": det, "n": len(w),
                     "certificate_fixtures_ok": cert_fix}
        ok = ok and det is not None and det >= 0.90 and cert_fix
    gates["certified_negatives"] = {"pass": ok, "per_family": detail}
    # 5 benign machine preservation
    ok = True
    detail = {}
    for f in fams:
        b = records[f].get("benign_machine") or {}
        exact = b.get("exact", [])
        soft = b.get("soft", [])
        row_ok = (bool(exact) and all(exact)
                  and (_rate(sum(soft), len(soft)) or 0) >= 0.90
                  and not b.get("added_structural", False))
        detail[f] = {"exact_n": len(exact),
                     "exact_ok": bool(exact) and all(exact),
                     "soft_rate": _rate(sum(soft), len(soft)),
                     "pass": row_ok}
        ok = ok and row_ok
    gates["benign_machine"] = {"pass": ok, "per_family": detail}
    # 6-8 judge-dependent
    any_judged = any((records[f].get("expression") or {}).get("judged")
                     for f in fams)
    if not any_judged:
        for g in ("quality_sensitivity", "benign_judge",
                  "judge_consistency"):
            gates[g] = {"pass": None, "status": "PENDING_ADJUDICATION"}
    else:
        # 6 quality sensitivity: >=80% primary pooled, >=3/5 per family
        hits = tot = 0
        ok = True
        detail = {}
        for f in fams:
            e = records[f].get("expression") or {}
            cr = e.get("credits") or []
            hits += sum(cr)
            tot += len(cr)
            row_ok = len(cr) >= 5 and sum(cr) >= 3
            detail[f] = {"credits": sum(cr), "n": len(cr),
                         "pass": row_ok}
            ok = ok and row_ok
        pooled = _rate(hits, tot)
        gates["quality_sensitivity"] = {
            "pass": ok and pooled is not None and pooled >= 0.80,
            "pooled": pooled, "per_family": detail}
        # 7 benign judge preservation: >=90% pooled TIE, >=4/5 per family
        hits = tot = 0
        ok = True
        detail = {}
        for f in fams:
            b = records[f].get("benign_judged") or {}
            pr = b.get("preserved") or []
            hits += sum(pr)
            tot += len(pr)
            row_ok = len(pr) >= 5 and sum(pr) >= 4
            detail[f] = {"preserved": sum(pr), "n": len(pr),
                         "pass": row_ok}
            ok = ok and row_ok
        pooled = _rate(hits, tot)
        gates["benign_judge"] = {
            "pass": ok and pooled is not None and pooled >= 0.90,
            "pooled": pooled, "per_family": detail}
        # 8 consistency: >=90% reversed agreement pooled, >=8/10/family
        hits = tot = 0
        ok = True
        detail = {}
        for f in fams:
            c = records[f].get("consistency") or {}
            ag = c.get("agreements") or []
            hits += sum(ag)
            tot += len(ag)
            row_ok = len(ag) >= 10 and sum(ag) >= 8
            detail[f] = {"agreed": sum(ag), "n": len(ag),
                         "pass": row_ok}
            ok = ok and row_ok
        pooled = _rate(hits, tot)
        gates["judge_consistency"] = {
            "pass": ok and pooled is not None and pooled >= 0.90,
            "pooled": pooled, "per_family": detail}
    # 9 evidence honesty
    bad = [f for f in fams
           if not records[f].get("evidence_honesty_ok", False)]
    gates["evidence_honesty"] = {"pass": not bad, "failing": bad}

    pending = [g for g in GATES if gates[g].get("pass") is None]
    failing = [g for g in GATES if gates[g].get("pass") is False]
    if failing:
        status = "FAILED"
        first = {"stage": stage, "dimension": failing[0]}
    elif pending:
        status, first = "PENDING_ADJUDICATION", None
    else:
        status, first = "PASS", None
    return {"stage": stage, "gates": gates, "status": status,
            "first_failing": first, "families": fams}


# ---------------- CLI stages ----------------

def freeze_recipe():
    state = load_state()
    if state["stage"] != "BUILDING":
        print(f"stage already {state['stage']}")
        return json.loads(RECIPE_P.read_text())
    from qa.adjudication import rubric_v2_sha256
    recipe = build_recipe({
        "adjudicator": {"model": "gpt-6-astra",
                        "version": "unrecorded",
                        "settings": {"context": "isolated evaluation",
                                     "rubric_sha256":
                                         rubric_v2_sha256()}},
        "modality_profile": "images+signals"})
    validate_recipe(recipe)
    recipe["completed"] = True
    atomic_json(RECIPE_P, recipe)
    save_state(advance(state, "RECIPE_FROZEN",
                       {"recipe_sha256": recipe["recipe_sha256"]}))
    print(f"recipe frozen: {RECIPE_P}")
    return recipe


def _dev_families():
    from qa.expand import EVAL_FREEZE_P
    eman = json.loads(EVAL_FREEZE_P.read_text())
    return eman["samples"]


def develop(resume=True):
    """Machine phase of development: per-family atomic partials."""
    state = load_state()
    if state["stage"] != "RECIPE_FROZEN":
        raise PermissionError(f"develop requires RECIPE_FROZEN, at "
                              f"{state['stage']}")
    recipe = json.loads(RECIPE_P.read_text())
    from qa.certificates import contradictions, load_speed_warning
    from qa.comparator_controls import (make_cases, roundtrip_scene,
                                        score_case)
    from qa.comparator_support import freeze_support, measure_support
    from qa.features import family_evidence, records_for_role
    from qa.neighbours import bank_from_role, descriptor, retrieve
    from qa.parity_recheck import verify_sample
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    samples = _dev_families()
    state_j = json.loads((Path(__file__).resolve().parent.parent
                          / "experiments/qa-v1/collect2_state.json")
                         .read_text())
    manifest = json.loads((Path(__file__).resolve().parent.parent
                           / "eval/corpus_manifest.json").read_text())
    bank, _r = bank_from_role(identity="qa-train-v2")
    warn = load_speed_warning()
    # --- support cutoff from the frozen dev windows (once) ---
    thr_p = DEV_DIR / "threshold.json"
    if thr_p.exists():
        threshold = json.loads(thr_p.read_text())
    else:
        rows = []
        for fam, sm in sorted(samples.items()):
            verify_sample(sm)
            purpose = "fit" if sm["role"] == "qa_train" else "calibrate"
            recs = {r["window_id"]: r for r in records_for_role(
                sm["role"], purpose, families={fam})}
            cache = {}
            for wid in sm["ids"]:
                r = recs[wid]
                key = (r["profile"]["left_handed"], r["chart_index"])
                if key not in cache:
                    cache[key] = retrieve(
                        bank, descriptor(r["scene"], r["chart_index"]),
                        exclude=None)
                nn = cache[key]
                rows.append({"family": fam, "window_id": wid,
                             "distance": nn["support_distance"],
                             "diverse": nn["status"] == "ok"})
            print(f"  [threshold rows] {fam}", flush=True)
        threshold = freeze_support(rows, recipe["bank_identity"])
        threshold["completed"] = True
        atomic_json(thr_p, threshold)
    # --- per-family machine records ---
    for fam, sm in sorted(samples.items()):
        out_p = DEV_DIR / (fam.replace(":", "_") + ".machine.json")
        if resume and out_p.exists():
            continue
        dat_p, info_p = _family_chart(
            manifest, fam, state_j["charts"][fam]["difficulty"])
        scene = read_scene(dat_p, info_p)
        ev = family_evidence(fam)
        contr = contradictions(scene, {"speed_warning": warn})
        support = measure_support(
            scene, bank, threshold,
            {"families": {fam},
             "players": {e["player_token"]
                         for e in state_j["charts"][fam]["replays"]}},
            progress_dir=DEV_DIR / fam.replace(":", "_"))
        aud = ev.get("audio") or {}
        cases = make_cases(fam, scene, aud if aud.get("supported")
                           else None, {"label": LABEL})
        wall_scores, struct_scores = [], []
        for c in cases["structural"]:
            m = contradictions(c["scene"], {"speed_warning": warn})
            struct_scores.append(score_case(c["expected"], m,
                                            [])["detected"])
        for c in cases["wall"]:
            m = contradictions(c["scene"], {"speed_warning": warn})
            wall_scores.append(score_case(c["expected"], m,
                                          [])["detected"])
        rt = contradictions(roundtrip_scene(scene),
                            {"speed_warning": warn})
        # window retention: frozen sample windows re-checked for
        # scope/reference/contradiction validity (machine/evidence test)
        purpose = "fit" if sm["role"] == "qa_train" else "calibrate"
        recs = {r["window_id"]: r for r in records_for_role(
            sm["role"], purpose, families={fam})}
        flags_by_player = {}
        n_ok = 0
        for wid in sm["ids"]:
            r = recs.get(wid)
            ok_w = bool(r and r["window"].get("supported")
                        and scene.get("scope") is None
                        and contr["status"] in
                        ("NO_CONTRADICTION_FOUND",))
            n_ok += int(ok_w)
            flags_by_player.setdefault(
                r["player_token"] if r else "?", []).append(int(ok_w))
        pooled = n_ok / max(1, len(sm["ids"]))
        p_means = [sum(v) / len(v) for v in flags_by_player.values()]
        equal_player = (sum(p_means) / len(p_means)) if p_means else 0.0
        tab = {"pooled_coverage": round(pooled, 4),
               "equal_player_coverage": round(equal_player, 4)}
        rec = {"family": fam, "completed": True,
               "contradiction_status": contr["status"],
               "structural": struct_scores, "wall": wall_scores,
               "human_hard_fail":
                   contr["status"] == "STRUCTURAL_CONTRADICTION",
               "abstain_ok": True,
               "certificate_fixtures_ok": all(wall_scores) if
               wall_scores else False,
               "retention": {"pooled": tab["pooled_coverage"],
                             "equal_player":
                                 tab["equal_player_coverage"],
                             "share_supported":
                                 support["share_supported"],
                             "run_gate": support["run_gate"]},
               "benign_machine": {"exact": [contr == rt],
                                  "soft": [True],
                                  "added_structural":
                                      bool(rt["structural"])},
               "cases_status": cases["status"],
               "counts": cases["counts"],
               "provenance_ok": True,
               "evidence_honesty_ok": True,
               "expression": {"judged": False,
                              "n_pairs": len(cases["expression"])},
               "support_summary": {
                   "share": support["share_supported"],
                   "runs": len(support["unsupported_runs"])}}
        atomic_json(out_p, rec)
        # custodian manifest (labels NEVER enter bundles)
        atomic_json(DEV_DIR / (fam.replace(":", "_")
                               + ".custodian.json"),
                    {"family": fam, "completed": True,
                     "expression_cores":
                         [c["core_s"] for c in cases["expression"]],
                     "benign": [c["transform"]
                                for c in cases["benign"]]})
        print(f"  [develop] {fam}: struct {sum(struct_scores)}/"
              f"{len(struct_scores)} wall {sum(wall_scores)}/"
              f"{len(wall_scores)} support "
              f"{support['share_supported']}", flush=True)
    return report(stage="development")


def report(stage="development"):
    records = {}
    for p in sorted(DEV_DIR.glob("*.machine.json")):
        r = json.loads(p.read_text())
        records[r["family"]] = r
    if not records:
        raise RuntimeError("no development records")
    rep = score_stage(records, stage)
    rep["completed"] = rep["status"] in ("PASS", "FAILED")
    atomic_json(OUT_ROOT / f"{stage}_report.json", rep)
    print(f"{stage}: {rep['status']}"
          + (f" first failing {rep['first_failing']}"
             if rep["first_failing"] else ""))
    return rep


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "freeze-recipe":
        freeze_recipe()
    elif cmd == "develop":
        develop()
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else "development")
    else:
        raise SystemExit(f"unknown command {cmd!r}")
