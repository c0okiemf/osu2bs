"""Comparator-QA Task 1 (spec 2026-09-24 §§1-2,9): versioned contract,
staged lifecycle and freeze machinery for evaluator `comparator-v1`.

- build_recipe assembles the complete frozen identity (code/dependency
  hashes, seeds, budgets, reserved families, bank identity, geometry
  constants, adjudicator model/settings and executable modality profile).
  Absolute-kinematics/mixture outputs are DIAGNOSTIC ONLY and can never
  hold a decision role. validate_recipe is the freeze gate: unresolved
  adjudicator identity, missing code files or a tampered hash refuse.
- advance() drives BUILDING -> RECIPE_FROZEN -> DEVELOPMENT_PASS ->
  EVALUATOR_FROZEN -> FRESH_PASS -> CONFIRMED strictly in order, each
  step demanding its evidence hash; FAILED is reachable from anywhere
  with stage+dimension. No transition happens because a file exists.
- Payload acquisition and derived loaders call require_stage(); fresh
  payloads need EVALUATOR_FROZEN, seal access needs FRESH_PASS.
- The four unread reservations migrate to role qa_validate_comparator by
  an explicit lineage record preserving the original reservation hash —
  never by moving them toward training. Historical latent records stay.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "experiments" / "qa-v4" / "comparator"
STATE_P = OUT_ROOT / "state.json"
LINEAGE_P = OUT_ROOT / "reservation_lineage.json"
LABEL = "comparator-v1"
SEED = 20260924
RESERVED_FAMILIES = ("fam:11ec7", "fam:10ff1", "fam:658e", "fam:20cae")
STAGES = ("BUILDING", "RECIPE_FROZEN", "DEVELOPMENT_PASS",
          "EVALUATOR_FROZEN", "FRESH_PASS", "CONFIRMED")
STAGE_EVIDENCE = {"RECIPE_FROZEN": "recipe_sha256",
                  "DEVELOPMENT_PASS": "development_report_sha256",
                  "EVALUATOR_FROZEN": "evaluator_freeze_sha256",
                  "FRESH_PASS": "fresh_report_sha256",
                  "CONFIRMED": "seal_report_sha256"}
CODE_FILES = ("qa/comparator_contract.py", "qa/certificates.py",
              "qa/comparator_support.py", "qa/comparator_controls.py",
              "qa/comparator_run.py", "qa/evidence.py",
              "qa/adjudication.py", "qa/prompts/comparator-v1.md",
              "qa/scene.py", "qa/physics.py", "qa/mutations.py",
              "qa/neighbours.py", "qa/features.py",
              "qa/coverage_sample.py", "qa/contract.py",
              "docs/specs/2026-09-24-qa-comparator-design.md")
BUDGETS = {"work_deadline_min": 45, "hard_deadline_min": 55,
           "ram_headroom_gb": 2,
           "dev_pair_judgments_max": 160, "fresh_pair_judgments_max": 80,
           "seal_pair_judgments_max": 120,
           "fresh_replays_max": 24, "replay_cap_total": 583,
           "replay_cap_absolute": 600, "byte_cap_gb": 5}


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_obj(o):
    return _sha_bytes(json.dumps(o, sort_keys=True, default=str).encode())


def atomic_json(path, record):
    """fsync/replace write; an existing COMPLETED record with a different
    content hash refuses (no silent overwrite of finished work)."""
    path = Path(path)
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get("completed") and _sha_obj(prior) != _sha_obj(record):
            raise ValueError(f"completed record collision at {path.name}: "
                             "refusing overwrite with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(record, indent=1, sort_keys=True, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _dependency_versions():
    import matplotlib
    import numpy
    import torch
    try:
        import librosa
        lb = librosa.__version__
    except Exception:
        lb = "unavailable"
    return {"torch": torch.__version__, "numpy": numpy.__version__,
            "matplotlib": matplotlib.__version__, "librosa": lb}


def build_recipe(inputs):
    """Complete recipe record; missing pieces are RECORDED (build always
    succeeds) and refuse at validate_recipe (the freeze gate)."""
    code, missing = {}, []
    for rel in CODE_FILES:
        p = ROOT / rel
        if p.exists():
            code[rel] = _sha_bytes(p.read_bytes())
        else:
            missing.append(rel)
    artifacts = {}
    for name, rel in (("speed_threshold",
                       "experiments/qa-v1/speed_threshold.json"),
                      ("reservation",
                       "experiments/qa-v3/chart-latent/reservation.json"),
                      ("collect2_state",
                       "experiments/qa-v1/collect2_state.json")):
        p = ROOT / rel
        artifacts[name] = _sha_bytes(p.read_bytes()) if p.exists() \
            else None
    rec = {"label": LABEL, "seed": SEED, "schema_version": 1,
           "model_decision_role": "diagnostic_only",
           "code_sha256": code, "missing_code_files": missing,
           "dependencies": _dependency_versions(),
           "budgets": dict(BUDGETS),
           "reserved_families": list(RESERVED_FAMILIES),
           "bank_identity": {"identity": "qa-train-v2",
                             "state_sha256": artifacts["collect2_state"]},
           "geometry": {"heights_m": [1.4, 1.7, 2.0],
                        "grid": "x=(li-1.5)*0.6, y=0.6+ll*0.6"},
           "artifact_sha256": artifacts,
           "adjudicator": inputs.get("adjudicator") or {},
           "modality_profile": inputs.get("modality_profile"),
           "ruling": "review + spec 2f687f5"}
    rec["recipe_sha256"] = _sha_obj({k: v for k, v in rec.items()
                                     if k != "recipe_sha256"})
    return rec


def validate_recipe(recipe):
    """The freeze gate: raises ValueError on anything unresolved."""
    if recipe.get("model_decision_role") != "diagnostic_only":
        raise ValueError("mixture/latent outputs can only ever be "
                         "diagnostic_only in comparator-v1")
    if recipe.get("missing_code_files"):
        raise ValueError("cannot freeze with missing code files: "
                         f"{recipe['missing_code_files']}")
    adj = recipe.get("adjudicator") or {}
    for k in ("model", "version", "settings"):
        if not adj.get(k):
            raise ValueError(f"unresolved adjudicator identity: {k}")
    if recipe.get("modality_profile") not in ("images+signals",
                                              "images+signals+audio"):
        raise ValueError("executable modality profile unresolved")
    if list(recipe.get("reserved_families") or ()) != \
            list(RESERVED_FAMILIES):
        raise ValueError("reserved family set drifted")
    if recipe.get("artifact_sha256", {}).get("speed_threshold") is None:
        raise ValueError("frozen speed-warning artifact missing")
    want = _sha_obj({k: v for k, v in recipe.items()
                     if k != "recipe_sha256"})
    if recipe.get("recipe_sha256") != want:
        raise ValueError("recipe hash mismatch (tampered or stale)")


def load_state():
    if STATE_P.exists():
        return json.loads(STATE_P.read_text())
    return {"stage": "BUILDING", "label": LABEL, "history": []}


def advance(state, target, evidence):
    """Strict in-order stage transition with mandatory evidence."""
    cur = state.get("stage", "BUILDING")
    if target == "FAILED":
        for k in ("stage", "dimension"):
            if not evidence.get(k):
                raise ValueError(f"FAILED requires evidence {k!r}")
        new = {**state, "stage": "FAILED",
               "failure": dict(evidence),
               "history": state.get("history", []) + [
                   {"from": cur, "to": "FAILED", **evidence}]}
        return new
    if cur == "FAILED":
        raise PermissionError("packet already FAILED; a new authority "
                              "decision is required")
    if target not in STAGES:
        raise ValueError(f"unknown stage {target!r}")
    if STAGES.index(target) != STAGES.index(cur) + 1:
        raise PermissionError(f"cannot advance {cur} -> {target}: stages "
                              "are strictly ordered")
    key = STAGE_EVIDENCE[target]
    if not evidence.get(key):
        raise ValueError(f"transition to {target} requires evidence "
                         f"{key!r}")
    return {**state, "stage": target, key: evidence[key],
            "history": state.get("history", []) + [
                {"from": cur, "to": target, key: evidence[key]}]}


def save_state(state):
    atomic_json(STATE_P, state)
    return state


def require_stage(state, minimum):
    """Gate for payload acquisition / derived loaders: PermissionError
    below the required stage."""
    cur = state.get("stage", "BUILDING")
    if cur == "FAILED":
        raise PermissionError("packet FAILED; access denied")
    if STAGES.index(cur) < STAGES.index(minimum):
        raise PermissionError(f"stage {cur} < required {minimum}")


def migrate_reservation():
    """Explicit lineage: the four unread latent reservations become role
    qa_validate_comparator, preserving the original reservation hash.
    Idempotent; never moves families toward any training role."""
    if LINEAGE_P.exists():
        return json.loads(LINEAGE_P.read_text())
    res_p = ROOT / "experiments/qa-v3/chart-latent/reservation.json"
    res = json.loads(res_p.read_text())
    fams = [r["family"] for r in res["families"]]
    if sorted(fams) != sorted(RESERVED_FAMILIES):
        raise RuntimeError("reservation families do not match the "
                           "contract's reserved set")
    if res["status"] != "RESERVED":
        raise RuntimeError(f"reservation status {res['status']}")
    rec = {"from_role": "qa_validate_latent",
           "to_role": "qa_validate_comparator",
           "families": sorted(fams),
           "reservation_sha256": _sha_bytes(res_p.read_bytes()),
           "unread": True,
           "ruling": "review; spec 2f687f5",
           "note": "metadata-only lineage; payloads still require "
                   "EVALUATOR_FROZEN stage",
           "completed": True}
    atomic_json(LINEAGE_P, rec)
    return rec


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "lineage"
    if cmd == "lineage":
        print(json.dumps(migrate_reservation(), indent=1))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
