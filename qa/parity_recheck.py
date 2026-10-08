"""Feature-parity repair Task 4 (review spec 2026-09-24 §4): ONE frozen
corrected evaluation of the existing models on their actual fitted inputs.

- Fold recovery: run_folds never persisted fold states; at most four
  deterministic reconstructions (original tensors, seed SEED+k, original
  player removals, first 500 updates ONLY) are authorized, saved
  permanently under experiments/qa-v2/feature-parity/folds/, and each
  recovered macro NLL must match the recorded fold value within 1e-6 or
  the packet stops as REPRODUCTION_MISMATCH. An existing artifact is
  loaded, hash-checked and NEVER refit.
- freeze_recheck persists code/config/sample/checkpoint identity BEFORE
  any corrected coverage exists; run_recheck refuses to run if the code
  or samples changed after the freeze.
- One run over all 4 calibration families (final checkpoint) and all 16
  QA-train families (their own out-of-fold checkpoints/scalers/spans);
  fixed inflation grid {1,1.25,1.5,2}; >=85% pooled AND equal-player
  coverage and <=20% too-wide intervals in EVERY family for one global
  factor; no accuracy filtering, no family exclusion, no per-family
  factors. Spans use observed fitting-partition components only.
- Historical qa-v1 artifacts stay read-only; outcomes:
  FEATURE_PROVENANCE_UNRESOLVED / INSUFFICIENT / CALIBRATION_FAILED_V2 /
  CALIBRATION_REPAIRED (which only resumes pre-seal evaluator work —
  never a seal or E1 decision).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from qa.calibrate import _p95
from qa.coverage_sample import coverage_table, select_windows
from qa.model import (FEATURE_NAMES, MotionMixture, TARGET_NAMES,
                      mixture_quantiles, target_from_window)
from qa.train import SEED, _macro_nll

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "qa-v2" / "feature-parity"
FOLD_DIR = OUT_DIR / "folds"
RUN_DIR = OUT_DIR / "run"
FITS = ROOT / "experiments" / "qa-v1" / "model-fits"
INFLATION_GRID = (1.0, 1.25, 1.5, 2.0)
COVERAGE_MIN = 0.85
SHARPNESS_MAX_WIDE = 0.20
CAP = 2000
SAMPLE_SEED = "qa-parity-v2"
RECOVERY_UPDATES = 500
AUDIO_IDX = [i for i, n in enumerate(FEATURE_NAMES)
             if n.startswith("audio_")]
CODE_FILES = ("qa/features.py", "qa/coverage_sample.py", "qa/model.py",
              "qa/neighbours.py", "qa/train.py", "qa/parity_recheck.py",
              "qa/calibrate.py", "qa/telemetry_v2.py", "qa/scene.py",
              "qa/audio.py")


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _ids_sha(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# ---------------- masked reference spans ----------------

def observed_values(values, mask):
    """Only observed fitting-partition values; placeholder zeros for
    masked components are NOT data."""
    return [v for v, m in zip(values, mask) if m]


def observed_span(values, mask, lo=0.01, hi=0.99):
    ov = observed_values(values, mask)
    if not ov:
        return None
    t = torch.tensor(ov, dtype=torch.float64)
    return float(torch.quantile(t, hi) - torch.quantile(t, lo))


def component_spans(y, mask):
    """Per-target p1-p99 spans over observed values only."""
    out = []
    for j in range(y.shape[1]):
        col = y[:, j][mask[:, j]]
        out.append(float(torch.quantile(col.double(), 0.99)
                         - torch.quantile(col.double(), 0.01))
                   if len(col) else None)
    return out


# ---------------- fold recovery ----------------

def fold_manifest():
    rep = json.loads((FITS / "fold_report.json").read_text())
    if rep["chosen_updates"] != RECOVERY_UPDATES:
        raise RuntimeError("recorded chosen_updates changed")
    out = []
    for k in range(4):
        p = FOLD_DIR / f"fold{k}.pt"
        e = {"fold": k, "path": str(p), "updates": RECOVERY_UPDATES,
             "held": rep["folds"][k],
             "recorded_macro_nll": rep["per_update"]["500"][k]}
        if p.exists():
            e["sha256"] = _sha_file(p)
        out.append(e)
    return out


def load_or_recover_fold(entry, fams=None):
    """Load an existing hash-matching fold artifact, or perform exactly the
    authorized 500-update reconstruction. Never refits an existing fold;
    never reselects settings or budget."""
    p = Path(entry["path"])
    if p.exists():
        want = entry.get("sha256")
        if want and _sha_file(p) != want:
            raise ValueError(f"fold artifact hash mismatch: {p.name}")
        art = torch.load(p)
        if art["updates"] != entry["updates"] \
                or art["seed"] != SEED + entry["fold"] \
                or sorted(art["held"]) != sorted(entry["held"]):
            raise ValueError(f"fold artifact identity mismatch: {p.name}")
        return art
    if entry["updates"] != RECOVERY_UPDATES:
        raise ValueError("recovery cannot reselect updates: only the "
                         f"original {RECOVERY_UPDATES}-update snapshot may "
                         "be reconstructed")
    from qa.train import _fit, assemble_role, folds_of
    if fams is None:
        fams = assemble_role("qa_train")
    held = folds_of(fams)[entry["fold"]]
    if sorted(held) != sorted(entry["held"]):
        raise RuntimeError("fold membership drifted from the recorded "
                           "report — stop")
    train_f = {f: fams[f] for f in fams if f not in held}
    held_players = {pl for f in held for pl in fams[f]["players"]}
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    resume = FOLD_DIR / f"fold{entry['fold']}.resume.pt"
    _m, scaler, snaps = _fit(train_f, RECOVERY_UPDATES,
                             SEED + entry["fold"],
                             removed_players=held_players,
                             stop_after=RECOVERY_UPDATES,
                             state_path=resume, save_state_every=100)
    state = snaps[RECOVERY_UPDATES]
    m = MotionMixture()
    m.load_state_dict(state)
    m.eval()
    macro = _macro_nll(m, {f: fams[f] for f in held}, scaler)
    got = sum(macro.values()) / len(macro)
    if abs(got - entry["recorded_macro_nll"]) > 1e-6:
        raise RuntimeError(
            f"REPRODUCTION_MISMATCH fold {entry['fold']}: recomputed "
            f"{got!r} vs recorded {entry['recorded_macro_nll']!r}")
    art = {"state": state, "scaler": scaler, "updates": RECOVERY_UPDATES,
           "seed": SEED + entry["fold"], "held": sorted(held),
           "removed_players": sorted(held_players),
           "macro_nll_recorded": entry["recorded_macro_nll"],
           "macro_nll_recomputed": got,
           "compute_updates_spent": RECOVERY_UPDATES,
           "provenance": "deterministic reconstruction (original cached "
                         "tensors, original seed and player removals, "
                         "first 500 updates); recorded macro NLL matches "
                         "within 1e-6 — matching recorded metrics, not "
                         "proven byte identity to a lost artifact"}
    tmp = p.with_suffix(".tmp")
    torch.save(art, tmp)
    tmp.replace(p)
    if resume.exists():
        resume.unlink()
    print(f"[fold {entry['fold']}] reconstructed; macro NLL {got:.9f} "
          f"matches recorded", flush=True)
    return art


# ---------------- freeze ----------------

def verify_sample(sm):
    if _ids_sha(sm["ids"]) != sm["sha256"]:
        raise ValueError("sample manifest altered after freeze")


def build_sample_manifest():
    """Selected window IDs per family, BEFORE any NN retrieval or model
    scoring, so nothing downstream can censor the sample."""
    from qa.features import records_for_role, role_families
    sm = {}
    for role, purpose in (("qa_calib", "calibrate"), ("qa_train", "fit")):
        for fam in role_families(role):
            recs = [{"player_token": r["player_token"],
                     "window_id": r["window_id"]}
                    for r in records_for_role(role, purpose,
                                              families={fam})]
            ids = select_windows(recs, cap=CAP, seed=SAMPLE_SEED)
            sm[fam] = {"role": role, "n_eligible": len(recs),
                       "n_players": len({r["player_token"] for r in recs}),
                       "ids": ids, "sha256": _ids_sha(ids)}
            print(f"  [sample] {fam}: {len(ids)}/{len(recs)} windows",
                  flush=True)
    return sm


def freeze_recheck():
    """Persist complete identity BEFORE producing coverage. Idempotent:
    an existing freeze is verified, never overwritten."""
    from qa.features import family_identity
    man_p = OUT_DIR / "freeze_manifest.json"
    if man_p.exists():
        man = json.loads(man_p.read_text())
        for rel, want in man["code_sha256"].items():
            if _sha_file(ROOT / rel) != want:
                raise RuntimeError(f"code changed after freeze: {rel}")
        return man
    prov = json.loads((OUT_DIR / "provenance_qa_train.json").read_text())
    if prov["status"] != "OK":
        raise RuntimeError("FEATURE_PROVENANCE_UNRESOLVED — no corrected "
                           "run without reconciled provenance")
    folds = fold_manifest()
    for e in folds:
        if not Path(e["path"]).exists():
            raise RuntimeError("fold artifacts must be recovered before "
                               "the freeze (python -m qa.parity_recheck "
                               "recover)")
        e["sha256"] = _sha_file(e["path"])
    samples = build_sample_manifest()
    man = {"frozen_at": datetime.now(timezone.utc).isoformat(),
           "code_sha256": {rel: _sha_file(ROOT / rel)
                           for rel in CODE_FILES},
           "config": {"inflation_grid": list(INFLATION_GRID),
                      "coverage_min": COVERAGE_MIN,
                      "sharpness_max_wide": SHARPNESS_MAX_WIDE,
                      "cap": CAP, "sample_seed": SAMPLE_SEED,
                      "recovery_updates": RECOVERY_UPDATES},
           "checkpoints": {"final": _sha_file(FITS
                                              / "motion_mixture_final.pt"),
                           "fold_report": _sha_file(FITS
                                                    / "fold_report.json"),
                           "folds": {str(e["fold"]): e["sha256"]
                                     for e in folds}},
           "fold_entries": folds,
           "families": {fam: family_identity(fam, rec["role"])
                        for fam, rec in samples.items()},
           "samples": samples}
    tmp = man_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    tmp.replace(man_p)
    print(f"freeze written: {man_p}")
    return man


# ---------------- corrected run ----------------

def _quantiles_for(model, scaler, x):
    with torch.no_grad():
        lw, mu, ls = model((x - scaler[0]) / scaler[1])
    return mixture_quantiles(lw, mu, ls).float()   # [N,3,T]


def _eval_family(fam, sm, model, scaler, spans, bank, exclude, t_std):
    from qa.features import family_evidence, features_for, records_for_role
    from qa.neighbours import descriptor, retrieve
    purpose = "fit" if sm["role"] == "qa_train" else "calibrate"
    verify_sample(sm)
    recs = {r["window_id"]: r
            for r in records_for_role(sm["role"], purpose, families={fam})}
    missing = [i for i in sm["ids"] if i not in recs]
    if missing:
        raise RuntimeError(f"{fam}: {len(missing)} frozen sample ids no "
                           "longer resolve — population drift, stop")
    sel = [recs[i] for i in sm["ids"]]
    ev = family_evidence(fam)
    x = torch.stack([features_for(r, ev) for r in sel])
    ys, ms = zip(*(target_from_window(r["window"]) for r in sel))
    y, mask = torch.stack(ys), torch.stack(ms)
    q = _quantiles_for(model, scaler, x)
    p10, p50, p90 = q[:, 0], q[:, 1], q[:, 2]
    # report-only paired audio-omission check on the SAME windows: zeroing
    # the audio slots is exactly map_features(..., audio=None)
    q0 = None
    if ev["policy"] == "audio":
        x0 = x.clone()
        x0[:, AUDIO_IDX] = 0.0
        q0 = _quantiles_for(model, scaler, x0)
    # NN support (selection already frozen; failures censor nothing)
    desc_cache, nn = {}, []
    for r in sel:
        key = (r["profile"]["left_handed"], r["chart_index"])
        if key not in desc_cache:
            desc_cache[key] = descriptor(r["scene"], r["chart_index"])
        nn.append(retrieve(bank, desc_cache[key], exclude=exclude))
    sup = [n["support_distance"] for n in nn if n["status"] == "ok"]
    dis = []
    for i, n in enumerate(nn):
        if n["status"] != "ok":
            continue
        nb_med = torch.stack([nb["targets"] for nb in n["neighbours"]]) \
            .median(dim=0).values
        d = ((p50[i] - nb_med).abs() / t_std)[mask[i]]
        if len(d):
            dis.append(float(d.mean()))
    span_t = torch.tensor([s if s is not None else float("inf")
                           for s in spans])
    coverage, sharpness, omission = {}, {}, {}
    for f in INFLATION_GRID:
        lo = p50 - f * (p50 - p10)
        hi = p50 + f * (p90 - p50)
        inside = (y >= lo) & (y <= hi)
        obs = [{"family": fam, "player": sel[i]["player_token"],
                "window_id": sel[i]["window_id"],
                "inside": inside[i].tolist(), "observed": mask[i].tolist(),
                "nn_supported": nn[i]["status"] == "ok",
                "event_time": sel[i]["window"].get("event_time")}
               for i in range(len(sel))]
        tab = coverage_table(obs)["families"][fam]
        coverage[str(f)] = {"pooled": tab["pooled_coverage"],
                            "equal_player": tab["equal_player_coverage"],
                            "per_component": tab["per_component"],
                            "per_player": tab["per_player"],
                            "overlapping_window_count":
                                tab["overlapping_window_count"]}
        width = f * (p90 - p10)
        wide = (width > span_t[None, :]) & mask
        sharpness[str(f)] = {
            "wide_share": round(float(wide.sum()) / max(1,
                                                        int(mask.sum())), 4),
            "n_components": int(mask.sum())}
        if q0 is not None:
            lo0 = q0[:, 1] - f * (q0[:, 1] - q0[:, 0])
            hi0 = q0[:, 1] + f * (q0[:, 2] - q0[:, 1])
            in0 = ((y >= lo0) & (y <= hi0))[mask]
            omission[str(f)] = {
                "pooled_without_audio": round(float(in0.float().mean()), 4),
                "pooled_with_audio": coverage[str(f)]["pooled"]}
    resid = (y - p50) / t_std
    per_comp_resid = {}
    for j, name in enumerate(TARGET_NAMES):
        col = resid[:, j][mask[:, j]]
        if len(col):
            per_comp_resid[name] = {"mean": round(float(col.mean()), 4),
                                    "abs_mean":
                                        round(float(col.abs().mean()), 4),
                                    "n": int(len(col))}
    return {"family": fam, "role": sm["role"],
            "n_eligible": sm["n_eligible"], "n_selected": len(sel),
            "n_players_selected": len({r["player_token"] for r in sel}),
            "evidence_policy": ev["policy"],
            "sample_sha256": sm["sha256"],
            "support": {"nn_ok": len(sup),
                        "nn_insufficient": len(sel) - len(sup),
                        "support_p95": _p95(sup),
                        "disagree_p95": _p95(dis)},
            "coverage": coverage, "sharpness": sharpness,
            "residuals_std": per_comp_resid,
            "audio_omission_check_report_only": omission or None}


def choose_global_factor(results):
    for f in INFLATION_GRID:
        ok = True
        for r in results.values():
            c = r["coverage"][str(f)]
            s = r["sharpness"][str(f)]
            if c["pooled"] is None or c["equal_player"] is None \
                    or c["pooled"] < COVERAGE_MIN \
                    or c["equal_player"] < COVERAGE_MIN \
                    or s["wide_share"] > SHARPNESS_MAX_WIDE:
                ok = False
                break
        if ok:
            return f
    return None


def disposition(results):
    insufficient = [f for f, r in results.items()
                    if r["n_selected"] == 0
                    or r["coverage"][str(INFLATION_GRID[0])]["pooled"]
                    is None]
    if insufficient:
        return {"status": "INSUFFICIENT", "families": insufficient,
                "factor": None}
    factor = choose_global_factor(results)
    return {"status": "CALIBRATION_REPAIRED" if factor is not None
            else "CALIBRATION_FAILED_V2", "factor": factor}


def run_recheck(resume=True):
    from qa.neighbours import bank_from_role
    from qa.train import assemble_role, folds_of
    man = freeze_recheck()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    packs = assemble_role("qa_train")
    folds = folds_of(packs)
    fold_of = {f: k for k, fold in enumerate(folds) for f in fold}
    arts = {e["fold"]: load_or_recover_fold(e, fams=packs)
            for e in man["fold_entries"]}
    ck = torch.load(FITS / "motion_mixture_final.pt")
    final = MotionMixture()
    final.load_state_dict(ck["state"])
    final.eval()
    # observed-only standardization + fitting-partition spans
    ally = torch.cat([packs[f]["y"] for f in packs])
    allm = torch.cat([packs[f]["mask"] for f in packs])
    t_std = torch.tensor(
        [float(ally[:, j][allm[:, j]].std()) if int(allm[:, j].sum()) > 1
         else 1.0 for j in range(ally.shape[1])]).clamp(min=1e-6)
    final_spans = component_spans(ally, allm)
    fold_spans, fold_models = {}, {}
    for k, art in arts.items():
        rows_y, rows_m = [], []
        removed = set(art["removed_players"])
        for f in packs:
            if f in art["held"]:
                continue
            keep = [i for i, pl in enumerate(packs[f]["players"])
                    if pl not in removed]
            rows_y.append(packs[f]["y"][keep])
            rows_m.append(packs[f]["mask"][keep])
        fold_spans[k] = component_spans(torch.cat(rows_y),
                                        torch.cat(rows_m))
        m = MotionMixture()
        m.load_state_dict(art["state"])
        m.eval()
        fold_models[k] = (m, art["scaler"])
    bank, _records = bank_from_role()
    results = {}
    for fam, sm in sorted(man["samples"].items()):
        out_p = RUN_DIR / (fam.replace(":", "_") + ".json")
        if resume and out_p.exists():
            prior = json.loads(out_p.read_text())
            if prior["sample_sha256"] == sm["sha256"]:
                results[fam] = prior
                continue
            raise ValueError(f"{fam}: resumed result was produced from a "
                             "different frozen sample")
        if sm["role"] == "qa_train":
            k = fold_of[fam]
            model, scaler = fold_models[k]
            spans = fold_spans[k]
            exclude = {"families": {fam},
                       "players": set(packs[fam]["players"])}
            tag = f"fold-{k}"
        else:
            model, scaler = final, ck["scaler"]
            spans = final_spans
            exclude = None
            tag = "final"
        r = _eval_family(fam, sm, model, scaler, spans, bank, exclude,
                         t_std)
        r["model"] = tag
        tmp = out_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=1, sort_keys=True))
        tmp.replace(out_p)
        results[fam] = r
        c = r["coverage"]["2.0"]
        print(f"  [{fam}] {tag} pooled@2 {c['pooled']} equal@2 "
              f"{c['equal_player']} nn_ok {r['support']['nn_ok']}"
              f"/{r['n_selected']}", flush=True)
    disp = disposition(results)
    report = {"disposition": disp, "config": man["config"],
              "frozen_at": man["frozen_at"],
              "reconstruction_compute_updates":
                  sum(a.get("compute_updates_spent", 0)
                      for a in arts.values()),
              "families": {f: {k: v for k, v in r.items()
                               if k not in ("coverage",)}
                           for f, r in results.items()},
              "coverage": {f: r["coverage"] for f, r in results.items()},
              "note": "corrected run on actual fitted inputs; historical "
                      "qa-v1 results preserved read-only; REPAIRED only "
                      "resumes pre-seal evaluator validation — no seal or "
                      "E1 decision here"}
    if disp["status"] == "CALIBRATION_REPAIRED":
        calib = {f: r for f, r in results.items()
                 if r["role"] == "qa_calib"}
        report["thresholds_v2"] = {
            "T_support": max(r["support"]["support_p95"]
                             for r in calib.values()),
            "T_disagree": max(r["support"]["disagree_p95"]
                              for r in calib.values()),
            "inflation_factor": disp["factor"],
            "recipes": "unchanged max-family-p95; grid {1,1.25,1.5,2} "
                       ">=85% pooled AND equal-player in every family; "
                       "sharpness <=20% wide vs observed-only train "
                       "p1-p99 spans"}
        tp = OUT_DIR / "frozen_thresholds_v2.json"
        tmp = tp.with_suffix(".tmp")
        tmp.write_text(json.dumps(report["thresholds_v2"], indent=1,
                                  sort_keys=True))
        tmp.replace(tp)
    rp = OUT_DIR / "recheck_report.json"
    tmp = rp.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(rp)
    print(f"{disp['status']} (factor {disp['factor']}) -> {rp}")
    return report


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "recover":
        from qa.train import assemble_role
        fams = assemble_role("qa_train")
        for e in fold_manifest():
            load_or_recover_fold(e, fams=fams)
    elif cmd == "freeze":
        freeze_recheck()
    elif cmd == "run":
        run_recheck()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
