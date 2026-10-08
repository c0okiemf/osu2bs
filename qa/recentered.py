"""Comparator-anchored location correction.

ONE frozen variant, no fits/collection/tuning: for each query window,
retrieve the existing five QA-train neighbours (unchanged diversity/
support rules), and in log1p target space shift EVERY mixture-component
mean of the v2 predictor by `neighbour target median − original model
predictive median`, per observed target component. Mixture weights and
variances unchanged — a pure location shift, so every marginal quantile
moves by exactly the delta and interval widths (hence sharpness) are
untouched by construction.

Correction rules: neighbour medians use only OBSERVED neighbour
components (bank rows now carry masks; placeholder zeros are not data);
zero observed neighbours for a component, or an unsupported retrieval,
leaves that component/window UNCORRECTED and counted as unknown. No
query targets, chart-specific human averages, family IDs, accuracy
filtering or calib/seal neighbours.

Anti-circularity: the retained support/disagreement measurement stays
the ORIGINAL recipe against the UNCORRECTED median (post-shift agreement
is guaranteed by construction and earns no credit). This is one
composite predictor, not two systems confirming each other.

Evaluation: one frozen run on the same eight development/calibration
families and identical windows; original grid {1,1.25,1.5,2}; >=85%
pooled AND equal-player coverage plus <=20% wide vs the ORIGINAL frozen
spans in EVERY family. Pass -> advances only to human-control/
adversarial validation before evaluator freeze and sealed confirmation.
Fail -> stop. E1 held; B0 active; seal untouched.
"""
import json
from pathlib import Path

import torch

from qa.calibrate import _p95
from qa.coverage_sample import coverage_table
from qa.model import (MotionMixture, TARGET_NAMES, mixture_quantiles,
                      target_from_window)

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "qa-v2" / "expansion"
OUT_DIR = ROOT / "experiments" / "qa-v2" / "recentered"
RUN_DIR = OUT_DIR / "run"
FREEZE_P = OUT_DIR / "freeze.json"
INFLATION_GRID = (1.0, 1.25, 1.5, 2.0)
COVERAGE_MIN = 0.85
SHARPNESS_MAX_WIDE = 0.20
CODE_FILES = ("qa/recentered.py", "qa/neighbours.py", "qa/model.py",
              "qa/features.py", "qa/coverage_sample.py", "qa/expand.py")


def neighbour_delta(nns, p50):
    """Per-component location delta from OBSERVED neighbour targets;
    None-supported components stay uncorrected (delta 0, counted)."""
    delta = torch.zeros(len(TARGET_NAMES))
    unknown = []
    if nns["status"] != "ok":
        return delta, list(TARGET_NAMES)
    for j, name in enumerate(TARGET_NAMES):
        vals = [float(n["targets"][j]) for n in nns["neighbours"]
                if "mask" not in n or bool(n["mask"][j])]
        if not vals:
            unknown.append(name)
            continue
        nb_med = float(torch.tensor(vals).median())
        delta[j] = nb_med - float(p50[j])
    return delta, unknown


def run(resume=True):
    from qa.expand import CKPT_P, EVAL_FREEZE_P
    from qa.features import family_evidence, features_for, records_for_role
    from qa.neighbours import bank_from_role, descriptor, retrieve
    from qa.parity_recheck import _sha_file, verify_sample
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    eman = json.loads(EVAL_FREEZE_P.read_text())
    if FREEZE_P.exists():
        fman = json.loads(FREEZE_P.read_text())
        for rel, want in fman["code_sha256"].items():
            if _sha_file(ROOT / rel) != want:
                raise RuntimeError(f"code changed after freeze: {rel}")
    else:
        fman = {"ruling": "review", "recipe": __doc__.strip(),
                "code_sha256": {rel: _sha_file(ROOT / rel)
                                for rel in CODE_FILES},
                "checkpoint_v2": _sha_file(CKPT_P),
                "eval_freeze_sha256": _sha_file(EVAL_FREEZE_P),
                "sample_sha256": {f: sm["sha256"]
                                  for f, sm in eman["samples"].items()}}
        tmp = FREEZE_P.with_suffix(".tmp")
        tmp.write_text(json.dumps(fman, indent=1, sort_keys=True))
        tmp.replace(FREEZE_P)
    ck = torch.load(CKPT_P)
    model = MotionMixture()
    model.load_state_dict(ck["state"])
    model.eval()
    mu_s, sd_s = ck["scaler"]
    spans = torch.tensor([s if s is not None else float("inf")
                          for s in eman["frozen_spans_original16"]])
    t_std = torch.tensor(eman["frozen_t_std_original16"]).clamp(min=1e-6)
    bank, _r = bank_from_role(identity="qa-train-v2")
    results = {}
    for fam, sm in sorted(eman["samples"].items()):
        out_p = RUN_DIR / (fam.replace(":", "_") + ".json")
        if resume and out_p.exists():
            prior = json.loads(out_p.read_text())
            if prior["sample_sha256"] != sm["sha256"]:
                raise ValueError(f"{fam}: resumed result from a different "
                                 "frozen sample")
            results[fam] = prior
            continue
        verify_sample(sm)
        purpose = "fit" if sm["role"] == "qa_train" else "calibrate"
        recs = {r["window_id"]: r
                for r in records_for_role(sm["role"], purpose,
                                          families={fam})}
        sel = [recs[i] for i in sm["ids"]]
        ev = family_evidence(fam)
        x = torch.stack([features_for(r, ev) for r in sel])
        ys, ms = zip(*(target_from_window(r["window"]) for r in sel))
        y, mask = torch.stack(ys), torch.stack(ms)
        with torch.no_grad():
            lw, mus, ls = model((x - mu_s) / sd_s)
        q0 = mixture_quantiles(lw, mus, ls).float()      # uncorrected
        desc_cache, nns = {}, []
        for r in sel:
            key = (r["profile"]["left_handed"], r["chart_index"])
            if key not in desc_cache:
                desc_cache[key] = descriptor(r["scene"], r["chart_index"])
            nns.append(retrieve(bank, desc_cache[key], exclude=None))
        deltas, unknown_counts = [], {}
        for i, nn in enumerate(nns):
            d, unk = neighbour_delta(nn, q0[i, 1])
            deltas.append(d)
            for name in unk:
                unknown_counts[name] = unknown_counts.get(name, 0) + 1
        delta = torch.stack(deltas)
        # pure location shift: corrected quantiles = original + delta
        mus_c = mus + delta.double()[:, None, :]
        qc = mixture_quantiles(lw, mus_c, ls).float()
        # RETAINED original disagreement recipe vs the UNCORRECTED median
        sup, dis = [], []
        for i, nn in enumerate(nns):
            if nn["status"] != "ok":
                continue
            sup.append(nn["support_distance"])
            nb_med = torch.stack([n["targets"] for n in nn["neighbours"]]) \
                .median(dim=0).values
            dv = ((q0[i, 1] - nb_med).abs() / t_std)[mask[i]]
            if len(dv):
                dis.append(float(dv.mean()))
        coverage, sharpness = {}, {}
        for f in INFLATION_GRID:
            rows = {}
            for tag, q in (("corrected", qc), ("original", q0)):
                lo = q[:, 1] - f * (q[:, 1] - q[:, 0])
                hi = q[:, 1] + f * (q[:, 2] - q[:, 1])
                inside = (y >= lo) & (y <= hi)
                obs = [{"family": fam, "player": sel[i]["player_token"],
                        "window_id": sel[i]["window_id"],
                        "inside": inside[i].tolist(),
                        "observed": mask[i].tolist(),
                        "nn_supported": nns[i]["status"] == "ok"}
                       for i in range(len(sel))]
                tab = coverage_table(obs)["families"][fam]
                rows[tag] = {"pooled": tab["pooled_coverage"],
                             "equal_player": tab["equal_player_coverage"]}
            coverage[str(f)] = rows
            width = f * (qc[:, 2] - qc[:, 0])
            wide = (width > spans[None, :]) & mask
            sharpness[str(f)] = round(float(wide.sum())
                                      / max(1, int(mask.sum())), 4)
        resid_c = (y - qc[:, 1]) / t_std
        resid_0 = (y - q0[:, 1]) / t_std
        per_comp = {}
        for j, name in enumerate(TARGET_NAMES):
            col_c, col_0 = resid_c[:, j][mask[:, j]], \
                resid_0[:, j][mask[:, j]]
            if len(col_c):
                per_comp[name] = {
                    "resid_mean_original": round(float(col_0.mean()), 4),
                    "resid_mean_corrected": round(float(col_c.mean()), 4),
                    "n": int(len(col_c))}
        rec_out = {"family": fam, "role": sm["role"],
                   "n_selected": len(sel),
                   "sample_sha256": sm["sha256"],
                   "support": {"nn_ok": len(sup),
                               "nn_insufficient": len(sel) - len(sup),
                               "support_p95": _p95(sup),
                               "disagree_p95_uncorrected": _p95(dis)},
                   "unknown_component_counts": unknown_counts,
                   "coverage": coverage, "sharpness": sharpness,
                   "residuals_std": per_comp}
        tmp = out_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec_out, indent=1, sort_keys=True))
        tmp.replace(out_p)
        results[fam] = rec_out
        print(f"  [{fam}] corrected pooled@1.5 "
              f"{coverage['1.5']['corrected']['pooled']} (orig "
              f"{coverage['1.5']['original']['pooled']}) wide@1.5 "
              f"{sharpness['1.5']}", flush=True)
    chosen = None
    for f in INFLATION_GRID:
        ok = all(
            r["coverage"][str(f)]["corrected"]["pooled"] is not None
            and r["coverage"][str(f)]["corrected"]["pooled"] >= COVERAGE_MIN
            and r["coverage"][str(f)]["corrected"]["equal_player"]
            >= COVERAGE_MIN
            and r["sharpness"][str(f)] <= SHARPNESS_MAX_WIDE
            for r in results.values())
        if ok:
            chosen = f
            break
    status = "RECENTERED_CALIBRATED" if chosen is not None \
        else "RECENTERED_FAILED"
    report = {"status": status, "inflation_factor": chosen,
              "families": results,
              "note": "one composite predictor (model + neighbour "
                      "location correction); disagreement retained "
                      "against the UNCORRECTED median; pass advances only "
                      "to control/adversarial validation — no evaluator "
                      "freeze, seal or E1 authority"}
    rp = OUT_DIR / "recentered_report.json"
    tmp = rp.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(rp)
    print(f"{status} (factor {chosen}) -> {rp}")
    return report


if __name__ == "__main__":
    run()
