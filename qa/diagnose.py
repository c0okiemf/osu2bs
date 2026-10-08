"""Bounded frozen-model calibration diagnostic.

NO fitting decisions, NO threshold changes, NO accuracy filtering, NO seal
access. Fold models are DETERMINISTIC REPRODUCTIONS of the committed fold
fits (same seeds/budget -> identical weights) used solely to score
held-out families out-of-fold; that reproduction is recorded, not a new
fit. Outputs per-component / per-player / per-accuracy-band /
height-known coverage@2.0 and standardized residuals for all four calib
families and QA-train out-of-fold, plus implementation checks:
missing-height encoding, target units, mixture-quantile math, tracking
quality.
"""
import json
from pathlib import Path

import torch

from qa.model import (FEATURE_NAMES, MotionMixture, TARGET_NAMES,
                      mixture_quantiles, target_from_window)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "qa-v1" / "calibration_diagnostic.json"
ACC_BANDS = ((0.0, 0.80), (0.80, 0.90), (0.90, 1.01))
FACTOR = 2.0


def _acc_band(a):
    for lo, hi in ACC_BANDS:
        if lo <= (a or 0) < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "unknown"


def _pred(model, scaler, x, g=None):
    with torch.no_grad():
        lw, mus, ls = model(((x - scaler[0]) / scaler[1])[None])
    q = mixture_quantiles(lw, mus, ls)[0].float()
    return q[1], q[0], q[2]


def _accumulate(acc, key, inside, resid, mask):
    a = acc.setdefault(key, {"n": 0,
                             "cov": [0] * len(TARGET_NAMES),
                             "cnt": [0] * len(TARGET_NAMES),
                             "resid_sum": [0.0] * len(TARGET_NAMES),
                             "resid_abs": [0.0] * len(TARGET_NAMES)})
    a["n"] += 1
    for i in range(len(TARGET_NAMES)):
        if mask[i]:
            a["cnt"][i] += 1
            a["cov"][i] += int(inside[i])
            a["resid_sum"][i] += float(resid[i])
            a["resid_abs"][i] += abs(float(resid[i]))


def _finalize(acc):
    out = {}
    for key, a in acc.items():
        out[key] = {"n_windows": a["n"], "per_component": {}}
        for i, name in enumerate(TARGET_NAMES):
            c = max(1, a["cnt"][i])
            out[key]["per_component"][name] = {
                "coverage@2": round(a["cov"][i] / c, 4),
                "resid_mean": round(a["resid_sum"][i] / c, 4),
                "resid_abs_mean": round(a["resid_abs"][i] / c, 4),
                "n": a["cnt"][i]}
        covs = [a["cov"][i] / max(1, a["cnt"][i])
                for i in range(len(TARGET_NAMES)) if a["cnt"][i]]
        out[key]["coverage_overall"] = round(sum(covs) / len(covs), 4) \
            if covs else None
    return out


def _score_role(model, scaler, t_std, role, fams=None, max_per_family=2000):
    from qa.features import family_evidence, features_for, records_for_role
    by_family, by_player, by_acc, by_height = {}, {}, {}, {}
    fam_ev, fam_n = {}, {}
    purpose = "fit" if role == "qa_train" else "calibrate"
    for r in records_for_role(role, purpose, families=fams):
        fam = r["family"]
        if fam_n.get(fam, 0) >= max_per_family:
            continue
        ev = fam_ev.setdefault(fam, family_evidence(fam))
        y, mask = target_from_window(r["window"])
        x = features_for(r, ev)
        med, p10, p90 = _pred(model, scaler, x)
        lo = med - FACTOR * (med - p10)
        hi = med + FACTOR * (p90 - med)
        inside = ((y >= lo) & (y <= hi)).tolist()
        resid = ((y - med) / t_std).tolist()
        _accumulate(by_family, fam, inside, resid, mask.tolist())
        _accumulate(by_player, f"{fam}/{r['player_token'][:8]}",
                    inside, resid, mask.tolist())
        _accumulate(by_acc, _acc_band(r["accuracy"]), inside,
                    resid, mask.tolist())
        _accumulate(by_height,
                    "known" if r["profile"]["height_known"] else "unknown",
                    inside, resid, mask.tolist())
        fam_n[fam] = fam_n.get(fam, 0) + 1
    return {"by_family": _finalize(by_family),
            "by_player": _finalize(by_player),
            "by_accuracy_band": _finalize(by_acc),
            "by_height_status": _finalize(by_height)}


def implementation_checks(model, scaler):
    """Height encoding, unit ranges, mixture-quantile math, tracking."""
    checks = {}
    # (1) missing-height encoding: feature vector literally zeros the slots
    from qa.features import features_for
    sc = {"scope": None, "notes": [(1.0, 1, 0, 1, 1), (1.5, 2, 1, 0, 0)],
          "bombs": [], "walls": [], "settings": {}}
    xu = features_for({"family": "probe", "scene": sc, "chart_index": 0,
                       "profile": {"height": None, "height_known": False,
                                   "left_handed": False}},
                      {"family": "probe", "policy": "no_audio:probe",
                       "audio": None})
    hi_ = FEATURE_NAMES.index("body_height")
    ki_ = FEATURE_NAMES.index("body_height_known")
    checks["missing_height_zeroed"] = bool(xu[hi_] == 0.0 and xu[ki_] == 0.0)
    # (2) mixture-quantile math: single-component mixture => ANALYTIC
    # quantiles equal normal quantiles (deterministic CDF inversion)
    m = MotionMixture()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    x = torch.zeros(len(FEATURE_NAMES))
    med, p10, p90 = _pred(m, (torch.zeros_like(x), torch.ones_like(x)), x)
    # zeroed net -> mu 0, log_scale 0 (clamped in [-4,2] ok) -> N(0,1)
    z90 = 1.2815515655
    checks["quantile_math"] = {
        "p90_expected": z90, "p90_observed": round(float(p90[0]), 6),
        "ok": abs(float(p90[0]) - z90) < 1e-4}
    # (3) target unit sanity from train packs
    packs = sorted((ROOT / "experiments/qa-v1/model-data/qa_train")
                   .glob("*.pt"))
    ys = torch.cat([torch.load(p)["y"] for p in packs[:4]])
    checks["target_ranges_log1p"] = {
        TARGET_NAMES[i]: [round(float(ys[:, i].min()), 3),
                          round(float(ys[:, i].max()), 3)]
        for i in range(len(TARGET_NAMES))}
    # (4) tracking quality proxy: window frame counts per calib family
    from qa.features import records_for_role
    fps = {}
    for r in records_for_role("qa_calib", "calibrate"):
        arr = fps.setdefault(r["family"], [])
        if len(arr) < 50:
            arr.append(r["window"]["n_frames"])
    checks["calib_window_frames"] = {
        f: {"mean": round(sum(v) / len(v), 1), "min": min(v)}
        for f, v in fps.items() if v}
    return checks


def run():
    ck = torch.load(ROOT / "experiments/qa-v1/model-fits/"
                    "motion_mixture_final.pt")
    final = MotionMixture()
    final.load_state_dict(ck["state"])
    final.eval()
    scaler = ck["scaler"]
    packs = sorted((ROOT / "experiments/qa-v1/model-data/qa_train")
                   .glob("*.pt"))
    t_std = torch.cat([torch.load(p)["y"] for p in packs]).std(dim=0) \
        .clamp(min=1e-6)
    out = {"note": "frozen-model diagnostic; deterministic fold "
                   "reproductions; no fitting decisions, no thresholds, "
                   "no filtering, no seal access",
           "calib": _score_role(final, scaler, t_std, "qa_calib")}
    # out-of-fold: deterministic reproduction of the committed fold fits
    from qa.train import _fit, assemble_role, folds_of, SEED
    fams = assemble_role("qa_train")
    folds = folds_of(fams)
    oof = {}
    for k, held in enumerate(folds):
        train_f = {f: fams[f] for f in fams if f not in held}
        held_players = {p for f in held for p in fams[f]["players"]}
        _m, fscaler, snaps = _fit(train_f, 2000, SEED + k,
                                  removed_players=held_players)
        fm = MotionMixture()
        fm.load_state_dict(snaps[500])         # committed chosen updates
        fm.eval()
        part = _score_role(fm, fscaler, t_std, "qa_train", fams=set(held),
                           max_per_family=800)
        for section, data in part.items():
            oof.setdefault(section, {}).update(data)
    out["train_out_of_fold"] = oof
    out["implementation_checks"] = implementation_checks(final, scaler)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True))
    tmp.replace(OUT)
    print("diagnostic written:", OUT)
    return out


if __name__ == "__main__":
    run()
