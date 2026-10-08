"""Independent-QA Task 5 (calibration): frozen threshold recipes (spec §6).

The RECIPES are committed here before any calibration score is inspected;
running this module applies them ONCE against the four qa-calib families:

- T_support   = max over calib families of the family p95 control support
                distance (query vs the QA-train bank);
- T_disagree  = same max-of-family-p95 for the standardized absolute
                difference between the model predictive median and the
                empirical neighbour target median, averaged over supported
                components (standardization = QA-train target std);
- inflation   = smallest factor in {1, 1.25, 1.5, 2} such that the model's
                [p10, p90] interval inflated about the median covers >=85%
                of observed components in EVERY calib family; no factor
                reaching it = calibration failure;
- sharpness   = inflated intervals wider than the QA-train p1-p99 target
                span for >20% of components fail.

Thresholds are engineering values from four independent families — never a
population confidence claim. E1/seal outcomes never enter here.
"""
import json
from pathlib import Path

import torch

from qa.model import MotionMixture, mixture_quantiles
from qa.neighbours import bank_from_role, descriptor, retrieve

ROOT = Path(__file__).resolve().parent.parent
FITS = ROOT / "experiments" / "qa-v1" / "model-fits"
OUT = ROOT / "experiments" / "qa-v1" / "frozen_thresholds.json"
INFLATION_GRID = (1.0, 1.25, 1.5, 2.0)
COVERAGE_MIN = 0.85
SHARPNESS_MAX_WIDE = 0.20


def _p95(v):
    s = sorted(v)
    return s[min(len(s) - 1, int(0.95 * len(s)))] if s else None


def run_calibration(max_windows_per_family=2000):
    from qa.features import family_evidence, features_for, records_for_role
    from qa.model import target_from_window
    ck = torch.load(FITS / "motion_mixture_final.pt")
    model = MotionMixture()
    model.load_state_dict(ck["state"])
    model.eval()
    mu_s, sd_s = ck["scaler"]
    bank, records = bank_from_role()
    train_targets = torch.stack([r["targets"] for r in records])
    t_std = train_targets.std(dim=0).clamp(min=1e-6)
    t_p1 = torch.quantile(train_targets, 0.01, dim=0)
    t_p99 = torch.quantile(train_targets, 0.99, dim=0)
    fam_support, fam_disagree = {}, {}
    fam_cov = {}
    widths, spans = [], []            # per-component raw interval widths
    fam_ev = {}
    for r in records_for_role("qa_calib", "calibrate"):
        fam, sc, ci = r["family"], r["scene"], r["chart_index"]
        ev = fam_ev.setdefault(fam, family_evidence(fam))
        sup = fam_support.setdefault(fam, [])
        dis = fam_disagree.setdefault(fam, [])
        cov = fam_cov.setdefault(fam, {k: [] for k in INFLATION_GRID})
        if len(sup) >= max_windows_per_family:
            continue
        q = descriptor(sc, ci)
        ns = retrieve(bank, q, exclude=None)
        if ns["status"] != "ok":
            continue
        sup.append(ns["support_distance"])
        y, mask = target_from_window(r["window"])
        x = features_for(r, ev)
        with torch.no_grad():
            lw, mus, ls = model(((x - mu_s) / sd_s)[None])
        q = mixture_quantiles(lw, mus, ls)[0].float()
        p10, med, p90 = q[0], q[1], q[2]
        nb_med = torch.stack([n["targets"] for n in
                              ns["neighbours"]]).median(dim=0).values
        dvec = ((med - nb_med).abs() / t_std)[mask]
        if len(dvec):
            dis.append(float(dvec.mean()))
        for factor in INFLATION_GRID:
            lo = med - factor * (med - p10)
            hi = med + factor * (p90 - med)
            inside = ((y >= lo) & (y <= hi))[mask]
            cov[factor] += inside.tolist()
        widths += (p90 - p10)[mask].tolist()
        spans += (t_p99 - t_p1)[mask].tolist()
    t_support = max(_p95(v) for v in fam_support.values())
    t_disagree = max(_p95(v) for v in fam_disagree.values())
    chosen_factor = None
    for factor in INFLATION_GRID:
        if all((sum(c[factor]) / max(1, len(c[factor]))) >= COVERAGE_MIN
               for c in fam_cov.values()):
            chosen_factor = factor
            break
    # sharpness at the chosen factor: inflated width vs train p1-p99 span
    sharp_fail = None
    wide_share = None
    if chosen_factor is not None:
        wide = [w * chosen_factor > s for w, s in zip(widths, spans)]
        wide_share = sum(wide) / max(1, len(wide))
        sharp_fail = wide_share > SHARPNESS_MAX_WIDE
    status = "CALIBRATED" if (chosen_factor is not None
                              and not sharp_fail) else "CALIBRATION_FAILED"
    out = {"status": status, "T_support": t_support,
           "T_disagree": t_disagree, "inflation_factor": chosen_factor,
           "per_family": {
               f: {"n": len(fam_support[f]),
                   "support_p95": _p95(fam_support[f]),
                   "disagree_p95": _p95(fam_disagree[f]),
                   "coverage": {str(k): round(sum(v) / max(1, len(v)), 4)
                                for k, v in fam_cov[f].items()}}
               for f in fam_support},
           "sharpness": {"wide_share": wide_share,
                         "fail": bool(sharp_fail)},
           "recipes": "max-family-p95; inflation grid {1,1.25,1.5,2} "
                      ">=85% every family; sharpness <=20% wide vs train "
                      "p1-p99; frozen before inspection",
           "note": "engineering thresholds from four independent families"}
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True))
    tmp.replace(OUT)
    print(f"{status}: T_support {t_support:.3f} T_disagree "
          f"{t_disagree:.3f} inflation {chosen_factor}")
    return out


def chart_support_check(per_head_support, per_head_disagree, thresholds,
                        head_times=None):
    """Serving-time usable-support rule: >=90% eligible heads within BOTH
    thresholds and no unsupported run longer than 2 s containing >=4
    heads; the rest must be explicitly reviewed, never dropped."""
    ok = [s is not None and d is not None
          and s <= thresholds["T_support"] and d <= thresholds["T_disagree"]
          for s, d in zip(per_head_support, per_head_disagree)]
    share = sum(ok) / max(1, len(ok))
    long_run = False
    if head_times is not None:
        run_start, run_len = None, 0
        for i, good in enumerate(ok):
            if not good:
                if run_start is None:
                    run_start = head_times[i]
                run_len += 1
                if run_len >= 4 and head_times[i] - run_start > 2.0:
                    long_run = True
            else:
                run_start, run_len = None, 0
    return {"share_supported": share, "usable": share >= 0.90
            and not long_run, "long_unsupported_run": long_run,
            "unsupported_indices": [i for i, g in enumerate(ok) if not g]}


if __name__ == "__main__":
    run_calibration()
