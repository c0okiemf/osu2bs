"""Global-factor feasibility calculation.

CPU-only; no fit, no collection, no per-family factors, no asymmetric
adjustment. FROZEN amended development-calibration recipe:

1. For every observed component of the 8 gate families' frozen windows
   under the v2 refit, the required inflation to include its target:
   below median (median-t)/(median-p10); above (t-median)/(p90-median);
   t == median needs 0. Zero-width sides are handled EXPLICITLY: target
   off a zero-width side can never be covered (+inf); never an epsilon.
2. The smallest single global factor meeting >=85% coverage in ALL 8
   families under BOTH aggregations (pooled = unit weights; equal-player
   = weight 1/(n_players*n_components_p*n_obs_pc)), via exact weighted
   empirical quantiles with deterministic tie handling (smallest value
   whose cumulative weight >= 0.85), restricted to [1, 2].
3. At that factor, the unchanged per-family sharpness limit (<=20% wide
   vs the ORIGINAL frozen 16-family spans).

Acceptance = both pass everywhere -> a PROPOSED new calibration version,
NOT evaluator acceptance. Sharpness failing at the minimal
coverage-satisfying factor proves no larger symmetric global factor can
pass these gates on these observations (wide-share is nondecreasing in
the factor; smaller factors fail coverage) -> INCOMPATIBLE, stop.
EXPANSION_FAILED stays the historical verdict. Seal untouched; E1 held.
"""
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "qa-v2" / "expansion"
OUT_P = EXP / "factor_feasibility.json"
COVERAGE_MIN = 0.85
SHARPNESS_MAX_WIDE = 0.20
FACTOR_LO, FACTOR_HI = 1.0, 2.0


def required_factor(target, p10, median, p90):
    """Minimal symmetric inflation covering one observed component."""
    if target == median:
        return 0.0
    if target < median:
        width = median - p10
        return (median - target) / width if width > 0 else math.inf
    width = p90 - median
    return (target - median) / width if width > 0 else math.inf


def weighted_min_factor(freqs, weights, q=COVERAGE_MIN):
    """Smallest value f with sum(weights[freq <= f]) >= q * total —
    exact weighted empirical quantile, deterministic ties (all equal
    values accumulate together)."""
    total = sum(weights)
    pairs = sorted(zip(freqs, weights))
    acc, i = 0.0, 0
    while i < len(pairs):
        v = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == v:
            acc += pairs[i][1]
            i += 1
        if acc >= q * total:
            return v
    return math.inf


def family_min_factors(obs):
    """obs: list of {player, component, f_req}. Returns (pooled, equal)."""
    freqs = [o["f_req"] for o in obs]
    pooled = weighted_min_factor(freqs, [1.0] * len(freqs))
    players = {}
    for o in obs:
        players.setdefault(o["player"], {}).setdefault(o["component"],
                                                       []).append(o["f_req"])
    fr, w = [], []
    n_p = len(players)
    for comps in players.values():
        n_c = len(comps)
        for vals in comps.values():
            for v in vals:
                fr.append(v)
                w.append(1.0 / (n_p * n_c * len(vals)))
    return pooled, weighted_min_factor(fr, w)


def run():
    from qa.expand import CKPT_P, EVAL_FREEZE_P
    from qa.features import family_evidence, features_for, records_for_role
    from qa.model import (MotionMixture, TARGET_NAMES, mixture_quantiles,
                          target_from_window)
    from qa.parity_recheck import verify_sample
    man = json.loads(EVAL_FREEZE_P.read_text())
    ck = torch.load(CKPT_P)
    model = MotionMixture()
    model.load_state_dict(ck["state"])
    model.eval()
    mu_s, sd_s = ck["scaler"]
    spans = man["frozen_spans_original16"]
    report = {"recipe": __doc__.strip(), "families": {}}
    fam_obs, fam_widths = {}, {}
    for fam, sm in sorted(man["samples"].items()):
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
        q = mixture_quantiles(lw, mus, ls)          # float64 [N,3,T]
        obs, widths = [], []
        for i, r in enumerate(sel):
            for j, name in enumerate(TARGET_NAMES):
                if not bool(mask[i, j]):
                    continue
                p10, med, p90 = (float(q[i, 0, j]), float(q[i, 1, j]),
                                 float(q[i, 2, j]))
                obs.append({"player": r["player_token"], "component": name,
                            "f_req": required_factor(float(y[i, j]), p10,
                                                     med, p90)})
                widths.append((p90 - p10, spans[j]))
        fam_obs[fam], fam_widths[fam] = obs, widths
        pooled, equal = family_min_factors(obs)
        report["families"][fam] = {"n_observed": len(obs),
                                   "min_factor_pooled": pooled,
                                   "min_factor_equal_player": equal,
                                   "n_uncoverable": sum(
                                       1 for o in obs
                                       if o["f_req"] == math.inf)}
        print(f"  [{fam}] min f pooled {pooled:.4f} equal {equal:.4f}",
              flush=True)
    f_star = max(max(r["min_factor_pooled"], r["min_factor_equal_player"])
                 for r in report["families"].values())
    f_star = max(f_star, FACTOR_LO)
    report["global_min_factor"] = f_star
    if f_star > FACTOR_HI or f_star == math.inf:
        report["status"] = "INCOMPATIBLE"
        report["reason"] = (f"minimal coverage-satisfying factor "
                           f"{f_star} exceeds the frozen [1,2] range")
    else:
        sharp = {}
        ok = True
        for fam, widths in fam_widths.items():
            wide = sum(1 for w, s in widths
                       if s is not None and f_star * w > s)
            share = wide / max(1, len(widths))
            sharp[fam] = round(share, 4)
            ok = ok and share <= SHARPNESS_MAX_WIDE
        report["sharpness_at_factor"] = sharp
        report["status"] = "FEASIBLE_FACTOR" if ok else "INCOMPATIBLE"
        if not ok:
            report["reason"] = ("sharpness fails at the MINIMAL "
                               "coverage-satisfying factor; wide-share is "
                               "nondecreasing in the factor and smaller "
                               "factors fail coverage, so no symmetric "
                               "global factor passes these gates on these "
                               "observations")
    report["note"] = ("development-calibration evidence only; a feasible "
                      "factor is a PROPOSED calibration version, not "
                      "evaluator acceptance; EXPANSION_FAILED stands as "
                      "the historical verdict")
    tmp = OUT_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(OUT_P)
    print(f"{report['status']} (global min factor "
          f"{report['global_min_factor']:.4f}) -> {OUT_P}")
    return report


if __name__ == "__main__":
    run()
