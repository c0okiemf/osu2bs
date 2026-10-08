"""Chart-latent Task 5 (spec §6): one-shot development evaluation,
snapshot selection and candidate freeze.

- Development families = the 8 exposed calib families on their EXACT
  frozen expansion windows, scored with the FINAL 24-family local base
  (fold models were training inputs only). Contexts are canonical-chart
  only, standardized by the adapter's train-family scaler.
- ONE snapshot selected by lowest equal-family (equal-player) marginal
  observed NLL across all 8 families; ties within 1e-6 pick the earlier
  update; a step-0 winner is a nonlearning outcome -> DEVELOPMENT_FAILED.
- Frozen development gates: (1) candidate mean NLL no worse than the
  local base AND >=0.01 nats/observed-component better than the
  constant-prior ablation; (2) one factor of {1,1.25,1.5,2} meets >=85%
  pooled AND equal-player coverage and <=20% wide (ORIGINAL frozen
  spans) in EACH family; (3) complete denominators + support rules
  (max-family-p95 thresholds over the 8 development families, >=90%
  within-support retention, no long unsupported runs). Gate (4) — the
  blinded human-control/adversarial adjudication suite — is a separate
  explicit step: freeze_candidate REFUSES until its marker records a
  pass. Development failure stops BEFORE any fresh payload download.
"""
import json
import math
from pathlib import Path

import torch

from qa.calibrate import _p95, chart_support_check
from qa.chart_latent import ChartLatentAdapter, constant_prior
from qa.coverage_sample import coverage_table
from qa.model import (MotionMixture, TARGET_NAMES, mixture_quantiles,
                      target_from_window)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "qa-v3" / "chart-latent"
DEV_DIR = OUT_DIR / "dev"
DEV_REPORT_P = OUT_DIR / "dev_report.json"
ADJUDICATION_P = OUT_DIR / "adjudication_gate.json"
CANDIDATE_P = OUT_DIR / "candidate_freeze.json"
INFLATION_GRID = (1.0, 1.25, 1.5, 2.0)
COVERAGE_MIN = 0.85
SHARPNESS_MAX_WIDE = 0.20
NLL_LIFT_MIN = 0.01
SNAPSHOTS = (0, 500, 1000, 2000)


def marginal_nll_rows(lw, mu, ls, y, mask):
    """Per-window observed-component-normalized marginal NLL [N]."""
    y_ = y[:, None, :].double()
    m_ = mask[:, None, :]
    lp = -0.5 * (((y_ - mu.double()) / ls.double().exp()) ** 2) \
        - ls.double() - 0.5 * math.log(2 * math.pi)
    lp = torch.where(m_, lp, torch.zeros_like(lp)).sum(-1)
    row = -torch.logsumexp(lw.double() + lp, dim=-1) \
        / mask.sum(-1).clamp(min=1).double()
    return row, mask.any(-1)


def select_snapshot(mean_by_snapshot):
    """Lowest mean NLL; ties within 1e-6 pick the earlier update; a
    step-0 winner is a nonlearning outcome."""
    best = min(SNAPSHOTS,
               key=lambda u: (round(mean_by_snapshot[u], 6), u))
    return best, ("DEVELOPMENT_FAILED" if best == 0 else "OK")


def equal_player_mean(rows, keep, players):
    by_p = {}
    for i, p in enumerate(players):
        if bool(keep[i]):
            by_p.setdefault(p, []).append(float(rows[i]))
    means = [sum(v) / len(v) for v in by_p.values()]
    return sum(means) / len(means) if means else None


def _dev_family_inputs(fam, sm, base_model, base_scaler):
    """Base params + targets + meta for one dev family (cached)."""
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    p = DEV_DIR / ("base_" + fam.replace(":", "_") + ".pt")
    if p.exists():
        d = torch.load(p)
        if d["sample_sha256"] == sm["sha256"]:
            return d
        raise ValueError(f"{fam}: cached base params from a different "
                         "frozen sample")
    from qa.features import family_evidence, features_for, records_for_role
    from qa.parity_recheck import verify_sample
    verify_sample(sm)
    purpose = "fit" if sm["role"] == "qa_train" else "calibrate"
    recs = {r["window_id"]: r
            for r in records_for_role(sm["role"], purpose,
                                      families={fam})}
    sel = [recs[i] for i in sm["ids"]]
    ev = family_evidence(fam)
    x = torch.stack([features_for(r, ev) for r in sel])
    ys, ms = zip(*(target_from_window(r["window"]) for r in sel))
    with torch.no_grad():
        lw, mu, ls = base_model((x - base_scaler[0]) / base_scaler[1])
    d = {"family": fam, "sample_sha256": sm["sha256"],
         "lw": lw, "mu": mu, "ls": ls,
         "y": torch.stack(ys), "mask": torch.stack(ms),
         "players": [r["player_token"] for r in sel],
         "event_times": [float(r["window"].get("event_time", 0.0))
                         for r in sel],
         "nn_keys": [(r["profile"]["left_handed"], r["chart_index"])
                     for r in sel]}
    tmp = p.with_suffix(".tmp")
    torch.save(d, tmp)
    tmp.replace(p)
    return d


def _dev_context(fam):
    from qa.chart_context import chart_context
    from qa.features import family_evidence
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    state = json.loads((ROOT / "experiments/qa-v1/collect2_state.json")
                       .read_text())
    manifest = json.loads((ROOT / "eval/corpus_manifest.json").read_text())
    dat_p, info_p = _family_chart(manifest, fam,
                                  state["charts"][fam]["difficulty"])
    scene = read_scene(dat_p, info_p)             # canonical, unmirrored
    return chart_context(scene, family_evidence(fam))


def _adapter_from(state, s):
    a = ChartLatentAdapter(s)
    a.load_state_dict(state)
    a.eval()
    return a


def evaluate_development():
    from qa.chart_context import standardize
    from qa.expand import CKPT_P, EVAL_FREEZE_P
    from qa.neighbours import bank_from_role, descriptor, retrieve
    if DEV_REPORT_P.exists():
        return json.loads(DEV_REPORT_P.read_text())
    eman = json.loads(EVAL_FREEZE_P.read_text())
    ck = torch.load(CKPT_P)
    base = MotionMixture()
    base.load_state_dict(ck["state"])
    base.eval()
    ack = torch.load(OUT_DIR / "adapter.pt")
    s = torch.tensor(json.loads(
        (OUT_DIR / "design_manifest.json").read_text()
    )["frozen_t_std_original16"]).clamp(min=1e-6)
    design = json.loads((OUT_DIR / "design_manifest.json").read_text())
    spans = torch.tensor([v if v is not None else float("inf")
                          for v in design["frozen_spans_original16"]])
    t_std = torch.tensor(design["frozen_t_std_original16"]) \
        .clamp(min=1e-6)
    fams = sorted(eman["samples"])
    inputs, contexts, clipped = {}, {}, {}
    for fam in fams:
        inputs[fam] = _dev_family_inputs(fam, eman["samples"][fam], base,
                                         ck["scaler"])
        z, cl = standardize(_dev_context(fam).tensor(),
                            ack["context_scaler"])
        contexts[fam] = z
        if cl:
            clipped[fam] = cl
        print(f"  [dev inputs] {fam}", flush=True)
    # ---- snapshot selection on equal-family equal-player marginal NLL
    per_snap = {}
    for u in SNAPSHOTS:
        adapter = _adapter_from(ack["snapshots"][u], s)
        fam_nll = {}
        for fam in fams:
            d = inputs[fam]
            lw12, mu12, ls12 = adapter.marginal(d["lw"], d["mu"], d["ls"],
                                                contexts[fam])
            rows, keep = marginal_nll_rows(lw12, mu12, ls12, d["y"],
                                           d["mask"])
            fam_nll[fam] = equal_player_mean(rows, keep, d["players"])
        per_snap[u] = {"per_family": fam_nll,
                       "mean": sum(fam_nll.values()) / len(fam_nll)}
        print(f"  [snapshot {u}] mean NLL {per_snap[u]['mean']:.6f}",
              flush=True)
    best, sel_status = select_snapshot({u: per_snap[u]["mean"]
                                        for u in SNAPSHOTS})
    report = {"snapshot_nll": {str(u): per_snap[u] for u in SNAPSHOTS},
              "selected_snapshot": best,
              "clipped_context_dims": clipped}
    if sel_status == "DEVELOPMENT_FAILED":
        report["status"] = "DEVELOPMENT_FAILED"
        report["reason"] = "step-0 winner: adapter did not learn"
        _write(report)
        return report
    adapter = _adapter_from(ack["snapshots"][best], s)
    const = constant_prior(adapter,
                           [z for f, z in sorted(
                               _train_contexts(ack).items())])
    # ---- gate 1: NLL vs base and vs constant prior
    base_nll, const_nll = {}, {}
    for fam in fams:
        d = inputs[fam]
        rows, keep = marginal_nll_rows(d["lw"], d["mu"], d["ls"], d["y"],
                                       d["mask"])
        base_nll[fam] = equal_player_mean(rows, keep, d["players"])
        n = d["mu"].shape[0]
        lwc = (const["log_p"].to(d["lw"].dtype)[None, :, None]
               + d["lw"][:, None, :]).reshape(n, -1)
        muc = (d["mu"][:, None] + adapter.delta().to(d["mu"].dtype)
               [None, :, None, :]).reshape(n, -1, d["mu"].shape[-1])
        lsc = (d["ls"][:, None] + adapter.a().to(d["ls"].dtype)
               [None, :, None, :]).reshape(n, -1, d["ls"].shape[-1])
        rows, keep = marginal_nll_rows(lwc, muc, lsc, d["y"], d["mask"])
        const_nll[fam] = equal_player_mean(rows, keep, d["players"])
    cand_mean = per_snap[best]["mean"]
    base_mean = sum(base_nll.values()) / len(base_nll)
    const_mean = sum(const_nll.values()) / len(const_nll)
    gate1 = (cand_mean <= base_mean
             and const_mean - cand_mean >= NLL_LIFT_MIN)
    report["nll"] = {"candidate": cand_mean, "base": base_mean,
                     "constant_prior": const_mean,
                     "per_family_base": base_nll,
                     "per_family_constant": const_nll, "gate1": gate1}
    # ---- gate 2: coverage/sharpness on the original grid + spans
    coverage, factor = {}, None
    for fam in fams:
        d = inputs[fam]
        q = adapter.quantiles(d["lw"], d["mu"], d["ls"],
                              contexts[fam]).float()
        cov_f = {}
        for f in INFLATION_GRID:
            lo = q[:, 1] - f * (q[:, 1] - q[:, 0])
            hi = q[:, 1] + f * (q[:, 2] - q[:, 1])
            inside = (d["y"] >= lo) & (d["y"] <= hi)
            obs = [{"family": fam, "player": d["players"][i],
                    "window_id": str(i), "inside": inside[i].tolist(),
                    "observed": d["mask"][i].tolist(), "nn_supported":
                    True} for i in range(len(d["players"]))]
            tab = coverage_table(obs)["families"][fam]
            width = f * (q[:, 2] - q[:, 0])
            wide = (width > spans[None, :]) & d["mask"]
            cov_f[str(f)] = {"pooled": tab["pooled_coverage"],
                             "equal_player": tab["equal_player_coverage"],
                             "wide_share": round(float(wide.sum())
                                                 / max(1, int(d["mask"]
                                                              .sum())), 4)}
        coverage[fam] = cov_f
        inputs[fam]["q"] = q
    for f in INFLATION_GRID:
        ok = all(c[str(f)]["pooled"] is not None
                 and c[str(f)]["pooled"] >= COVERAGE_MIN
                 and c[str(f)]["equal_player"] >= COVERAGE_MIN
                 and c[str(f)]["wide_share"] <= SHARPNESS_MAX_WIDE
                 for c in coverage.values())
        if ok:
            factor = f
            break
    report["coverage"] = coverage
    report["selected_factor"] = factor
    # ---- gate 3: support thresholds + retention (existing recipes)
    bank, _r = bank_from_role(identity="qa-train-v2")
    support = {}
    for fam in fams:
        d = inputs[fam]
        cache, sup, dis, dis_base = {}, [], [], []
        from qa.features import records_for_role
        purpose = "fit" if eman["samples"][fam]["role"] == "qa_train" \
            else "calibrate"
        recs = {r["window_id"]: r for r in records_for_role(
            eman["samples"][fam]["role"], purpose, families={fam})}
        sel = [recs[i] for i in eman["samples"][fam]["ids"]]
        base_q = mixture_quantiles(d["lw"], d["mu"], d["ls"]).float()
        for i, r in enumerate(sel):
            key = d["nn_keys"][i]
            if key not in cache:
                cache[key] = retrieve(bank,
                                      descriptor(r["scene"],
                                                 r["chart_index"]),
                                      exclude=None)
            nn = cache[key]
            if nn["status"] != "ok":
                sup.append(None)
                dis.append(None)
                continue
            nb = torch.stack([n["targets"] for n in nn["neighbours"]]) \
                .median(dim=0).values
            m = d["mask"][i]
            sup.append(nn["support_distance"])
            dis.append(float(((d["q"][i, 1] - nb).abs() / t_std)[m]
                             .mean()) if int(m.sum()) else None)
            dis_base.append(float(((base_q[i, 1] - nb).abs() / t_std)[m]
                                  .mean()) if int(m.sum()) else None)
        support[fam] = {"support": sup, "disagree": dis,
                        "support_p95": _p95([v for v in sup
                                             if v is not None]),
                        "disagree_p95": _p95([v for v in dis
                                              if v is not None]),
                        "disagree_p95_base": _p95([v for v in dis_base
                                                   if v is not None]),
                        "times": d["event_times"]}
        print(f"  [support] {fam} p95 "
              f"{support[fam]['support_p95']:.3f}", flush=True)
    thresholds = {"T_support": max(s_["support_p95"]
                                   for s_ in support.values()),
                  "T_disagree": max(s_["disagree_p95"]
                                    for s_ in support.values()),
                  "recipe": "max-family-p95 over the 8 development "
                            "families; frozen on write"}
    retention = {}
    for fam in fams:
        s_ = support[fam]
        chk = chart_support_check(s_["support"], s_["disagree"],
                                  thresholds, head_times=s_["times"])
        retention[fam] = {"share_supported": chk["share_supported"],
                          "usable": chk["usable"],
                          "long_unsupported_run":
                              chk["long_unsupported_run"]}
    gate3 = all(r["usable"] for r in retention.values())
    report["thresholds"] = thresholds
    report["retention"] = retention
    report["support_p95"] = {f: s_["support_p95"]
                             for f, s_ in support.items()}
    report["disagree_p95"] = {f: {"prior_predictive": s_["disagree_p95"],
                                  "local_base": s_["disagree_p95_base"]}
                              for f, s_ in support.items()}
    gates_ok = gate1 and factor is not None and gate3
    report["gates"] = {"1_nll": gate1, "2_factor": factor,
                       "3_support_retention": gate3,
                       "4_adjudication": "PENDING_ASTRA"}
    report["status"] = "DEVELOPMENT_GATES_1_3_PASSED" if gates_ok \
        else "DEVELOPMENT_FAILED"
    _write(report)
    return report


def _train_contexts(ack):
    """Standardized contexts of the 24 training charts (for the constant
    prior); recomputed deterministically from canonical charts."""
    from qa.chart_context import standardize
    design = json.loads((OUT_DIR / "design_manifest.json").read_text())
    out = {}
    for fam in design["train_families"]:
        z, _cl = standardize(_dev_context(fam).tensor(),
                             ack["context_scaler"])
        out[fam] = z
    return out


def _write(report):
    tmp = DEV_REPORT_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(DEV_REPORT_P)
    print(f"{report['status']} -> {DEV_REPORT_P}")


def freeze_candidate():
    """Only after gates 1-3 AND the recorded adjudication gate pass."""
    from qa.parity_recheck import _sha_file
    if CANDIDATE_P.exists():
        return json.loads(CANDIDATE_P.read_text())
    rep = json.loads(DEV_REPORT_P.read_text())
    if rep["status"] != "DEVELOPMENT_GATES_1_3_PASSED":
        raise RuntimeError(f"development status {rep['status']}")
    if not ADJUDICATION_P.exists():
        raise RuntimeError("adjudication gate has no recorded result — "
                           "gate 4 must pass before the candidate freeze")
    adj = json.loads(ADJUDICATION_P.read_text())
    if adj.get("status") != "PASSED":
        raise RuntimeError(f"adjudication gate: {adj.get('status')}")
    from qa.expand import CKPT_P
    man = {"selected_snapshot": rep["selected_snapshot"],
           "inflation_factor": rep["selected_factor"],
           "thresholds": rep["thresholds"],
           "adapter_sha256": _sha_file(OUT_DIR / "adapter.pt"),
           "base_sha256": _sha_file(CKPT_P),
           "design_sha256": _sha_file(OUT_DIR / "design_manifest.json"),
           "dev_report_sha256": _sha_file(DEV_REPORT_P),
           "adjudication_sha256": _sha_file(ADJUDICATION_P),
           "code_sha256": {rel: _sha_file(ROOT / rel) for rel in
                           ("qa/chart_latent.py", "qa/chart_context.py",
                            "qa/chart_latent_run.py",
                            "qa/chart_latent_train.py", "qa/model.py",
                            "qa/features.py", "qa/latent_validation.py")}}
    tmp = CANDIDATE_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    tmp.replace(CANDIDATE_P)
    print(f"candidate frozen: {CANDIDATE_P}")
    return man


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    {"dev": evaluate_development, "freeze": freeze_candidate}[cmd]()
