"""Chart-latent Task 1 (review spec 2026-09-24 §2): frozen variant
identity + metadata-only fresh-validation reservation.

- freeze_design pins the immutable baseline (24-family local base ckpt,
  scaler, NN bank identity, ORIGINAL spans/std, the 8 development
  families' sample hashes, six-fold hash assignment, all budgets/gates)
  BEFORE any fit.
- reserve_validation reserves exactly four fresh ExpertPlus Standard
  families by SHA256("qa-chart-latent-validation-v1:<family>") over the
  bounded eligibility pool, ≤60 live metadata probes, chart-side scope
  checks only — NO kinematics, replay payloads or target inspection.
  Shortfall = INVENTORY_INCOMPLETE, stop; no substitution afterwards.
- fetch_validation downloads the ≤24 reserved replay slots (2 players
  per performance third, ≤6/family) ONLY under a committed candidate
  freeze; role qa_validate_latent permits one frozen validation job and
  never fit/reference/calibration. Seal unread; E1 held; B0 active.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "qa-v3" / "chart-latent"
DESIGN_P = OUT_DIR / "design_manifest.json"
RESERVE_P = OUT_DIR / "reservation.json"
SALT = "qa-chart-latent-validation-v1"
FOLD_SALT = "qa-chart-latent-fold-v1"
N_VALIDATE = 4
MAX_PROBES = 60
MAX_PAYLOADS_PER_FAMILY = 6
MAX_PAYLOADS = 24
N_STATES = 3
CONTEXT_DIM = 21
ADAPTER_UPDATES = 2000
FOLD_UPDATES = 500
FOLD_SEED_BASE = 20260924 + 4000


def _h(salt, s):
    return hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def latent_folds(train_fams):
    """Six predetermined folds of four, sorted fold-salt round-robin."""
    order = sorted(train_fams, key=lambda f: _h(FOLD_SALT, f))
    folds = [[] for _ in range(6)]
    for i, f in enumerate(order):
        folds[i % 6].append(f)
    return [sorted(f) for f in folds]


def freeze_design():
    """Idempotent; everything pinned BEFORE any fit."""
    if DESIGN_P.exists():
        return json.loads(DESIGN_P.read_text())
    from qa.expand import CKPT_P, EVAL_FREEZE_P, SEL_P
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eman = json.loads(EVAL_FREEZE_P.read_text())
    ck = torch.load(CKPT_P)
    train_fams = sorted(ck["families"])
    if len(train_fams) != 24:
        raise RuntimeError(f"expected 24 train families, got "
                           f"{len(train_fams)}")
    man = {"ruling": "review (spec 2026-09-24 chart-latent)",
           "local_base_sha256": _sha_file(CKPT_P),
           "eval_freeze_sha256": _sha_file(EVAL_FREEZE_P),
           "selection_sha256": _sha_file(SEL_P),
           "train_families": train_fams,
           "development_families": {f: sm["sha256"]
                                    for f, sm in eman["samples"].items()},
           "frozen_spans_original16": eman["frozen_spans_original16"],
           "frozen_t_std_original16": eman["frozen_t_std_original16"],
           "bank_identity": "qa-train-v2",
           "folds": latent_folds(train_fams),
           "config": {"states": N_STATES, "context_dim": CONTEXT_DIM,
                      "adapter_updates": ADAPTER_UPDATES,
                      "adapter_seed": 20260924,
                      "fold_updates": FOLD_UPDATES,
                      "fold_seed_base": FOLD_SEED_BASE,
                      "lr_prior": 0.01, "lr_state": 0.001,
                      "grad_clip": 5.0,
                      "reg": "1e-3*mean(W^2)+1e-3*mean((delta/s)^2+a^2)",
                      "validation_salt": SALT,
                      "max_probes": MAX_PROBES,
                      "n_validate": N_VALIDATE,
                      "max_payloads": MAX_PAYLOADS,
                      "coverage_min": 0.85, "sharpness_max_wide": 0.20,
                      "inflation_grid": [1.0, 1.25, 1.5, 2.0],
                      "nll_context_lift_min": 0.01}}
    tmp = DESIGN_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    tmp.replace(DESIGN_P)
    print(f"design frozen: {DESIGN_P}")
    return man


def _used_families():
    """Families in ANY existing role/state (train/calib/gen/seal/pilot +
    expansion exclusions) — the reservation must be disjoint from all."""
    state = json.loads((ROOT / "experiments/qa-v1/collect2_state.json")
                       .read_text())
    pilot = json.loads((ROOT / "experiments/qa-v1/pilot_state.json")
                       .read_text())
    used = set(state["charts"]) | set(pilot["charts"])
    sel_p = ROOT / "experiments/qa-v2/expansion/selection.json"
    if sel_p.exists():
        sel = json.loads(sel_p.read_text())
        used |= {a["excluded"] for a in sel.get("amendments", [])}
    return used


def default_pool():
    from qa.expand import candidates
    used = _used_families()
    return [(f, d) for f, d in candidates() if f not in used]


def _probe_live(fam, map_dir):
    """ExpertPlus-Standard-only leaderboard metadata probe (counts only;
    no easiest-difficulty fallback, no payloads)."""
    from qa.collect2 import MIN_SCORES
    from qa.replays import API, _get, level_hash
    try:
        h = level_hash(map_dir)
        lb = _get(f"{API}/leaderboards/hash/{h}")
        l = next((x for x in (lb.get("leaderboards") or [])
                  if x["difficulty"]["modeName"] == "Standard"
                  and x["difficulty"]["difficultyName"] == "ExpertPlus"),
                 None)
        if l is None:
            return {"family": fam, "usable": False,
                    "reason": "no_expertplus_standard_lb"}
        d = _get(f"{API}/leaderboard/{l['id']}?page=1&count=50")
        n_ok = sum(1 for s in (d.get("scores") or [])
                   if not s.get("modifiers"))
        if n_ok < MIN_SCORES:
            return {"family": fam, "usable": False,
                    "reason": f"thin_leaderboard:{n_ok}"}
        return {"family": fam, "dir": str(map_dir), "hash": h,
                "leaderboard": l["id"], "difficulty": "ExpertPlus",
                "n_modifier_free": n_ok, "usable": True}
    except Exception as e:
        return {"family": fam, "usable": False,
                "reason": f"probe_error:{type(e).__name__}"}


def _local_ok(fam, map_dir):
    """Chart-side eligibility only: supported ExpertPlus Standard scene
    (scope None — includes the modchart heuristic). Never kinematics."""
    from qa.expand import _local_chart
    from qa.scene import read_scene
    dat_p, info_p = _local_chart(map_dir, "ExpertPlus")
    if dat_p is None or not dat_p.exists():
        return "no_local_expertplus"
    scope = read_scene(dat_p, info_p)["scope"]
    return f"scope:{scope}" if scope is not None else None


def reserve_validation(design=None, pool=None, prober=None):
    """Metadata-only reservation of exactly four families, frozen."""
    if RESERVE_P.exists():
        r = json.loads(RESERVE_P.read_text())
        if r["status"] != "RESERVED":
            raise RuntimeError(f"reservation status {r['status']}")
        return r
    design = design or freeze_design()
    pool = pool if pool is not None else default_pool()
    prober = prober or _probe_live
    ranked = sorted(pool, key=lambda fd: _h(SALT, fd[0]))
    used = _used_families()
    picked, census, probes = [], [], 0
    for fam, map_dir in ranked:
        if len(picked) >= N_VALIDATE:
            break
        if fam in used:
            census.append({"family": fam, "usable": False,
                           "reason": "role_overlap"})
            continue
        local = _local_ok(fam, map_dir)
        if local is not None:
            census.append({"family": fam, "usable": False,
                           "reason": local})
            continue
        if probes >= MAX_PROBES:
            break
        probes += 1
        r = prober(fam, map_dir)
        census.append({k: v for k, v in r.items() if k != "dir"})
        if r.get("usable"):
            picked.append(r)
    status = "RESERVED" if len(picked) == N_VALIDATE \
        else "INVENTORY_INCOMPLETE"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"status": status, "salt": SALT, "families": picked,
           "census": census, "n_probes": probes,
           "player_rule": "two per performance third, <=6 players, one "
                          "replay each, no accuracy floor",
           "caps": {"per_family": MAX_PAYLOADS_PER_FAMILY,
                    "total": MAX_PAYLOADS},
           "role": "qa_validate_latent",
           "note": "metadata-only; payloads/derivation require the "
                   "committed candidate freeze; no substitution after "
                   "reservation"}
    tmp = RESERVE_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True))
    tmp.replace(RESERVE_P)
    if status != "RESERVED":
        raise RuntimeError(f"INVENTORY_INCOMPLETE: {len(picked)}/"
                           f"{N_VALIDATE} usable in {probes} probes")
    print(f"reserved: {[r['family'] for r in picked]} ({probes} probes)")
    return out


def fetch_validation(reservation=None, candidate_freeze=None):
    """Download the reserved replay slots — only under a committed
    candidate freeze (spec §7). Resumable per family; no substitutions;
    unsupported outcomes stay explicit."""
    if candidate_freeze is None:
        raise PermissionError("fresh validation payloads require the "
                              "committed candidate freeze")
    cf = Path(candidate_freeze)
    if not cf.exists():
        raise PermissionError(f"candidate freeze not found: {cf}")
    reservation = reservation or json.loads(RESERVE_P.read_text())
    if reservation["status"] != "RESERVED":
        raise RuntimeError(f"reservation status {reservation['status']}")
    from qa.collect2 import ROOTS, STATE_P, _fetch_family
    from qa.expand import thirds_picks
    from qa.replays import _save, _secret_key
    state = json.loads(STATE_P.read_text())
    key = _secret_key()
    fams = {r["family"] for r in reservation["families"]}
    for entry in reservation["families"]:
        fam = entry["family"]
        if fam in state["charts"]:
            continue
        got = sum(len(state["charts"][f].get("replays", []))
                  for f in fams if f in state["charts"])
        room = MAX_PAYLOADS - got
        if room <= 0:
            break
        ROOTS["qa_validate_latent"].mkdir(parents=True, exist_ok=True)
        rec = _fetch_family(entry, "qa_validate_latent", state, key,
                            picks_fn=thirds_picks,
                            max_picks=MAX_PAYLOADS_PER_FAMILY,
                            max_replays=min(MAX_PAYLOADS_PER_FAMILY,
                                            room))
        state["charts"][fam] = rec
        _save(STATE_P, state)
        print(f"  [validate] {fam:<12} {rec['status']:<5} replays "
              f"{len(rec['replays'])}", flush=True)
    got = sum(len(state["charts"][f].get("replays", []))
              for f in fams if f in state["charts"])
    report = {"payloads": got, "cap": MAX_PAYLOADS,
              "families": sorted(fams & set(state["charts"]))}
    print(f"validation collection: {report}")
    return report


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "freeze"
    if cmd == "freeze":
        freeze_design()
    elif cmd == "reserve":
        reserve_validation()
    else:
        raise SystemExit(f"unknown command {cmd!r} (fetch happens via "
                         "qa.chart_latent_run after candidate freeze)")
