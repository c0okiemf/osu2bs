"""Independent-QA Task 4 (training): fixed folds, baselines, gates.

Dataset: per-family feature/target tensors assembled from validated
windows_v2 records + independent scene + audio evidence; left-handed
replays use the MIRRORED scene view so features match their canonical
kinematics. Cached per family, keyed by telemetry/model versions.

Fits (budget: nine, ≤2000 updates each, seed 20260924): four sha-ordered
round-robin family folds (train 12, hold out 4, remove held-out players
from training), four unconditional body-profile-only baselines (same
budget), one final 16-family fit at the update count chosen from
{500, 1000, 2000} by fold-macro held-out NLL. Gate: conditional beats
baseline by 0.05*max(|baseline|, 1) on the mean AND improves >=3/4 folds
— failure stops before any unseal, with no architecture/seed sweep.
"""
import hashlib
import json
import random
from pathlib import Path

import torch

from qa.model import (FEATURE_NAMES, MotionMixture, mixture_nll,
                      target_from_window)
from qa.scene import _mirror_note
from qa.telemetry_v2 import TELEMETRY_VERSION

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "qa-v1" / "model-data"
FITS = ROOT / "experiments" / "qa-v1" / "model-fits"
SEED = 20260924
UPDATE_GRID = (500, 1000, 2000)
BATCH = 256
LR = 1e-3
GRAD_CLIP = 5.0
DATA_VERSION = 1
BODY_FEATS = [FEATURE_NAMES.index(n) for n in
              ("body_height", "body_height_known", "body_left_handed")]


def mirror_scene(scene):
    out = dict(scene)
    out["notes"] = [_mirror_note(n) for n in scene["notes"]]
    return out


def assemble_role(role):
    """Per-family tensors {x, y, mask, player} with atomic caching.
    Features come from the SHARED provider (qa.features) — the same path
    calibration/diagnostics use, so no train/serve skew can reopen."""
    from qa.features import (family_evidence, features_for,
                             records_for_role, role_families)
    DATA.mkdir(parents=True, exist_ok=True)
    purpose = "fit" if role == "qa_train" else "calibrate"
    fams, todo = {}, []
    for fam in role_families(role):
        key = hashlib.sha256(
            f"{fam}:{TELEMETRY_VERSION}:{DATA_VERSION}".encode()
        ).hexdigest()[:16]
        cp = DATA / role / f"{key}.pt"
        if cp.exists():
            fams[fam] = torch.load(cp)
        else:
            todo.append((fam, cp))
    for fam, cp in todo:
        ev = family_evidence(fam)
        xs, ys, masks, players = [], [], [], []
        for r in records_for_role(role, purpose, families={fam}):
            y, mask = target_from_window(r["window"])
            xs.append(features_for(r, ev))
            ys.append(y)
            masks.append(mask)
            players.append(r["player_token"])
        if not xs:
            continue
        pack = {"x": torch.stack(xs), "y": torch.stack(ys),
                "mask": torch.stack(masks), "players": players,
                "family": fam}
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        torch.save(pack, tmp)
        tmp.replace(cp)
        fams[fam] = pack
        print(f"  [{role}] {fam}: {len(xs)} windows", flush=True)
    return fams


def folds_of(families):
    """Four predetermined sha-ordered round-robin folds."""
    order = sorted(families, key=lambda f: hashlib.sha256(
        f"{SEED}:{f}".encode()).hexdigest())
    folds = [[] for _ in range(4)]
    for i, f in enumerate(order):
        folds[i % 4].append(f)
    return folds


def _standardize(xs):
    mu = xs.mean(dim=0)
    sd = xs.std(dim=0).clamp(min=1e-6)
    return mu, sd


def _sample_stream(fams, rng):
    """family -> player -> window uniform sampling."""
    fam_list = sorted(fams)
    by_player = {f: {} for f in fam_list}
    for f in fam_list:
        for i, p in enumerate(fams[f]["players"]):
            by_player[f].setdefault(p, []).append(i)
    while True:
        f = fam_list[rng.randrange(len(fam_list))]
        ps = sorted(by_player[f])
        p = ps[rng.randrange(len(ps))]
        idxs = by_player[f][p]
        yield f, idxs[rng.randrange(len(idxs))]


def _macro_nll(model, fams, scaler, body_only=False):
    model.eval()
    out = {}
    with torch.no_grad():
        for f, pack in fams.items():
            x = (pack["x"] - scaler[0]) / scaler[1]
            if body_only:
                keep = torch.zeros_like(x)
                keep[:, BODY_FEATS] = x[:, BODY_FEATS]
                x = keep
            w, mu, ls = model(x)
            out[f] = float(mixture_nll(w, mu, ls, pack["y"], pack["mask"]))
    return out


def _fit(fams_train, updates, rng_seed, body_only=False, removed_players=(),
         stop_after=None, state_path=None, save_state_every=None):
    """stop_after/state_path/save_state_every exist ONLY for the budgeted
    historical fold recovery (parity-repair spec §4): early stop at the
    already-selected snapshot and resumable state every N updates. The
    update math is unchanged; defaults reproduce the original behavior."""
    torch.manual_seed(rng_seed)
    rng = random.Random(rng_seed)
    use = {}
    for f, pack in fams_train.items():
        keep = [i for i, p in enumerate(pack["players"])
                if p not in removed_players]
        if not keep:
            continue
        use[f] = {"x": pack["x"][keep], "y": pack["y"][keep],
                  "mask": pack["mask"][keep],
                  "players": [pack["players"][i] for i in keep]}
    if len(use) < 8:
        raise RuntimeError(f"split-support failure: only {len(use)} "
                           "training families survive")
    allx = torch.cat([p["x"] for p in use.values()])
    scaler = _standardize(allx)
    model = MotionMixture()
    opt = torch.optim.Adam(model.parameters(), LR)
    stream = _sample_stream(use, rng)
    snap = {}
    start_u = 1
    if state_path is not None and Path(state_path).exists():
        st = torch.load(state_path)
        if st["seed"] != rng_seed:
            raise ValueError("resume state seed mismatch")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        snap = st["snap"]
        start_u = st["update"] + 1
    for u in range(start_u, max(UPDATE_GRID) + 1):
        idx = [next(stream) for _ in range(BATCH)]
        xb = torch.stack([use[f]["x"][i] for f, i in idx])
        yb = torch.stack([use[f]["y"][i] for f, i in idx])
        mb = torch.stack([use[f]["mask"][i] for f, i in idx])
        xb = (xb - scaler[0]) / scaler[1]
        if body_only:
            keep = torch.zeros_like(xb)
            keep[:, BODY_FEATS] = xb[:, BODY_FEATS]
            xb = keep
        w, mu, ls = model(xb)
        loss = mixture_nll(w, mu, ls, yb, mb)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at update {u}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        if u in UPDATE_GRID:
            snap[u] = {k: v.clone() for k, v in model.state_dict().items()}
        if save_state_every and state_path is not None \
                and u % save_state_every == 0:
            tmp = Path(state_path).with_suffix(".tmp")
            torch.save({"seed": rng_seed, "update": u,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "py_rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "snap": snap}, tmp)
            tmp.replace(state_path)
        if stop_after is not None and u >= stop_after:
            break
        if u >= updates and updates not in UPDATE_GRID:
            break
    return model, scaler, snap


def run_folds(role="qa_train"):
    """The nine budgeted fits + gates. Deterministic; results persisted."""
    FITS.mkdir(parents=True, exist_ok=True)
    fams = assemble_role(role)
    folds = folds_of(fams)
    report = {"seed": SEED, "folds": [sorted(f) for f in folds],
              "per_update": {}, "baseline": {}}
    for updates in UPDATE_GRID:
        report["per_update"][str(updates)] = []
    for k, held in enumerate(folds):
        train_f = {f: fams[f] for f in fams if f not in held}
        held_f = {f: fams[f] for f in held}
        held_players = {p for f in held for p in fams[f]["players"]}
        _m, scaler, snaps = _fit(train_f, max(UPDATE_GRID), SEED + k,
                                 removed_players=held_players)
        for updates, state in snaps.items():
            m = MotionMixture()
            m.load_state_dict(state)
            macro = _macro_nll(m, held_f, scaler)
            report["per_update"][str(updates)].append(
                sum(macro.values()) / len(macro))
        bm, bscaler, bsnaps = _fit(train_f, max(UPDATE_GRID),
                                   SEED + 100 + k, body_only=True,
                                   removed_players=held_players)
        bmacro = _macro_nll(bm, held_f, bscaler, body_only=True)
        report["baseline"].setdefault("folds", []).append(
            sum(bmacro.values()) / len(bmacro))
        print(f"[fold {k}] held {sorted(held)} baseline "
              f"{report['baseline']['folds'][-1]:.4f}", flush=True)
    chosen = min(UPDATE_GRID,
                 key=lambda u: sum(report["per_update"][str(u)]) / 4)
    report["chosen_updates"] = chosen
    model_means = report["per_update"][str(chosen)]
    base_means = report["baseline"]["folds"]
    mean_m = sum(model_means) / 4
    mean_b = sum(base_means) / 4
    margin = 0.05 * max(abs(mean_b), 1.0)
    improved = sum(1 for a, b in zip(model_means, base_means) if a < b)
    gate = (mean_b - mean_m >= margin) and improved >= 3
    report["gate"] = {"model_mean": mean_m, "baseline_mean": mean_b,
                      "margin": margin, "folds_improved": improved,
                      "pass": gate}
    if gate:
        final, scaler, snaps = _fit(fams, chosen, SEED + 999)
        fstate = snaps.get(chosen) or final.state_dict()
        torch.save({"state": fstate, "scaler": scaler,
                    "updates": chosen, "seed": SEED,
                    "feature_names": list(FEATURE_NAMES),
                    "telemetry_version": TELEMETRY_VERSION},
                   FITS / "motion_mixture_final.pt")
    (FITS / "fold_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True))
    print(f"updates={chosen}; model {mean_m:.4f} vs baseline {mean_b:.4f} "
          f"(margin {margin:.4f}, {improved}/4 folds) -> "
          f"{'GATE PASS' if gate else 'GATE FAIL — stop before unseal'}")
    return report


if __name__ == "__main__":
    run_folds()
