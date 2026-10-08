"""Chart-latent Tasks 3-4 (spec §5): cross-fitted local predictions and
the single adapter fit.

Task 3: six predetermined fold fits of the UNCHANGED local MotionMixture
(fresh init, 500 updates, seed 20260924+4000+k, held-out players removed
BEFORE scaler fitting — qa.train._fit already orders it that way), each
held-out family's windows scored ONLY by its own fold model. These OOF
parameters are adapter TRAINING inputs, never adapter validation. The
24-family base, its scaler and the NN bank stay untouched.

Task 4 (training driver): one adapter fit on chart bags (4 distinct
families x 3 distinct players x 8 windows >=0.8s apart), exact shared-
latent bag likelihood (prior paid once per bag), Adam group LRs
0.01/0.001, clip 5, seed 20260924, max 2000 updates, snapshots
0/500/1000/2000, resumable every 100 updates. No base gradients, no
sweeps, no extra states.
"""
import json
import random
from pathlib import Path

import torch

from qa.latent_validation import (DESIGN_P, FOLD_SEED_BASE, FOLD_UPDATES,
                                  OUT_DIR, latent_folds)
from qa.model import MotionMixture

FOLD_DIR = OUT_DIR / "folds"
OOF_DIR = OUT_DIR / "oof"
ROOT = Path(__file__).resolve().parent.parent
MIN_TRAIN_FAMS = 12
MIN_PLAYERS = 3
MIN_WINDOWS = 100


def make_local_folds(train_fams):
    folds = latent_folds(train_fams)
    return [{"fold": k, "held": folds[k], "seed": FOLD_SEED_BASE + k,
             "updates": FOLD_UPDATES} for k in range(len(folds))]


def fold_partition(packs, held, removed_players):
    """Kept row indices per training family AFTER held-player removal;
    the scaler may only ever see these rows."""
    kept = {}
    for f, pack in packs.items():
        if f in held:
            continue
        idx = [i for i, p in enumerate(pack["players"])
               if p not in removed_players]
        if idx:
            kept[f] = idx
    return kept


def fit_local_fold(fold, packs=None):
    """Fresh-init bounded fold fit with full identity; idempotent."""
    from qa.train import _fit
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    p = FOLD_DIR / f"fold{fold['fold']}.pt"
    if p.exists():
        art = torch.load(p)
        if art["seed"] != fold["seed"] or art["updates"] != \
                fold["updates"] or sorted(art["held"]) != \
                sorted(fold["held"]):
            raise ValueError(f"fold artifact identity mismatch: {p.name}")
        return art
    if packs is None:
        from qa.train import assemble_role
        packs = assemble_role("qa_train")
    held_players = {pl for f in fold["held"]
                    for pl in packs[f]["players"]}
    for f in fold["held"]:
        players = set(packs[f]["players"])
        if len(players) < MIN_PLAYERS or len(packs[f]["players"]) \
                < MIN_WINDOWS:
            raise RuntimeError(f"DATA_SUPPORT_INSUFFICIENT: held {f}")
    train_f = {f: packs[f] for f in packs if f not in fold["held"]}
    kept = fold_partition(packs, fold["held"], held_players)
    if len(kept) < MIN_TRAIN_FAMS:
        raise RuntimeError(f"DATA_SUPPORT_INSUFFICIENT: only {len(kept)} "
                           "training families after exclusions")
    _m, scaler, snaps = _fit(train_f, fold["updates"], fold["seed"],
                             removed_players=held_players,
                             stop_after=fold["updates"],
                             state_path=FOLD_DIR
                             / f"fold{fold['fold']}.resume.pt",
                             save_state_every=100)
    art = {"kind": "latent-fold", "fold": fold["fold"],
           "held": sorted(fold["held"]), "seed": fold["seed"],
           "updates": fold["updates"], "init": "fresh",
           "state": snaps[fold["updates"]], "scaler": scaler,
           "removed_players": sorted(held_players),
           "train_families": sorted(train_f) }
    tmp = p.with_suffix(".tmp")
    torch.save(art, tmp)
    tmp.replace(p)
    resume = FOLD_DIR / f"fold{fold['fold']}.resume.pt"
    if resume.exists():
        resume.unlink()
    print(f"[latent fold {fold['fold']}] fitted (held {art['held']})",
          flush=True)
    return art


def family_window_meta(fam, role="qa_train"):
    """window ids + event times in provider (== pack row) order."""
    from qa.features import records_for_role
    ids, times, players = [], [], []
    purpose = {"qa_train": "fit", "qa_calib": "calibrate",
               "qa_calib2": "calibrate",
               "qa_validate_latent": "validate"}[role]
    for r in records_for_role(role, purpose, families={fam}):
        ids.append(r["window_id"])
        times.append(float(r["window"].get("event_time", 0.0)))
        players.append(r["player_token"])
    return ids, times, players


def cache_oof_predictions(artifact, fam, pack):
    """Frozen fold-model parameters for ONE held-out family; refuses any
    artifact whose partition did not hold this family out."""
    if artifact.get("kind") != "latent-fold" \
            or fam not in artifact.get("held", ()):
        raise ValueError(f"partition mismatch: artifact does not hold out "
                         f"{fam}")
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    out_p = OOF_DIR / (fam.replace(":", "_") + ".pt")
    if out_p.exists():
        return torch.load(out_p)
    m = MotionMixture()
    m.load_state_dict(artifact["state"])
    m.eval()
    mu_s, sd_s = artifact["scaler"]
    with torch.no_grad():
        lw, mu, ls = m((pack["x"] - mu_s) / sd_s)
    ids, times, players = family_window_meta(fam)
    if len(ids) != len(pack["x"]) or players != pack["players"]:
        raise RuntimeError(f"{fam}: provider order no longer matches the "
                           "cached pack — provenance broken, stop")
    store = {"family": fam, "fold": artifact["fold"],
             "window_ids": ids, "event_times": times,
             "players": pack["players"],
             "lw": lw, "mu": mu, "ls": ls,
             "y": pack["y"], "mask": pack["mask"]}
    tmp = out_p.with_suffix(".tmp")
    torch.save(store, tmp)
    tmp.replace(out_p)
    return store


def build_oof(packs=None):
    """All six folds + 24 OOF family stores (resumable per item)."""
    from qa.train import assemble_role
    design = json.loads(DESIGN_P.read_text())
    packs = packs or assemble_role("qa_train")
    if sorted(packs) != design["train_families"]:
        raise RuntimeError("train families drifted from the frozen design")
    for fold in make_local_folds(design["train_families"]):
        art = fit_local_fold(fold, packs)
        for fam in fold["held"]:
            cache_oof_predictions(art, fam, packs[fam])
            print(f"  oof cached: {fam} (fold {fold['fold']})",
                  flush=True)
    return sorted(f.name for f in OOF_DIR.glob("*.pt"))


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "oof"
    if cmd == "oof":
        build_oof()
    elif cmd == "adapter":
        from qa.chart_latent import train_adapter
        train_adapter()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
