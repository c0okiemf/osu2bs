"""Q3 planner training: experiment-only destinations, fixed budget (spec §6).

Model: two-layer temporal convolution, width 64, over the window sequence
(receptive field 5 windows >= the 32-beat context). Inputs per window: the 13
audio/MI/tempo features + the requested workload vector (training: the
family's own measured target means — conditional generation) + the previous
window's plan (teacher-forced targets; fit 2 would schedule sampled ones) + a
4-dim neutral style vector (zeros). Outputs: bounded scalar heads (Huber,
family-normalized weighting) + quiet/accent probabilities (BCE).

Budget: one seed 20260921, Adam 1e-3, <=200 epochs x 30 minibatches, batch 32
sequences; val windows span start/middle/end. Normalization fit on TRAIN only.
Checkpoints + history under experiments/quality-v1/q3-planner/ — never any
shipped file.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
WROOT = ROOT / "experiments" / "quality-v1" / "q3-planner"
SEED = 20260921
N_SCALAR, N_BIN = 10, 2
STYLE_DIM = 4
CROP = 16                      # training sequence length (windows)


class Planner(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.c1 = nn.Conv1d(n_in, 64, 3, padding=2)   # causal via left-pad
        self.c2 = nn.Conv1d(64, 64, 3, padding=2)
        self.scalar = nn.Linear(64, N_SCALAR)
        self.binary = nn.Linear(64, N_BIN)

    def forward(self, x):                              # x: [B, W, F]
        h = x.transpose(1, 2)
        h = F.relu(self.c1(h)[..., :x.shape[1]])       # causal crop
        h = F.relu(self.c2(h)[..., :x.shape[1]])
        h = h.transpose(1, 2)
        return torch.sigmoid(self.scalar(h)) * 2.0, self.binary(h)


def load_windows():
    fams = []
    for p in sorted((WROOT / "windows").glob("*.pt")):
        d = torch.load(p)
        fams.append(d)
    tr = [d for d in fams if d["meta"]["split"] == "train"]
    va = [d for d in fams if d["meta"]["split"] == "val"]
    return tr, va


def _sequence(d, norm=None):
    """Assemble the full input sequence for one family: features + workload
    conditioning (family target means) + teacher-forced previous plan + zero
    style."""
    x, y = d["x"], d["y"]
    workload = y[:, :N_SCALAR].mean(dim=0, keepdim=True).expand(len(x), -1)
    prev = torch.cat([torch.zeros(1, y.shape[1]), y[:-1]], dim=0)
    style = torch.zeros(len(x), STYLE_DIM)
    feats = torch.cat([x, workload, prev, style], dim=1)
    if norm is not None:
        mu, sd = norm
        feats = (feats - mu) / sd
    return feats, y


def fit_norm(train):
    cat = torch.cat([_sequence(d)[0] for d in train])
    return cat.mean(dim=0), cat.std(dim=0).clamp(min=1e-6)


def losses(model, feats, y):
    sc, bl = model(feats)
    huber = F.smooth_l1_loss(sc, y[..., :N_SCALAR])
    bce = F.binary_cross_entropy_with_logits(bl, y[..., N_SCALAR:])
    return huber + bce, huber, bce


def constant_plan_error(train, val):
    """Control: the per-tier constant plan (train target means) vs val."""
    mean = torch.cat([d["y"][:, :N_SCALAR] for d in train]).mean(dim=0)
    errs = [F.smooth_l1_loss(mean.expand(len(d["y"]), -1),
                             d["y"][:, :N_SCALAR]).item() for d in val]
    return sum(errs) / len(errs)


N_FEAT = 13                    # audio/MI/tempo features before the workload
PREV_LO, PREV_HI = N_FEAT + N_SCALAR, N_FEAT + N_SCALAR + N_SCALAR + N_BIN


def rollout_val_error(model, norm, va):
    """Autoregressive val scalar error: previous-plan inputs are the model's
    OWN predictions (training conditioning: the family's target means). The
    fit-1 vs fit-2 selection metric — no decodes, whole sequences."""
    mu, sd = norm
    model.eval()
    errs = []
    for d in va:
        x, y = d["x"], d["y"]
        workload = y[:, :N_SCALAR].mean(dim=0)
        prev = torch.zeros(N_SCALAR + N_BIN)
        hist, preds = [], []
        for w in range(len(x)):
            f = torch.cat([x[w], workload, prev, torch.zeros(STYLE_DIM)])
            hist.append((f - mu) / sd)
            with torch.no_grad():
                sc, bl = model(torch.stack(hist)[None])
            prev = torch.cat([sc[0, -1], (bl[0, -1] > 0).float()])
            preds.append(sc[0, -1])
        errs.append(float(F.smooth_l1_loss(torch.stack(preds),
                                           y[:, :N_SCALAR])))
    return sum(errs) / len(errs)


def train_planner(max_epochs=200, steps=30, batch=32, fit2=False):
    """fit2=True is the spec's declared branch after a loss-improved/
    rollout-failed fit 1: scheduled sampling of the previous-plan input —
    with probability ramping 0->1 over the first half of training, each
    window's prev input is replaced by the model's own prediction."""
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = load_windows()
    assert tr and va, f"windows missing: train={len(tr)} val={len(va)}"
    norm = fit_norm(tr)
    seqs = [_sequence(d, norm) for d in tr]
    seqs = [(f, y) for f, y in seqs if len(f) > CROP]
    # val: start/middle/end crops per family (never only the first crop)
    vsets = []
    for d in va:
        f, y = _sequence(d, norm)
        if len(f) <= CROP:
            continue
        for off in (0, max(0, (len(f) - CROP) // 2), len(f) - CROP):
            vsets.append((f[off:off + CROP], y[off:off + CROP]))
    xva = torch.stack([f for f, _ in vsets]).to(dev)
    yva = torch.stack([y for _, y in vsets]).to(dev)
    model = Planner(seqs[0][0].shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    hist = {"epochs": [], "const_plan_val_error":
            constant_plan_error(tr, va), "seed": SEED}
    with torch.no_grad():
        model.eval()
        v0 = losses(model, xva, yva)
        hist["initial_val"] = [float(x) for x in v0]
    best, best_state, patience = 1e9, None, 0
    mu_p = norm[0][PREV_LO:PREV_HI].to(dev)
    sd_p = norm[1][PREV_LO:PREV_HI].to(dev)
    for ep in range(max_epochs):
        model.train()
        for _ in range(steps):
            xs, ys = [], []
            for f, y in rng.choices(seqs, k=batch):
                i = rng.randrange(len(f) - CROP)
                xs.append(f[i:i + CROP])
                ys.append(y[i:i + CROP])
            xb, yb = torch.stack(xs).to(dev), torch.stack(ys).to(dev)
            if fit2:
                p = min(1.0, ep / (max_epochs / 2))
                with torch.no_grad():
                    for w in range(CROP - 1):
                        sc, bl = model(xb[:, :w + 1])
                        pred = torch.cat([sc[:, -1],
                                          (bl[:, -1] > 0).float()], dim=1)
                        mask = torch.rand(len(xb), device=dev) < p
                        repl = (pred - mu_p) / sd_p
                        xb[mask, w + 1, PREV_LO:PREV_HI] = repl[mask]
            loss, _h, _b = losses(model, xb, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl, vh, vb = losses(model, xva, yva)
        hist["epochs"].append({"epoch": ep, "val": float(vl),
                               "huber": float(vh), "bce": float(vb)})
        if float(vl) < best - 1e-4:
            best, patience = float(vl), 0
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience += 1
        if ep % 10 == 0 or patience > 25:
            print(f"[planner] epoch {ep}: val {float(vl):.4f} "
                  f"(huber {float(vh):.4f} bce {float(vb):.4f})", flush=True)
        if patience > 25:
            break
    hist["best_val"] = best
    tag = "-fit2" if fit2 else ""
    out = WROOT / f"planner{tag}-seed20260921.pt"
    torch.save({"state": best_state, "norm": norm,
                "n_in": seqs[0][0].shape[1], "fit2": fit2}, out)
    (WROOT / f"planner{tag}-history.json").write_text(json.dumps(hist, indent=1))
    # Q3 gate component: >=10% better val scalar error than the constant plan
    model.load_state_dict(best_state)
    model.to(dev).eval()
    with torch.no_grad():
        sc, _ = model(xva)
        model_err = float(F.smooth_l1_loss(sc, yva[..., :N_SCALAR]))
    const_err = hist["const_plan_val_error"]
    verdict = model_err <= 0.9 * const_err
    ro = rollout_val_error(model.cpu(), norm, va)
    print(f"planner{tag}: best val {best:.4f}; scalar err {model_err:.4f} vs "
          f"constant-plan {const_err:.4f} -> "
          f"{'PASS (>=10% better)' if verdict else 'FAIL'}; "
          f"val ROLLOUT err {ro:.4f}")
    return {"best_val": best, "model_err": model_err, "const_err": const_err,
            "beats_constant_by_10pct": verdict, "rollout_val_err": ro}


def compare_rollout():
    """fit-1 vs fit-2 selection: val rollout scalar error for each ckpt."""
    tr, va = load_windows()
    for tag in ("", "-fit2"):
        p = WROOT / f"planner{tag}-seed20260921.pt"
        if not p.exists():
            print(f"planner{tag}: (missing)")
            continue
        d = torch.load(p)
        m = Planner(d["n_in"])
        m.load_state_dict(d["state"])
        print(f"planner{tag}: val rollout err "
              f"{rollout_val_error(m, d['norm'], va):.4f}")


if __name__ == "__main__":
    import sys
    if "fit2" in sys.argv:
        train_planner(fit2=True)
    elif "rollout" in sys.argv:
        compare_rollout()
    else:
        train_planner()
