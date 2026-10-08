"""Chart-latent Task 4 (spec §4-§5): exact three-state shared chart
latent over the frozen local mixture.

- prior pi(c) = softmax(W c + b), W 3x21, on the standardized 21-field
  chart context; no hidden layer, no body/target input.
- state location delta_zj = 2 * s_j * tanh(u_zj); state log-scale
  multiplier a_zj = log(2) * tanh(v_zj); s_j = the ORIGINAL frozen
  observed-component target std (design manifest), never refit.
- conditional p(y|x,z) shifts every local component mean by delta_z and
  multiplies scales by exp(a_z); serving marginal integrates the PRIOR
  exactly: 12 Gaussian components with weights pi_z(c) * w_ik. No
  posterior over human targets exists at serving.
- a simulated chart draws z ONCE for all notes (per-note redraws destroy
  the shared latent even though one-note marginals look identical).
- bag likelihood pays the prior ONCE per chart bag:
  L_B = -logsumexp_z[log pi_z(c) + sum_i log p(y_i,obs|x_i,z)] / n_obs.
- init: W,b=0; u = atanh(offset/2) for offsets (-0.25, 0, +0.25) on the
  first seven pinned targets (0 elsewhere); a = 0. States are z0/z1/z2,
  numerical capacity — never mapper personalities; usage is reported,
  not forced. Regularizer 1e-3*mean(W^2) + 1e-3*mean((delta/s)^2 + a^2).
"""
import json
import math
import random
from pathlib import Path

import torch
from torch import nn

from qa.model import N_TARGETS, mixture_quantiles

N_STATES = 3
CONTEXT_DIM = 21
FIRST_SEVEN = 7
INIT_OFFSETS = (-0.25, 0.0, 0.25)
ADAPTER_SEED = 20260924
ADAPTER_UPDATES = 2000
SNAPSHOTS = (0, 500, 1000, 2000)
BAG_FAMILIES = 4
BAG_PLAYERS = 3
BAG_WINDOWS = 8
BAG_MIN_SEP_S = 0.8
LR_PRIOR = 0.01
LR_STATE = 0.001
GRAD_CLIP = 5.0
REG = 1e-3


class ChartLatentAdapter(nn.Module):
    def __init__(self, s, context_dim=CONTEXT_DIM, states=N_STATES,
                 targets=N_TARGETS):
        super().__init__()
        self.register_buffer("s", s.clone().clamp(min=1e-6))
        self.W = nn.Parameter(torch.zeros(states, context_dim))
        self.b = nn.Parameter(torch.zeros(states))
        u0 = torch.zeros(states, targets)
        for z, off in enumerate(INIT_OFFSETS):
            u0[z, :FIRST_SEVEN] = math.atanh(off / 2.0)
        self.u = nn.Parameter(u0)
        self.v = nn.Parameter(torch.zeros(states, targets))

    def prior(self, context):
        """log pi(c); context [.., 21] standardized."""
        return torch.log_softmax(context @ self.W.T + self.b, dim=-1)

    def delta(self):
        return 2.0 * self.s[None, :] * torch.tanh(self.u)   # [Z,T]

    def a(self):
        return math.log(2.0) * torch.tanh(self.v)           # [Z,T]

    def marginal(self, lw, mu, ls, context):
        """12-component serving marginal (prior integrated exactly).
        lw [N,K], mu/ls [N,K,T], context [21] -> (lw12, mu12, ls12)."""
        log_pi = self.prior(context)                        # [Z]
        d, av = self.delta(), self.a()
        n, k, t = mu.shape
        lw12 = (log_pi[None, :, None] + lw[:, None, :]) \
            .reshape(n, N_STATES * k)
        mu12 = (mu[:, None, :, :] + d[None, :, None, :]) \
            .reshape(n, N_STATES * k, t)
        ls12 = (ls[:, None, :, :] + av[None, :, None, :]) \
            .reshape(n, N_STATES * k, t)
        return lw12, mu12, ls12

    def quantiles(self, lw, mu, ls, context, probs=(0.1, 0.5, 0.9)):
        return mixture_quantiles(*self.marginal(lw, mu, ls, context),
                                 probs=probs)

    def bag_nll(self, lw, mu, ls, y, mask, context):
        """Exact shared-latent bag likelihood; batched over bags.
        lw [B,N,K], mu/ls [B,N,K,T], y/mask [B,N,T], context [B,21]."""
        d, av = self.delta(), self.a()
        mu_z = mu[:, None] + d[None, :, None, None, :]       # [B,Z,N,K,T]
        ls_z = ls[:, None] + av[None, :, None, None, :]
        y_ = y[:, None, :, None, :]
        normal_lp = -0.5 * (((y_ - mu_z) / ls_z.exp()) ** 2) \
            - ls_z - 0.5 * math.log(2 * math.pi)
        m_ = mask[:, None, :, None, :]
        component_lp = torch.where(m_, normal_lp,
                                   torch.zeros_like(normal_lp)).sum(-1)
        note_lp = torch.logsumexp(lw[:, None, :, :] + component_lp,
                                  dim=-1)                    # [B,Z,N]
        log_pi = self.prior(context)                         # [B,Z]
        chart_lp = torch.logsumexp(log_pi + note_lp.sum(-1), dim=-1)
        n_obs = mask.sum((1, 2)).float()
        if bool((n_obs == 0).any()):
            raise ValueError("bag with zero observed target components")
        return (-chart_lp / n_obs).mean()

    def regularizer(self):
        return REG * self.W.pow(2).mean() + REG * (
            (self.delta() / self.s[None, :]).pow(2) + self.a().pow(2)
        ).mean()

    def sample_chart(self, lw, mu, ls, context, rng):
        """ONE latent draw for the whole chart; descriptor samples only,
        never a body trajectory."""
        pi = self.prior(context).exp()
        z = int(torch.multinomial(pi, 1, generator=rng))
        d, av = self.delta(), self.a()
        comp = torch.multinomial(lw.exp(), 1, generator=rng).squeeze(-1)
        idx = torch.arange(len(comp))
        mu_s = mu[idx, comp] + d[z][None, :]
        sd_s = (ls[idx, comp] + av[z][None, :]).exp()
        eps = torch.randn(mu_s.shape, generator=rng)
        return {"latent_state_per_note": [z] * len(comp),
                "samples": mu_s + eps * sd_s,
                "space": "log1p_SI_descriptors"}


def constant_prior(adapter, train_contexts):
    """Arithmetic mean of pi over the 24 TRAINING chart contexts — the
    frozen no-context ablation, not a separately optimized model."""
    with torch.no_grad():
        p = torch.stack([adapter.prior(c).exp()
                         for c in train_contexts]).mean(dim=0)
    return {"p": p / p.sum(), "log_p": (p / p.sum()).log(),
            "source_roles": {"qa_train"},
            "n_contexts": len(train_contexts)}


# ---------------- training driver ----------------

def _bag_windows(store, rng):
    """3 distinct players x 8 windows >=0.8s apart (within player)."""
    by_p = {}
    for i, p in enumerate(store["players"]):
        by_p.setdefault(p, []).append(i)
    players = sorted(by_p)
    if len(players) < BAG_PLAYERS:
        return None
    rng.shuffle(players)
    rows = []
    for p in players:
        if len(rows) >= BAG_PLAYERS * BAG_WINDOWS:
            break
        idxs = by_p[p][:]
        rng.shuffle(idxs)
        picked, times = [], []
        for i in idxs:
            t = store["event_times"][i]
            if all(abs(t - u) >= BAG_MIN_SEP_S for u in times):
                picked.append(i)
                times.append(t)
                if len(picked) == BAG_WINDOWS:
                    break
        if len(picked) < BAG_WINDOWS:
            continue
        rows += picked
    return rows if len(rows) == BAG_PLAYERS * BAG_WINDOWS else None


def load_training_inputs():
    """OOF stores + standardized contexts for the 24 train families."""
    from qa.chart_context import chart_context, fit_context_scaler, \
        standardize
    from qa.chart_latent_train import OOF_DIR
    from qa.features import family_evidence
    from qa.latent_validation import DESIGN_P
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    design = json.loads(DESIGN_P.read_text())
    stores, contexts = {}, {}
    state = json.loads((Path(__file__).resolve().parent.parent
                        / "experiments/qa-v1/collect2_state.json")
                       .read_text())
    manifest = json.loads((Path(__file__).resolve().parent.parent
                           / "eval/corpus_manifest.json").read_text())
    for fam in design["train_families"]:
        stores[fam] = torch.load(OOF_DIR / (fam.replace(":", "_") + ".pt"))
        dat_p, info_p = _family_chart(manifest, fam,
                                      state["charts"][fam]["difficulty"])
        scene = read_scene(dat_p, info_p)          # canonical, unmirrored
        contexts[fam] = chart_context(scene, family_evidence(fam))
    scaler = fit_context_scaler([contexts[f]
                                 for f in design["train_families"]])
    std_ctx, clipped = {}, {}
    for fam, rec in contexts.items():
        z, cl = standardize(rec.tensor(), scaler)
        std_ctx[fam] = z
        if cl:
            clipped[fam] = cl
    return design, stores, std_ctx, scaler, clipped


def train_adapter():
    from qa.chart_latent_train import OUT_DIR
    ck_p = OUT_DIR / "adapter.pt"
    if ck_p.exists():
        print(f"adapter exists: {ck_p}")
        return torch.load(ck_p)
    design, stores, std_ctx, ctx_scaler, clipped = load_training_inputs()
    torch.manual_seed(ADAPTER_SEED)
    rng = random.Random(ADAPTER_SEED)
    s = torch.tensor(design["frozen_t_std_original16"]).clamp(min=1e-6)
    adapter = ChartLatentAdapter(s)
    opt = torch.optim.Adam([
        {"params": [adapter.W, adapter.b], "lr": LR_PRIOR},
        {"params": [adapter.u, adapter.v], "lr": LR_STATE}])
    snaps = {0: {k: v.clone() for k, v in adapter.state_dict().items()}}
    state_p = OUT_DIR / "adapter.resume.pt"
    start_u, insufficient = 1, 0
    if state_p.exists():
        st = torch.load(state_p)
        adapter.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        snaps = st["snaps"]
        start_u = st["update"] + 1
        insufficient = st.get("insufficient", 0)
    fams = design["train_families"]
    for u in range(start_u, ADAPTER_UPDATES + 1):
        bags = []
        while len(bags) < BAG_FAMILIES:
            picks = rng.sample(fams, BAG_FAMILIES - len(bags))
            for fam in picks:
                rows = _bag_windows(stores[fam], rng)
                if rows is None:
                    insufficient += 1
                    continue
                st_ = stores[fam]
                idx = torch.tensor(rows)
                bags.append((st_["lw"][idx], st_["mu"][idx],
                             st_["ls"][idx], st_["y"][idx],
                             st_["mask"][idx], std_ctx[fam]))
        lw = torch.stack([b[0] for b in bags])
        mu = torch.stack([b[1] for b in bags])
        ls = torch.stack([b[2] for b in bags])
        y = torch.stack([b[3] for b in bags])
        mask = torch.stack([b[4] for b in bags])
        ctx = torch.stack([b[5] for b in bags])
        loss = adapter.bag_nll(lw, mu, ls, y, mask, ctx) \
            + adapter.regularizer()
        if not torch.isfinite(loss):
            raise RuntimeError(f"TRAINING_DIVERGED at update {u}")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
        opt.step()
        if u in SNAPSHOTS:
            snaps[u] = {k: v.clone()
                        for k, v in adapter.state_dict().items()}
        if u % 100 == 0:
            tmp = state_p.with_suffix(".tmp")
            torch.save({"update": u, "model": adapter.state_dict(),
                        "opt": opt.state_dict(),
                        "py_rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "snaps": snaps,
                        "insufficient": insufficient}, tmp)
            tmp.replace(state_p)
            print(f"  [adapter] update {u} loss {float(loss):.4f}",
                  flush=True)
    ck = {"snapshots": snaps, "seed": ADAPTER_SEED,
          "updates": ADAPTER_UPDATES,
          "context_scaler": ctx_scaler,
          "clipped_context_dims": clipped,
          "insufficient_bags": insufficient,
          "s_source": "frozen_t_std_original16"}
    tmp = ck_p.with_suffix(".tmp")
    torch.save(ck, tmp)
    tmp.replace(ck_p)
    if state_p.exists():
        state_p.unlink()
    print(f"adapter saved: {ck_p} (insufficient bags: {insufficient})")
    return ck
