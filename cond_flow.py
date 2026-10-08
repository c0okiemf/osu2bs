"""Q4: schedule-aware conditioned Flow (spec §7, fit A).

One shared geometry model with the EXISTING Flow capacity plus condition
inputs: the complete upcoming schedule (next same-hand gap ms, next
other-hand gap ms, next simultaneous event gap ms — timing only, upcoming
human GEOMETRY is a target, never input), the current 8-beat window intent
(12 window-target dims; human aggregates at train, planner output at
inference), the 10-dim workload vector and an 8-dim style embedding
(vocabulary resolved to neutral K=1 — the channel stays for schema, style
dropout is then a no-op).

ONE canonical feature builder (`cond_block`) serves training and inference.
The decode path is the complete-schedule seam: `CondFlowShim` exposes the
exact Flow member surface groom_notes uses (hidden/dir_head/col_head/
lay_head/chain_head/eval) with the condition rows precomputed from the fixed
EventSchedule, so flow_decode.decode_geometry drives it unchanged.

Fit A: family-normalized CE (one densest-Standard chart per family rep,
weight 1, + L/R mirror) + 0.1 x expected one-step reposition/pair cost over
the legal action distribution (teacher prev event + known timing) + a
bounded motion-risk auxiliary head. Teacher-forced val CE every epoch;
windowed rollout selection every 25 epochs (24 val families x 3 fixed
32-beat windows x 2 seeds, 8-beat warmup) — selection by rollout quality
under hard checks, CE tie-break. Checkpoints under
experiments/quality-v1/q4-flow/ only; flow.pt is never written.
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from groom import (ANCHOR, CTX, DCOL, DLAY, D_MODEL, MAX_CHAIN, NTOK,
                   N_HEAD, N_LAYER, events_to_xy, mirror_events)
from phrase_planner import WIN, window_targets, accent_target

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "quality-v1" / "q4-flow"
SEED = 20260921
GAP_EDGES = (150.0, 250.0, 350.0, 500.0, 700.0, 1000.0, 2000.0)  # ms, 8 bins
N_GAP = len(GAP_EDGES) + 1
N_INTENT, N_WORK, N_STYLE_EMB = 12, 10, 8
NCOND = 3 * N_GAP + N_INTENT + N_WORK
RISK_W = 0.1
SPEED_LIM = 12.0            # cells/s before reposition cost accrues
ROLLOUT_EVERY = 25
N_VAL_FAMS = 24
ROLL_SEEDS = (0, 1)


def _gap_oh(ms):
    """Bucketed one-hot of a forward gap in ms; None (no next event) shares
    the overflow bucket."""
    b = N_GAP - 1
    if ms is not None:
        for i, e in enumerate(GAP_EDGES):
            if ms < e:
                b = i
                break
    return F.one_hot(torch.tensor(b), N_GAP).float()


def cond_block(events_sh, times_ms, intent_rows, workload):
    """[N, NCOND] condition rows for an emission-ordered (step, hand) event
    list. TIMING of upcoming events only — geometry never enters. intent_rows:
    [n_windows, 12] window intent; workload: [10]."""
    n = len(events_sh)
    next_same = [None] * n
    next_other = [None] * n
    # forward gaps by scanning from the right
    seen = {0: None, 1: None}
    for i in range(n - 1, -1, -1):
        s, h = events_sh[i]
        next_same[i] = seen.get(h)
        next_other[i] = seen.get(1 - h)
        seen[h] = times_ms[i]
    # next simultaneous instant: timestamps carrying both hands
    by_t = {}
    for (s, h), t in zip(events_sh, times_ms):
        by_t.setdefault(t, set()).add(h)
    sim_ts = sorted(t for t, hs in by_t.items() if len(hs) == 2)
    import bisect
    rows = []
    for i, ((s, h), t) in enumerate(zip(events_sh, times_ms)):
        g_same = None if next_same[i] is None else next_same[i] - t
        g_other = None if next_other[i] is None else next_other[i] - t
        j = bisect.bisect_right(sim_ts, t)
        g_sim = sim_ts[j] - t if j < len(sim_ts) else None
        wi = min(int(s) // WIN, len(intent_rows) - 1) if len(intent_rows) \
            else 0
        intent = intent_rows[wi] if len(intent_rows) \
            else torch.zeros(N_INTENT)
        rows.append(torch.cat([_gap_oh(g_same), _gap_oh(g_other),
                               _gap_oh(g_sim), intent, workload]))
    return torch.stack(rows) if rows else torch.zeros(0, NCOND)


class ConditionedFlow(nn.Module):
    """Flow's architecture with a widened input (base token + condition block
    + style embedding) and a bounded motion-risk auxiliary head."""

    def __init__(self, n_styles=1):
        super().__init__()
        self.inp = nn.Linear(NTOK + NCOND + N_STYLE_EMB, D_MODEL)
        self.style_emb = nn.Embedding(max(1, n_styles), N_STYLE_EMB)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEAD, 4 * D_MODEL, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, N_LAYER)
        self.dir_head = nn.Linear(D_MODEL, 9)
        self.col_head = nn.Linear(D_MODEL + 9, DCOL)
        self.lay_head = nn.Linear(D_MODEL + 9 + DCOL, DLAY)
        self.chain_head = nn.Linear(D_MODEL + 9, MAX_CHAIN)
        self.risk_head = nn.Linear(D_MODEL + 9, 1)

    def hidden(self, x, style_idx=None):     # x: [B, N, NTOK+NCOND]
        b, n = x.shape[:2]
        sid = torch.zeros(b, dtype=torch.long, device=x.device) \
            if style_idx is None else style_idx
        emb = self.style_emb(sid)[:, None, :].expand(b, n, N_STYLE_EMB)
        h = self.inp(torch.cat([x, emb], -1)) \
            + self.pos(torch.arange(n, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n).to(x.device)
        return self.tr(h, mask=mask, is_causal=True)

    def forward(self, x, dir_oh, dcol_oh, style_idx=None):
        h = self.hidden(x, style_idx)
        return (self.dir_head(h),
                self.col_head(torch.cat([h, dir_oh], -1)),
                self.lay_head(torch.cat([h, dir_oh, dcol_oh], -1)),
                self.chain_head(torch.cat([h, dir_oh], -1)),
                self.risk_head(torch.cat([h, dir_oh], -1)))


class CondFlowShim:
    """Drop-in flow for groom_notes' complete-schedule geometry decode: the
    exact member surface (hidden + 4 heads + eval), with the condition rows
    precomputed from the FIXED schedule in emission order. hidden(x) receives
    the base [1, n, NTOK] token history; rows i<n get cond[i]."""

    def __init__(self, model, cond, style_idx=0):
        self._m, self._cond = model, cond
        self._sid = torch.tensor([style_idx], dtype=torch.long)

    def eval(self):
        self._m.eval()
        return self

    def hidden(self, x):
        n = x.shape[1]
        assert n <= len(self._cond), \
            f"schedule shorter than decode history ({n} > {len(self._cond)})"
        full = torch.cat([x, self._cond[:n][None].to(x.dtype)], -1)
        return self._m.hidden(full, self._sid)

    def __getattr__(self, k):
        if k in ("dir_head", "col_head", "lay_head", "chain_head",
                 "risk_head"):
            return getattr(self._m, k)
        raise AttributeError(k)


def shim_for_schedule(model, schedule, step_ms, intent_rows, workload,
                      style_idx=0):
    """Build the shim for an EventSchedule: emission order = ascending step,
    hands ascending — matching the production scheduler's emission."""
    ev = [(s, h) for s in sorted(schedule.entries)
          for h in schedule.entries[s]["hands"]]
    times = [s * step_ms for s, _h in ev]
    cond = cond_block(ev, times, intent_rows, workload)
    return CondFlowShim(model, cond, style_idx)


# ---------------- dataset ----------------

def _family_sequences(dirs, tag):
    """One densest-Standard chart per family-rep dir + its L/R mirror:
    (x_base, cond, y, aux) sequences, weight 1 per family (family-normalized
    by construction). aux: per-event (dt_same_ms, prev_col, prev_lay,
    other_col_if_double) for the risk term."""
    import groom
    seqs, skipped = [], 0
    for dp in dirs:
        d = Path(dp)
        samples = groom.load_map_all(d)
        if not samples:
            skipped += 1
            continue
        name = sorted(samples, key=lambda n: len(samples[n][2]))[-1]
        inp, pres, events, wl, wr = samples[name][:5]
        info_p = next((p for p in d.iterdir()
                       if p.name.lower() == "info.dat"), None)
        if info_p is None:
            skipped += 1
            continue
        info = json.loads(info_p.read_text(encoding="utf-8-sig"))
        bpm = info.get("_beatsPerMinute")
        if not bpm:
            skipped += 1
            continue
        step_ms = 60000.0 / float(bpm) / 4
        T = len(pres)
        step_times = [s * step_ms for s in range(T)]
        afeat = inp[:, :4]
        intents = []
        for w0 in range(0, max(T - WIN + 1, 1), WIN):
            w1 = min(w0 + WIN, T)
            tg = window_targets(pres, events, step_times, w0, w1)
            tg["accent_occupied"] = accent_target(afeat, pres, w0, w1)
            intents.append(torch.tensor(
                [tg[k] for k in ("occ_rate", "coincidence", "cad_l", "cad_r",
                                 "burst", "travel", "axis_vert", "axis_horiz",
                                 "axis_diag", "motif_conc", "quiet",
                                 "accent_occupied")], dtype=torch.float32))
        intents = torch.stack(intents)
        workload = intents[:, :N_WORK].mean(dim=0)
        energy = inp[:, 12]
        for ev, a, b in ((events, wl, wr), (mirror_events(events), wr, wl)):
            x, y = events_to_xy(ev, a, b, energy)
            ev_sh = [(e[0], e[1]) for e in ev]
            times = [e[0] * step_ms for e in ev]
            cond = cond_block(ev_sh, times, intents, workload)
            aux = _aux_rows(ev, step_ms)
            seqs.append({"x": x, "cond": cond, "y": y, "aux": aux,
                         "fam": str(dp), "tag": tag})
    return seqs, skipped


def _aux_rows(events, step_ms):
    """Per event: [dt_same_ms, prev_col, prev_lay, other_col (or -1), hand] —
    teacher context for the physical risk term."""
    lastn = {0: None, 1: None}
    by_step = {}
    for s, h, _d, c, _l, _k in events:
        by_step.setdefault(s, {})[h] = c
    rows = []
    for s, h, d, c, l, k in events:
        p = lastn[h]
        dt = (s - p[0]) * step_ms if p else 1e6
        pc, pl = (p[1], p[2]) if p else ANCHOR[h]
        oc = by_step[s].get(1 - h, -1)
        rows.append((dt, pc, pl, float(oc), float(h)))
        lastn[h] = (s, c, l)
    return torch.tensor(rows, dtype=torch.float32)


def expected_risk(c_lg, l_lg, aux):
    """Expected one-step reposition/pair cost over the LEGAL delta action
    distribution (factored col/layer marginals), teacher prev event + known
    timing. Bounded to [0, ~1.5]."""
    B, N = c_lg.shape[:2]
    dt = aux[..., 0].clamp(min=80.0) / 1000.0        # s
    pc, pl, oc = aux[..., 1], aux[..., 2], aux[..., 3]
    dcol = torch.arange(DCOL, device=c_lg.device).float() - 3.0
    dlay = torch.arange(DLAY, device=c_lg.device).float() - 2.0
    nc = pc[..., None] + dcol                        # [B,N,DCOL]
    nl = pl[..., None] + dlay                        # [B,N,DLAY]
    legal_c = (nc >= 0) & (nc <= 3)
    legal_l = (nl >= 0) & (nl <= 2)
    pcol = torch.softmax(c_lg.masked_fill(~legal_c, -1e9), -1)
    play = torch.softmax(l_lg.masked_fill(~legal_l, -1e9), -1)
    dist = (dcol.abs()[None, None, :, None] ** 2
            + dlay.abs()[None, None, None, :] ** 2).sqrt()   # [1,1,C,L]
    speed = dist / dt[..., None, None]
    repo = ((speed - SPEED_LIM).clamp(min=0.0) / SPEED_LIM).clamp(max=1.0)
    # pair crossing risk: doubles whose new col crosses the other hand —
    # left hand (h=0) crossing = new col RIGHT of the other, right hand the
    # mirror; sign flip by actual hand identity
    hand = aux[..., 4]
    have_o = (oc >= 0).float()[..., None]
    side = torch.where(hand > 0.5, -1.0, 1.0)[..., None]   # left:+, right:-
    cross = ((nc - oc[..., None]).sign() * side).clamp(min=0.0)
    pair = 0.5 * have_o * cross                       # [B,N,DCOL]
    e_repo = (pcol[..., :, None] * play[..., None, :] * repo).sum((-1, -2))
    e_pair = (pcol * pair).sum(-1)
    return e_repo + e_pair


def _loss(model, batch, dev, style_drop=0.25, rng=None, risk_w=RISK_W,
          aux_w=0.1):
    x = torch.stack([b["x"] for b in batch]).to(dev)
    cond = torch.stack([b["cond"] for b in batch]).to(dev)
    y = torch.stack([b["y"] for b in batch]).to(dev)
    aux = torch.stack([b["aux"] for b in batch]).to(dev)
    sid = torch.zeros(len(batch), dtype=torch.long, device=dev)
    if rng is not None:                     # style dropout (no-op at K=1)
        drop = torch.tensor([rng.random() < style_drop for _ in batch],
                            device=dev)
        sid = torch.where(drop, torch.zeros_like(sid), sid)
    d_oh = F.one_hot(y[..., 0], 9).float()
    c_oh = F.one_hot(y[..., 1], DCOL).float()
    d_lg, c_lg, l_lg, k_lg, r_lg = model(
        torch.cat([x, cond], -1), d_oh, c_oh, sid)
    ce = (F.cross_entropy(d_lg.reshape(-1, 9), y[..., 0].reshape(-1))
          + F.cross_entropy(c_lg.reshape(-1, DCOL), y[..., 1].reshape(-1))
          + F.cross_entropy(l_lg.reshape(-1, DLAY), y[..., 2].reshape(-1))
          + F.cross_entropy(k_lg.reshape(-1, MAX_CHAIN),
                            y[..., 3].reshape(-1)))
    er = expected_risk(c_lg, l_lg, aux)
    # bounded risk head: supervise on the HUMAN action's own physical cost
    with torch.no_grad():
        human_cost = _human_cost(y, aux)
    aux_bce = F.binary_cross_entropy_with_logits(
        r_lg.squeeze(-1), (human_cost > 0.25).float())
    return ce + risk_w * er.mean() + aux_w * aux_bce, ce, er.mean()


def _human_cost(y, aux):
    dcol = (y[..., 1].float() - 3.0)
    dlay = (y[..., 2].float() - 2.0)
    dt = aux[..., 0].clamp(min=80.0) / 1000.0
    speed = (dcol ** 2 + dlay ** 2).sqrt() / dt
    return ((speed - SPEED_LIM).clamp(min=0.0) / SPEED_LIM).clamp(max=1.0)


# ---------------- windowed rollout selection ----------------

def _prev_block(h, d, c, l, k):
    """The first 26 base-token dims: one-hots of the previous event's
    (hand, dir, col-bucket, layer-bucket, chain) — token_vec's layout."""
    pk_idx = min(int(k), MAX_CHAIN) - 1
    return torch.cat([F.one_hot(torch.tensor(int(h)), 3).float(),
                      F.one_hot(torch.tensor(int(d)), 10).float(),
                      F.one_hot(torch.tensor(int(min(max(c, 0), 4))),
                                5).float(),
                      F.one_hot(torch.tensor(int(min(max(l, 0), 3))),
                                4).float(),
                      F.one_hot(torch.tensor(pk_idx), 4).float()])


def _window_variety_excess(tokens, dts, doubles, env):
    """Matched-envelope excess of ONE rollout window's sampled tokens (frozen
    approximation, fit2_policy.json): max-4gram share of the sampled token
    stream vs the matched human bin p90; cad ~ 2000/mean same-hand dt."""
    from eval.motif_envelope import matched_p90
    n = len(tokens) - 3
    if n < 4:
        return None
    grams = {}
    for i in range(n):
        g = tuple(tokens[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    share = max(grams.values()) / n
    dbl = sum(doubles) / len(doubles) if doubles else 0.0
    cad = 2000.0 / max(1.0, sum(dts) / len(dts)) if dts else 0.0
    return max(0.0, share - matched_p90(env, n, dbl, cad))


def rollout_score(model, seqs, dev, temp=1.0, env=None):
    """SAMPLED-HISTORY windowed rollouts on the human schedule: 3 fixed
    48-event windows/seq (start/middle/end), 2 fixed seeds, 16-event teacher
    warmup — after that, each sampled event is written back into the next
    token's previous-event block and per-hand positions, so exposure error
    accumulates as it would at decode. Score = mean sampled-action physical
    cost (reposition + pair crossing); illegal actions are masked so any
    residual illegal is a hard-check hit. With `env`, also returns the mean
    matched-envelope variety excess of the sampled windows (fit B's
    selection metric). Lower is better; deterministic."""
    model.eval()
    total_cost, total_illegal, n_ev = 0.0, 0, 0
    var_excesses = []
    zero_sid = torch.zeros(1, dtype=torch.long, device=dev)
    for si, sq in enumerate(seqs):
        x, cond, aux, y = sq["x"], sq["cond"], sq["aux"], sq["y"]
        n = len(x)
        if n < 48:
            continue
        for w0 in (0, max(0, (n - 48) // 2), n - 48):
            for seed in ROLL_SEEDS:
                g = torch.Generator().manual_seed(SEED + seed * 997 + si)
                lo, hi = w0, min(w0 + 48, n)
                xr = x[lo:hi].clone()
                lastpos = {}                      # hand -> sampled (col, lay)
                w_toks, w_dts, w_dbls = [], [], []
                with torch.no_grad():
                    for i in range(16, hi - lo):
                        full = torch.cat([xr[:i + 1], cond[lo:lo + i + 1]],
                                         -1)[None].to(dev)
                        hid = model.hidden(full, zero_sid)[0, -1]
                        hand = int(aux[lo + i, 4])
                        pc, pl = lastpos.get(hand,
                                             (float(aux[lo + i, 1]),
                                              float(aux[lo + i, 2])))
                        d_lg = model.dir_head(hid)
                        d = int(torch.multinomial(
                            torch.softmax(d_lg / temp, -1).cpu(), 1,
                            generator=g))
                        d_oh = F.one_hot(torch.tensor(d, device=dev),
                                         9).float()
                        c_lg = model.col_head(torch.cat([hid, d_oh]))
                        nc = pc + torch.arange(DCOL).float() - 3.0
                        legal_c = (nc >= 0) & (nc <= 3)
                        if not legal_c.any():
                            total_illegal += 1
                            n_ev += 1
                            continue
                        ci = int(torch.multinomial(torch.softmax(
                            c_lg.cpu().masked_fill(~legal_c, -1e9) / temp,
                            -1), 1, generator=g))
                        l_lg = model.lay_head(torch.cat(
                            [hid, d_oh, F.one_hot(
                                torch.tensor(ci, device=dev), DCOL).float()]))
                        nl = pl + torch.arange(DLAY).float() - 2.0
                        legal_l = (nl >= 0) & (nl <= 2)
                        li = int(torch.multinomial(torch.softmax(
                            l_lg.cpu().masked_fill(~legal_l, -1e9) / temp,
                            -1), 1, generator=g))
                        newc, newl = pc + ci - 3.0, pl + li - 2.0
                        dt = max(float(aux[lo + i, 0]), 80.0) / 1000.0
                        speed = math.hypot(ci - 3.0, li - 2.0) / dt
                        total_cost += min(max(speed - SPEED_LIM, 0.0)
                                          / SPEED_LIM, 1.0)
                        oc = float(aux[lo + i, 3])
                        if oc >= 0:
                            if (hand == 0 and newc > oc) or \
                                    (hand == 1 and newc < oc):
                                total_cost += 0.5
                        lastpos[hand] = (newc, newl)
                        w_toks.append((hand, d, ci - 3, li - 2, oc >= 0))
                        w_dts.append(max(float(aux[lo + i, 0]), 80.0))
                        w_dbls.append(oc >= 0)
                        # write the SAMPLED event into the next token's
                        # previous-event block (chain kept at teacher's)
                        if i + 1 < hi - lo:
                            k = int(y[lo + i, 3]) + 1
                            xr[i + 1, :26] = _prev_block(
                                hand, d, int(newc), int(newl), k)
                        n_ev += 1
                if env is not None:
                    ve = _window_variety_excess(w_toks, w_dts, w_dbls, env)
                    if ve is not None:
                        var_excesses.append(ve)
    if n_ev == 0:
        return {"score": float("inf"), "illegal": 0, "events": 0,
                "variety_excess": None}
    return {"score": total_cost / n_ev + 10.0 * total_illegal / n_ev,
            "illegal": total_illegal, "events": n_ev,
            "variety_excess": (sum(var_excesses) / len(var_excesses)
                               if var_excesses else None)}


def train_cond_flow(max_epochs=600, steps=30, batch=32, resume=True,
                    fit="A"):
    """fit="B": the variety-targeted variant
    (fit2_policy.json, FROZEN before training): risk weight 0, aux 0,
    motion-gated eligibility (0 illegal, physical score <= 0.0162 =
    1.10 x fit A's selected), rank by matched-envelope variety excess then
    val CE. fit="A" behavior is unchanged."""
    from eval import corpus
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "dataset.pt"
    if cache.exists():
        data = torch.load(cache)
        trs, vas = data["train"], data["val"]
    else:
        trs, sk_t = _family_sequences(
            sorted(corpus.train_families_rep("train")), "train")
        va_dirs = sorted(corpus.train_families_rep("val"))
        vas, sk_v = _family_sequences(va_dirs, "val")
        torch.save({"train": trs, "val": vas,
                    "skipped": {"train": sk_t, "val": sk_v}}, cache)
    trs = [s for s in trs if len(s["x"]) > CTX]
    va_ce = [s for s in vas if len(s["x"]) >= CTX]
    va_roll = [s for s in vas if len(s["x"]) >= 48][:2 * N_VAL_FAMS]
    fitb = fit == "B"
    risk_w, aux_w = (0.0, 0.0) if fitb else (RISK_W, 0.1)
    env = None
    if fitb:
        env = json.loads((ROOT / "experiments/quality-v1/q4-style/"
                          "human_motif_envelope.json").read_text())
        assert env.get("version") == 2, "fit B needs the matched envelope v2"
    tag = "b" if fitb else "a"
    print(f"[cond-flow-{tag}] {len(trs)} train seqs, {len(va_ce)} val CE "
          f"seqs, {len(va_roll)} val rollout seqs; dev={dev}", flush=True)
    model = ConditionedFlow().to(dev)
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    ck = OUT / f"condflow-{tag}-seed20260921.pt"
    state_p = OUT / ("train_state.pt" if not fitb else "train_state-b.pt")
    hist_p = OUT / f"condflow-{tag}-history.json"
    hist = {"epochs": [], "rollouts": [], "seed": SEED, "fit": fit}
    start_ep = 0
    if resume and state_p.exists():
        st = torch.load(state_p)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])
        start_ep = st["epoch"] + 1
        hist = json.loads(hist_p.read_text())
        print(f"[cond-flow] resumed at epoch {start_ep}", flush=True)

    def val_ce():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va_ce), 16):
                bt = [{k: s[k][:CTX] for k in ("x", "cond", "y", "aux")}
                      for s in va_ce[i:i + 16]]
                _l, ce, _er = _loss(model, bt, dev)
                tot += float(ce) * len(bt)
        return tot / len(va_ce)

    PHYS_MAX = 0.0162          # fit B motion gate: 1.10 x fit A's selected

    def _rank(r):
        """Selection key. fit A: (physical score, CE). fit B: motion-gated
        eligibility first, then (matched variety excess, CE)."""
        if not fitb:
            return (0, r["score"], r["ce"])
        eligible = r["illegal"] == 0 and r["score"] <= PHYS_MAX
        return (0 if eligible else 1,
                r["variety_excess"] if r["variety_excess"] is not None
                else float("inf"), r["ce"])

    best = None
    if hist["rollouts"]:
        best = min(hist["rollouts"], key=_rank)
    for ep in range(start_ep, max_epochs):
        model.train()
        for _ in range(steps):
            bt = []
            for s in rng.choices(trs, k=batch):
                i = rng.randrange(len(s["x"]) - CTX)
                bt.append({k: s[k][i:i + CTX]
                           for k in ("x", "cond", "y", "aux")})
            loss, _ce, _er = _loss(model, bt, dev, rng=rng,
                                   risk_w=risk_w, aux_w=aux_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vce = val_ce()
        hist["epochs"].append({"epoch": ep, "val_ce": vce})
        if ep % ROLLOUT_EVERY == 0 or ep == max_epochs - 1:
            rs = rollout_score(model, va_roll, dev, env=env)
            rs.update(epoch=ep, ce=vce)
            hist["rollouts"].append(rs)
            ve = rs.get("variety_excess")
            print(f"[cond-flow-{tag}] ep {ep}: val CE {vce:.4f} rollout "
                  f"{rs['score']:.4f} (illegal {rs['illegal']}"
                  + (f", variety {ve:.4f}" if ve is not None else "")
                  + ")", flush=True)
            if best is None or _rank(rs) < _rank(best):
                best = rs
                torch.save({"state": {k: v.cpu()
                                      for k, v in model.state_dict().items()},
                            "epoch": ep, "rollout": rs, "n_styles": 1,
                            "fit": fit}, ck)
        elif ep % 10 == 0:
            print(f"[cond-flow-{tag}] ep {ep}: val CE {vce:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                    "epoch": ep}, state_p)
        hist_p.write_text(json.dumps(hist, indent=1))
    sel_ok = (not fitb) or (best is not None and _rank(best)[0] == 0)
    print(f"[cond-flow-{tag}] best @ep{best['epoch']}: score "
          f"{best['score']:.4f}"
          + (f" variety {best['variety_excess']:.4f}"
             if best.get("variety_excess") is not None else "")
          + ("" if sel_ok else " — NO MOTION-ELIGIBLE CHECKPOINT (explicit)")
          + f" -> {ck}", flush=True)
    return hist


if __name__ == "__main__":
    import sys
    eps = next((int(a) for a in sys.argv[1:] if a.isdigit()), 600)
    train_cond_flow(max_epochs=eps,
                    fit="B" if "fitb" in sys.argv else "A")
