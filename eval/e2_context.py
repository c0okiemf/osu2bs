"""E2: schedule-aware flow conditioning.

ContextFlow = shipped B0 Flow + ONE zero-initialized projection added to
the input embedding, so initial logits equal B0 exactly. Its extra input
per event is built by ONE shared feature path used in training and decode:

- schedule_rows(): from the FIXED schedule only, the next four head events
  after the current one (emission order = step, then hand): gap seconds
  (clipped 2 s, scaled), hand, coincidence (another head at that step),
  follower count, present flag. The current event's own followers are a
  training TARGET (chain head) and are never included. No future geometry.
- opposite_head(): computed inside the model from the token row itself:
  the previous event's direction/column/layer, gated on "previous event is
  the opposite hand at the SAME step" (already emitted, so no leakage).

Training reuses E1's objective: CE + 0.05 KL-to-B0 on the general stratum,
approved/general 50/50, LR 3e-5, batch 8, <=3600 updates, snapshots
0/600/1800/3600, selection by held-out family-macro CE. One fit.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from groom import CTX, D_MODEL, DCOL, DLAY, MAX_CHAIN, NTOK, Flow

ROOT = Path(__file__).resolve().parent.parent
E2 = ROOT / "experiments" / "expressive-v1" / "e2"
N_NEXT = 4
SCHED_DIM = 5 * N_NEXT
OPP_DIM = 10 + 5 + 4
CTX_DIM = SCHED_DIM + OPP_DIM
GAP_CLIP_S = 2.0


def schedule_rows(sched, step_ms):
    """sched: [(step, hand, followers)] in emission order -> [N, 20]."""
    n = len(sched)
    steps = [s for s, _h, _k in sched]
    heads_at = {}
    for s, _h, _k in sched:
        heads_at[s] = heads_at.get(s, 0) + 1
    rows = torch.zeros(n, SCHED_DIM)
    for i, (s, _h, _k) in enumerate(sched):
        for j in range(N_NEXT):
            t = i + 1 + j
            if t >= n:
                break
            s2, h2, k2 = sched[t]
            gap = min((s2 - s) * step_ms / 1000.0, GAP_CLIP_S) / GAP_CLIP_S
            rows[i, 5 * j:5 * j + 5] = torch.tensor(
                [gap, float(h2), float(heads_at[s2] > 1),
                 min(k2, 2) / 2.0, 1.0])
    return rows


def opposite_head(x):
    """x [..., NTOK] token rows -> [..., 19] gated prev dir/col/layer.
    Token layout (groom.token_vec): prev hand 0:3, prev dir 3:13, prev col
    13:18, prev layer 18:22, prev chain 22:26, hand 26:28, dt_same 28:38,
    dt_any 38:48 (index 0 = same step)."""
    same_hand = x[..., 0] * x[..., 26] + x[..., 1] * x[..., 27]
    gate = x[..., 38] * (1.0 - same_hand)
    return gate[..., None] * x[..., 3:22]


class ContextFlow(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        self.base = base or Flow()
        self.proj = nn.Linear(CTX_DIM, D_MODEL)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def hidden(self, x, sched):
        n = x.shape[1]
        feat = torch.cat([sched, opposite_head(x)], -1)
        h = self.base.inp(x) + self.proj(feat) \
            + self.base.pos(torch.arange(n, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n).to(x.device)
        return self.base.tr(h, mask=mask, is_causal=True)

    def forward(self, x, sched, dir_oh, dcol_oh):
        h = self.hidden(x, sched)
        b = self.base
        return (b.dir_head(h), b.col_head(torch.cat([h, dir_oh], -1)),
                b.lay_head(torch.cat([h, dir_oh, dcol_oh], -1)),
                b.chain_head(torch.cat([h, dir_oh], -1)))


class ContextShim:
    """Drop-in flow for groom_notes' fixed-schedule decode. groom_notes
    calls hidden() once per emitted head with the last <=CTX token rows;
    a counter maps rows to absolute schedule indices."""

    def __init__(self, model, rows):
        self._m, self._rows, self._i = model, rows, -1

    def eval(self):
        self._m.eval()
        return self

    def hidden(self, x):
        self._i += 1
        n = x.shape[1]
        if n < CTX and self._i != n - 1:
            raise RuntimeError(f"schedule index drift: call {self._i}, "
                               f"history {n}")
        if self._i >= len(self._rows):
            raise RuntimeError("decode emitted more heads than scheduled")
        lo = self._i - n + 1
        return self._m.hidden(x, self._rows[lo:self._i + 1][None]
                              .to(x.dtype))

    def __getattr__(self, k):
        if k in ("dir_head", "col_head", "lay_head", "chain_head"):
            return getattr(self._m.base, k)
        raise AttributeError(k)


def shim_for_schedule(model, schedule, step_ms):
    sched = [(s, h, schedule.entries[s]["k"].get(h, 0))
             for s in sorted(schedule.entries)
             for h in schedule.entries[s]["hands"]]
    return ContextShim(model, schedule_rows(sched, step_ms))


# ---------------- dataset + training ----------------

def _events_sched(events):
    return [(s, h, max(chain - 1, 0)) for s, h, _d, _c, _l, chain in events]


def build_dataset():
    """E1's dataset families/charts plus schedule rows (own cache)."""
    import hashlib
    import groom
    from groom import events_to_xy, mirror_events
    from eval import corpus
    from eval.expressive_manifest import family_strata
    cache = E2 / "dataset.pt"
    if cache.exists():
        return torch.load(cache)
    E2.mkdir(parents=True, exist_ok=True)
    m = corpus._load_validated()
    strata_all = family_strata(m)
    fam_dir = {}
    for r in m["maps"]:
        if r["eligible"] == "ok" and r.get("family_rep"):
            fam_dir.setdefault((r["split"], r["family"]), r["dir"])
    data = {"train": {"approved": {}, "general": {}},
            "val": {"approved": {}, "general": {}}, "charts": {}}
    for (split, fam), dp in sorted(fam_dir.items()):
        if split not in ("train", "val"):
            continue
        stratum = "approved" if fam in set(strata_all["approved"]) else \
            "rejected" if fam in set(strata_all["rejected"]) else "general"
        if stratum == "rejected":
            continue
        d = Path(dp)
        info_p = next((p for p in d.iterdir()
                       if p.name.lower() == "info.dat"), None)
        if info_p is None:
            continue
        bpm = json.loads(info_p.read_text(encoding="utf-8-sig")) \
            .get("_beatsPerMinute")
        samples = groom.load_map_all(d)
        if not samples or not bpm:
            continue
        step_ms = 60000.0 / float(bpm) / 4
        surv = []
        for name, s in sorted(samples.items()):
            inp, _pres, events, wl, wr = s[:5]
            energy = inp[:, 12]
            xo, yo = events_to_xy(events, wl, wr, energy)
            if len(xo) <= CTX:
                continue
            ev_m = mirror_events(events)
            xm, ym = events_to_xy(ev_m, wr, wl, energy)
            cid = hashlib.sha256(f"{fam}:{name}".encode()).hexdigest()[:16]
            data["charts"][cid] = {
                "x": xo, "y": yo,
                "r": schedule_rows(_events_sched(events), step_ms)}
            data["charts"][cid + ":m"] = {
                "x": xm, "y": ym,
                "r": schedule_rows(_events_sched(ev_m), step_ms)}
            surv.append(cid)
        if surv:
            data[split][stratum][fam] = surv
        print(f"  [e2 data] {fam}", flush=True)
    torch.save(data, cache)
    return data


def _ce(model, xb, rb, yb):
    d_oh = F.one_hot(yb[..., 0], 9).float()
    c_oh = F.one_hot(yb[..., 1], DCOL).float()
    out = model(xb, rb, d_oh, c_oh)
    ce = sum(F.cross_entropy(lg.reshape(-1, n), yb[..., j].reshape(-1))
             for j, (lg, n) in enumerate(zip(out, (9, DCOL, DLAY,
                                                   MAX_CHAIN))))
    return ce, out, d_oh, c_oh


def _val(model, data, dev):
    model.eval()
    res = {}
    with torch.no_grad():
        for stratum in ("approved", "general"):
            fam_ce = []
            for fam, cids in data["val"][stratum].items():
                tot, n = 0.0, 0
                for cid in cids:
                    ch = data["charts"][cid]
                    L = len(ch["x"])
                    for o in sorted({0, max(0, (L - CTX) // 2), L - CTX}):
                        ce, *_ = _ce(model, ch["x"][o:o + CTX][None].to(dev),
                                     ch["r"][o:o + CTX][None].to(dev),
                                     ch["y"][o:o + CTX][None].to(dev))
                        tot += float(ce)
                        n += 1
                if n:
                    fam_ce.append(tot / n)
            res[f"{stratum}_macro"] = sum(fam_ce) / len(fam_ce)
    res["overall_macro"] = 0.5 * (res["approved_macro"]
                                  + res["general_macro"])
    return res


def train(max_updates=3600, resume=True, out_dir=None, init_ckpt=None,
          pool=None, pool_p=0.0, seed_offset=250):
    """E2 defaults. E3 passes out_dir, init_ckpt (a ContextFlow state) and
    a frozen turning pool {family: [(chart_key, start)]} sampled with
    probability pool_p (family-balanced)."""
    import random as _random
    run_dir = out_dir or E2
    from eval.expressive_train import (BATCH, KL_W, LR, SNAPSHOTS,
                                       SampleStream, select_snapshot,
                                       teacher_kl)
    from eval.expressive_manifest import load_run
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = load_run(ROOT / "experiments/expressive-v1/e1")["config"]["seed"]
    data = build_dataset()
    stream = SampleStream({k: data["train"][k]
                           for k in ("approved", "general")}, 0.5,
                          seed=seed * 100 + seed_offset)
    pool_rng = _random.Random(seed * 100 + seed_offset + 1)
    pool_fams = sorted(pool or {})
    pool_strata = {f: ("approved" if f in data["train"]["approved"]
                       else "general") for f in pool_fams}
    state = torch.load(ROOT / "flow.pt", map_location="cpu")
    student = ContextFlow()
    student.base.load_state_dict(state)
    if init_ckpt is not None:
        student.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
    teacher = Flow()
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.to(dev)
    teacher.to(dev)
    opt = torch.optim.Adam(student.parameters(), LR)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_p, hist_p = run_dir / "train_state.pt", run_dir / "history.json"
    history = {"updates": []}
    u = 0
    if resume and state_p.exists():
        st = torch.load(state_p)
        student.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["torch_rng"])
        stream.load_state_dict(st["sampler"])
        if "pool_rng" in st:
            pool_rng.setstate(st["pool_rng"])
        u = st["update"]
        history = json.loads(hist_p.read_text())
    else:
        torch.manual_seed(seed)

    def snapshot(k):
        p = run_dir / f"snapshot-{k}.pt"
        if not p.exists():
            torch.save({k2: v.cpu() for k2, v in student.state_dict().items()},
                       p)
        history["updates"].append({"update": k,
                                   "val": _val(student, data, dev)})
        hist_p.write_text(json.dumps(history, indent=1))
        print(f"[e2] u{k}: {history['updates'][-1]['val']}", flush=True)

    if u == 0:
        snapshot(0)
    while u < max_updates:
        student.train()
        xs, rs, ys, gen = [], [], [], []
        for _ in range(BATCH):
            s = stream.next()
            ch = data["charts"][s["chart"] + (":m" if s["mirror"] else "")]
            i = int(s["crop_u"] * (len(ch["x"]) - CTX))
            if pool_fams and pool_rng.random() < pool_p:
                fam = pool_fams[pool_rng.randrange(len(pool_fams))]
                key, i = pool[fam][pool_rng.randrange(len(pool[fam]))]
                ch = data["charts"][key]
                s = {"stratum": pool_strata[fam]}
            xs.append(ch["x"][i:i + CTX])
            rs.append(ch["r"][i:i + CTX])
            ys.append(ch["y"][i:i + CTX])
            gen.append(s["stratum"] == "general")
        xb, rb, yb = (torch.stack(v).to(dev) for v in (xs, rs, ys))
        ce, out, d_oh, c_oh = _ce(student, xb, rb, yb)
        loss = ce
        gm = torch.tensor(gen, device=dev)
        if gm.any():
            with torch.no_grad():
                tout = teacher(xb[gm], d_oh[gm], c_oh[gm])
            loss = loss + KL_W * sum(
                teacher_kl(so[gm].reshape(-1, n), to.reshape(-1, n))
                for so, to, n in zip(out, tout, (9, DCOL, DLAY, MAX_CHAIN))
            ) / 4
        if not torch.isfinite(loss):
            raise RuntimeError(f"TRAINING_DIVERGED at u{u}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        u += 1
        if u % 100 == 0 or u == max_updates:
            torch.save({"model": student.state_dict(),
                        "opt": opt.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "sampler": stream.state_dict(),
                        "pool_rng": pool_rng.getstate(), "update": u},
                       state_p)
        if u in SNAPSHOTS:
            snapshot(u)
    sel = select_snapshot(history["updates"], 5,
                          len(data["val"]["approved"]))
    history["selected"] = sel
    hist_p.write_text(json.dumps(history, indent=1))
    print(f"[e2] selected {sel}")
    return history


def load_selected():
    hist = json.loads((E2 / "history.json").read_text())
    sel = hist["selected"]
    m = ContextFlow()
    m.load_state_dict(torch.load(E2 / f"snapshot-{sel['update']}.pt",
                                 map_location="cpu"))
    m.eval()
    return m, sel


def evaluate():
    """The one preregistered evaluation: six E2 seeds + six matched
    shipped-flow controls per song on B0's schedules, then P1-P4."""
    from eval.e1_machine_ab import run as machine_ab
    from eval.expressive_eval import run_e1_eval
    model, sel = load_selected()
    if sel["step_zero"]:
        print("[e2] step-zero winner: no checkpoint -> NOT_SHOWN")
        return {"status": "STEP_ZERO"}
    run_e1_eval("E2", flow_factory=lambda sch, ms: shim_for_schedule(
        model, sch, ms), tag=f"e2@{sel['update']}")
    run_e1_eval("ctrl", flow_factory=lambda sch, ms: None,
                tag="shipped-ctrl")
    return machine_ab(tags=("E2", "ctrl"),
                      pairs=[("E2", "b0"), ("E2", "ctrl"), ("ctrl", "b0")],
                      report_name="report_e2.json")


if __name__ == "__main__":
    import sys
    {"data": build_dataset, "train": train,
     "evaluate": evaluate}[sys.argv[1]]()
