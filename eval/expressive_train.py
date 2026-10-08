"""E1 Task 3: two shipped-checkpoint adaptation arms (spec §E1).

Both arms start from the EXACT shipped Flow weights. Objective: original
teacher-forced dir/col/layer/chain CE plus 0.05 x KL(teacher||student) to a
frozen B0 Flow on GENERAL-stratum samples only. Sampling hierarchy:
stratum (approved p / general 1-p) -> uniform family -> uniform surviving
chart -> original/mirror equally -> uniform valid CTX crop. Adam 3e-5,
batch 8, <=3600 updates, snapshots {0,600,1800,3600}, recoverable state
every 100 updates (model/optimizer/RNG/sampler cursor). Selection: lowest
family-macro held-out val CE with approved/general strata equally weighted
(fallback overall macro was frozen pre-training; unused: 11 approved val
families >= 5). No physical-risk loss, motif penalty, entropy bonus or
synthetic fun reward. Outputs under experiments/expressive-v1/e1/.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from groom import CTX, DCOL, DLAY, MAX_CHAIN, Flow, events_to_xy, \
    mirror_events

ROOT = Path(__file__).resolve().parent.parent
E1 = ROOT / "experiments" / "expressive-v1" / "e1"
KL_W = 0.05
LR, BATCH, MAX_UPDATES = 3e-5, 8, 3600
SNAPSHOTS = (0, 600, 1800, 3600)
STATE_EVERY = 100


def teacher_kl(student_logits, teacher_logits):
    """Batch-mean KL(softmax(teacher) || softmax(student)); no gradient
    flows into the teacher."""
    t = F.softmax(teacher_logits.detach(), dim=-1)
    return (t * (torch.log(t.clamp(min=1e-12))
                 - F.log_softmax(student_logits, dim=-1))).sum(-1).mean()


def initialized_models(ckpt_path):
    """(student, frozen teacher), both loaded from the same checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu")
    student, teacher = Flow(), Flow()
    student.load_state_dict(state)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return student, teacher


class SampleStream:
    """Deterministic explicit-hierarchy sampler with a serializable cursor:
    stratum -> uniform family -> uniform chart -> mirror coin -> crop is
    drawn later against the chart's actual length (crop_u in [0,1))."""

    def __init__(self, strata, approved_probability, seed):
        for name in ("approved", "general"):
            if not strata.get(name):
                raise ValueError(f"missing stratum {name!r} — unsupported "
                                 "configuration, not a silent mixture change")
        self.strata = {k: {f: list(cs) for f, cs in sorted(v.items())}
                       for k, v in strata.items()}
        self.fams = {k: sorted(v) for k, v in self.strata.items()}
        self.p = approved_probability
        self.rng = random.Random(seed)
        self.count = 0

    def next(self):
        stratum = "approved" if self.rng.random() < self.p else "general"
        fam = self.fams[stratum][self.rng.randrange(
            len(self.fams[stratum]))]
        charts = self.strata[stratum][fam]
        chart = charts[self.rng.randrange(len(charts))]
        mirror = self.rng.random() < 0.5
        crop_u = self.rng.random()
        self.count += 1
        return {"stratum": stratum, "family": fam, "chart": chart,
                "mirror": mirror, "crop_u": crop_u}

    def state_dict(self):
        return {"rng": self.rng.getstate(), "count": self.count}

    def load_state_dict(self, st):
        self.rng.setstate(_as_rng_state(st["rng"]))
        self.count = st["count"]


def _as_rng_state(s):
    if isinstance(s, tuple):
        return s
    a, b, c = s
    return (a, tuple(b), c)


def select_snapshot(history, min_approved_families, approved_families_n):
    """Lowest stratified family-macro val CE; frozen fallback = overall
    macro when the approved val stratum is thin. Ties within 1e-6 choose the
    earlier update. update 0 winning = no new checkpoint (step_zero)."""
    use_fallback = approved_families_n < min_approved_families
    metric = "overall_macro" if use_fallback else "stratified"

    def key(rec):
        v = rec["val"]
        if use_fallback:
            return v["overall_macro"]
        return 0.5 * v["approved_macro"] + 0.5 * v["general_macro"]
    best = min(history, key=lambda r: (round(key(r), 6), r["update"]))
    return {"update": best["update"], "metric": metric,
            "value": key(best), "step_zero": best["update"] == 0}


# ---------------- dataset ----------------

def build_dataset():
    """Per-family chart sequences for BOTH strata of the train split, plus
    fixed val crops. One rep dir per family; ALL supported charts in it;
    original + mirror precomputed. Cached by manifest identity."""
    import hashlib
    import groom
    from eval import corpus
    from eval.expressive_manifest import family_strata, load_run
    run = load_run(E1)
    cache = E1 / "dataset.pt"
    if cache.exists():
        d = torch.load(cache)
        if d.get("identity") == run["identity"]:
            return d
    m = corpus._load_validated()
    strata_all = family_strata(m)
    fam_dir = {}
    for r in m["maps"]:
        if r["eligible"] == "ok" and r.get("family_rep"):
            fam_dir.setdefault((r["split"], r["family"]), r["dir"])
    data = {"train": {"approved": {}, "general": {}},
            "val": {"approved": {}, "general": {}}, "charts": {},
            "identity": run["identity"]}
    for (split, fam), dp in sorted(fam_dir.items()):
        if split not in ("train", "val"):
            continue
        stratum = "approved" if fam in set(strata_all["approved"]) else \
            "rejected" if fam in set(strata_all["rejected"]) else "general"
        if stratum == "rejected":
            continue
        samples = groom.load_map_all(Path(dp))
        if not samples:
            continue
        surv = []
        for name, s in sorted(samples.items()):
            inp, _pres, events, wl, wr = s[:5]
            energy = inp[:, 12]
            xo, yo = events_to_xy(events, wl, wr, energy)
            if len(xo) <= CTX:                 # original and mirror share len
                continue
            xm, ym = events_to_xy(mirror_events(events), wr, wl, energy)
            cid = hashlib.sha256(f"{fam}:{name}".encode()).hexdigest()[:16]
            data["charts"][cid] = {"x": xo, "y": yo}
            data["charts"][cid + ":m"] = {"x": xm, "y": ym}
            surv.append(cid)
        if surv:
            data[split][stratum][fam] = surv
    torch.save(data, cache)
    return data


def _resolve_chart(data, cid, mirror):
    return data["charts"][cid + ":m"] if mirror else data["charts"][cid]


def train_arm(arm, approved_probability, resume=True, max_updates=MAX_UPDATES,
              dev=None):
    """One adaptation arm. Resumable per 100 updates; snapshots at the
    frozen schedule; divergence stops the arm preserving the last valid
    artifact."""
    from eval.expressive_manifest import load_run
    run = load_run(E1)
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = run["config"]["seed"]
    data = build_dataset()
    strata = {k: data["train"][k] for k in ("approved", "general")}
    stream = SampleStream(strata, approved_probability,
                          seed=seed * 100 + int(approved_probability * 100))
    student, teacher = initialized_models(ROOT / "flow.pt")
    student.to(dev)
    teacher.to(dev)
    opt = torch.optim.Adam(student.parameters(), LR)
    arm_dir = E1 / f"arm-{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    state_p = arm_dir / "train_state.pt"
    hist_p = arm_dir / "history.json"
    history = {"arm": arm, "p_approved": approved_probability,
               "updates": [], "diverged": False}
    start = 0
    if resume and state_p.exists():
        st = torch.load(state_p)
        student.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["torch_rng"])
        stream.load_state_dict(st["sampler"])
        start = st["update"]
        history = json.loads(hist_p.read_text())
        print(f"[arm-{arm}] resumed at update {start}", flush=True)
    else:
        torch.manual_seed(seed)
    val_sets = _val_sets(data)

    def snapshot(u):
        p = arm_dir / f"snapshot-{u}.pt"
        if not p.exists():
            torch.save({k: v.cpu() for k, v in student.state_dict().items()},
                       p)
        rec = {"update": u, "val": _val_ce(student, val_sets, dev)}
        history["updates"].append(rec)
        print(f"[arm-{arm}] u{u}: val approved {rec['val']['approved_macro']:.4f} "
              f"general {rec['val']['general_macro']:.4f}", flush=True)

    if start == 0:
        snapshot(0)
    u = max(start, 0)
    while u < max_updates:
        student.train()
        xs, ys, gen_mask = [], [], []
        for _ in range(BATCH):
            s = stream.next()
            ch = _resolve_chart(data, s["chart"], s["mirror"])
            x, y = ch["x"], ch["y"]
            i = int(s["crop_u"] * (len(x) - CTX))
            xs.append(x[i:i + CTX])
            ys.append(y[i:i + CTX])
            gen_mask.append(s["stratum"] == "general")
        xb = torch.stack(xs).to(dev)
        yb = torch.stack(ys).to(dev)
        d_oh = F.one_hot(yb[..., 0], 9).float()
        c_oh = F.one_hot(yb[..., 1], DCOL).float()
        d_lg, c_lg, l_lg, k_lg = student(xb, d_oh, c_oh)
        ce = (F.cross_entropy(d_lg.reshape(-1, 9), yb[..., 0].reshape(-1))
              + F.cross_entropy(c_lg.reshape(-1, DCOL),
                                yb[..., 1].reshape(-1))
              + F.cross_entropy(l_lg.reshape(-1, DLAY),
                                yb[..., 2].reshape(-1))
              + F.cross_entropy(k_lg.reshape(-1, MAX_CHAIN),
                                yb[..., 3].reshape(-1)))
        loss = ce
        gm = torch.tensor(gen_mask, device=dev)
        if gm.any():
            with torch.no_grad():
                td, tc, tl, tk = teacher(xb[gm], d_oh[gm], c_oh[gm])
            sd, sc, sl, sk = d_lg[gm], c_lg[gm], l_lg[gm], k_lg[gm]
            kl = (teacher_kl(sd.reshape(-1, 9), td.reshape(-1, 9))
                  + teacher_kl(sc.reshape(-1, DCOL), tc.reshape(-1, DCOL))
                  + teacher_kl(sl.reshape(-1, DLAY), tl.reshape(-1, DLAY))
                  + teacher_kl(sk.reshape(-1, MAX_CHAIN),
                               tk.reshape(-1, MAX_CHAIN))) / 4
            loss = loss + KL_W * kl
        if not torch.isfinite(loss):
            history["diverged"] = True
            hist_p.write_text(json.dumps(history, indent=1))
            print(f"[arm-{arm}] DIVERGED at u{u} — stopping, last valid "
                  "artifacts preserved", flush=True)
            return history
        opt.zero_grad()
        loss.backward()
        opt.step()
        u += 1
        if u % STATE_EVERY == 0 or u == max_updates:
            torch.save({"model": student.state_dict(),
                        "opt": opt.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "sampler": stream.state_dict(), "update": u},
                       state_p)
        if u in SNAPSHOTS:
            snapshot(u)
            hist_p.write_text(json.dumps(history, indent=1))
    hist_p.write_text(json.dumps(history, indent=1))
    from eval.expressive_manifest import load_run as _lr
    sel = select_snapshot(
        history["updates"], 5,
        _lr(E1)["config"]["approved_val_families_n"])
    history["selected"] = sel
    hist_p.write_text(json.dumps(history, indent=1))
    print(f"[arm-{arm}] selected update {sel['update']} "
          f"({sel['metric']}, {sel['value']:.4f})"
          + (" — STEP ZERO: no new checkpoint" if sel["step_zero"] else ""),
          flush=True)
    return history


def _val_sets(data):
    """Fixed start/middle/end CTX crops per val chart, family-tagged; no
    duplicated crops for short charts."""
    sets = {"approved": {}, "general": {}}
    for stratum in sets:
        for fam, cids in data["val"][stratum].items():
            crops = []
            for cid in cids:
                ch = data["charts"][cid]
                x, y = ch["x"], ch["y"]
                n = len(x)
                offs = sorted({0, max(0, (n - CTX) // 2), n - CTX})
                for o in offs:
                    crops.append((x[o:o + CTX], y[o:o + CTX]))
            if crops:
                sets[stratum][fam] = crops
    return sets


def _val_ce(model, val_sets, dev):
    model.eval()
    out = {}
    fam_ces = {"approved": [], "general": []}
    with torch.no_grad():
        for stratum, fams in val_sets.items():
            for fam, crops in fams.items():
                tot = 0.0
                for x, y in crops:
                    xb, yb = x[None].to(dev), y[None].to(dev)
                    d_oh = F.one_hot(yb[..., 0], 9).float()
                    c_oh = F.one_hot(yb[..., 1], DCOL).float()
                    d_lg, c_lg, l_lg, k_lg = model(xb, d_oh, c_oh)
                    tot += float(
                        F.cross_entropy(d_lg.reshape(-1, 9),
                                        yb[..., 0].reshape(-1))
                        + F.cross_entropy(c_lg.reshape(-1, DCOL),
                                          yb[..., 1].reshape(-1))
                        + F.cross_entropy(l_lg.reshape(-1, DLAY),
                                          yb[..., 2].reshape(-1))
                        + F.cross_entropy(k_lg.reshape(-1, MAX_CHAIN),
                                          yb[..., 3].reshape(-1)))
                fam_ces[stratum].append(tot / len(crops))
    a = fam_ces["approved"]
    g = fam_ces["general"]
    out["approved_macro"] = sum(a) / len(a) if a else None
    out["general_macro"] = sum(g) / len(g) if g else None
    allv = a + g
    out["overall_macro"] = sum(allv) / len(allv) if allv else None
    return out


if __name__ == "__main__":
    import sys
    arm = sys.argv[1] if len(sys.argv) > 1 else "A"
    p = {"A": 0.50, "B": 0.75}[arm]
    train_arm(arm, p)
