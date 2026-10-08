"""Learned map critic: human-vs-generated discriminator over event tokens.

Positives are the HAND-APPROVED maps only (groom.APPROVED_DIRS) — the critic
is the style judge, so it learns the user's taste, not the whole corpus.
Negatives are OUR decoder's output on the same songs' rhythms (matched pairs),
so it learns the style gap, not the song.
Used as the best-of-N selection score in convert.py when critic.pt exists.

  .venv/bin/python critic.py gen     # decode negatives on approved maps (~20 min)
  .venv/bin/python critic.py train   # ~5 min, writes critic.pt
  .venv/bin/python critic.py check
"""
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import groom
from groom import (NTOK, APPROVED_DIRS, events_to_xy, out_to_events,
                   groom_notes, load_map)

CRITIC_PT = Path(__file__).parent / "critic.pt"
NEG_PT = Path(__file__).parent / "negatives.pt"
WIN = 128           # events per judged window
D, LAYERS, HEADS = 128, 3, 4
GEN_STRIDE = 1      # approved set is small: decode a negative for every map


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(NTOK, D)
        self.pos = nn.Embedding(WIN, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.1,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tr = nn.TransformerEncoder(layer, LAYERS)
        self.out = nn.Linear(D, 1)

    def forward(self, x):  # [B, n<=WIN, NTOK] -> [B] logit (human-like)
        n = x.shape[1]
        h = self.tr(self.inp(x) + self.pos(torch.arange(n, device=x.device)))
        return self.out(h.mean(1)).squeeze(-1)


def _bpm_of(d):
    info_p = next(p for p in d.iterdir() if p.name.lower() == "info.dat")
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    return info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm", 120)


def map_dirs():
    """Approved maps that are in the TRAIN split (family-deduped) — the
    critic's positives must be split-clean, never dev/val/test/mapper_eval
    (phase 5A). Fails closed via the manifest."""
    import eval.corpus as corpus
    train = set(corpus.train_families_rep("train"))
    return [d for md in APPROVED_DIRS if Path(md).exists()
            for d in sorted(Path(md).iterdir())
            if d.is_dir() and not groom.held_out(d) and str(d) in train]


def _wall_arrays(wall_runs, T):
    wl = torch.zeros(T, dtype=torch.bool)
    wr = torch.zeros(T, dtype=torch.bool)
    for s0, ln, col in wall_runs:
        (wl if col == 0 else wr)[s0:s0 + ln] = True
    return wl, wr


def gen():
    """Decode our pipeline on human maps' rhythms -> matched negatives."""
    negs = torch.load(NEG_PT) if NEG_PT.exists() else {}
    dirs = map_dirs()[::GEN_STRIDE]
    for i, d in enumerate(dirs):
        if d.name in negs:
            continue
        m = load_map(d)
        if m is None:
            continue
        inp = m[0]
        steps = set(torch.nonzero(inp[:, 0] > 0).flatten().tolist())
        step_ms = 60000.0 / _bpm_of(d) / groom.STEPS_PER_BEAT
        try:  # seed 1: silences the replay print, decorrelates from convert
            out, walls = groom_notes(steps, len(inp), step_ms,
                                     afeat=inp[:, 9:13], seed=1)
        except Exception as e:
            print(f"  decode failed {d.name[:40]}: {e}")
            continue
        negs[d.name] = (out, walls, len(inp))
        if (i + 1) % 20 == 0:
            torch.save(negs, NEG_PT)
            print(f"{i + 1}/{len(dirs)} maps decoded")
    torch.save(negs, NEG_PT)
    print(f"negatives on disk: {len(negs)}")


def build_pairs():
    """[(human tokens X, generated tokens X)] per map with a negative."""
    negs = torch.load(NEG_PT)
    pairs = []
    for d in map_dirs():
        if d.name not in negs:
            continue
        m = load_map(d)
        if m is None:
            continue
        inp, _, events, wl, wr = m
        energy = inp[:, 12]
        xp, _ = events_to_xy(events, wl, wr, energy)
        out, walls, T = negs[d.name]
        wla, wra = _wall_arrays(walls, T)
        xn, _ = events_to_xy(out_to_events(out), wla, wra, energy)
        if len(xp) > WIN and len(xn) > WIN:
            pairs.append((xp, xn))
    print(f"pairs: {len(pairs)}")
    return pairs


def train():
    pairs = build_pairs()
    rng = random.Random(0)
    rng.shuffle(pairs)
    nval = max(2, len(pairs) // 10)
    val, tr = pairs[:nval], pairs[nval:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Critic().to(dev)
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    xva, yva = [], []
    for xp, xn in val:  # fixed val windows
        for x, y in ((xp, 1.0), (xn, 0.0)):
            for i in (0, (len(x) - WIN) // 2, len(x) - WIN):
                xva.append(x[i:i + WIN])
                yva.append(y)
    xva = torch.stack(xva).to(dev)
    yva = torch.tensor(yva).to(dev)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(80):
        model.train()
        for _ in range(30):
            xs, ys = [], []
            for _ in range(16):
                xp, xn = rng.choice(tr)
                for x, y in ((xp, 1.0), (xn, 0.0)):
                    i = rng.randrange(len(x) - WIN)
                    xs.append(x[i:i + WIN])
                    ys.append(y)
            loss = F.binary_cross_entropy_with_logits(
                model(torch.stack(xs).to(dev)), torch.tensor(ys).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lg = model(xva)
            vloss = F.binary_cross_entropy_with_logits(lg, yva).item()
            acc = ((lg > 0) == (yva > 0.5)).float().mean().item()
        if vloss < best - 1e-4:
            best, patience, best_state = vloss, 0, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 5 == 0 or patience > 10:
            print(f"[critic] epoch {epoch}: val loss {vloss:.4f} acc {acc:.2f}")
        if patience > 10:
            break
    torch.save(best_state, CRITIC_PT)
    print(f"saved {CRITIC_PT} (best val loss {best:.4f})")


_C = None


def score(out, wall_runs, T, afeat):
    """Mean human-likeness LOGIT of a decoded map, windowed. Logits, not
    probabilities: the critic separates classes near-perfectly, so sigmoids
    saturate and ranking among candidates would happen in noise."""
    global _C
    if _C is None:
        _C = Critic()
        _C.load_state_dict(torch.load(CRITIC_PT, map_location="cpu"))
        _C.eval()
    wl, wr = _wall_arrays(wall_runs, T)
    x, _ = events_to_xy(out_to_events(out), wl, wr,
                        afeat[:, 3] if afeat is not None else None)
    logits = []
    with torch.no_grad():
        for i in range(0, max(1, len(x) - WIN + 1), WIN // 2):
            logits.append(float(_C(x[i:i + WIN][None])))
    return sum(logits) / len(logits)


def check():
    assert NEG_PT.exists(), "run `critic.py gen` first"
    pairs = build_pairs()
    xp, xn = pairs[0]
    lg = Critic()(xp[:WIN][None])
    assert lg.shape == (1,)
    if CRITIC_PT.exists():
        negs = torch.load(NEG_PT)
        key = next(iter(negs))
        out, walls, T = negs[key]
        s = score(out, walls, T, None)
        print(f"critic score of a known-generated map: {s:.2f} (low = working)")
    print("check ok")


if __name__ == "__main__":
    {"gen": gen, "train": train, "check": check}[sys.argv[1] if len(sys.argv) > 1
                                                 else "check"]()
