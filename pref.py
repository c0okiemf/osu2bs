"""Q4: small preference tie-breaker (spec §7, preference bullet).

A two-layer MLP over explicit whole-chart summary features, fit on
TRAIN-only matched pairs: (human chart, deliberately degraded variant of the
SAME chart — same audio, same span, same padding). Six construction-known
degradations: hard-invalid motion, repeated-one-pattern, flattened quiet
sections, randomized valid directions, poor seams, density-matched doubles
saturation. Bradley-Terry pairwise loss.

Gates: >=90% discrimination on HELD-OUT (val-family) construction-known
degradation pairs; >=70% on held-out metric-dominance generated pairs
(candidates from real decodes where one dominates the other on flags, pairs
and dup8 simultaneously). This is NOT human-preference accuracy. The score
may break ties only after physical/workload/structure gates, and critic.pt
is never replaced.

Artifacts: experiments/quality-v1/q4-pref/ only.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "quality-v1" / "q4-pref"
SEED = 20260921


class UniformGrid:
    """Minimal grid shim for human charts saved on a uniform 1/4 grid."""

    def __init__(self, step_ms):
        self.step_ms = float(step_ms)

    def time(self, s):
        return s * self.step_ms


FEATS = ["flags_00", "flags_25", "flags_50", "flags_75", "narrow_k",
         "conv_k", "broad_k", "opp_k", "max4gram", "tokrun", "dotburden_k",
         "p95speed", "vert", "longrun", "lat", "dbl_share", "heads_per_s"]


def feat_vec(raw, walls, grid, bpm):
    """Explicit whole-chart summaries (count features per 1k heads)."""
    from eval.quality_metrics import quality_metrics
    m = quality_metrics(raw, walls, grid, bpm)
    heads = max(1, m["n_heads"])
    k = 1000.0 / heads
    ts = sorted(grid.time(s) for s, *_ in raw)
    span = max((ts[-1] - ts[0]) / 1000.0, 1e-6) if len(ts) > 1 else 1e-6
    by_step = {}
    for s, h, *_ in raw:
        by_step.setdefault(s, set()).add(h)
    dbl = sum(1 for hs in by_step.values() if len(hs) == 2) \
        / max(1, len(by_step))
    fx = m["flags_by_ext"]
    exts = sorted(fx)
    return torch.tensor([
        *(float(fx[e]) for e in exts),
        m["narrow"] * k, m["converging"] * k, m["broad"] * k,
        m["opposite_horizontal"] * k,
        m["four_gram"]["max_4gram_share"],
        float(m["longest_token_run"]["length"]
              if isinstance(m["longest_token_run"], dict)
              else m["longest_token_run"]),
        m["unknown_dot_burden"] * k,
        m["p95_speed_unrounded"],
        m["style"]["vert"], m["style"]["longrun"], m["style"]["lat"],
        dbl, heads / span], dtype=torch.float32)


# ---------------- construction-known degradations ----------------
# Each takes raw [(s,h,c,l,d)] + rng and returns a SAME-SPAN variant.

def deg_invalid_motion(raw, rng):
    """Teleport ~15% of notes across the grid — extreme one-step speeds."""
    out = list(raw)
    for i in range(len(out)):
        if rng.random() < 0.15:
            s, h, c, l, d = out[i]
            out[i] = (s, h, 3 - c, 2 - l, d)
    return out


def deg_one_pattern(raw, rng):
    """Overwrite geometry with one repeated pattern (down-cuts, cols 1/2)."""
    return [(s, h, 1 if h == 0 else 2, 0, 1) for s, h, _c, _l, _d in raw]


def deg_flatten_quiet(raw, rng, win=32):
    """Fill empty 8-beat windows with filler notes — kills rests."""
    steps = {s for s, *_ in raw}
    if not steps:
        return list(raw)
    out = list(raw)
    for w0 in range(0, max(steps) + 1, win):
        if not any(w0 <= s < w0 + win for s in steps):
            for s in range(w0, w0 + win, 4):
                out.append((s, s // 4 % 2, 1 + s // 4 % 2, 0, 1))
    return sorted(out)


def deg_random_dirs(raw, rng):
    """Uniformly random (valid-range) cut directions — parity chaos."""
    return [(s, h, c, l, rng.randrange(9)) for s, h, c, l, _d in raw]


def deg_poor_seams(raw, rng, win=32):
    """Mirror columns in alternating 8-beat blocks — hard discontinuities at
    every seam."""
    return [(s, h, 3 - c if (s // win) % 2 else c, l, d)
            for s, h, c, l, d in raw]


def deg_doubles_saturation(raw, rng):
    """Density-matched doubles saturation: pair up singles into doubles while
    dropping the notes in between — same total note count."""
    out = []
    singles = [n for n in raw]
    i = 0
    while i + 1 < len(singles):
        s, h, c, l, d = singles[i]
        out.append((s, 0, 1, l, d))
        out.append((s, 1, 2, l, d))
        i += 2
    if i < len(singles):
        out.append(singles[i])
    return sorted(out)


DEGRADATIONS = {"invalid_motion": deg_invalid_motion,
                "one_pattern": deg_one_pattern,
                "flatten_quiet": deg_flatten_quiet,
                "random_dirs": deg_random_dirs,
                "poor_seams": deg_poor_seams,
                "doubles_sat": deg_doubles_saturation}


class PrefMLP(nn.Module):
    def __init__(self, n_in=len(FEATS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _family_pairs(dirs, tag, rng):
    """(preferred_feats, degraded_feats, kind) per family x degradation.
    Idempotent per-family cache (1-hour rule): a killed pass resumes."""
    import hashlib
    import groom
    cache_dir = OUT / "pairs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs, skipped = [], 0
    for dp in dirs:
        cp = cache_dir / (hashlib.sha256(dp.encode()).hexdigest()[:16] + ".pt")
        if cp.exists():
            rec = torch.load(cp)
            pairs += rec["pairs"]
            skipped += rec["skipped"]
            continue
        fam_pairs, fam_skip = _one_family(dp, tag, rng)
        tmp = cp.with_suffix(".tmp")
        torch.save({"pairs": fam_pairs, "skipped": fam_skip}, tmp)
        tmp.replace(cp)
        pairs += fam_pairs
        skipped += fam_skip
    return pairs, skipped


def _one_family(dp, tag, rng):
    import groom
    d = Path(dp)
    samples = groom.load_map_all(d)
    if not samples:
        return [], 1
    name = sorted(samples, key=lambda n: len(samples[n][2]))[-1]
    _inp, pres, events, wl, wr = samples[name][:5]
    info_p = next((p for p in d.iterdir()
                   if p.name.lower() == "info.dat"), None)
    if info_p is None:
        return [], 1
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    bpm = info.get("_beatsPerMinute")
    if not bpm:
        return [], 1
    grid = UniformGrid(60000.0 / float(bpm) / 4)
    raw = [(e[0], e[1], e[3], e[4], e[2]) for e in events]
    if len(raw) < 32:
        return [], 1
    try:
        base = feat_vec(raw, [], grid, float(bpm))
    except Exception:
        return [], 1
    pairs = []
    for kind, fn in DEGRADATIONS.items():
        variant = fn(raw, rng)
        if variant == raw:          # identity (e.g. no quiet window to fill)
            continue
        try:
            bad = feat_vec(variant, [], grid, float(bpm))
        except Exception:
            continue
        if torch.equal(bad, base):   # metrically indistinguishable
            continue
        pairs.append({"good": base, "bad": bad, "kind": kind,
                      "fam": str(dp), "tag": tag})
    return pairs, 0


def fit_pref(max_steps=2000):
    from eval import corpus
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "pairs.pt"
    if cache.exists():
        data = torch.load(cache)
    else:
        tr, sk_t = _family_pairs(sorted(corpus.train_families_rep("train")),
                                 "train", rng)
        va, sk_v = _family_pairs(sorted(corpus.train_families_rep("val")),
                                 "val", rng)
        data = {"train": tr, "val": va,
                "skipped": {"train": sk_t, "val": sk_v}}
        torch.save(data, cache)
    tr, va = data["train"], data["val"]
    # spec: training pairs = current-policy candidate pairs + degraded
    # variants. Train-split metric-dominance pairs supply the former.
    tr = tr + dominance_pairs("train", n_songs=30)
    print(f"[pref] pairs: train {len(tr)} (incl. dominance) val {len(va)}",
          flush=True)
    g = torch.stack([p["good"] for p in tr])
    b = torch.stack([p["bad"] for p in tr])
    mu = torch.cat([g, b]).mean(dim=0)
    sd = torch.cat([g, b]).std(dim=0).clamp(min=1e-6)
    model = PrefMLP()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    zg, zb = (g - mu) / sd, (b - mu) / sd
    for step in range(max_steps):
        idx = torch.tensor(rng.sample(range(len(tr)),
                                      min(64, len(tr))))
        diff = model(zg[idx]) - model(zb[idx])
        loss = F.softplus(-diff).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    def acc(pairs):
        if not pairs:
            return None, {}
        gg = (torch.stack([p["good"] for p in pairs]) - mu) / sd
        bb = (torch.stack([p["bad"] for p in pairs]) - mu) / sd
        with torch.no_grad():
            ok = (model(gg) > model(bb))
        by_kind = {}
        for p, o in zip(pairs, ok.tolist()):
            by_kind.setdefault(p["kind"], []).append(o)
        return float(ok.float().mean()), \
            {k: round(sum(v) / len(v), 3) for k, v in sorted(by_kind.items())}

    tr_acc, tr_kinds = acc(tr)
    va_acc, va_kinds = acc(va)
    gate = va_acc is not None and va_acc >= 0.90
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd,
                "feats": FEATS, "seed": SEED}, OUT / "pref.pt")
    report = {"train_acc": tr_acc, "train_by_kind": tr_kinds,
              "val_acc": va_acc, "val_by_kind": va_kinds,
              "gate_90_construction": gate,
              "n_train_pairs": len(tr), "n_val_pairs": len(va),
              "note": "metric-dominance generated-pair gate evaluated "
                      "separately (needs decodes); critic.pt untouched"}
    (OUT / "pref_report.json").write_text(json.dumps(report, indent=1))
    print(f"[pref] train {tr_acc:.3f} val {va_acc:.3f} "
          f"{'PASS' if gate else 'FAIL'} (>=0.90 construction) "
          f"by-kind {va_kinds}", flush=True)
    return report


def _dominates(a, b):
    fa, fb = a["metrics"]["flags_by_ext"], b["metrics"]["flags_by_ext"]
    le = all(fa[e] <= fb[e] for e in fa)
    pa = a["metrics"]["narrow"] + a["metrics"]["conv"]
    pb = b["metrics"]["narrow"] + b["metrics"]["conv"]
    da, db = a["metrics"]["dup8"], b["metrics"]["dup8"]
    strict = sum(fa.values()) < sum(fb.values()) or pa < pb or da < db
    return le and pa <= pb and da <= db and strict


def dominance_candidates(split, n_songs):
    """Real decodes on MI inputs of `split` families: per-song candidate
    metric+feature bundles (idempotent per-song cache). Used to BUILD
    training pairs (split=train — 'matched candidate pairs from current
    policies') and to EVALUATE gate 2 (split=val, held out)."""
    import torch as _t
    _t.set_num_threads(4)
    from convert import convert_groomed, grid_steps, parse_osu
    from eval.clean_rhythm import _cand_metrics
    mi_root = ROOT / "experiments" / "quality-v1" / "q3-mi"
    cache_dir = OUT / f"dominance-{split}"
    legacy = OUT / "dominance"                 # first val run's cache name
    if split == "val" and legacy.exists() and not cache_dir.exists():
        legacy.rename(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    songs = []
    for fam in sorted(mi_root.iterdir()):
        prov_p = fam / "provenance.json"
        if not (fam / "gen.osu").exists() or not prov_p.exists():
            continue
        prov = json.loads(prov_p.read_text())
        if prov.get("split") == split:
            songs.append((fam, Path(prov["corpus_dir"])))
    out = []
    for fam, corpus_dir in songs[:n_songs]:
        cp = cache_dir / (fam.name + ".pt")
        if cp.exists():
            out.append((fam.name, torch.load(cp)["cands"]))
            continue
        audio = next((p for p in corpus_dir.iterdir()
                      if p.suffix in (".egg", ".ogg", ".mp3")), None)
        if audio is None:
            continue
        _m, objects, bpm, offset = parse_osu(fam / "gen.osu")
        _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
        recs = []
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off",
                        collect=recs, thin=True, calibrate=True)
        beat_ms = 60000.0 / bpm
        cands = [{"metrics": _cand_metrics(r["notes"], r["motion"], grid,
                                           beat_ms),
                  "feats": feat_vec(r["notes"], r["walls"], grid, bpm)}
                 for r in recs]
        tmp = cp.with_suffix(".tmp")
        torch.save({"cands": cands}, tmp)
        tmp.replace(cp)
        out.append((fam.name, cands))
        print(f"  [dom-{split}] {fam.name}: {len(cands)} cands", flush=True)
    return out


def dominance_pairs(split, n_songs):
    pairs = []
    for fam_name, cands in dominance_candidates(split, n_songs):
        for i in range(len(cands)):
            for j in range(len(cands)):
                if i != j and _dominates(cands[i], cands[j]):
                    pairs.append({"good": cands[i]["feats"],
                                  "bad": cands[j]["feats"],
                                  "kind": "dominance", "fam": fam_name,
                                  "tag": split})
    return pairs


def dominance_eval(n_songs=10):
    """Held-out gate 2: >=70% of val metric-dominance pairs ordered
    correctly under the saved model."""
    d = torch.load(OUT / "pref.pt")
    model = PrefMLP()
    model.load_state_dict(d["state"])
    model.eval()
    mu, sd = d["mu"], d["sd"]
    pairs = dominance_pairs("val", n_songs)
    n_ok = 0
    for p in pairs:
        with torch.no_grad():
            n_ok += int(float(model(((p["good"] - mu) / sd)[None]))
                        > float(model(((p["bad"] - mu) / sd)[None])))
    acc = n_ok / len(pairs) if pairs else None
    gate = acc is not None and acc >= 0.70
    rep = json.loads((OUT / "pref_report.json").read_text())
    rep["dominance"] = {"n_pairs": len(pairs), "acc": acc,
                        "gate_70_dominance": gate}
    (OUT / "pref_report.json").write_text(json.dumps(rep, indent=1))
    print(f"[pref] dominance: {n_ok}/{len(pairs)} = "
          f"{acc if acc is None else round(acc, 3)} "
          f"{'PASS' if gate else 'FAIL'} (>=0.70)", flush=True)
    return rep


def score(raw, walls, grid, bpm):
    """Tie-break score under the saved model — used ONLY after hard/
    physical/workload gates."""
    d = torch.load(OUT / "pref.pt")
    m = PrefMLP()
    m.load_state_dict(d["state"])
    m.eval()
    z = (feat_vec(raw, walls, grid, bpm) - d["mu"]) / d["sd"]
    with torch.no_grad():
        return float(m(z[None]))


if __name__ == "__main__":
    import sys
    if "dominance" in sys.argv:
        dominance_eval()
    else:
        fit_pref()
