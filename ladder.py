"""Learned difficulty ladder: how much harder each tier is than the last.

Fit over every multi-difficulty corpus map: per-rank swing rates give
log-rate increments per rank unit, regressed against rank — the gap SHRINKS
as difficulty rises (Easy->Normal ~1.45x, Expert->Expert+ ~1.26x), so a
constant per-tier multiplier mis-extrapolates. The fitted line extrapolates
Expert++, +++, ... indefinitely (ranks 11, 13, ...). NJS fits the same way.

  .venv/bin/python ladder.py fit   # writes ladder.json
Used by convert.diff_spec at decode.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
LADDER = HERE / "ladder.json"
RANKS = {"Easy": 1, "Normal": 3, "Hard": 5, "Expert": 7, "ExpertPlus": 9}
MIN_G = 0.02  # per-rank log increment floor: tiers never stop getting harder


def _rates(d):
    info_p = next((p for p in d.iterdir() if p.name.lower() == "info.dat"), None)
    if not info_p:
        return {}
    try:
        info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    bpm = info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm")
    if not bpm:
        return {}
    out = {}
    for s in info.get("_difficultyBeatmapSets", []):
        if s.get("_beatmapCharacteristicName", "Standard") != "Standard":
            continue
        for dm in s.get("_difficultyBeatmaps", []):
            name, f = dm.get("_difficulty"), dm.get("_beatmapFilename")
            if name not in RANKS or not f or not (d / f).exists():
                continue
            try:
                dat = json.loads((d / f).read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            notes = dat.get("_notes") or [
                {"_time": n.get("b", 0), "_type": n.get("c"),
                 "_cutDirection": n.get("d", 8)}
                for n in dat.get("colorNotes", [])]
            ts = sorted(n["_time"] for n in notes if n.get("_type") in (0, 1)
                        and n.get("_cutDirection", 8) != 8)
            if len(ts) < 60:
                continue
            span = (ts[-1] - ts[0]) * 60 / bpm
            if span > 30:
                njs = dm.get("_noteJumpMovementSpeed") or 0
                out[RANKS[name]] = (len(ts) / span, njs)
    return out


def _fit_dirs():
    """Train-split family representatives from the corpus manifest — the
    ladder is a learned artifact and must never see dev/test/mapper_eval
    (phase 5A). FAILS CLOSED: a missing/stale manifest raises rather than
    falling back to a leaky all-dirs scan (review finding 5)."""
    import eval.corpus as corpus
    return [Path(p) for p in corpus.train_families_rep("train")]


def fit():
    incs, njss = [], {r: [] for r in RANKS.values()}
    for d in _fit_dirs():
        per = _rates(d)
        for r, (rate, njs) in per.items():
            if njs:
                njss[r].append(njs)
        ranks = sorted(per)
        for a, b in zip(ranks, ranks[1:]):
            g = (math.log(per[b][0]) - math.log(per[a][0])) / (b - a)
            incs.append(((a + b) / 2, g))
    # per-rank log increment vs rank midpoint, least squares
    n = len(incs)
    mx = sum(x for x, _ in incs) / n
    my = sum(y for _, y in incs) / n
    beta = (sum((x - mx) * (y - my) for x, y in incs)
            / max(1e-9, sum((x - mx) ** 2 for x, _ in incs)))
    alpha = my - beta * mx
    med_njs = {r: sorted(v)[len(v) // 2] for r, v in njss.items() if v}
    # NJS vs rank, least squares over medians (extrapolates above rank 9)
    xs, ys = list(med_njs), [med_njs[r] for r in med_njs]
    mnx, mny = sum(xs) / len(xs), sum(ys) / len(ys)
    nb = (sum((x - mnx) * (y - mny) for x, y in zip(xs, ys))
          / max(1e-9, sum((x - mnx) ** 2 for x in xs)))
    na = mny - nb * mnx
    LADDER.write_text(json.dumps({
        "alpha": alpha, "beta": beta, "pairs": n,
        "njs_a": na, "njs_b": nb, "njs_med": med_njs}, indent=2))
    print(f"ladder.json: g(rank) = {alpha:.4f} {beta:+.5f}*rank "
          f"({n} tier pairs), njs = {na:.1f} {nb:+.2f}*rank")


_L = None


def _load():
    global _L
    if _L is None:
        _L = json.loads(LADDER.read_text())
    return _L


def g(rank):
    """Fitted per-rank log-rate increment at a rank midpoint (floored)."""
    l = _load()
    return max(MIN_G, l["alpha"] + l["beta"] * rank)


def scale(rank):
    """rate(rank) / rate(9), from the fitted increments (any rank)."""
    if rank == 9:
        return 1.0
    lo, hi = sorted((rank, 9))
    s = math.exp(sum(g(r + 0.5) for r in range(lo, hi)))
    return s if rank > 9 else 1.0 / s


def njs(rank):
    l = _load()
    m = l["njs_med"].get(str(rank)) or l["njs_med"].get(rank)
    return int(round(m if m else l["njs_a"] + l["njs_b"] * rank))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit()
    else:
        for r in (1, 3, 5, 7, 9, 11, 13, 15):
            print(f"rank {r:2d}: scale {scale(r):.2f}  njs {njs(r)}")
