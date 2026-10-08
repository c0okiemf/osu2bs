"""Thinning-effect decision analysis over the pilot MI subset (Packet D).

review's provisional rule for the paired thinned/dense arm: measure the
fraction of union occupied grid cells that `_hits` thinning changes, per song
and per 10-second window. If every supported subgroup has median changed
occupancy <1%, no song has a 10s window >5%, and the changed cells reveal no
systematic protected-accent/rest problem, DROP the separate thinning-trained
arm as low-information. Otherwise concentrate the paired comparison where it
bites.

  .venv/bin/python -m eval.pilot_thinning
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import convert

MI = Path(__file__).parent.parent / "experiments" / "pilot-mi"
OUT = Path(__file__).parent.parent / "experiments" / "pilot-thinning.json"


def _cells(grid, hits):
    return {grid.snap(h[0])[0] for h in hits}


def analyze(gen_osu):
    meta, objects, bpm, offset = convert.parse_osu(gen_osu)
    steps, T, step_ms, off2, grid = convert.grid_steps(
        objects, bpm, offset, meta.get("_timing"))
    un = _cells(grid, convert._hits(objects, thin=False))
    th = _cells(grid, convert._hits(objects, thin=True))
    changed = un ^ th
    union = un | th
    # per-10s-window changed fraction
    span = max((grid.time(s) for s in union), default=0)
    win = 10000.0
    nb = int(span / win) + 1
    wc = [0] * nb
    wt = [0] * nb
    for s in union:
        w = min(int(grid.time(s) / win), nb - 1)
        wt[w] += 1
        if s in changed:
            wc[w] += 1
    max_win = max((c / t for c, t in zip(wc, wt) if t >= 10), default=0.0)
    return {
        "grid": grid.decision.get("grid"),
        "unthinned_cells": len(un), "thinned_cells": len(th),
        "changed_cells": len(changed),
        "changed_pct_union": round(100 * len(changed) / max(1, len(union)), 3),
        "max_10s_window_changed_pct": round(100 * max_win, 2),
    }


def main():
    rows = []
    for d in sorted(MI.iterdir()):
        osu = d / "gen.osu"
        prov = d / "provenance.json"
        if not (osu.exists() and prov.exists()):
            continue
        p = json.loads(prov.read_text())
        a = analyze(osu)
        a.update(family=p["family"], genre=p["genre"], split=p["split"],
                 song=p["song"])
        rows.append(a)
    from collections import defaultdict
    bygenre = defaultdict(list)
    for r in rows:
        bygenre[r["genre"]].append(r["changed_pct_union"])

    def median(xs):
        xs = sorted(xs)
        if not xs:
            return 0.0
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    subgroup_med = {g: round(median(v), 3) for g, v in bygenre.items()}
    worst_win = max((r["max_10s_window_changed_pct"] for r in rows), default=0)
    all_sub_under_1 = all(v < 1.0 for v in subgroup_med.values())
    no_win_over_5 = worst_win <= 5.0
    drop_arm = all_sub_under_1 and no_win_over_5
    verdict = {
        "n_songs": len(rows),
        "subgroup_median_changed_pct": subgroup_med,
        "worst_10s_window_changed_pct": worst_win,
        "rule_all_subgroups_under_1pct": all_sub_under_1,
        "rule_no_10s_window_over_5pct": no_win_over_5,
        "recommendation": ("DROP the thinning-trained arm as low-information "
                           "(review rule met); focus the paired experiment on "
                           "scheduling/hand-allocation/rests/section-dynamics"
                           if drop_arm else
                           "KEEP the thinning arm; some subgroup/window exceeds "
                           "the threshold — concentrate the comparison there"),
        "caveat": "accent/rest-boundary inspection of changed cells still "
                  "required before final drop; this is the aggregate gate only",
    }
    OUT.write_text(json.dumps({"verdict": verdict, "songs": rows}, indent=1))
    print("=== thinning effect on fixed grid (MI pilot subset) ===")
    for r in sorted(rows, key=lambda x: -x["changed_pct_union"]):
        print(f"  {r['genre']:11s} {str(r['song'])[:30]:30s} "
              f"changed {r['changed_pct_union']:5.2f}% union  "
              f"max10s {r['max_10s_window_changed_pct']:5.2f}%  ({r['grid']})")
    print(f"\nsubgroup medians: {subgroup_med}")
    print(f"worst 10s window: {worst_win}%")
    print(f"-> {verdict['recommendation']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
