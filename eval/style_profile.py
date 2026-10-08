"""Style profile: direction variety, vertical-flip runs, density contrast."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home()) + "/app/osu2bs")
from parity import ang_dist


def profile(paths):
    vert = diag = lat = heads = flips = trans = 0
    runs, dbl, times_all = [], 0, 0
    win_stats = []
    for path in paths:
        try:
            d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        notes = d.get("_notes") or [
            {"_time": n.get("b", 0), "_type": n.get("c"),
             "_cutDirection": n.get("d", 8)} for n in d.get("colorNotes", [])]
        notes = [n for n in notes if n.get("_type") in (0, 1) and 0 <= n.get("_cutDirection", -1) <= 8]
        hd = sorted((n["_time"], n["_type"], n["_cutDirection"]) for n in notes
                    if n["_cutDirection"] != 8)
        if len(hd) < 100:
            continue
        heads += len(hd)
        by_t = {}
        for t, h, c in hd:
            vert += c in (0, 1)
            diag += c in (4, 5, 6, 7)
            lat += c in (2, 3)
            by_t.setdefault(round(t, 3), []).append(h)
        dbl += sum(1 for v in by_t.values() if len(v) == 2)
        times_all += len(by_t)
        last, run = {}, {}
        for t, h, c in hd:
            if h in last:
                pt, pc = last[h]
                if t - pt <= 1.0:
                    trans += 1
                    if ang_dist(c, pc) == 180:
                        flips += 1
                        if c in (0, 1) and pc in (0, 1):
                            run[h] = run.get(h, 1) + 1
                        else:
                            if run.get(h, 0) >= 2:
                                runs.append(run[h])
                            run[h] = 1
                    else:
                        if run.get(h, 0) >= 2:
                            runs.append(run[h])
                        run[h] = 1
                else:  # long gap breaks the run — flush, don't carry it over
                    if run.get(h, 0) >= 2:
                        runs.append(run[h])
                    run.pop(h, None)
            last[h] = (t, c)
        runs.extend(r for r in run.values() if r >= 2)  # EOF flush
        # density contrast: notes per 4-beat window across the mapped span
        t0, t1 = hd[0][0], hd[-1][0]
        nw = max(1, int((t1 - t0) / 4))
        counts = [0] * nw
        for t, _, _ in hd:
            counts[min(nw - 1, int((t - t0) / 4))] += 1
        m = sum(counts) / nw
        var = sum((c - m) ** 2 for c in counts) / nw
        cv = var ** 0.5 / m if m else 0
        rest = sum(1 for c in counts if c <= 1) / nw
        win_stats.append((cv, rest))
    n = max(1, heads)
    runs.sort()
    longrun = sum(1 for r in runs for _ in range(r) if r >= 6) / max(1, trans)
    cv = sum(w[0] for w in win_stats) / max(1, len(win_stats))
    rest = sum(w[1] for w in win_stats) / max(1, len(win_stats))
    return (f"vert {100*vert/n:4.1f}%  diag {100*diag/n:4.1f}%  lat {100*lat/n:4.1f}%  "
            f"180flip {100*flips/max(1,trans):4.1f}%  vrun-med {runs[len(runs)//2] if runs else 0}  "
            f"in-longrun(>=6) {100*longrun:4.1f}%  densityCV {cv:.2f}  rest4b {100*rest:4.1f}%")


if __name__ != "__main__":
    SETS = {}
else:
    SETS = {
    "GEMS": [str(Path.home()) + "/app/beat-saber-map-gen/input/bytrius/19909*/ExpertPlus*.dat",
             str(Path.home()) + "/app/beat-saber-map-gen/input/bytrius/46d4*/ExpertPlus*.dat",
             str(Path.home()) + "/app/beat-saber-map-gen/input/input/25f *(*/ExpertPlus*.dat",
             str(Path.home()) + "/app/beat-saber-map-gen/input/input/3741*/ExpertPlus*.dat"],
    "approved": [str(Path.home()) + "/app/beat-saber-map-gen/input/*/*/ExpertPlus*.dat"],
    "generated": [str(Path.home()) + "/app/osu2bs/out/*/ExpertPlus.dat"],
}
for name, globs in SETS.items():
    files = [f for g in globs for f in glob.glob(g)]
    print(f"{name:10s} ({len(files):3d} maps) {profile(files)}")
