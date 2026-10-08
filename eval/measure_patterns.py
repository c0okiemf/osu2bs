"""Rate of (a) up@bottom-lane -> down@top-lane same-hand within a beat and
(b) 90-degree same-hand turns within a beat, per ExpertPlus.dat glob."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home()) + "/app/osu2bs")
from parity import ang_dist

UP, DOWN = {0, 4, 5}, {1, 6, 7}


def rates(path):
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    notes = d.get("_notes") or [
        {"_time": n.get("b", 0), "_type": n.get("c"), "_lineIndex": n.get("x"),
         "_lineLayer": n.get("y"), "_cutDirection": n.get("d")}
        for n in d.get("colorNotes", [])]
    notes = [n for n in notes if n.get("_type") in (0, 1)
             and n.get("_cutDirection", 8) != 8]
    notes.sort(key=lambda n: n["_time"])
    last, sh, t90, trans = {}, 0, 0, 0
    for n in notes:
        p = last.get(n["_type"])
        if p and n["_time"] - p["_time"] <= 1.0:
            trans += 1
            if ang_dist(n["_cutDirection"], p["_cutDirection"]) == 90:
                t90 += 1
            if (p["_cutDirection"] in UP and p["_lineLayer"] == 0
                    and n["_cutDirection"] in DOWN and n["_lineLayer"] == 2):
                sh += 1
        last[n["_type"]] = n
    return sh, t90, trans, len(notes)


if __name__ == "__main__":
    tot = [0, 0, 0, 0]
    files = sorted(Path(p) for g in sys.argv[1:] for p in __import__("glob").glob(g))
    for f in files:
        try:
            r = rates(f)
        except Exception:
            continue
        tot = [a + b for a, b in zip(tot, r)]
        if len(files) <= 20:
            print(f"{f.parent.name[:40]:40s} shoulder {r[0]:3d}   "
                  f"90deg {r[1]:4d}/{r[2]:4d}")
    n = len(files)
    print(f"TOTAL ({n} maps): shoulder {tot[0]} ({tot[0]/max(1,n):.1f}/map), "
          f"90deg {tot[1]}/{tot[2]} ({100*tot[1]/max(1,tot[2]):.1f}% of in-combo "
          f"transitions), notes {tot[3]}")
