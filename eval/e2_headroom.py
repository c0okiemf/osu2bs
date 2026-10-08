"""E2 cached-candidate selection-headroom audit .

Every saved E2 attempt is measured with the unchanged P1-P4 + protections.
Exhaustive oracle: can choosing ONE actual candidate per song satisfy the
frozen six-song rule against BOTH B0 and the matched control? Reported for
admitted-only and all-ok pools, next to the existing selector's choice.
Diagnostic upper bound only: never a selector, never promotion evidence.
"""
import itertools
import json

from eval.e1_machine_ab import E1, OUT, _grid, decide, measure

PROPS = ("P1", "P2", "P3")


def _passes(per_song):
    """Frozen rule over per-song decide() dicts."""
    c = {k: sum(d[k] for d in per_song) for k in PROPS + ("guard",)}
    prot = sum(d["protect"] for d in per_song)
    return (all(c[k] >= 5 for k in PROPS) and c["guard"] >= 5
            and prot == len(per_song)), c


def run():
    from qa.certificates import load_speed_warning
    from qa.neighbours import bank_from_role
    cfg = json.loads((E1 / "run.json").read_text())["config"]
    thr = json.loads((E1.parent.parent / "qa-v4/comparator-v2/development/"
                                         "threshold.json").read_text())
    bank, _r = bank_from_role(identity="qa-train-v2")
    warn = load_speed_warning()
    part = E1 / "partial"
    songs = {}
    for song in cfg["songs"]:
        sid = song["song"]
        grid = _grid(song)
        b0 = json.loads((OUT / f"{sid}.b0.json").read_text())
        ctrl_p = OUT / f"{sid}.armctrl.json"
        ctrl = json.loads(ctrl_p.read_text()) if ctrl_p.exists() else b0
        sel = json.loads((part / f"{sid}:armE2.json").read_text())[
            "selection"]
        cands = []
        for sd in range(6):
            a = json.loads((part / f"{sid}:armE2:s{sd}.json").read_text())
            if not a.get("ok"):
                continue
            m = measure(sid, f"armE2s{sd}", a["notes"], grid, bank, thr, warn)
            cands.append({"id": f"s{sd}", "admitted": bool(a.get("eligible")),
                          "vs_b0": decide(b0, m), "vs_ctrl": decide(ctrl, m),
                          "props": m["props"]})
        songs[sid] = {"selected": sel.get("id"), "cands": cands}
    report = {"note": "diagnostic oracle upper bound; not a selector, not "
                      "promotion evidence", "songs": {}, "oracle": {}}
    for sid, s in songs.items():
        report["songs"][sid] = {
            "selected": s["selected"],
            "candidates": {c["id"]: {"admitted": c["admitted"],
                                     "vs_b0": c["vs_b0"],
                                     "vs_ctrl": c["vs_ctrl"]}
                           for c in s["cands"]}}
    for pool in ("admitted", "all_ok"):
        choices = [[c for c in s["cands"]
                    if pool == "all_ok" or c["admitted"]]
                   for s in songs.values()]
        if any(not ch for ch in choices):
            report["oracle"][pool] = {"feasible": False,
                                      "reason": "a song has no candidate"}
            continue
        best = None
        for combo in itertools.product(*choices):
            ok_b, cb = _passes([c["vs_b0"] for c in combo])
            ok_c, cc = _passes([c["vs_ctrl"] for c in combo])
            score = (ok_b and ok_c,
                     min(min(cb[k] for k in PROPS), min(cc[k] for k in PROPS)))
            if best is None or score > best[0]:
                best = (score, [c["id"] for c in combo], cb, cc)
        report["oracle"][pool] = {
            "feasible": best[0][0], "best_min_property_count": best[0][1],
            "choice": dict(zip(songs, best[1])),
            "counts_vs_b0": best[2], "counts_vs_ctrl": best[3]}
        print(f"oracle[{pool}]: feasible={best[0][0]} min_prop="
              f"{best[0][1]} vs_b0={best[2]} vs_ctrl={best[3]}")
    (OUT / "headroom_e2.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    run()
