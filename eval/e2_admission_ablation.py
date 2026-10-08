"""Admission-only ablation on cached E2 candidates (prereg addendum in docs/specs/e1-machine-ab-prereg.md).

Only the admitted bit changes: comparator-v2 machine verdict
MACHINE_PASS_QUALITY_NOT_EVALUATED replaces the Q1-era motion ceilings.
Selector, decodes and P1-P4 rule unchanged; control re-admitted the same.
"""
import json

from eval.e1_machine_ab import E1, OUT, _grid, decide, measure
from eval.e2_headroom import _passes
from eval.expressive_eval import select_arm

PASS = "MACHINE_PASS_QUALITY_NOT_EVALUATED"


def readmit(sid, arm, grid, bank, thr, warn):
    """Re-admitted pool + B0-selector choice for one arm on one song."""
    part = E1 / "partial"
    pool, detail = [], {}
    for sd in range(6):
        a = json.loads((part / f"{sid}:arm{arm}:s{sd}.json").read_text())
        if not a.get("ok"):
            detail[f"s{sd}"] = {"ok": False}
            continue
        m = measure(sid, f"arm{arm}s{sd}", a["notes"], grid, bank, thr, warn)
        adm = m["verdict"] == PASS
        detail[f"s{sd}"] = {"ok": True, "new_admitted": adm,
                            "verdict": m["verdict"],
                            "old_admitted": bool(a.get("eligible")),
                            "old_reasons": (a.get("admission") or {})
                            .get("reasons")}
        if adm:
            pool.append({"id": f"s{sd}", "eligible": True, "measure": m,
                         "production_key": (True, a["production_key"][1])})
    choice = select_arm(pool, selector="b0") if pool else \
        {"id": None, "status": "fallback_b0"}
    chosen = next((r["measure"] for r in pool if r["id"] == choice["id"]),
                  None)
    return chosen, choice, detail


def run():
    from qa.certificates import load_speed_warning
    from qa.neighbours import bank_from_role
    cfg = json.loads((E1 / "run.json").read_text())["config"]
    thr = json.loads((E1.parent.parent / "qa-v4/comparator-v2/development/"
                                         "threshold.json").read_text())
    bank, _r = bank_from_role(identity="qa-train-v2")
    warn = load_speed_warning()
    rows, rep = [], {"songs": {}}
    for song in cfg["songs"]:
        sid = song["song"]
        grid = _grid(song)
        b0 = json.loads((OUT / f"{sid}.b0.json").read_text())
        e2, e2c, e2d = readmit(sid, "E2", grid, bank, thr, warn)
        ct, ctc, ctd = readmit(sid, "ctrl", grid, bank, thr, warn)
        ct_eff = ct or b0                      # fallback = no change
        rows.append({"vs_b0": decide(b0, e2), "vs_ctrl": decide(ct_eff, e2)})
        rep["songs"][sid] = {"E2": {"selection": e2c, "candidates": e2d},
                             "ctrl": {"selection": ctc, "candidates": ctd}}
        print(f"  {sid}: E2 -> {e2c['id']} ctrl -> {ctc['id']}", flush=True)
    for k in ("vs_b0", "vs_ctrl"):
        ok, counts = _passes([r[k] for r in rows])
        rep[k] = {"pass": ok, "counts_of_6": counts,
                  "protect": sum(r[k]["protect"] for r in rows)}
    rep["verdict"] = ("IMPROVEMENT_SHOWN" if rep["vs_b0"]["pass"]
                      and rep["vs_ctrl"]["pass"] else "NOT_SHOWN")
    (OUT / "admission_ablation_e2.json").write_text(json.dumps(rep, indent=1))
    print(f"{rep['verdict']} vs_b0={rep['vs_b0']} vs_ctrl={rep['vs_ctrl']}")
    return rep


if __name__ == "__main__":
    run()
