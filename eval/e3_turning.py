"""E3: human turning-sequence oversampling from E2 .

Pool = every CTX-length human TRAIN window (stride 64, original and
mirror) that contains >=1 arc-like run under the P1 predicate
(eval.expression_profile._arcs), excluding every QA-owned family; frozen
to experiments/expressive-v1/e3/pool.json before training. Training =
E2's loop initialized from E2's selected checkpoint, 25% pool / 75% E2
distribution. Same evaluation and acceptance as E2 (prereg addendum).
"""
import hashlib
import json
from pathlib import Path

import torch

from eval.e2_context import E2, ContextFlow, shim_for_schedule, train

ROOT = Path(__file__).resolve().parent.parent
E3 = ROOT / "experiments" / "expressive-v1" / "e3"
POOL_P = E3 / "pool.json"
STRIDE = 64


def qa_owned_families():
    fams = set()
    for name in ("collect2_state.json", "pilot_state.json"):
        p = ROOT / "experiments/qa-v1" / name
        if p.exists():
            for fam, c in json.loads(p.read_text())["charts"].items():
                if c.get("side") != "gen":
                    fams.add(fam)
    return fams


def build_pool():
    if POOL_P.exists():
        return json.loads(POOL_P.read_text())
    import groom
    from groom import CTX, mirror_events
    from eval import corpus
    from eval.expression_profile import _arcs
    data = torch.load(E2 / "dataset.pt")
    train_fams = {f: s for s in ("approved", "general")
                  for f in data["train"][s]}
    banned = qa_owned_families()
    m = corpus._load_validated()
    fam_dir = {}
    for r in m["maps"]:
        if r["eligible"] == "ok" and r.get("family_rep") \
                and r["split"] == "train" and r["family"] in train_fams:
            fam_dir.setdefault(r["family"], r["dir"])
    pool, excluded = {}, sorted(set(fam_dir) & banned)
    for fam, dp in sorted(fam_dir.items()):
        if fam in banned:
            continue
        d = Path(dp)
        info = json.loads(next(p for p in d.iterdir()
                               if p.name.lower() == "info.dat")
                          .read_text(encoding="utf-8-sig"))
        step_ms = 60000.0 / float(info["_beatsPerMinute"]) / 4
        wins = []
        for name, smp in sorted(groom.load_map_all(d).items()):
            cid = hashlib.sha256(f"{fam}:{name}".encode()).hexdigest()[:16]
            if cid not in data["charts"]:
                continue
            for suffix, ev in (("", smp[2]), (":m", mirror_events(smp[2]))):
                for i in range(0, len(ev) - CTX + 1, STRIDE):
                    heads = [(s * step_ms / 1000.0, h, c, l, dd)
                             for s, h, dd, c, l, _k in ev[i:i + CTX]
                             if dd != 8]
                    if _arcs(heads)["n_runs"] >= 1:
                        wins.append([cid + suffix, i])
        if wins:
            pool[fam] = wins
        print(f"  [e3 pool] {fam}: {len(wins)} windows", flush=True)
    E3.mkdir(parents=True, exist_ok=True)
    rec = {"pool": pool, "excluded_qa_families": excluded,
           "n_windows": sum(len(v) for v in pool.values()),
           "n_families": len(pool), "stride": STRIDE,
           "predicate": "eval.expression_profile._arcs n_runs >= 1"}
    POOL_P.write_text(json.dumps(rec))
    print(f"pool frozen: {rec['n_families']} families, "
          f"{rec['n_windows']} windows, excluded {len(excluded)} QA fams")
    return rec


def fit():
    rec = build_pool()
    sel = json.loads((E2 / "history.json").read_text())["selected"]
    pool = {f: [tuple(w) for w in ws] for f, ws in rec["pool"].items()}
    return train(out_dir=E3, init_ckpt=E2 / f"snapshot-{sel['update']}.pt",
                 pool=pool, pool_p=0.25, seed_offset=375)


def evaluate():
    from eval.e1_machine_ab import run as machine_ab
    from eval.expressive_eval import run_e1_eval
    hist = json.loads((E3 / "history.json").read_text())
    sel = hist["selected"]
    if sel["step_zero"]:
        print("[e3] step-zero winner: no checkpoint -> NOT_SHOWN")
        return {"status": "STEP_ZERO"}
    model = ContextFlow()
    model.load_state_dict(torch.load(E3 / f"snapshot-{sel['update']}.pt",
                                     map_location="cpu"))
    model.eval()
    run_e1_eval("E3", flow_factory=lambda sch, ms: shim_for_schedule(
        model, sch, ms), tag=f"e3@{sel['update']}")
    return machine_ab(tags=("E3", "ctrl", "E2"),
                      pairs=[("E3", "b0"), ("E3", "ctrl"), ("E3", "E2")],
                      report_name="report_e3.json")


if __name__ == "__main__":
    import sys
    {"pool": build_pool, "fit": fit, "evaluate": evaluate}[sys.argv[1]]()
