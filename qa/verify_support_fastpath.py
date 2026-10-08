"""Verify the reference-validation fast path against STORED outputs only
(the slow implementation is never re-executed).

A) every experiments/expressive-v1/e1/machine_ab/<song>.<tag>.json record is
   regenerated with eval.e1_machine_ab.measure into a scratch dir and
   compared byte-for-byte;
B) every comparator-v2 development/fresh/seal chart's per-head support
   (reason + distance per head, evidence errors) is recomputed and compared
   with its stored support_progress.json.
Resumable: results accumulate in experiments/qa-v4/fastpath_verify.json.
"""
import json
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "experiments/expressive-v1/e1/machine_ab"
SCRATCH = ROOT / "experiments/qa-v4/fastpath_scratch"
RESULT_P = ROOT / "experiments/qa-v4/fastpath_verify.json"
V2 = ROOT / "experiments/qa-v4/comparator-v2"


def _save(res):
    tmp = RESULT_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True))
    tmp.replace(RESULT_P)


def _notes(sid, tag):
    part = ROOT / "experiments/expressive-v1/e1/partial"
    if tag == "b0":
        return json.loads((part / f"{sid}:b0.json").read_text())["notes"]
    arm = tag[3:]
    if len(arm) > 2 and arm[-1].isdigit() and arm[-2] == "s":
        arm, sd = arm[:-2], arm[-1]
        return json.loads((part / f"{sid}:arm{arm}:s{sd}.json")
                          .read_text())["notes"]
    sel = json.loads((part / f"{sid}:arm{arm}.json").read_text())["selection"]
    return json.loads((part / f"{sid}:arm{arm}:{sel['id']}.json")
                      .read_text())["notes"]


def verify_records(bank, res):
    import eval.e1_machine_ab as ab
    from qa.certificates import load_speed_warning
    cfg = json.loads((ab.E1 / "run.json").read_text())["config"]
    songs = {s["song"]: s for s in cfg["songs"]}
    thr = json.loads((V2 / "development/threshold.json").read_text())
    warn = load_speed_warning()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    grids = {}
    for p in sorted(AB.glob("*.*.json")):
        sid, tag = p.name[:-5].split(".", 1)
        key = f"record:{p.name}"
        if key in res or sid not in songs:
            continue
        grid = grids.setdefault(sid, ab._grid(songs[sid]))
        with mock.patch.object(ab, "OUT", SCRATCH):
            ab.measure(sid, tag, _notes(sid, tag), grid, bank, thr, warn)
        res[key] = (SCRATCH / p.name).read_bytes() == p.read_bytes()
        _save(res)
        print(f"  {p.name}: {'IDENTICAL' if res[key] else 'DIFFERS'}",
              flush=True)


def _stage_charts():
    """(stage, family, scene, exclusions, threshold) for dev/fresh/seal."""
    from qa.comparator_seal import seal_families
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    manifest = json.loads((ROOT / "eval/corpus_manifest.json").read_text())
    st = json.loads((ROOT / "experiments/qa-v1/collect2_state.json")
                    .read_text())
    dev_thr = json.loads((V2 / "development/threshold.json").read_text())
    fz_thr = {"T_support": json.loads((V2 / "evaluator_freeze.json")
                                      .read_text())["threshold"]}
    charts = dict(st["charts"])
    charts.update({f: c for f, c in seal_families()})
    for stage in ("development", "fresh", "seal"):
        for d in sorted((V2 / stage).glob("fam_*")):
            if not (d / "support_progress.json").exists():
                continue
            fam = "fam:" + d.name[4:]
            c = charts[fam]
            dat_p, info_p = _family_chart(manifest, fam, c["difficulty"])
            excl = {"families": {fam},
                    "players": {e["player_token"] for e in c["replays"]}}
            yield (stage, fam, read_scene(dat_p, info_p), excl,
                   dev_thr if stage == "development" else fz_thr, d)


def verify_per_head(bank, res):
    from qa.comparator_support import measure_support
    for stage, fam, scene, excl, thr, d in _stage_charts():
        key = f"per_head:{stage}:{fam}"
        if key in res:
            continue
        stored = json.loads((d / "support_progress.json").read_text())
        got = measure_support(scene, bank, thr, excl)
        res[key] = (got["per_head"] == stored["per_head"]
                    and got["evidence_errors"] == stored["evidence_errors"])
        _save(res)
        print(f"  {key}: {'IDENTICAL' if res[key] else 'DIFFERS'}",
              flush=True)


def run():
    from qa.neighbours import bank_from_role
    res = json.loads(RESULT_P.read_text()) if RESULT_P.exists() else {}
    bank, _r = bank_from_role(identity="qa-train-v2")
    verify_per_head(bank, res)
    verify_records(bank, res)
    bad = sorted(k for k, v in res.items() if not v)
    print(f"{len(res)} checked, {len(bad)} differ: {bad}")
    return res


if __name__ == "__main__":
    run()
