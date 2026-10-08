"""Q5 scoped release: the evaluated neutral bundle as an installable
generation policy (an explicit product-scope
amendment, NOT a pass of the frozen Q5 contract; the recorded C
confirmation FAIL stands and ships disclosed in the release manifest).

Claims: C-confirmed motion-flag reduction and pair-exposure reduction ONLY.
No variety, personality or achieved-workload claims; workload calibration
is descriptive only; style is neutral.

Serving path (byte-exact transcription of the evaluated run_q5 loop,
verified against stored evaluation outputs in eval/test_release.py):
B0 production decode -> frozen EventSchedule -> planner window intents +
fixed tier request -> fit-B conditioned geometry (CondFlowShim, seeds
0/1/2) -> Q1-V2 repair -> hard validation + emitted-motion admission vs B0
-> lowest-motif selection with pref tie-break -> explicit B0 fallback.

Install/rollback are transactional; artifacts verify against
release_manifest.json on every load (wrong-version refusal).
  python -m quality_policy install    # verify isolated, then atomic
  python -m quality_policy rollback   # one command back to B0
  python -m quality_policy status
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY_FILE = ROOT / "quality_policy.json"
MANIFEST = ROOT / "release_manifest.json"
ARTIFACTS = {"condflow.pt": "experiments/quality-v1/q4-flow/"
                            "condflow-b-seed20260921.pt",
             "planner.pt": "experiments/quality-v1/q3-planner/"
                           "planner-seed20260921.pt",
             "pref_model.pt": "experiments/quality-v1/q4-pref/pref.pt"}
BUNDLE_SEEDS = (0, 1, 2)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def active_policy():
    """The installed generation policy; missing/invalid file means B0."""
    try:
        return json.loads(POLICY_FILE.read_text()).get("policy", "b0")
    except (OSError, ValueError):
        return "b0"


def _verify_artifacts():
    man = json.loads(MANIFEST.read_text())
    for name, sha in man["artifacts"].items():
        p = ROOT / name
        if not p.exists() or _sha(p) != sha:
            raise RuntimeError(
                f"bundle artifact {name} missing or hash-mismatched — "
                "run `python -m quality_policy rollback`")
    return man


_CACHE = {}


def _models():
    if "m" not in _CACHE:
        import torch
        from cond_flow import ConditionedFlow
        man = _verify_artifacts()
        ckd = torch.load(ROOT / "condflow.pt")
        if ckd.get("fit") != "B":
            raise RuntimeError("wrong-version geometry checkpoint (want fit B)")
        cm = ConditionedFlow(n_styles=ckd.get("n_styles", 1))
        cm.load_state_dict(ckd["state"])
        cm.eval()
        pd = torch.load(ROOT / "planner.pt")
        from eval.quality_train import Planner
        pm = Planner(pd["n_in"])
        pm.load_state_dict(pd["state"])
        pm.eval()
        prd = torch.load(ROOT / "pref_model.pt")
        from pref import PrefMLP
        pref_m = PrefMLP()
        pref_m.load_state_dict(prd["state"])
        pref_m.eval()
        _CACHE["m"] = (cm, pm, pd["norm"], pref_m, prd, man)
    return _CACHE["m"]


def _motif_mean(raw, walls, grid, bpm):
    from eval.quality_metrics import four_gram_stats, to_ms
    fg = four_gram_stats(to_ms(raw, grid), 60000.0 / bpm)
    wins = [w for w in fg["windows"].values() if w["n_grams"] >= 4]
    return (sum(w["max_4gram_share"] for w in wins) / len(wins)
            if wins else None)


def bundle_song(objects, bpm, offset, audio_path, diff="ExpertPlus",
                timing=None):
    """The evaluated bundle path for one song. Returns (notes, walls, info)
    in convert_groomed's output format; explicit B0 fallback inside."""
    import torch
    from convert import (NPS_CAP, check, convert_groomed, diff_spec,
                         grid_steps)
    import groom
    import motion as motion_mod
    from cond_flow import shim_for_schedule
    from flow_decode import EventSchedule, decode_geometry
    from phrase_planner import (onset_counts, planner_trajectory,
                                song_window_inputs)
    from pref import feat_vec
    from quality_repair import RepairConfig, repair_winner
    from eval.clean_rhythm import _cand_metrics
    from eval.phrase_plan import _cand_gates

    cm, pm, pnorm, pref_m, prd, man = _models()
    request = torch.tensor(man["request"], dtype=torch.float32)
    spec = diff_spec(diff)
    band = spec["band"]
    cap = NPS_CAP * spec["scale"]
    steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset, timing,
                                               thin=True)
    step_times = [grid.time(s) for s in range(T)]
    afeat = groom.cached_audio_features(audio_path, step_times) \
        if audio_path else None
    recs = []
    convert_groomed(objects, bpm, offset, audio_path=audio_path, diff=diff,
                    replay_mode="off", collect=recs, thin=True,
                    calibrate=True, timing=timing)
    sel = next(r for r in recs if r["selected"])
    raw0, walls0 = sel["notes"], sel["walls"]
    beat_ms = 60000.0 / bpm
    m_b0 = _cand_metrics(raw0, motion_mod.report(
        [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw0]),
        grid, beat_ms)
    sched = EventSchedule.from_raw(raw0, walls0, grid, T, step_ms)
    mi = onset_counts(objects, step_times)
    x = song_window_inputs(afeat, mi, step_times, bpm)
    intents = planner_trajectory(pm, pnorm, x, request)
    shim = shim_for_schedule(cm, sched, step_ms, intents, request)
    qcfg = RepairConfig(variant="paired-direction")
    cands, rejects = [], []
    for sd in BUNDLE_SEEDS:
        p = decode_geometry(sched, steps, T, step_ms, off2, grid, afeat,
                            spec, flow_model=shim, seed=sd,
                            checkpoint="condflow-b")
        if not p.honored or p.infeasible:
            rejects.append({"seed": sd, "reason": "not_honored"
                            if not p.honored else "infeasible"})
            continue
        pr = repair_winner(p.raw, p.walls, grid, bpm, cap, qcfg)
        raw_c = pr.raw if pr is not None else p.raw
        notes_d = [{"t": grid.time(s), "hand": h, "col": c, "layer": l,
                    "dir": d} for s, h, c, l, d in raw_c]
        walls_d = [{"t": grid.time(s0),
                    "dur": grid.time(s0 + ln) - grid.time(s0), "col": col}
                   for s0, ln, col in p.walls]
        fails = list(check(notes_d, bpm, walls_d, cap))
        if not fails:
            m_c = _cand_metrics(raw_c, motion_mod.report(
                [(grid.time(s), h, c, l, d) for s, h, c, l, d in raw_c]),
                grid, beat_ms)
            fails = _cand_gates(m_b0, m_c, band)
        if fails:
            rejects.append({"seed": sd, "reason": fails})
            continue
        with torch.no_grad():
            z = (feat_vec(raw_c, p.walls, grid, bpm) - prd["mu"]) / prd["sd"]
            pscore = float(pref_m(z[None]))
        cands.append({"seed": sd, "raw": raw_c, "walls": p.walls,
                      "motif": _motif_mean(raw_c, p.walls, grid, bpm),
                      "pref": pscore})
    if cands:
        chosen = min(cands, key=lambda c: (c["motif"] if c["motif"]
                                           is not None else 1.0, -c["pref"]))
        raw_sel, walls_sel = chosen["raw"], chosen["walls"]
        info = {"source": f"bundle:{chosen['seed']}", "rejects": rejects,
                "admitted": [c["seed"] for c in cands]}
    else:
        raw_sel, walls_sel = raw0, walls0
        info = {"source": "b0_fallback", "rejects": rejects, "admitted": []}
    notes = [{"t": grid.time(s), "hand": h, "col": col, "layer": lay,
              "dir": d} for s, h, col, lay, d in raw_sel]
    walls = [{"t": grid.time(s0),
              "dur": grid.time(s0 + ln) - grid.time(s0), "col": col}
             for s0, ln, col in walls_sel]
    print(f"  quality policy bundle-v1: {info['source']} "
          f"(admitted {len(info['admitted'])}/{len(BUNDLE_SEEDS)})")
    return notes, walls, info


# ---------------- transactional install / rollback ----------------

def install():
    """Verify the bundle in an isolated copy, then atomically install the
    artifacts + release manifest and flip the policy. Never touches
    groom.pt/flow.pt/critic.pt/ladder.json."""
    import shutil
    import tempfile
    import torch
    from phrase_planner import train_mean_targets
    request = [round(float(v), 6) for v in train_mean_targets()[:10]]
    sources = {n: ROOT / p for n, p in ARTIFACTS.items()}
    for n, p in sources.items():
        if not p.exists():
            raise RuntimeError(f"source artifact missing: {p}")
    # isolated verification BEFORE touching anything shipped
    with tempfile.TemporaryDirectory() as td:
        for n, p in sources.items():
            shutil.copy2(p, Path(td) / n)
        ckd = torch.load(Path(td) / "condflow.pt")
        if ckd.get("fit") != "B":
            raise RuntimeError("isolated verify: geometry ckpt is not fit B")
        from cond_flow import ConditionedFlow, NCOND
        from groom import NTOK
        m = ConditionedFlow(n_styles=ckd.get("n_styles", 1))
        m.load_state_dict(ckd["state"])
        m.eval()
        with torch.no_grad():
            m.hidden(torch.zeros(1, 4, NTOK + NCOND))     # smoke forward
    manifest = {
        "version": 1, "policy": "bundle-v1",
        "scope": "motion-and-variety geometry bundle, SCOPED release",
        "claims": ["C-confirmed motion flag reduction (CI excludes zero)",
                   "C-confirmed pair-exposure reduction (CI excludes zero)"],
        "non_claims": ["variety (C: -15.2% vs >=20% gate, CI includes zero)",
                       "style personality (vocabulary neutral K=1)",
                       "achieved workload (calibration descriptive only)"],
        "c_confirmation": "recorded FAIL under the frozen Q5 contract; this "
                          "release is a post-confirmation product-"
                          "scope amendment, disclosed",
        "artifacts": {n: _sha(p) for n, p in sources.items()},
        "request": request,
        "rollback": "python -m quality_policy rollback",
    }
    # atomic installs: temp then replace
    for n, p in sources.items():
        tmp = ROOT / (n + ".tmp")
        shutil.copy2(p, tmp)
        os.replace(tmp, ROOT / n)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, MANIFEST)
    _verify_artifacts()                       # post-install hash check
    tmp = POLICY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"policy": "bundle-v1"}, indent=1))
    os.replace(tmp, POLICY_FILE)              # the default flips LAST
    print("installed bundle-v1 (policy active); rollback: "
          "`python -m quality_policy rollback`")


def rollback():
    """One command back to B0: flips the policy only (artifacts stay for a
    later re-enable; they are inert while the policy is b0)."""
    tmp = POLICY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"policy": "b0"}, indent=1))
    os.replace(tmp, POLICY_FILE)
    print("policy rolled back to b0 (B0 production path)")


def status():
    pol = active_policy()
    print(f"policy: {pol}")
    if MANIFEST.exists():
        try:
            man = _verify_artifacts()
            print(f"artifacts: OK ({', '.join(sorted(man['artifacts']))})")
        except RuntimeError as e:
            print(f"artifacts: {e}")


if __name__ == "__main__":
    import sys
    {"install": install, "rollback": rollback,
     "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
