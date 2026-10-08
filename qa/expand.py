"""QA training-coverage expansion (2026-09-24).

One bounded packet after CALIBRATION_FAILED_V2: widen QA-train coverage
toward the observed failure (directional intensity over-prediction for
economical-motion families) WITHOUT widening intervals.

- +8 QA-train families: 4 nearest to fam:1716 in the FROZEN explicit
  map-context descriptor space (mean qa.neighbours.descriptor over the
  local ExpertPlus chart, stride-sampled <=400 notes — chart-side only,
  selected before inspecting any replay kinematics) + 4 diverse controls
  by deterministic family hash. 4 fresh calibration families reserved
  independently by hash BEFORE any download; the old calib four remain
  disclosed development/stress cases (role qa_calib; fresh = qa_calib2).
- Up to 6 players per new family, two per performance THIRD of the
  modifier-free leaderboard (no accuracy floor), one replay each; max 72
  new replays (cumulative <=575 of the 600/5GB caps). Player/family
  separation from GEN and seal enforced via the shared state.
- ONE refit: same architecture/features/objective, seed SEED+999, the
  selected 500-update budget, on the expanded 24-family set; empirical
  bank rebuilt from that set only. No alternative fits.
- Acceptance: existing inflation grid meets >=85% pooled AND equal-player
  coverage and <=20% wide in EACH fresh-calib and stress family, with the
  ORIGINAL 16-family observed spans frozen for the sharpness comparison
  (expanded spans report-only). Old-vs-new on identical frozen windows.
  No family substitutions after outcomes. Seal untouched; E1 held.
"""
import hashlib
import json
from pathlib import Path

import torch

from qa.collect2 import (ALLOC_P, REPLAY_CAP, ROOTS, STATE_P, MIN_SCORES,
                         _fetch_family)
from qa.replays import API, _chart_notes_for, _get, _secret_key, _save, \
    level_hash

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "qa-v2" / "expansion"
SEL_P = OUT_DIR / "selection.json"
DESC_P = OUT_DIR / "desc_cache.json"
REF_FAMILY = "fam:1716"
SALT_CTRL = "qa-v3-expand"
SALT_CALIB = "qa-v3-calib"
N_NEAREST = N_CTRL = N_CALIB = 4
MAX_PLAYERS = 6
MAX_NEW_REPLAYS = 72
MAX_LIVE_PROBES = 60
DESC_MAX_NOTES = 400
CKPT_P = OUT_DIR / "motion_mixture_v2.pt"


def _h(salt, fam):
    return hashlib.sha256(f"{salt}:{fam}".encode()).hexdigest()


def _local_chart(map_dir, difficulty="ExpertPlus"):
    d = Path(map_dir)
    info_p = next((p for p in d.iterdir()
                   if p.name.lower() == "info.dat"), None)
    if info_p is None:
        return None, None
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    for s in info.get("_difficultyBeatmapSets", []) or []:
        if s.get("_beatmapCharacteristicName") != "Standard":
            continue
        for b in s.get("_difficultyBeatmaps", []) or []:
            if b.get("_difficulty") == difficulty:
                return d / b["_beatmapFilename"], info_p
    return None, None


def family_descriptor(map_dir):
    """FROZEN explicit map-context descriptor for one family: mean of the
    comparator's per-note scene descriptor over the local ExpertPlus chart
    (deterministic stride sample of <=400 notes). Chart-side only — no
    replay kinematics enter selection."""
    from qa.neighbours import descriptor
    from qa.scene import read_scene
    dat_p, info_p = _local_chart(map_dir)
    if dat_p is None or not dat_p.exists():
        return None
    scene = read_scene(dat_p, info_p)
    if scene["scope"] is not None or len(scene["notes"]) < 20:
        return None
    n = len(scene["notes"])
    stride = max(1, n // DESC_MAX_NOTES)
    idxs = list(range(0, n, stride))[:DESC_MAX_NOTES]
    return torch.stack([descriptor(scene, i) for i in idxs]).mean(dim=0)


def _used_families():
    alloc = json.loads(ALLOC_P.read_text())
    used = set(alloc["assigned"]) | set(alloc["fixed_from_pilot"])
    unusable = {c["family"] for c in alloc["inventory_census"]
                if not c.get("usable")}
    return used, unusable


def candidates():
    from eval import corpus
    used, unusable = _used_families()
    m = corpus._load_validated()
    return sorted({r["family"]: r["dir"] for r in m["maps"]
                   if "/beatsaver/" in r["dir"] and r["eligible"] == "ok"
                   and r.get("family_rep") and r["family"] not in used
                   and r["family"] not in unusable}.items())


def _probe(fam, map_dir, cache):
    """Leaderboard-usability probe (hardest difficulty with a local chart
    and >=20 modifier-free scores). Counts only — no replay content."""
    if fam in cache:
        return cache[fam]
    out = None
    try:
        h = level_hash(map_dir)
        lb = _get(f"{API}/leaderboards/hash/{h}")
        std = {l["difficulty"]["difficultyName"]: l
               for l in (lb.get("leaderboards") or [])
               if l["difficulty"]["modeName"] == "Standard"}
        for diff in ("ExpertPlus", "Expert", "Hard", "Normal", "Easy"):
            l = std.get(diff)
            if l is None:
                continue
            notes, _bpm = _chart_notes_for(map_dir, diff)
            if not notes:
                continue
            d = _get(f"{API}/leaderboard/{l['id']}?page=1&count=50")
            n_ok = sum(1 for s in (d.get("scores") or [])
                       if not s.get("modifiers"))
            if n_ok >= MIN_SCORES:
                out = {"family": fam, "dir": map_dir, "hash": h,
                       "leaderboard": l["id"], "difficulty": diff,
                       "n_modifier_free": n_ok}
                break
    except Exception as e:
        out = {"family": fam, "error": type(e).__name__}
    cache[fam] = out if out else {"family": fam, "usable": False}
    return cache[fam]


def _first_usable(ranked, k, probes, taken, budget):
    picked = []
    for fam, map_dir in ranked:
        if len(picked) >= k:
            break
        if fam in taken:
            continue
        if fam not in probes and budget[0] <= 0:
            raise RuntimeError("live-probe budget exhausted before "
                               "selection filled — enlarge deliberately, "
                               "not silently")
        if fam not in probes:
            budget[0] -= 1
        r = _probe(fam, map_dir, probes)
        if r.get("leaderboard"):
            picked.append(r)
            taken.add(fam)
    if len(picked) < k:
        raise RuntimeError(f"only {len(picked)}/{k} usable families in "
                           "ranked list")
    return picked


def select():
    """Stage 1: frozen selection BEFORE any replay download. Idempotent."""
    if SEL_P.exists():
        return json.loads(SEL_P.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = candidates()
    # descriptors (cached incrementally; chart-side only)
    cache = json.loads(DESC_P.read_text()) if DESC_P.exists() else {}
    from eval import corpus
    m = corpus._load_validated()
    ref_dir = next(r["dir"] for r in m["maps"]
                   if r["family"] == REF_FAMILY and r.get("family_rep")
                   and "/beatsaver/" in r["dir"])
    if REF_FAMILY not in cache:
        cache[REF_FAMILY] = family_descriptor(ref_dir).tolist()
    descs = {}
    for i, (fam, map_dir) in enumerate(pool):
        if fam not in cache:
            d = family_descriptor(map_dir)
            cache[fam] = d.tolist() if d is not None else None
            if i % 25 == 0:
                DESC_P.write_text(json.dumps(cache))
        if cache[fam] is not None:
            descs[fam] = torch.tensor(cache[fam])
    DESC_P.write_text(json.dumps(cache))
    dm = torch.stack(list(descs.values()))
    mu, sd = dm.mean(dim=0), dm.std(dim=0).clamp(min=1e-6)
    ref = (torch.tensor(cache[REF_FAMILY]) - mu) / sd
    dist = {fam: float(torch.linalg.norm((d - mu) / sd - ref))
            for fam, d in descs.items()}
    dirs = dict(pool)
    near_rank = [(f, dirs[f]) for f in sorted(dist, key=lambda f:
                                              (dist[f], f))]
    ctrl_rank = [(f, dirs[f]) for f in sorted(descs,
                                              key=lambda f: _h(SALT_CTRL,
                                                               f))]
    calib_rank = [(f, dirs[f]) for f in sorted(descs,
                                               key=lambda f: _h(SALT_CALIB,
                                                                f))]
    probes, taken, budget = {}, set(), [MAX_LIVE_PROBES]
    near = _first_usable(near_rank, N_NEAREST, probes, taken, budget)
    ctrl = _first_usable(ctrl_rank, N_CTRL, probes, taken, budget)
    calib = _first_usable(calib_rank, N_CALIB, probes, taken, budget)
    sel = {"ruling": "review", "ref_family": REF_FAMILY,
           "descriptor": f"mean neighbours.descriptor over local "
                         f"ExpertPlus chart, stride<= {DESC_MAX_NOTES} "
                         "notes, z-scored over candidate pool",
           "salts": {"controls": SALT_CTRL, "calib": SALT_CALIB},
           "n_candidates": len(descs),
           "train_nearest": near, "train_controls": ctrl,
           "calib_fresh": calib,
           "distances_selected": {r["family"]: dist[r["family"]]
                                  for r in near + ctrl + calib},
           "probes": {f: (p if p.get("leaderboard") else p)
                      for f, p in probes.items()},
           "player_rule": "two per performance third of the modifier-free "
                          "leaderboard, <=6 players, 1 replay each, no "
                          "accuracy floor",
           "caps": {"max_new_replays": MAX_NEW_REPLAYS,
                    "cumulative_max": 575}}
    tmp = SEL_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(sel, indent=1, sort_keys=True))
    tmp.replace(SEL_P)
    print(f"selection frozen: nearest {[r['family'] for r in near]} "
          f"controls {[r['family'] for r in ctrl]} "
          f"calib {[r['family'] for r in calib]}")
    return sel


def replace_calib(bad_fam, reason):
    """review: ONE next-ranked eligibility-only replacement for a
    calib family whose chart failed validity at ingestion (before any
    kinematics/coverage outcome). Deterministic: first usable family in
    the frozen qa-v3-calib hash rank not already taken; the excluded
    family and its reason stay recorded in the selection manifest."""
    sel = json.loads(SEL_P.read_text())
    if any(a["excluded"] == bad_fam for a in sel.get("amendments", [])):
        return sel
    if bad_fam not in {r["family"] for r in sel["calib_fresh"]}:
        raise ValueError(f"{bad_fam} is not a selected calib family")
    cache = json.loads(DESC_P.read_text())
    dirs = dict(candidates())
    ranked = [(f, dirs[f]) for f in sorted(
        (f for f in cache if f in dirs and cache[f] is not None),
        key=lambda f: _h(SALT_CALIB, f))]
    taken = ({r["family"] for r in sel["train_nearest"]
              + sel["train_controls"] + sel["calib_fresh"]}
             | {a["excluded"] for a in sel.get("amendments", [])})
    probes = dict(sel.get("probes", {}))
    repl = _first_usable(ranked, 1, probes, taken, [20])[0]
    sel["amendments"] = sel.get("amendments", []) + [
        {"ruling": "review", "excluded": bad_fam,
         "reason": reason, "replacement": repl["family"]}]
    sel["calib_fresh"] = [r for r in sel["calib_fresh"]
                          if r["family"] != bad_fam] + [repl]
    sel["probes"] = probes
    tmp = SEL_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(sel, indent=1, sort_keys=True))
    tmp.replace(SEL_P)
    print(f"calib replacement: {bad_fam} -> {repl['family']} "
          f"({reason})")
    return sel


def thirds_picks(eligible):
    """Two players per performance third (leaderboard rank order)."""
    n = len(eligible)
    picks, seen = [], set()
    for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
        third = eligible[int(lo * n):max(int(hi * n), int(lo * n) + 1)]
        got = 0
        for s in third:
            pid = s.get("playerId") or (s.get("player") or {}).get("id")
            if not pid or pid in seen or got >= 2:
                continue
            picks.append((s, pid))
            seen.add(pid)
            got += 1
    return picks


def collect():
    """Stage 2: fetch the frozen selection (resumable per family)."""
    sel = json.loads(SEL_P.read_text())
    state = json.loads(STATE_P.read_text())
    key = _secret_key()
    jobs = ([(r, "qa_train") for r in sel["train_nearest"]
             + sel["train_controls"]]
            + [(r, "qa_calib2") for r in sel["calib_fresh"]])
    # excluded families' replays still count against the expansion cap
    exp_fams = ({r["family"] for r, _ in jobs}
                | {a["excluded"] for a in sel.get("amendments", [])})
    for entry, side in jobs:
        fam = entry["family"]
        if fam in state["charts"]:
            continue
        added = sum(len(state["charts"][f].get("replays", []))
                    for f in exp_fams if f in state["charts"])
        room = MAX_NEW_REPLAYS - added
        if room <= 0:
            print("expansion replay cap reached")
            break
        ROOTS[side].mkdir(parents=True, exist_ok=True)
        rec = _fetch_family(entry, side, state, key,
                            picks_fn=thirds_picks, max_picks=MAX_PLAYERS,
                            max_replays=min(MAX_PLAYERS, room))
        state["charts"][fam] = rec
        _save(STATE_P, state)
        print(f"  [{side:>9}] {fam:<12} {rec['status']:<5} replays "
              f"{len(rec['replays'])}", flush=True)
    added = sum(len(state["charts"][f].get("replays", []))
                for f in exp_fams if f in state["charts"])
    total = sum(len(c.get("replays", []))
                for c in state["charts"].values()) + state["pilot_replays"]
    print(f"expansion: +{added} replays (cap {MAX_NEW_REPLAYS}); "
          f"cumulative {total}/{REPLAY_CAP}, "
          f"{state['bytes'] / 1e9:.2f} GB")
    return state


def prepare_records():
    """Stage 3: sanitize new originals + derive validated windows."""
    from qa.contract import migrate_originals
    from qa.telemetry_v2 import derive_role
    migrate_originals()
    return {"qa_train": derive_role("qa_train"),
            "qa_calib2": derive_role("qa_calib2")}


def refit():
    """Stage 4: the ONE authorized refit on the 24-family set (same
    architecture/features/objective, seed SEED+999, 500 updates)."""
    from qa.features import family_identity
    from qa.model import FEATURE_NAMES
    from qa.train import SEED, _fit, assemble_role
    from qa.telemetry_v2 import TELEMETRY_VERSION
    if CKPT_P.exists():
        print(f"refit exists: {CKPT_P}")
        return torch.load(CKPT_P)
    fams = assemble_role("qa_train")
    sel = json.loads(SEL_P.read_text())
    want = 16 + len(sel["train_nearest"]) + len(sel["train_controls"])
    if len(fams) != want:
        raise RuntimeError(f"expected {want} train families, got "
                           f"{len(fams)} — no silent refit on a partial "
                           "set")
    _m, scaler, snaps = _fit(fams, 500, SEED + 999, stop_after=500)
    ck = {"state": snaps[500], "scaler": scaler, "updates": 500,
          "seed": SEED + 999, "families": sorted(fams),
          "feature_names": list(FEATURE_NAMES),
          "telemetry_version": TELEMETRY_VERSION,
          "identity": {f: family_identity(f, "qa_train")
                       for f in sorted(fams)},
          "ruling": "review: one refit, no checkpoint search"}
    tmp = CKPT_P.with_suffix(".tmp")
    torch.save(ck, tmp)
    tmp.replace(CKPT_P)
    print(f"refit saved: {CKPT_P} ({len(fams)} families)")
    return ck


EVAL_FREEZE_P = OUT_DIR / "eval_freeze.json"
EVAL_RUN_DIR = OUT_DIR / "run"
PARITY_FREEZE_P = ROOT / "experiments/qa-v2/feature-parity/freeze_manifest.json"


def freeze_eval():
    """Freeze the acceptance evaluation BEFORE any coverage: gate families
    = 4 fresh calib + 4 existing stress calib; stress families reuse their
    EXACT parity-run window ids (old-vs-new on identical windows); spans
    and t_std stay the ORIGINAL 16-family observed values. Idempotent."""
    from qa.coverage_sample import select_windows
    from qa.features import records_for_role
    from qa.parity_recheck import (CODE_FILES, _ids_sha, _sha_file,
                                   component_spans)
    if EVAL_FREEZE_P.exists():
        man = json.loads(EVAL_FREEZE_P.read_text())
        for rel, want in man["code_sha256"].items():
            if _sha_file(ROOT / rel) != want:
                raise RuntimeError(f"code changed after eval freeze: {rel}")
        return man
    from qa.train import assemble_role
    all_packs = assemble_role("qa_train")
    sel = json.loads(SEL_P.read_text())
    parity = json.loads(PARITY_FREEZE_P.read_text())
    samples = {}
    for r in sel["calib_fresh"]:
        fam = r["family"]
        recs = [{"player_token": x["player_token"],
                 "window_id": x["window_id"]}
                for x in records_for_role("qa_calib2", "calibrate",
                                          families={fam})]
        ids = select_windows(recs, cap=2000)
        samples[fam] = {"role": "qa_calib2", "n_eligible": len(recs),
                        "ids": ids, "sha256": _ids_sha(ids)}
    for fam, sm in parity["samples"].items():
        if sm["role"] == "qa_calib":
            samples[fam] = {**sm, "note": "identical windows as the "
                                          "parity run (stress family)"}
    # ORIGINAL frozen spans/std: 16 original train families only
    sel_new = {r["family"] for r in sel["train_nearest"]
               + sel["train_controls"]}
    packs = {f: p for f, p in all_packs.items() if f not in sel_new}
    if len(packs) != 16:
        raise RuntimeError(f"expected 16 original packs, got {len(packs)}")
    y = torch.cat([p["y"] for p in packs.values()])
    m = torch.cat([p["mask"] for p in packs.values()])
    t_std = [float(y[:, j][m[:, j]].std()) for j in range(y.shape[1])]
    man = {"code_sha256": {rel: _sha_file(ROOT / rel)
                           for rel in CODE_FILES + ("qa/expand.py",)},
           "checkpoint_v2": _sha_file(CKPT_P),
           "checkpoint_old": _sha_file(ROOT / "experiments/qa-v1/"
                                              "model-fits/"
                                              "motion_mixture_final.pt"),
           "selection_sha256": _sha_file(SEL_P),
           "samples": samples,
           "frozen_spans_original16": component_spans(y, m),
           "frozen_t_std_original16": t_std,
           "gate": "existing grid {1,1.25,1.5,2}: >=0.85 pooled AND "
                   "equal-player, <=0.20 wide vs ORIGINAL spans, in EACH "
                   "of the 8 gate families; expanded spans report-only"}
    tmp = EVAL_FREEZE_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True))
    tmp.replace(EVAL_FREEZE_P)
    print(f"eval freeze written: {EVAL_FREEZE_P} "
          f"({len(samples)} gate families)")
    return man


def evaluate(resume=True):
    """Stage 5: ONE acceptance evaluation of the v2 refit."""
    from qa.model import MotionMixture
    from qa.neighbours import bank_from_role
    from qa.parity_recheck import _eval_family, disposition
    man = freeze_eval()
    EVAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT_P)
    v2 = MotionMixture()
    v2.load_state_dict(ck["state"])
    v2.eval()
    old_ck = torch.load(ROOT / "experiments/qa-v1/model-fits/"
                               "motion_mixture_final.pt")
    old = MotionMixture()
    old.load_state_dict(old_ck["state"])
    old.eval()
    bank, _r = bank_from_role(identity="qa-train-v2")
    spans = man["frozen_spans_original16"]
    t_std = torch.tensor(man["frozen_t_std_original16"]).clamp(min=1e-6)
    results, results_old = {}, {}
    for fam, sm in sorted(man["samples"].items()):
        out_p = EVAL_RUN_DIR / (fam.replace(":", "_") + ".json")
        if resume and out_p.exists():
            prior = json.loads(out_p.read_text())
            if prior["new"]["sample_sha256"] != sm["sha256"]:
                raise ValueError(f"{fam}: resumed result from a different "
                                 "frozen sample")
            results[fam] = prior["new"]
            results_old[fam] = prior["old"]
            continue
        rn = _eval_family(fam, sm, v2, ck["scaler"], spans, bank, None,
                          t_std)
        ro = _eval_family(fam, sm, old, old_ck["scaler"], spans, bank,
                          None, t_std)
        rn["model"], ro["model"] = "v2-24fam", "old-16fam"
        tmp = out_p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"new": rn, "old": ro}, indent=1,
                                  sort_keys=True))
        tmp.replace(out_p)
        results[fam], results_old[fam] = rn, ro
        print(f"  [{fam}] new pooled@2 {rn['coverage']['2.0']['pooled']} "
              f"(old {ro['coverage']['2.0']['pooled']})", flush=True)
    disp = disposition(results)
    if disp["status"] == "CALIBRATION_REPAIRED":
        disp["status"] = "EXPANSION_ACCEPTED"
    elif disp["status"] == "CALIBRATION_FAILED_V2":
        disp["status"] = "EXPANSION_FAILED"
    report = {"disposition": disp,
              "gate_families": sorted(results),
              "old_vs_new": {
                  fam: {f: {"new": results[fam]["coverage"][f]["pooled"],
                            "old": results_old[fam]["coverage"][f]
                            ["pooled"]} for f in ("1.5", "2.0")}
                  for fam in results},
              "note": "one refit, one evaluation; "
                      "original spans frozen for sharpness; stress "
                      "families scored on identical parity-run windows; "
                      "success resumes evaluator validation only — no "
                      "seal/E1 authority"}
    if disp["status"] == "EXPANSION_ACCEPTED":
        fresh = {f: r for f, r in results.items()
                 if r["role"] == "qa_calib2"}
        report["thresholds_v3"] = {
            "T_support": max(r["support"]["support_p95"]
                             for r in fresh.values()),
            "T_disagree": max(r["support"]["disagree_p95"]
                              for r in fresh.values()),
            "inflation_factor": disp["factor"],
            "source": "fresh calibration families under the unchanged "
                      "max-family-p95 recipes; stress values reported "
                      "alongside in per-family artifacts"}
    rp = OUT_DIR / "expansion_report.json"
    tmp = rp.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(rp)
    print(f"{disp['status']} (factor {disp.get('factor')}) -> {rp}")
    return report


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "select"
    {"select": select, "collect": collect, "prepare": prepare_records,
     "refit": refit, "freeze-eval": freeze_eval,
     "evaluate": evaluate}[cmd]()
