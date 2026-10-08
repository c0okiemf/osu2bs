"""Phase 5B clean-split rhythm rebuild + frozen A/B (experiment runner).

One from-scratch rhythm Groomer trained on the phase-5A clean train split,
checkpoint-selected on val, then A/B'd against the shipped rhythm weights under
the unchanged production decode policy. ALL artifacts live under a fresh run
directory; shipped groom.pt/flow.pt/critic.pt/ladder.json and production caches
are NEVER written. See docs/specs/2026-09-21-clean-rhythm-pilot-design.md.

Task 1 here: safe experiment-output guard + model-injection seam. prepare/train/
evaluate/report stages follow in later tasks.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# shipped artifacts that must never be written by this experiment
PROTECTED_FILES = [ROOT / "groom.pt", ROOT / "flow.pt", ROOT / "critic.pt",
                   ROOT / "ladder.json"]
SEED = 20260921
# code whose identity binds the prepared tensors / trained checkpoint
CODE_FILES = ["groom.py", "convert.py", "timing.py", "motion.py", "parity.py",
              "eval/corpus.py", "eval/clean_rhythm.py", "eval/map_load_pool.py",
              "eval/prepared_dataset.py"]


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _code_hashes():
    return {f: _sha256(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()}


def _protected_digests():
    import groom
    caches = [groom._FEAT_CACHE, ROOT / "feats_cache4.pt"]
    files = PROTECTED_FILES + [c for c in caches if Path(c).exists()]
    return {str(Path(p).name): _sha256(p) for p in files if Path(p).exists()}


def guard_checkpoint_destination(out_path, run_dir, protected_paths):
    """Raise ValueError (before any training/save) unless out_path is a FRESH
    file strictly inside run_dir and is not — directly, by symlink, or by
    hardlink — any protected shipped file. Returns the resolved path."""
    out = Path(out_path)
    run = Path(run_dir).resolve()
    rp = out.resolve()                       # resolve() follows symlinks
    if run != rp and run not in rp.parents:
        raise ValueError(f"{out} is outside the run dir {run}")
    for prot in protected_paths:
        pp = Path(prot)
        if rp == pp.resolve():
            raise ValueError(f"{out} resolves to protected {prot}")
        if out.exists() and pp.exists() and os.path.samefile(out, pp):
            raise ValueError(f"{out} is a hardlink to protected {prot}")
    if out.exists():
        raise ValueError(f"{out} already exists; use a fresh run path")
    return rp


# --- Task 2: freeze inputs + clean rebuild --------------------------------

def _verify_panel(panel):
    """Every panel osu/audio must exist and match its frozen hash."""
    bad = []
    for e in panel["entries"]:
        for kind, ph in (("osu", "osu_sha256"), ("audio", "audio_sha256")):
            p = Path(e[kind])
            if not p.exists():
                bad.append({"song": e["song"], "reason": f"{kind}_missing"})
            elif _sha256(p) != e[ph]:
                bad.append({"song": e["song"], "reason": f"{kind}_sha_mismatch"})
    return bad


def _inventory(tr, val):
    def fams(rows):
        return sorted({m[6] for m in rows if len(m) > 6})
    return {"n_train_charts": len(tr), "n_val_charts": len(val),
            "n_train_families": len(fams(tr)), "n_val_families": len(fams(val)),
            "train_family_reps": fams(tr), "val_family_reps": fams(val),
            "train_weight_sum": round(sum(m[5] for m in tr), 6)}


def _dataset_identity(run, rec):
    """Semantic identity binding a prepared dataset to this run's snapshot, code
    and feature cache. Computed identically by prepare (write) and train (read)."""
    from eval.prepared_dataset import build_identity
    feat = Path(rec["feat_cache"])
    consumed = {"feat_cache_file": _sha256(feat)} if feat.exists() else {}
    return build_identity(rec["manifest_snapshot_sha256"], rec["inventory"],
                          consumed, rec["code_hashes"])


def prepare(out, dataset_cache=False, loader_workers="1"):
    """Freeze the manifest snapshot, panel, protected digests and dataset
    inventory into a fresh run dir. Isolates the feature cache (copies the
    shipped audio-feature cache read-only — audio features are content-addressed
    by audio+FEAT_VERSION, manifest-independent — never writes the shipped one).
    dataset_cache=True additionally persists the exact post-_split rows for reuse
    by train --use-prepared-dataset. loader_workers selects the (opt-in) parallel
    map loader."""
    import eval.corpus as corpus
    import groom
    run = Path(out)
    run.mkdir(parents=True, exist_ok=True)
    snap = run / "corpus_manifest.snapshot.json"
    snap.write_bytes(Path(corpus.MANIFEST).read_bytes())   # working copy (901 census)
    panel_src = ROOT / "eval/clean_rhythm_panel.json"
    shutil.copy2(panel_src, run / "clean_rhythm_panel.json")
    panel = json.loads(panel_src.read_text())
    panel_bad = _verify_panel(panel)
    # read-only baseline snapshots of shipped weights/ladder (never their originals)
    (run / "baseline").mkdir(exist_ok=True)
    for p in PROTECTED_FILES:
        if p.exists():
            shutil.copy2(p, run / "baseline" / p.name)
    feat_run = run / "feats_cache.pt"
    if groom._FEAT_CACHE and Path(groom._FEAT_CACHE).exists() and not feat_run.exists():
        shutil.copy2(groom._FEAT_CACHE, feat_run)          # valid audio features, isolated

    orig_manifest, orig_feat = corpus.MANIFEST, groom._FEAT_CACHE
    try:
        corpus.MANIFEST = snap
        corpus.check()                                     # version+family+hash checker
        groom._FEAT_CACHE = feat_run
        groom._FEATS.clear()
        tr, val, dev, rng, prewarm = _split_maybe_parallel(loader_workers, feat_run)
        assert tr and val, f"empty split train={len(tr)} val={len(val)}"
        inv = _inventory(tr, val)
        split_state = rng.getstate()
    finally:
        corpus.MANIFEST, groom._FEAT_CACHE = orig_manifest, orig_feat
        groom._FEATS.clear()

    run_rec = {
        "stage": "prepared", "seed": SEED,
        "manifest_snapshot_sha256": _sha256(snap),
        "panel_sha256": _sha256(panel_src),
        "panel_bad": panel_bad,
        "benchmark_sha256": _sha256(ROOT / "eval/benchmark.json"),
        "code_hashes": _code_hashes(),
        "protected_digests": _protected_digests(),
        "feat_cache": str(feat_run),
        "loader_workers": loader_workers,
        "dataset_cached": bool(dataset_cache),
        "inventory": inv,
    }
    (run / "run.json").write_text(json.dumps(run_rec, indent=1, sort_keys=True))
    if prewarm is not None:
        (run / "feature_prewarm.json").write_text(
            json.dumps(prewarm, indent=1, sort_keys=True, default=list))
    if dataset_cache:
        from eval.prepared_dataset import write_prepared
        identity = _dataset_identity(run, run_rec)
        write_prepared(run, tr, val, split_state, identity)
        print(f"  persisted prepared dataset (identity-bound) to {run}")
    print(f"prepared {run}: train {inv['n_train_charts']} charts / "
          f"{inv['n_train_families']} fam, val {inv['n_val_charts']}/"
          f"{inv['n_val_families']}; panel_bad={len(panel_bad)}")
    if panel_bad:
        print("  PANEL FAILURES:", panel_bad)
    return run_rec


def _bind_and_split():
    import eval.corpus as corpus
    import groom
    tr, val, dev, rng = groom._split(groom.MAPS_DIRS, manifest=True)
    return tr, val, dev


def _split_maybe_parallel(loader_workers, feat_cache):
    """Serial when loader_workers=='1'; otherwise load maps through the ordered
    CPU pool with parallel feature-miss prewarming (byte-identical output).
    'auto' -> min(8, cpus); explicit N is clamped to the effective CPU count.
    Returns (tr, val, dev, rng, prewarm_stats|None)."""
    import groom
    if str(loader_workers) == "1":
        return (*groom._split(groom.MAPS_DIRS, manifest=True), None)
    from eval.map_load_pool import OrderedMapPool, auto_workers, effective_cpu_count
    if str(loader_workers) == "auto":
        workers = auto_workers(10 ** 6)
    else:
        workers = min(int(loader_workers), effective_cpu_count())
        if workers < 1:
            raise ValueError(f"invalid loader_workers {loader_workers}")
    pool = OrderedMapPool(workers, feat_cache)
    with pool:                                   # publishes the warmed cache on exit
        tr, val, dev, rng = groom._split(groom.MAPS_DIRS, manifest=True,
                                         map_executor=pool.map)
    return tr, val, dev, rng, dict(pool.stats)


def train(run, use_prepared_dataset=False, target="rhythm"):
    """Train one from-scratch model (target='rhythm' Groomer or 'flow' Flow) on
    the frozen split to a separate checkpoint. Protected shipped files are
    verified unchanged before and after. use_prepared_dataset reuses the verified
    cached rows (no live reload)."""
    import numpy as np
    import torch
    import eval.corpus as corpus
    import groom
    run = Path(run)
    rec = json.loads((run / "run.json").read_text())
    # protected files must be byte-identical to what prepare froze
    before = _protected_digests()
    for k, v in rec["protected_digests"].items():
        if before.get(k) != v:
            raise SystemExit(f"protected file {k} changed since prepare; abort")
    ckpt = run / "checkpoints" / f"{target}-seed20260921.pt"
    hist = run / "checkpoints" / f"{target}-history.json"
    guard_checkpoint_destination(ckpt, run, PROTECTED_FILES)

    orig_manifest, orig_feat = corpus.MANIFEST, groom._FEAT_CACHE
    t0 = time.time()
    try:
        corpus.MANIFEST = run / "corpus_manifest.snapshot.json"
        assert _sha256(corpus.MANIFEST) == rec["manifest_snapshot_sha256"]
        groom._FEAT_CACHE = Path(rec["feat_cache"])
        groom._FEATS.clear()
        if use_prepared_dataset:
            # reuse the verified cached rows; NO _split/load_dataset/features
            from eval.prepared_dataset import read_prepared
            tr, val, _state = read_prepared(run, _dataset_identity(run, rec))
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  reused prepared dataset: {len(tr)} train / {len(val)} val")
        else:
            tr, val, dev = _bind_and_split()
        # seed everything AFTER data load, BEFORE model init + sampling
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        if target == "rhythm":
            history = groom.train_rhythm(tr, val, dev, random.Random(SEED),
                                         out_path=ckpt, history_path=hist,
                                         max_epochs=200, steps_per_epoch=30)
        else:
            history = groom.train_flow(tr, val, dev, random.Random(SEED),
                                       out_path=ckpt, history_path=hist,
                                       max_epochs=600, steps_per_epoch=30)
    finally:
        corpus.MANIFEST, groom._FEAT_CACHE = orig_manifest, orig_feat
        groom._FEATS.clear()

    # reload the saved checkpoint independently and re-measure selected val
    import torch.nn as nn
    if target == "rhythm":
        m = groom.Groomer()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            [groom.POS_WEIGHT, groom.POS_WEIGHT, groom.WALL_POS_WEIGHT,
             groom.WALL_POS_WEIGHT]))
        reload_val, f1n, f1w = groom.eval_rhythm(m, val, "cpu", lossf)
        reload_extra = {"reload_note_f1": f1n, "reload_wall_f1": f1w}
    else:
        m = groom.Flow()
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        reload_val, dacc = groom.eval_flow(m, val, "cpu")
        reload_extra = {"reload_dir_acc": dacc}
    learned = (history["best_val_loss"] < history["initial_val_loss"]
               and all(x == x for x in (history["best_val_loss"],)))  # finite
    after = _protected_digests()
    if after != before:
        raise SystemExit("protected files changed during training; abort")
    summary = {
        "stage": "trained", "target": target, "checkpoint": str(ckpt),
        "checkpoint_sha256": _sha256(ckpt),
        "initial_val_loss": history["initial_val_loss"],
        "best_val_loss": history["best_val_loss"], "best_epoch": history["best_epoch"],
        "updates": history["updates"], "reload_val_loss": reload_val,
        "learned": bool(learned),
        "runtime_s": round(time.time() - t0, 1), "device": dev,
        "protected_unchanged": True, **reload_extra,
    }
    (run / "train_summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"trained: best val {history['best_val_loss']:.4f} @epoch "
          f"{history['best_epoch']} (init {history['initial_val_loss']:.4f}), "
          f"reload val {reload_val:.4f}, learned={learned}, {summary['runtime_s']}s")
    return summary


# --- Task 3: frozen two-checkpoint A/B ------------------------------------

NOMINAL_EYE = None   # (unused; kept for symmetry with other packets)


def _load_groomer(state_path):
    import torch
    import groom
    m = groom.Groomer()
    m.load_state_dict(torch.load(state_path, map_location="cpu"))
    m.eval()
    return m


def _load_flow(state_path):
    import torch
    import groom
    m = groom.Flow()
    m.load_state_dict(torch.load(state_path, map_location="cpu"))
    m.eval()
    return m


def _cand_metrics(notes_steps, motion, grid, beat_ms):
    """Metric bundle for one decoded candidate: dup8 + rate (map_reader on
    beat-indexed notes) and motion (from the collect record's report)."""
    from eval.map_reader import metrics as mr_metrics
    beat_notes = sorted((grid.time(s) / beat_ms, h, c, l, d)
                        for s, h, c, l, d in notes_steps)
    mm = mr_metrics(beat_notes, beat_ms)
    lr = motion["lr_doubles"]
    return {"dup8": mm.get("raw_dup8_maxphase_pct", 0.0),
            "raw_sps": mm.get("raw_sps", 0.0),
            "flags_by_ext": {e: v for e, v in motion["flags_by_ext"].items()},
            "narrow": len(lr["narrow"]), "conv": lr["converging"],
            "longest_run": motion["workload"]["longest_run"]["count"]}


def _arm_metrics(recs, winner, grid, beat_ms, band):
    """Combine the production winner + the mean of the paired initial six into
    one gate_song-shaped dict for an arm."""
    import motion as motion_mod
    six = [r for r in recs if r.get("pass_index") == 0
           and abs(r.get("rate_scale", 1.0) - 1.0) < 1e-9 and 0 <= r["seed"] <= 5]
    six_m = [_cand_metrics(r["notes"], r["motion"], grid, beat_ms) for r in six]
    win_ms = [(grid.time(s), h, c, l, d) for s, h, c, l, d in winner]
    win_motion = motion_mod.report(win_ms)
    wm = _cand_metrics(winner, win_motion, grid, beat_ms) if winner else None
    exts = ("0.0", "0.25", "0.5", "0.75")

    def mean(key_fn):
        vals = [key_fn(m) for m in six_m]
        return sum(vals) / len(vals) if vals else 0.0
    valid = bool(winner) and any(r.get("ok") for r in recs)
    in_band = bool(wm) and band[0] * 0.98 <= wm["raw_sps"] <= band[1] * 1.02
    return {
        "valid": valid, "in_band": in_band,
        "dup8_winner": wm["dup8"] if wm else 1e9,
        "dup8_six_mean": mean(lambda m: m["dup8"]),
        "flags_by_ext": {e: (wm["flags_by_ext"].get(e, 0.0) if wm else 1e9,
                             mean(lambda m: m["flags_by_ext"].get(e, 0.0)))
                         for e in exts},
        "narrow_winner": wm["narrow"] if wm else 1e9,
        "narrow_six_mean": mean(lambda m: m["narrow"]),
        "conv_winner": wm["conv"] if wm else 1e9,
        "conv_six_mean": mean(lambda m: m["conv"]),
        "longest_run_winner": wm["longest_run"] if wm else 1e9,
        "longest_run_six_mean": mean(lambda m: m["longest_run"]),
        "raw_sps_winner": wm["raw_sps"] if wm else None,
        "n_six": len(six),
    }


def _winner_quiet(winner, grid, T):
    from eval.clean_rhythm_eval import shared_rest_bins
    notes = [{"t": grid.time(s), "dir": d} for s, h, c, l, d in winner]
    start, end = grid.time(0), grid.time(T - 1)
    return shared_rest_bins(notes, start, end)


def _resolve(rel):
    """Panel osu/audio paths are stored repo-relative; bind them to ROOT so the
    stage is CWD-independent."""
    return ROOT / rel


def evaluate(run, target="rhythm", rhythm_ckpt=None, flow_ckpt=None):
    """Decode all 8 panel songs under the shipped (A) and new (B) model for the
    given target, with the unchanged production policy and the OTHER models
    frozen at their shipped weights; record per-song A/B metrics. target='both'
    compares shipped-both (A) vs clean-both (B) from explicit rhythm_ckpt +
    flow_ckpt (critic/ladder stay shipped). target='flow'
    swaps the Flow (rhythm/critic/ladder stay shipped); 'rhythm' swaps the
    Groomer (flow/critic/ladder stay shipped)."""
    import torch
    import eval.corpus as corpus
    import groom
    from convert import convert_groomed, grid_steps, parse_osu, diff_spec
    run = Path(run)
    rec = json.loads((run / "run.json").read_text())
    ts_path = run / "train_summary.json"
    ts = json.loads(ts_path.read_text()) if ts_path.exists() else {}
    before = _protected_digests()
    torch.set_num_threads(4)                         # tiny-GEMM decode gotcha
    # per-arm convert kwargs: arm A = shipped, arm B = the retrained checkpoint(s).
    # 'both' injects BOTH clean checkpoints (the 4th 2x2 cell); rhythm/critic/
    # ladder (or flow, per target) stay shipped for both arms.
    ckpt = run / "checkpoints" / f"{target}-seed20260921.pt"
    if target == "rhythm":
        a_kw = {"rhythm_model": _load_groomer(ROOT / "groom.pt")}
        b_kw = {"rhythm_model": _load_groomer(ckpt)}
        ck_sha = {"checkpoint_sha256": _sha256(ckpt)}
    elif target == "flow":
        a_kw = {"flow_model": _load_flow(ROOT / "flow.pt")}
        b_kw = {"flow_model": _load_flow(ckpt)}
        ck_sha = {"checkpoint_sha256": _sha256(ckpt)}
    else:  # both: shipped-both vs clean-both, from two explicit checkpoints
        rck, fck = Path(rhythm_ckpt), Path(flow_ckpt)
        a_kw = {}                                    # shipped rhythm + shipped flow
        b_kw = {"rhythm_model": _load_groomer(rck), "flow_model": _load_flow(fck)}
        ck_sha = {"rhythm_ckpt_sha256": _sha256(rck), "flow_ckpt_sha256": _sha256(fck)}
    panel = json.loads((run / "clean_rhythm_panel.json").read_text())
    band = diff_spec("ExpertPlus")["band"]
    orig_feat = groom._FEAT_CACHE
    groom._FEAT_CACHE = Path(rec["feat_cache"])
    groom._FEATS.clear()
    songs, raw = {}, {}
    try:
        for e in panel["entries"]:
            meta, objects, bpm, offset = parse_osu(_resolve(e["osu"]))
            steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset, thin=True)
            beat_ms = 60000.0 / bpm
            arms = {}
            for tag, kw in (("a", a_kw), ("b", b_kw)):
                recs = []
                convert_groomed(objects, bpm, offset,
                                audio_path=str(_resolve(e["audio"])),
                                diff="ExpertPlus", replay_mode="off",
                                collect=recs, thin=True, calibrate=True,
                                **kw)
                # convert_groomed returns (note-dicts, wall-dicts); the metric
                # helpers want the winner's step 5-tuples, which convert marks in
                # collect via `selected` (identity with the chosen raw candidate)
                winner = next((r["notes"] for r in recs if r.get("selected")), [])
                arms[tag] = _arm_metrics(recs, winner, grid, beat_ms, band)
                arms[tag + "_winner_len"] = len(winner)
                arms[tag + "_quiet_obj"] = _winner_quiet(winner, grid, T)
            # rest gate: A/B quiet ids on the shared grid domain
            arms["b"]["quiet"] = {"a_ids": arms["a_quiet_obj"]["quiet_ids"],
                                  "b_ids": arms["b_quiet_obj"]["quiet_ids"],
                                  "n_bins": arms["b_quiet_obj"]["n_bins"]}
            arms["a"]["quiet"] = arms["b"]["quiet"]
            songs[e["song"]] = {"a": arms["a"], "b": arms["b"]}
            print(f"  {e['song']}: A valid={arms['a']['valid']} "
                  f"B valid={arms['b']['valid']} A_sps={arms['a']['raw_sps_winner']} "
                  f"B_sps={arms['b']['raw_sps_winner']}")
    finally:
        groom._FEAT_CACHE = orig_feat
        groom._FEATS.clear()
    if _protected_digests() != before:
        raise SystemExit("protected files changed during A/B; abort")
    # 'both' has no local train_summary (both checkpoints were trained in their
    # own runs and verified learned there) -> treat as learned=True.
    learned = ts.get("learned", target == "both")
    out = {"songs": songs, "target": target, "learned": bool(learned),
           "integrity_ok": True, **ck_sha}
    (run / "ab_report.json").write_text(json.dumps(out, indent=1, sort_keys=True,
                                                   default=lambda o: None))
    print(f"A/B decoded {len(songs)} songs -> {run}/ab_report.json")
    return out


def report(run):
    from eval.clean_rhythm_eval import evaluate_acceptance
    run = Path(run)
    ab = json.loads((run / "ab_report.json").read_text())
    ts_path = run / "train_summary.json"
    ts = json.loads(ts_path.read_text()) if ts_path.exists() else {}
    verdict = evaluate_acceptance(ab)
    # 'both' runs carry no local train_summary; the ab_report records learned.
    learned = ab.get("learned", ts.get("learned"))
    baseline = "BASELINE_BUILT" if learned else "INCOMPLETE"
    out = {"baseline_state": baseline, "ab": verdict,
           "best_val_loss": ts.get("best_val_loss"),
           "initial_val_loss": ts.get("initial_val_loss"),
           "reload_val_loss": ts.get("reload_val_loss"),
           "checkpoint_sha256": ts.get("checkpoint_sha256")}
    (run / "verdict.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"VERDICT: {baseline} / {verdict['status']} | failures "
          f"{verdict.get('failures')}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset-cache", action="store_true")
    p.add_argument("--loader-workers", default="1")
    for name in ("train", "evaluate", "report"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name == "train":
            s.add_argument("--model", choices=("rhythm", "flow"),
                           default="rhythm")
            s.add_argument("--use-prepared-dataset", action="store_true")
        if name == "evaluate":
            s.add_argument("--model", choices=("rhythm", "flow", "both"),
                           default="rhythm")
            s.add_argument("--rhythm-ckpt")   # required for --model both
            s.add_argument("--flow-ckpt")     # required for --model both
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.out, dataset_cache=a.dataset_cache,
                loader_workers=a.loader_workers)
    elif a.cmd == "train":
        train(a.run, use_prepared_dataset=a.use_prepared_dataset,
              target=a.model)
    elif a.cmd == "evaluate":
        if a.model == "both" and not (a.rhythm_ckpt and a.flow_ckpt):
            ap.error("--model both requires --rhythm-ckpt and --flow-ckpt")
        evaluate(a.run, target=a.model, rhythm_ckpt=a.rhythm_ckpt,
                 flow_ckpt=a.flow_ckpt)
    else:
        report(a.run)


if __name__ == "__main__":
    main()
