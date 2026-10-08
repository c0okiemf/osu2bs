"""Bounded acceptance runner for the dataset pipeline speed packet.

Proves the parallel loader and the prepared-dataset cache produce a BIT-IDENTICAL
dataset and checkpoint versus the serial path, and measures the speedup, then
classifies the outcome. No full retrain / A/B decode / production write. See
docs/specs/2026-09-21-pipeline-speed-design.md.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260921


def classify(exact, repeatable, serial_prepare, parallel_prepare, live_load,
             cached_load, original_combined, optimized_combined,
             cache_only_combined):
    """Reduce measurements to one of the four spec states.
    exact: {cache, pool, checkpoint_cache, checkpoint_pool} booleans."""
    if not repeatable:
        return "INCONCLUSIVE"
    cache_exact = bool(exact.get("cache") and exact.get("checkpoint_cache"))
    pool_exact = bool(exact.get("pool") and exact.get("checkpoint_pool"))
    pool_ok = serial_prepare and parallel_prepare and \
        serial_prepare / parallel_prepare >= 1.5
    reuse_ok = live_load and cached_load is not None and \
        cached_load / live_load <= 0.25
    both_ok = original_combined and optimized_combined and \
        original_combined / optimized_combined >= 2.0
    cache_ok = original_combined and cache_only_combined and \
        original_combined / cache_only_combined >= 1.7
    if cache_exact and pool_exact and reuse_ok and pool_ok and both_ok:
        return "CACHE_AND_POOL_ACCEPTED"
    if cache_exact and reuse_ok and cache_ok:
        return "CACHE_ONLY_ACCEPTED"
    return "NOT_ACCEPTED"


def _prepare_arm(arm_dir, workers, dataset_cache=True):
    """Run a fresh clean_rhythm prepare into arm_dir; return elapsed seconds."""
    import eval.clean_rhythm as cr
    t0 = time.time()
    cr.prepare(str(arm_dir), dataset_cache=dataset_cache, loader_workers=workers)
    return time.time() - t0


def _load_dataset_digest(run_dir):
    """dataset.pt canonical digest for a prepared run."""
    import torch
    from eval.prepared_dataset import dataset_digest
    tr, val, state = torch.load(run_dir / "dataset.pt", map_location="cpu")
    return dataset_digest(tr, val, state), (tr, val, state)


def _rehearsal_checkpoint(tr, val, out):
    """Deterministic tiny CPU rehearsal: first 8 train / 4 val, 2 epochs x 3
    updates, seed 20260921. Returns the checkpoint bytes."""
    import numpy as np
    import torch
    import groom
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    groom.train_rhythm(tr[:8], val[:4], "cpu", random.Random(SEED),
                       out_path=out, history_path=str(out) + ".hist.json",
                       max_epochs=2, steps_per_epoch=3)
    return Path(out).read_bytes()


def run(out, workers="auto"):
    import torch
    from eval.prepared_dataset import read_prepared
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; use a fresh dir")
    out.mkdir(parents=True)
    prot_before = _protected()

    # timing: interleaved S,P,P,S, independent run dirs
    order = [("serial", "1"), ("parallel", workers),
             ("parallel", workers), ("serial", "1")]
    times = {"serial": [], "parallel": []}
    digests = {}
    for i, (mode, w) in enumerate(order):
        d = out / f"arm{i}-{mode}"
        el = _prepare_arm(d, w, dataset_cache=True)
        times[mode].append(el)
        digests[f"arm{i}-{mode}"] = _load_dataset_digest(d)[0]
        print(f"  {mode} prepare #{i}: {el:.1f}s digest {digests[f'arm{i}-{mode}'][:12]}")

    # exactness: all serial digests equal each other, all parallel equal serial
    serial_digs = [digests[k] for k in digests if k.endswith("serial")]
    par_digs = [digests[k] for k in digests if k.endswith("parallel")]
    serial_repeatable = len(set(serial_digs)) == 1
    cache_exact = serial_repeatable            # cache round-trip proven below too
    pool_exact = serial_repeatable and set(par_digs) == set(serial_digs)

    # cache reuse timing + exactness: read prepared from the first serial arm
    ref = out / "arm0-serial"
    rec = json.loads((ref / "run.json").read_text())
    import eval.clean_rhythm as cr
    ident = cr._dataset_identity(ref, rec)
    t0 = time.time()
    tr_c, val_c, st_c = read_prepared(ref, ident)
    cached_load = time.time() - t0
    from eval.prepared_dataset import dataset_digest
    cache_exact = cache_exact and dataset_digest(tr_c, val_c, st_c) == digests["arm0-serial"]

    # checkpoint rehearsal: serial-live vs pool-built vs cache-reloaded
    _, (tr_s, val_s, _) = _load_dataset_digest(ref)
    _, (tr_p, val_p, _) = _load_dataset_digest(out / "arm1-parallel")
    ck_s = _rehearsal_checkpoint(tr_s, val_s, out / "ck_serial.pt")
    ck_s2 = _rehearsal_checkpoint(tr_s, val_s, out / "ck_serial2.pt")
    ck_p = _rehearsal_checkpoint(tr_p, val_p, out / "ck_parallel.pt")
    ck_c = _rehearsal_checkpoint(tr_c, val_c, out / "ck_cache.pt")
    ck_repeatable = ck_s == ck_s2
    exact = {"cache": cache_exact, "pool": pool_exact,
             "checkpoint_cache": ck_c == ck_s, "checkpoint_pool": ck_p == ck_s}

    serial_med = _median(times["serial"])
    par_med = _median(times["parallel"])
    live_load = serial_med           # a live split load ~ a serial prepare load
    original_combined = 2 * serial_med                 # two live loads (prepare+train)
    optimized_combined = par_med + cached_load         # parallel prepare + cached reuse
    cache_only_combined = serial_med + cached_load     # serial prepare + cached reuse
    verdict = classify(exact, serial_repeatable and ck_repeatable, serial_med,
                       par_med, live_load, cached_load, original_combined,
                       optimized_combined, cache_only_combined)

    prot_after = _protected()
    report = {
        "verdict": verdict, "workers": workers,
        "serial_prepare_median_s": round(serial_med, 1),
        "parallel_prepare_median_s": round(par_med, 1),
        "cached_load_s": round(cached_load, 3),
        "speedup_prepare": round(serial_med / par_med, 2) if par_med else None,
        "combined_speedup_pool": round(original_combined / optimized_combined, 2)
        if optimized_combined else None,
        "combined_speedup_cache_only": round(original_combined / cache_only_combined, 2)
        if cache_only_combined else None,
        "exact": exact, "serial_repeatable": serial_repeatable,
        "checkpoint_repeatable": ck_repeatable,
        "serial_times_s": [round(t, 1) for t in times["serial"]],
        "parallel_times_s": [round(t, 1) for t in times["parallel"]],
        "protected_unchanged": prot_before == prot_after,
        "code_rev": _rev(),
    }
    (out / "speed_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"VERDICT: {verdict} | prepare {serial_med:.0f}s->{par_med:.0f}s "
          f"({report['speedup_prepare']}x) | combined pool "
          f"{report['combined_speedup_pool']}x cache-only "
          f"{report['combined_speedup_cache_only']}x | exact={exact}")
    return report


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _protected():
    import eval.clean_rhythm as cr
    return cr._protected_digests()


def _rev():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", default="auto")
    a = ap.parse_args(argv)
    return run(a.out, a.workers)


if __name__ == "__main__":
    main()
