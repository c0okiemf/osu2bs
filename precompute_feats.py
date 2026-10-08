"""Parallel audio-feature precompute over the training corpus.
Workers write per-shard caches (seeded from the main cache so reruns skip
done songs), then shards merge back into the main cache. Resumable; a killed
run's partial shards are salvaged on the next start."""
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import torch

import groom

HERE = Path(__file__).parent
MAIN = groom._FEAT_CACHE
NPROC = 6


def merge_shards():
    cache = torch.load(MAIN) if MAIN.exists() else {}
    for sp in HERE.glob("feats_shard_*.pt"):
        try:
            cache.update(torch.load(sp))
        except Exception:
            pass  # half-written shard from a kill; recomputed below
        sp.unlink()
    torch.save(cache, MAIN)
    return cache


def work(arg):
    i, ds = arg
    groom._FEAT_CACHE = HERE / f"feats_shard_{i}.pt"
    for j, d in enumerate(ds):
        try:
            groom.load_map_all(d)
        except Exception as e:
            print(f"shard{i} {d.name[:40]}: {type(e).__name__} {e}", flush=True)
        if (j + 1) % 20 == 0:
            print(f"shard{i}: {j + 1}/{len(ds)}", flush=True)


if __name__ == "__main__":
    ds = [d for md in groom.MAPS_DIRS if Path(md).exists()
          for d in sorted(Path(md).iterdir()) if d.is_dir()]
    print(f"{len(ds)} map dirs", flush=True)
    merge_shards()  # salvage leftovers from an interrupted run
    for i in range(NPROC):
        if MAIN.exists():
            shutil.copyfile(MAIN, HERE / f"feats_shard_{i}.pt")
    with Pool(NPROC) as p:
        p.map(work, [(i, ds[i::NPROC]) for i in range(NPROC)])
    print(f"cache: {len(merge_shards())} songs", flush=True)
