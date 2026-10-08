"""Opt-in ordered CPU map pool with PARALLEL feature-miss prewarming.

Workers parse a directory and build tensors using a read-only feature-cache
snapshot; a miss returns a structured request. The PARENT deduplicates genuine
misses (in canonical directory order), dispatches one feature job per unique key
to the SAME spawned workers (librosa is bit-deterministic across processes — see
the prewarm design/tests), merges results in request order, then retries only
the affected maps with the resolved tensor supplied. Only the parent publishes
the run-local cache, once, after all splits. Dataset order/weights are identical
to serial. See docs/specs/2026-09-21-parallel-feature-prewarm-design.md.
"""
import multiprocessing as mp
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_WCACHE = {}                 # per-worker read-only feature snapshot (from init)
_DELTA = {}                  # per-call resolved-feature delta for retries


def effective_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def auto_workers(n_dirs):
    return max(1, min(8, effective_cpu_count(), max(1, n_dirs)))


def _worker_init(cache_path):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)
    global _WCACHE
    import torch as _t
    _WCACHE = _t.load(cache_path) if Path(cache_path).exists() else {}


def _provider_factory(delta):
    def provider(path, step_times):
        import groom
        key = groom._feat_key(path, step_times)
        v = delta.get(key)
        if v is None:
            v = _WCACHE.get(key)
        if v is None or len(v) != len(step_times):
            raise groom.FeatureMiss(path, step_times)
        return v
    return provider


def _worker(dir_str):
    import groom
    try:
        samples = groom.load_map_all(Path(dir_str),
                                     feature_provider=_provider_factory({}))
        return {"tag": "ok", "samples": samples}
    except groom.FeatureMiss as m:
        return {"tag": "miss", "dir": dir_str, "path": m.path,
                "step_times": m.step_times, "key": groom._feat_key(m.path, m.step_times)}


def _feature_worker(request):
    import groom
    try:
        t0 = time.perf_counter()
        feats = groom.audio_features(request["path"], request["step_times"])
        return {"key": request["key"], "tensor": feats,
                "elapsed_s": time.perf_counter() - t0, "pid": os.getpid()}
    except Exception as e:                        # ordinary audio failure (tagged)
        return {"key": request["key"], "error": f"{type(e).__name__}: {e}",
                "path": request["path"]}


def _retry_worker(task):
    import groom
    try:
        samples = groom.load_map_all(Path(task["dir"]),
                                     feature_provider=_provider_factory(task["delta"]))
        return {"tag": "ok", "samples": samples}
    except groom.FeatureMiss:
        return {"tag": "miss", "dir": task["dir"]}   # second miss = failure


def _bounded(ex, fn, items, window):
    """Submit in order, keep <= window in flight, yield results in input order."""
    it, futs, out = iter(items), deque(), []
    for _ in range(max(1, window)):
        try:
            futs.append(ex.submit(fn, next(it)))
        except StopIteration:
            break
    while futs:
        out.append(futs.popleft().result())
        try:
            futs.append(ex.submit(fn, next(it)))
        except StopIteration:
            pass
    return out


class OrderedMapPool:
    """map_executor: `.map(dirs)` returns one samples-dict per input dir, in
    input order, warming missing features in parallel. Publishes the merged run
    cache once on a successful context exit."""

    def __init__(self, workers, feature_cache, inventory_sink=None):
        self.workers = int(workers)
        self.cache = str(feature_cache)
        self.inventory_sink = inventory_sink
        self._ex = None
        self._parent = {}          # authoritative run-cache (persists across .map)
        self._dirty = False
        self.stats = {"workers": self.workers, "ok": 0, "miss": 0,
                      "unique_features_computed": 0, "parent_feature_calls": 0,
                      "parent_hits_on_stale_miss": 0, "feature_errors": 0,
                      "retries": 0, "cache_writes": 0, "job_durations_s": [],
                      "job_pids": []}

    def __enter__(self):
        import torch
        ctx = mp.get_context("spawn")
        self._ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx,
                                       initializer=_worker_init,
                                       initargs=(self.cache,))
        self._parent = torch.load(self.cache) if Path(self.cache).exists() else {}
        self._init_hash = _dict_len(self._parent)
        return self

    def map(self, dirs):
        dirs = list(dirs)
        res = _bounded(self._ex, _worker, [str(d) for d in dirs], 2 * self.workers)
        out = [None] * len(dirs)
        misses = []                                  # (idx, dir, path, step_times, key)
        for i, r in enumerate(res):
            if r["tag"] == "ok":
                self.stats["ok"] += 1
                out[i] = r["samples"]
            else:
                self.stats["miss"] += 1
                misses.append((i, r["dir"], r["path"], r["step_times"], r["key"]))
        # split misses into parent-hits (warmed by a prior .map) vs genuine misses,
        # deduplicating genuine keys in canonical (first-directory) order
        import groom
        n_audio = groom.N_AUDIO
        ordered_keys, aliases = [], {}
        for i, d, path, st, key in misses:
            if key in self._parent and len(self._parent[key]) == len(st):
                self.stats["parent_hits_on_stale_miss"] += 1
                continue
            if key not in aliases:
                aliases[key] = []
                ordered_keys.append(key)
            aliases[key].append((path, st))

        def _valid(t, st):
            return list(t.shape) == [len(st), n_audio] and str(t.dtype) == "torch.float32"

        def _accept(key, r, st):
            if not _valid(r["tensor"], st):
                raise ValueError(f"bad feature result shape/dtype for {key}")
            self._parent[key] = r["tensor"]           # merge in canonical order
            self._dirty = True
            self.stats["job_durations_s"].append(r["elapsed_s"])
            self.stats["job_pids"].append(r["pid"])

        # compute EVERY unique missing key ONCE, in PARALLEL (first alias each)
        reqs = [{"key": k, "path": aliases[k][0][0], "step_times": aliases[k][0][1]}
                for k in ordered_keys]
        results = _bounded(self._ex, _feature_worker, reqs, 2 * self.workers)
        for k, r in zip(ordered_keys, results):
            self.stats["unique_features_computed"] += 1
            if "error" not in r:
                _accept(k, r, aliases[k][0][1])
                continue
            # first alias failed: advance through remaining aliases (rare)
            for path, st in aliases[k][1:]:
                rr = _bounded(self._ex, _feature_worker,
                              [{"key": k, "path": path, "step_times": st}], 1)[0]
                if "error" not in rr:
                    _accept(k, rr, st)
                    break
            else:
                self.stats["feature_errors"] += 1
        # retry the missed maps with their resolved feature delta (no serial parse)
        retry_tasks, retry_idx = [], []
        for i, d, path, st, key in misses:
            if key in self._parent and len(self._parent[key]) == len(st):
                retry_tasks.append({"dir": d, "delta": {key: self._parent[key]}})
                retry_idx.append(i)
            else:
                out[i] = {}                           # feature failed -> excluded
        if retry_tasks:
            rres = _bounded(self._ex, _retry_worker, retry_tasks, 2 * self.workers)
            for i, r in zip(retry_idx, rres):
                self.stats["retries"] += 1
                if r["tag"] == "miss":
                    raise RuntimeError(f"second miss on retry: {r['dir']}")
                out[i] = r["samples"]
        return out

    def close(self, publish=True):
        import torch
        if self._ex is not None:
            if publish and self._dirty:
                tmp = Path(self.cache + ".tmp")
                torch.save(self._parent, tmp)
                tmp.replace(self.cache)
                self.stats["cache_writes"] += 1
                import groom
                groom._FEATS.clear()
                groom._FEATS.update(self._parent)     # sync parent memory
            self._ex.shutdown()
            self._ex = None

    def __exit__(self, exc_type, exc, tb):
        # publish only on a clean exit; a failure leaves the run cache intact
        self.close(publish=exc_type is None)
        return False


def _dict_len(d):
    return len(d)
