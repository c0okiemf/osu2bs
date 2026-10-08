# Bounded pipeline speed packet: prepare once, load maps concurrently

> **Superseded for cache misses (2026-09-21):** Profiling measured a roughly 50% miss
> rate, dominated by librosa. The parent-only feature restriction below is
> replaced by the [parallel prewarm correction](2026-09-21-parallel-feature-prewarm-design.md).
> Its bounded acceptance replaces this document's repeated timing schedule.

Date: 2026-09-21.

**Decision: optimize only the clean-rhythm data preparation/reuse path in this packet.** Add an identity-bound prepared dataset and opt-in CPU map loading with canonical reassembly. Keep the serial path as the reference. Resume the clean-rhythm pilot after this packet's equivalence and speed gates pass.

## Evidence, scope and alternatives

Use the recorded profile without another discovery benchmark: approximately 12 minutes per sequential split load, called once in prepare and again in train, with an almost-warm 33–34MB feature cache. Target redundant loading and per-map CPU work; do not optimize librosa, change features or alter training math.

Cache reuse alone removes the second full build; combine it with a bounded map-worker pool to improve the first. Defer A/B multiprocessing: `eval.clean_rhythm` currently has prepare/train but evaluate/report are stubs, so there is no implemented serial A/B runner to preserve yet. Building that evaluator inside a performance packet would mix two projects. Defer GPU decode, dual-GPU training, batching, mixed precision, compiler/model changes and new numerical kernels. GPU availability is not a reason to change CPU decode semantics.

**Code correction to the profile:** `groom._feat_key` already hashes audio contents plus the grid rounded to 3 decimals plus `FEAT_VERSION`. Preserve it exactly, including rounding; do not replace it with a path key or “fix” its grid semantics in this packet.

## Boundaries

- Production defaults remain serial and unchanged. New experiment flags are opt-in: prepare `--dataset-cache --loader-workers {1,auto,N}`, train `--use-prepared-dataset`. Workers default 1; no flags retains the existing path.
- All cache/checkpoint/profiling artifacts stay in fresh experiment directories. Never write shipped models, ladder, production caches or corpus source files. Preserve current uncommitted clean-rhythm implementation work.
- No training distribution, filtering, feature math, randomness, sample order, family weight, CROP/CTX behavior or decoder changes.
- No A/B generation, full retrain, held-out-score optimization or new dependency in this speed packet. A small deterministic training rehearsal is solely an equivalence check.

## 1. Prepared dataset reuse

At the end of successful prepare, persist the **exact returned train/val rows from `_split`**: train is already shuffled once by `Random(0)`, validation retains its existing order. Save before the trainer's CROP filter/renormalization. Preserve tuple/list types, seven fields, CPU tensor dtype/shape/value bits, event order, weights and family keys. Save the returned split RNG state too; never shuffle cached train rows again. The actual trainer still seeds its training RNGs after loading exactly as now.

Use `dataset.pt` for the CPU data payload and a separate canonical identity manifest. The identity includes:

- corpus snapshot bytes/hash and exact train/val representative sequence;
- each consumed Info/chart/audio path and content hash, including files resolved by fallback;
- emitted chart order per directory; loader-relevant directory inventories and absent-file sentinels for skipped/fallback candidates, so adding a formerly missing chart/audio invalidates reuse; loader/feature/timing/split code hashes, relevant configuration/constants and schema version;
- feature-cache version and keys/value digests actually consumed (not the entire cache's mutable unrelated contents);
- torch/NumPy versions, ordered dataset-content digest, and split RNG state.

Do not include worker count, completion order, run-directory name, PIDs or timing logs in semantic identity. Execution-mode metadata belongs in a sidecar. Changing an unrelated report writer does not invalidate tensors; changing loader/features/split math does. Adding an unused feature-cache entry does not invalidate tensors; changing a consumed feature value does.

On `train --use-prepared-dataset`, verify identity and source hashes, load CPU rows, restore the recorded split state where returned, and enter the existing trainer. Do not call `_split`, `load_dataset`, `load_map_all`, `cached_audio_features` or audio decoding in this branch. Device choice is made at training time, not frozen in the CPU payload. Missing, stale, corrupt or incomplete data fails explicitly; no silent 12-minute rebuild. Re-preparation is an explicit action in a fresh directory.

Write payload to a temporary file in the run, then atomic rename; publish the identity/complete marker last, referencing its checksum. Readers require both. Failure cannot publish a reusable partial dataset. Hash/schema checks are not an optional fast mode; include their cost in timings.

## 2. Ordered per-map CPU pool

Refactor only enough to share the serial parent assembly: enumerate directories exactly as today's loader does, skip held-out directories at the same point, preserve each `load_map_all` dictionary's insertion order, and compute approved/base weights and per-chart family budgets in the parent using the existing logic. Do not sort directories or difficulty names into a new “canonical” order. Here canonical means **the current serial order**, not lexicographic order.

Workers perform one directory's existing parsing/tensor construction. Tag tasks/results with the original ordinal. Consume/reassemble by ordinal, never completion time. Apply `_split`'s existing shuffle once after complete assembly. `_renorm_by_family` remains exclusively where it currently runs after the trainer's length filter. An empty-map result retains its position/status; a worker crash is not an empty map and cannot silently drop a family.

Use a spawned process pool; never fork a CUDA-initialized process. Workers are CPU-only, with torch intra-op/inter-op and BLAS/OpenMP threads 1, set before numerical work (environment before child imports). Restore parent environment after launch; do not modify parent training threads, CUDA settings or RNG state. Existing feature work done by the parent uses its unchanged serial numerical environment.

Auto worker count: `min(8, effective_cpu_count, n_eligible_dirs)`, minimum 1. Effective CPUs respect process affinity and, where present, the CPU quota; explicit N must not exceed the effective count. On an unconstrained 24-core box auto uses 8 workers. This is a conservative memory/process-startup bound, not a claim of optimal occupancy. Keep at most `2*workers` submitted-but-unassembled tasks, counting out-of-order completed results against the bound. Do not preload the entire tensor dataset into every worker or create a new pool for each map. Record combined parent/worker peak memory and actual settings.

### Feature-cache ownership

Workers must **never** call the disk-writing `cached_audio_features` path. Add an optional feature-provider argument to `load_map_all`; its default remains the current function. Parallel workers receive a read-only snapshot of the run's feature cache and a lookup-only provider using the unchanged key function and length check.

On a miss, return a structured request containing map ordinal, audio identity, full grid and expected key; no partial sample is accepted. The parent deduplicates requests in canonical map order, calls the unchanged `audio_features` serially, and supplies the new CPU tensors to retry the affected maps. This avoids making numerical feature equality depend on worker thread settings. `load_map_all` currently catches feature exceptions: propagate the dedicated cache-miss signal before that broad catch, rather than silently returning an empty map.

Use two bounded passes for this loader's one feature request per map: initial read-only loads; parent fills unique misses; retry only those maps with the relevant feature deltas. During the initial pass an ordinal miss placeholder counts as assembled into the parent result slots, releasing the queue bound; it is not a final dataset row. Resolve all placeholders before yielding final map dictionaries to sample assembly. This avoids a first-map miss blocking the bounded queue while the parent waits for a complete miss census. Keep the warm snapshot process-local; do not reread it for every map. A repeat miss after filling is an error, not an unbounded retry. Existing genuine feature-decode failure retains the serial loader's exclusion behavior/reason, and is distinguished from process failure.

Duplicate requests for one key reuse the first canonical request exactly as serial caching does. Conflicting existing tensor length triggers the same miss/replacement behavior as serial. Workers do not calculate new features or merge files. After all maps succeed/exclude consistently, the parent publishes **one** merged run-local feature cache atomically, preserving untouched entries, replacing invalid-length entries exactly as serial does, and retaining canonical insertion order of new keys. Compare duplicate-key tensors bitwise if independently supplied; never last-writer-wins. Shipped cache remains read-only.

## 3. Exactness acceptance

No `allclose`, tolerance-based equivalence, reordered-sample multiset comparison or averaged-loss substitute is acceptable.

1. Full frozen train/val corpus: serial original load versus parallel load produce identical ordered sample structures and **bit-identical tensor/scalar payloads**, before shuffle, after `_split`, and after CROP/family renormalization. Compare weights and RNG state as well as inputs/targets/walls/events.
2. Cached versus live serial loading returns that same post-shuffle dataset and split state. Prove the cached training branch makes zero parser/feature calls by failing stubs, not just a short runtime.
3. Compare canonical dataset artifact bytes. Serialization uses a shared deterministic normalization in both modes: ordered containers, contiguous CPU tensors and a fixed archive member prefix (for example torch serialization through a BytesIO stream). Volatile provenance stays outside the payload. Also compare tensor bytes directly; container metadata must not mask a numerical difference. A consistent representation normalization is allowed only at serialization, with round-trip type/value/order tests.
4. Checkpoint gate: same-device deterministic **CPU rehearsal**, same seed 20260921, same training thread settings, first 8 eligible post-shuffle train charts and first 4 val charts, two epochs × three updates, existing batch/corruption/loss/selection logic. Compare serial-live, parallel-built and cache-reloaded inputs. Require identical sampled indices/crop/mirror decisions, numerical history, best epoch and state-dict tensor bytes, plus byte-identical checkpoint files serialized with the same basename/prefix. This is a bounded checkpoint-equivalence check, not a claim that a full CUDA run was repeated.
5. Run serial-versus-serial rehearsal first as a repeatability control. If it is not bit-repeatable, report INCONCLUSIVE and resolve the test environment before certifying the optimization; do not relax to approximate equality. Do not change production CUDA determinism/math settings to conceal it. The later full pilot retains its existing single-device training settings and budget; no cross-GPU checkpoint-identity claim is made.
6. Adversarial fixtures: reversed completion order, mixed difficulty order, empty map, worker exception, partial write, cache miss caught as exception, duplicate miss key, invalid cached length, cold/warm caches, stale source/code identity, and resume attempts against incomplete artifacts. Hash shipped artifacts before/after.

## 4. Bounded speed acceptance

On the actual frozen train/val set, use serial and parallel preparations with separate copies of the same initial feature cache. Include spawn, IPC, misses, merge, source verification and serialization. Record sample counts/digests with every timing. Do not compare an old cold run against a new warm one.

Use two runs per mode in interleaved order `serial, parallel, parallel, serial`, fresh output directories, identical starting feature-cache bytes. Compute speedup as the median serial elapsed time / median parallel elapsed time. Record per-stage times and memory. No worker-count sweep in this packet; use auto 8 on this box, and report constraints/errors.

- Parallel preparation gate: **>=1.5x** end-to-end versus serial preparation with the same optional dataset serialization enabled.
- Prepared-load gate: source validation + cache load **<=25%** of a live serial split load, and zero parser/feature calls.
- Combined data-stage gate: **>=2x** for cache+pool, or **>=1.7x** for the cache-only fallback, for prepare+train-data-load versus the original two-live-load path. Exclude actual optimizer training equally; include original snapshot/inventory and optimized cache write/read/validation work. Report the exact sum of measured stages rather than mixing the supplied 12-minute estimate with new measurements.

If exactness fails, reject the affected optimization. If the pool is exact but misses its speed gate, keep `workers=1`; accept only the independently proven cache-reuse portion if its own exactness/load/combined gates pass. Label that outcome **CACHE_ONLY_ACCEPTED**, not parallel success. This is a predeclared bounded fallback, not permission for endless tuning. Otherwise accept **CACHE_AND_POOL_ACCEPTED**, or report **NOT_ACCEPTED / INCONCLUSIVE** with evidence. No change to production defaults in any outcome.

The cache-only threshold is lower because removing one of two equal loads has an ideal ceiling of 2x before cache overhead; requiring >=2x would make that fallback unattainable without another improvement.

## Exit and handoff

Deliver code/tests, a compact speed/equivalence report and approved opt-in clean-rhythm commands. No model-quality evaluation or A/B parallelism claim. Resume the existing clean-rhythm pilot immediately using only accepted options; its training/A/B budgets and quality gates remain unchanged. A later A/B scheduling packet can preserve complete song/arm jobs with 4-thread workers once the serial evaluator exists, but is not implemented here.
