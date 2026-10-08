# Phase 5B first packet: clean-split rhythm rebuild

Date: 2026-09-21.

**Verdict: execute one from-scratch rhythm-model rebuild on the phase-5A train split, select its checkpoint on val, and A/B against the shipped rhythm checkpoint. All outputs are separate experimental artifacts.** This advances the constructive phase; a newly discovered mapping defect is not a prerequisite.

## Scope and alternatives

The purpose is to obtain a reproducible, clean-trained rhythm baseline and measure its effect when substituted into the existing generator. The clean splits remove the known family leakage; this is not a claim of exhaustive audio-fingerprint deduplication.

Choose rhythm-only retraining over (1) another diagnostic without training or (2) simultaneously rebuilding rhythm, flow, critic and ladder. The former does not execute the requested phase; the latter makes a regression hard to attribute. One training seed, existing architecture and evidence schema, existing decoder policy, two rhythm checkpoints. No dense-evidence, physical-ms, phrase, style or difficulty-control ablation in this packet.

The frozen flow/critic/ladder retain their historical data contamination. Call the result **clean-trained rhythm in a mixed-provenance pipeline**, not a clean-trained full generator. One training seed cannot establish training-seed robustness or attribute all differences solely to removing leakage.

## Non-negotiable boundaries

- Never overwrite, rename or replace shipped `groom.pt`, `flow.pt`, `critic.pt`, `ladder.json`, or production feature caches. No installed-engine update, export into the desktop song folders, or push.
- All training checkpoints, dataset caches, features and decoded candidates live under a new `experiments/clean-rhythm-v1/` directory. Existing outputs are not overwritten; a mechanical rerun uses a new suffixed run directory.
- Train gradients use only manifest `train`; checkpoint/early-stop selection uses only `val`. Dev/benchmark evaluation occurs after checkpoint selection. No sealed-test tensor loading, scoring or fitting. Inventory integrity hashing is allowed and is not test evaluation.
- Preserve the feature layout, preprocessing, eligibility filters, model architecture, corruption/mirroring, loss weights, optimizer, conditioning and shipped decoding rules. Do not refit flow, critic or ladder.
- A/B results never trigger an automatic production replacement in this packet.

## Frozen inputs and preparation

Copy the current validated `eval/corpus_manifest.json` into the run before tensor extraction, and use that exact snapshot for all loaders. The working copy currently contains the corrected eligibility census (901 rather than 810); do not discard those uncommitted corrections, rebuild the splits or silently use the older committed census. Record snapshot SHA256 and the working-tree/code hashes. Preserve the unrelated dirty files.

Verify manifest version, family disjointness and chart hashes with the phase-5A checker. Record the actual train/val family representatives and emitted difficulty samples, their resolved chart/Info/audio hashes, lengths, pre/post-CROP eligibility and sampling weights. Use the existing family representatives and family weight renormalization after the CROP filter. Counts must come from the actual loader, not the directory census. Record excluded families/reasons and empty chart sets; empty train-after-CROP or empty val fails before training.

Redirect `groom._FEAT_CACHE` into the run and clear its in-memory feature view before extraction. Bind `eval.corpus.MANIFEST` to the snapshot within the isolated runner process. Keep tensor preparation on CPU; do not place the entire dataset on the GPU. Cached dataset artifacts must contain the snapshot/config/code/source identity that permits reuse; mismatch fails rather than reusing stale tensors. Do not run MI inference.

Record executing code per stage. Tensor-cache identity binds the actual loader/feature/timing code and configuration; checkpoint provenance additionally binds the trainer. A/B binds its decoder/metric code identically across arms. Adding the later report/evaluation command to the runner does not itself invalidate completed training, but changing preprocessing or training behavior requires a new prepared run. No silent mixed-code A/B.

### Eight-song frozen decode panel

Freeze `eval/clean_rhythm_panel.json` and copy it into the run before training. Exactly:

1. still_waiting
2. numb
3. rather_be
4. FENT_TG
5. if_i_lose_myself
6. rap_god
7. reality_check
8. spaceman

For the first five, bind `.osu`/audio paths and hashes to `experiments/flow-v1/replay-ab-motion/<song>/results.json`. For the last three, bind `eval/<song>/gen.osu` to the existing `stored_gen/*` entries in `eval/benchmark.json`, and audio to `eval.eval_gen.SONGS`, the existing MI-generation provenance also used by `eval.evidence_audit`. Freeze the actual audio hashes. Verify all inputs exist and match; missing inputs are a preparation failure, never replaced with a conveniently available song.

Freeze the full `eval/benchmark.json` hash as reference context. Its historical exports without retained upstream inputs are not fresh A/B pairs. Existing human charts remain descriptive references, not labels of correct generated timing or a fitting set. The eight songs are a development/regression panel, not an untouched test or a statistically representative genre sample. Record corpus-family overlap if any for the five sentinels; do not imply that every sentinel is held out from rhythm training. The three gems stay dev, excluded from fitting.

## Training contract

Create `checkpoints/rhythm-seed20260921.pt` from a newly initialized `Groomer`; **do not load shipped groom weights as initialization**. Preserve the raw state-dict layout so the existing architecture can load it. Keep training metadata in a sidecar rather than changing checkpoint interpretation.

Seed Python, NumPy and torch (including CUDA) with **20260921** before model initialization and training sampling; use a fresh seeded `random.Random` for the trainer. Record library/device versions and determinism settings; do not promise bit-identical GPU training across hardware. Stable input order, deterministic validation corruption (existing seed0), and recorded seeds are required.

Budget: existing Adam lr **1e-3**, batch **16**, crop **512**, **30 updates/epoch**, maximum **200 epochs / 6000 updates**, existing loss weights (notes3, walls15), improvement threshold **1e-4**, stop after **26 consecutive non-improving epochs**. Select the minimum existing validation BCE, not dev motion or critic score. Retain the current chart-mean validation reduction, clearly labeled; family normalization applies to training sampling. Do not introduce a loss-reweighting ablation inside this rebuild.

Log initial untrained validation loss, every epoch's train/val loss and note/wall F1, update count, best epoch, actual sample counts/weights, runtime and checkpoint hash. Save the best state, not the final epoch by accident. The best loss must be finite and lower than the initial untrained loss to label the rebuild learned successfully. Also evaluate the shipped rhythm checkpoint on the identical val tensors and corruption after training; report this comparison as descriptive because shipped weights may have seen these families. It is not the checkpoint-selection criterion or a clean generalization contest.

No automatic extra seed, longer run, hyperparameter search or warm-start fallback. A crash may be retried with unchanged settings in a fresh run and disclosed; a completed quality failure is a valid result, not a request to keep tuning until green.

## A/B contract

Use today's unchanged production algorithm for both arms. A = explicitly loaded frozen shipped rhythm state; B = the selected new rhythm state. Load both on the same inference device (CPU for this packet); keep the shipped flow, critic and ladder fixed and hash-verified. Add an optional rhythm-model injection seam to `convert_groomed` and forward it to the existing `groom_notes(model=...)`. Do not reassign `MODEL_PT` to select an arm or contaminate the model cache.

Settings: ExpertPlus, replay off, legacy thinning on, normal learned rest computation and density calibration on. **Each rhythm model computes its own probabilities/rest mask/calibration**; holding A's mask fixed would hide a real consequence of changing the rhythm model. Grid, expanded/thinned input steps, features, audio, duration and difficulty settings must match between arms and be hash-recorded.

Keep the existing six seeds 0–5 and temperature cycle .85/1/1.15, existing candidate gates, critic ranking and adaptive retry/rate loop. Maximum existing budget is **24 decodes per arm/song** (12 per pass, at most two passes), hence **384 decodes maximum** across the panel. Log every pass, seed, temperature, rate_scale and candidate, including failed and superseded candidates; stop on an unexpected budget overrun rather than extending it.

Two views, obtained from the same runs:

- **Paired initial six:** first pass, rate_scale1, seeds0–5 in both arms. Equal realized budget; compare song-level means and paired seed differences. Do not treat 48 candidates as 48 songs.
- **Production-policy winner:** the existing final valid winner per arm/song. Both arms have the same allowed budget and retry policy, but actual decode counts may differ; report that difference, do not claim equal realized counts. No reranking using new metrics.

All-invalid collect mode can return artifacts in current `convert_groomed`; the experiment must explicitly detect absence of a valid winner and record failure. Never report that artifact as a successful map. Do not call `write_map` or replace any production output.

Save every v2 candidate with BPM/timing source and full input/model provenance, hard-check reasons, flow metrics, motion report, step-addressable rest/calibration/scheduling trace, and declared final-selection status. Tensor traces must be serialized losslessly as arrays or separate `.pt` trace artifacts; never stringify tensors into opaque text. Historical Packet-B reports are reference context, not arm A: regenerate A now under identical code to B.

## Frozen measurement and acceptance

Reuse map-reader metric v2, `motion.report`, `convert.check`, and the closed difficulty vector where descriptive. Report raw and grouped rates, duplicate8 at fixed/max phases, doubles, hand workloads, per-hand cadence, longest runs, narrow/broad outward and converging pairs, dot-stack classifications, flags for all four extents, wall counts and validity, and current vert/axis-run/lateral gates. No new comfort oracle, scalar difficulty score or genre-equality target.

For rest comparisons add only a shared-domain counter: full 2s bins starting at `grid.time(0)`, ending within `grid.time(T-1)`, using directional heads and `<=1` head as quiet. A and B use the same bins; do not independently anchor at their first note. Report quiet-bin identities, gained/lost bins, and quiet share. Also report existing first-note-anchored rest metrics for continuity, separately labeled. Empty shared domains are not a pass. This is comparative preservation, not ground-truth musical silence.

The following are frozen **development non-regression tolerances**, not human limits. Evaluate per song; no pooling hides one broken song. A/B acceptance requires all:

| Gate | Rule |
|---|---|
| Integrity | All snapshots/checkpoints/features bound; no train/val family overlap or forbidden fitting; shipped files/caches unchanged |
| Learning | Nonempty eligible train/val; selected finite val BCE below untrained initial BCE; complete budget/history |
| Validity | Both arms produce a valid final winner on all8 songs; B's invalid initial-six count per song does not exceed A's. Use existing candidate `ok` (hard checks plus style/rate gates), and report hard-check failures separately |
| Rate/style | Every B winner within its existing ExpertPlus band including the existing 2% numerical allowance; existing peak-rate/vert/axis-run/lateral validity gates pass |
| Exact duplication | For each song, B winner and B initial-six mean `raw_dup8_maxphase_pct` each <= `max(corresponding A, 1.0%)` |
| Repositioning | For each song and each extent 0/.25/.5/.75, B winner and B initial-six mean flags/1k each <= `A + max(5, .10*A)` |
| Pair exposure | Each B winner's narrow `<>` and converging count each <= `max(A,1)`; initial-six mean of each count <= A mean + .5 |
| Hand monopoly | B winner longest one-hand run <= `max(A,12)`; initial-six mean <= `max(A mean,12)`. Report the metric's simultaneous-order limitation |
| Rest preservation | Each B winner quiet share differs from A by at most5 percentage points, and retains at least90% of A's quiet-bin identities. When A has none, identity-retention is not_applicable but share gate remains |

Density, rest and hard-validity gates prevent winning motion numbers by removing the chart. For initial-six motion/dup/exposure means include every candidate with a computable metric, including invalid candidates; missing/undefined required metrics fail comparability, never silently reduce the denominator. A baseline winner that itself fails current validation/band prerequisites makes that song inconclusive and prevents A/B acceptance; record the baseline failure rather than relaxing the gate.

Do not demand matching coincidence or matching per-hand medians across songs. Report those vectors to understand effects, not to force uniform choreography. Do not tune gates after seeing B. No minimum improvement quota is required: trustworthy clean-trained provenance plus non-regression is useful phase progress. A stronger quality-improvement claim requires actual supporting deltas and is limited to this panel/seed.

## Outcomes and bounded exit

1. **BASELINE_BUILT / AB_ACCEPTED:** successful clean rhythm artifact and all frozen A/B gates pass. Retain it as the accepted experimental rhythm baseline; shipped files remain unchanged in this packet.
2. **BASELINE_BUILT / AB_NOT_ACCEPTED:** training/integrity succeed but one or more decode gates fail. Keep the checkpoint and exact per-song failures; this still completes the rebuild experiment, without promotion or a claim of quality parity.
3. **INCOMPLETE:** missing/stale inputs, unsafe paths, split/loader failure, nonfinite training, no improvement over untrained initialization, incomplete evaluation or changed shipped artifacts. Report the concrete mechanical/learning failure; do not label a bad experiment an A/B rejection.

One packet ends with the frozen inputs, separate checkpoint, val history, all A/B candidates and verdict report. Completing or failing this experiment does not declare phase5B/the whole flow plan finished. Subsequent flow/critic rebuilds and phrase/difficulty/style construction receive their own bounded decisions; no newly discovered physical defect is required to proceed with those constructive phases.
