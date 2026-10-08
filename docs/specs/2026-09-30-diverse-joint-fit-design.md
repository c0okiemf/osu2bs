# Test source diversity with the repaired joint pipeline

Execute only
after v18 finishes and its aligned approved-only candidate remains negative. A
positive v18 instead proceeds to confirmation planning. Keep B0 unchanged.

## Why this is a different test

The approved-only model has strong training prediction and poor held-out prediction
even with correct histories; shorter histories and authored conditioning do not
resolve it. Boundary probability handling is repaired and checkpoint selection is
aligned with that decoder. Repeating that fit, changing temperatures or increasing
state length does not address source generalization. Existing eligible source data
provide336 train families, but no model has yet used that recovered, quarantined,
uniform-audio inventory with the corrected pipeline.

One controlled data-coverage ablation: train the same architecture/objective on
all336 eligible train sources (45 approved plus291 general). General means unlabeled
for personal preference, not negative. No approval flag, oversampling, architecture
change, new regularizer, extra data collection or additional model-size/seed sweep.
The target quality preference remains the same six approved validation sources.

## Fixed contract

Reuse recovered-inventory roles and component quarantine. Exclude fam:24227 and
every validation/development identity. Exact336 train membership must match v15's
uniform-audio train inventory. Keep the same approved6 validation set, preserving
all other validation sources as unused. Validate source/payload/view hashes and
approved-source provenance; all45 approved teacher tensors must remain bit-exact
with v16. Prepare291 additional train contexts through the unchanged teacher codec.

Same336 cached14800Hz mono renders and original extractor, original source native
timing/literal notes/obstacles/NJS/rates, same two-layer128-width JointModel, original
seed and fresh initialization.6000 updates, AdamW3e-4/wd.01/gradient cap1,16×128,
4 real-BOS plus12 uniform crops, uniform family sampling, non-BOS32-step warmup.
Snapshots0/1000/3000/6000, exact resumable optimizer/all RNG/counters. One fit only.

Keep v16's approved45-derived validation scales and median rate fixed for matched
selection; do not refit these on336. Validation inputs stay6 approved sources,
native authored BPM/duration, NJS18, empty obstacles and common-band audio. Four
snapshots×6×2 seeds at temp1 =48 rollouts through v17's censored decoder. Same
failure/error/earliest-checkpoint rule, no selection by teacher CE or development.
Report diagnostic train/validation teacher losses for the selected checkpoint,
but never substitute them for generated quality or admission.

If a learned checkpoint is selected, run the unchanged8-song development panel,
6 fixed seeds/temperatures, actual deployment timing/workload/full obstacles,
original-audio measurement, first admitted/B0 fallback and unchanged numerical
gate. Compare with the v18 approved-only candidate, v15 literal retrieval, original
retrieval and B0. If initialization wins, record no learned candidate rather than
generating a random-weight panel.48 development attempts maximum, no pool expansion.

## Verification and follow-through

Bind source roles/provenance/data/views, old fixed validation calibration, all code,
spec, controls and protected hashes in `experiments/joint-phrase-v19/diverse-joint/`.
Verify source/action round trips, all336 contexts, original45 exact tensors and
fresh initialization. Verify validation selection and every exported full scene,
unmodified raw timing targets, profiles/errors/admission/fallback and repeated
report. Preserve all unknown/failure cases and source-style caveats. No production
promotion without full acceptance and fresh confirmation.

45-minute work/55-minute hard limits per stage;>=2GB headroom; no dependencies,
playtests, annotation, judge batches or push. This isolates broader source coverage
within the repaired pipeline, not approval preference as a causal variable. If it
still fails, retire this small GRU recipe and plan the next representation from
actual failure evidence; no additional fits of this same recipe or temperature
sweeps. The broader quality task continues across this packet's terminal result.
