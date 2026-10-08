# Continuous joint generation from authenticated approved sources

Preserve B0 and frozen history.

## Evidence and question

The old joint fit used24 approved of316 training families, with one approved
validation family. Recovery provides45 approved train and6 approved validation
families, including5 newly reserved groups. The known cross-role duplicate is
quarantined. Uniform audio improves retrieval geometry19.5% on all8 songs but
fails the full gate; longer native segments also failed. Return to continuous
joint generation to test learned musical schedules and transitions.

One fresh approved-only fit is a combined corpus/frontend experiment, not a
controlled isolation against the old mixed-corpus model. Compare deployment outputs
to B0, original retrieval and v15 uniform approved retrieval. No inherited weights.
Approval applies to sources, not outputs;6 validation maps are not confirmation.

## Frozen training contract

Exactly45 approved train and6 approved validation members from recovered inventory.
Validate original approved provenance or recovery's exact-chart/parsed-Info proofs.
Exclude fam:24227 and all development families from fit/selection. Validation:
fam:30fd, fam:12db3, fam:1a4dd, fam:21f92, fam:a253, fam:b7aa.
Use existing14800Hz mono FLOAT rendering then unchanged22050Hz feature extraction.
Reuse authenticated v15 train/development views; render6 validation views identically.
Original native audio duration, literal events, obstacles and timing stay exact.
Keep original audio separately for measurement. Rebuild teacher contexts; assert
exact action/source round trips. Do not round/merge/delete near-simultaneous notes.

Existing two-layer128-width JointModel, seed20260930, AdamW3e-4/wd.01, gradient
cap1,6000 updates, snapshots0/1000/3000/6000, batches16×128. Four actual BOS crops,
twelve uniform valid crops; uniform family sampling. Reuse v3 crop/loss functions:
all BOS targets supervised; first32 warmup steps masked only for non-BOS crops.
Save optimizer/all RNG/model/counters every100 updates and at deadline; resume
only the same identity. One fit, no seed/architecture/learning-rate sweep.

## Selection and development

Validation:4 snapshots×6 songs×seeds0/1, temperature1 =48 full-song attempts.
Unchanged legal-timing rollout. Authored BPM/native duration, empty obstacles,
median training hand-event rate; acknowledged selection proxy, as in prior fits.
Fit validation scales only on these45 training sources' original-audio profiles.
Rank by failed rollouts, mean rhythm+geometry error, earliest snapshot. No choice
by development or training loss. Report step0 selection honestly.

Development:actual audio/MI/B0 adapter, original native duration, B0 full obstacles
and workload. Six attempts per song, seeds0..5, temperatures[.85,1,1.15,.85,1,1.15].
Continuous recurrent state, existing legal timing/literal decoder. No parity
repair, minimum-gap clamp, transplant or post-generation repair. First scoped
machine-admitted result, otherwise B0; unknown is not admission.
Keep existing8-song numerical gate and historical train scales for comparability,
explicitly retaining their old exposure. B0/original retrieval/QA retain exposure;
new weights do not inherit it. Report v15 differences separately, full-scene QA,
directional continuity and smallest same-hand gaps as diagnostics, not new gates.
Export/read back before measurement; original audio; preserve seconds when mapping
to reference BPM. Historic v3 numbers are descriptive, not a controlled ablation.

## Verification and resources

Bind source payloads/provenance/roles, views, recipes, spec/code, B0 and comparator
inputs. Verify disjoint fixed45/6 membership, action round trips and contexts.
Test rejection of nontraining roles before tensor construction and validation
selection preferring complete outputs over low partial-output errors. Reuse BOS
and decoder checks. Audit48 validation/48 development artifacts, checkpoint hash,
profiles/errors/selections and repeat report.
Namespace `experiments/joint-phrase-v16/approved-joint/`.45-minute work/55-minute
process limits per stage, resumable without expanding the fit;>=2GB headroom.
No dependencies, playtests, annotations, external judges or push. No production
replacement. Negative requires failure diagnosis before another model variant;
positive still needs fresh confirmation. Packet completion does not finish the task.
