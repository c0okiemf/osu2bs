# E1 vs B0 machine A/B: preregistration (review D4, frozen before any value is computed)

Claim scope: **measurable expressive-property change** on already-exposed development songs. Never "fun validated". No promotion follows from this; any release needs fresh-generation confirmation (review D4). B0 stays active.

Disclosure: one arm-A attempt's stored summary (FENT_TG s1: arc_runs 0.118, double_entropy 3.90, low_vertical_share 0.564) was visible in a file listing before this freeze. No B0 values and no selected-chart comparison had been seen.

## Data

- The six E1 panel songs cached in `experiments/expressive-v1/e1/partial/`: FENT_TG, numb, rap_god, rather_be, spaceman, still_waiting.
- Per song: the B0 winner (`<song>:b0.json` notes) vs each arm's **selected** attempt (`<song>:armX.json` → `selection.id`). `fallback_b0` (arm A on spaceman) compares B0 with itself and counts as **no change**, never a win.
- Seconds come from the same deterministic grid the E1 evaluator used. E1 decodes on B0's EventSchedule, so timestamps are identical across B0, A and B, and the timing source cannot bias the comparison.

## Measures (`eval.expression_profile.profile` only; no train reference, per the independence review)

Preregistered measurement directions: "mostly boring low up-downs, almost no arm spinning or fun double patterns"):

| id | property | expected direction |
|---|---|---|
| P1 | `features.arcs.runs_per_100_heads` | higher than B0 |
| P2 | `features.spatial.low_vertical_share` | lower than B0 |
| P3 | `double_vocabulary.entropy` | higher than B0 |
| P4 (guard) | `features.patterns.max_4gram_share` | not higher than B0 + 0.02 |

## Machine protections (comparator-v2 components; evaluator not yet validated, so descriptive)

For each chart: `qa.certificates.contradictions` (notes only; walls are not in the cached records, so wall certificates are not applicable) and `qa.comparator_support.measure_support` against the qa-train-v2 bank with the development T_support. Protection holds per song if the arm introduces no HARD_FAIL or model contradiction absent in B0, and the supported-head share is ≥ B0's minus 0.05.

## Decision rule per arm

The arm **shows measurable expressive-property improvement** iff P1, P2 and P3 each move in the expected direction in **≥5 of 6** songs, the P4 guard holds in ≥5 of 6, and protections hold in 6 of 6. Otherwise: **NOT_SHOWN** (per-property counts reported). Paired per-song deltas are reported for every property regardless of outcome. There are no significance claims on six songs.

## Addendum: E2 , frozen before any E2 fit or decode

- **Model:** one ContextFlow fit from shipped B0 flow (zero-initialized projection, logits verified equal to B0 at init), E1 objective, approved/general 50/50, LR 3e-5, batch 8, ≤3600 updates. One snapshot is selected among 0/600/1800/3600 by held-out family-macro CE (stratified), never by these six songs. A step-0 winner means no checkpoint and E2 is NOT_SHOWN.
- **Decodes:** per song, six seeds (TEMPS ladder) of the selected E2 model on B0's exact EventSchedule, plus six matched **shipped-flow controls** on the same schedule and seeds. The B0 selector is used for both, over admitted attempts only; fallback means no change. All attempts are preserved.
- **Comparisons** use the same P1–P4 rules and protections as above, reported separately: E2 vs B0 (primary), E2 vs control (checkpoint effect), and control vs B0 (what resampling alone does).
- **E2 shows measurable expressive-property improvement** iff the rule passes for **both** E2 vs B0 and E2 vs control. A gain that control also reproduces over B0 is attributed to resampling, not the checkpoint. SCOPE/SUPPORT_UNKNOWN stays unknown, never an unqualified QA pass.
- One fit and one evaluation, then stop.

## Addendum: E3 turning-sequence oversampling , frozen before the pool is built or any fit runs

- **Model:** one fit initialized from E2's selected checkpoint (update 3600). Same ContextFlow architecture, decoder, CE + 0.05 KL-to-B0 objective on the general stratum, LR, batch and ≤3600 updates. Selection is by held-out family-macro CE among 0/600/1800/3600; a step-0 winner means no checkpoint and NOT_SHOWN.
- **Sampling:** each training example comes with probability 0.75 from E2's distribution (approved/general 50/50) and 0.25 from the **turning pool**. The pool is every CTX-length human training window (train split, stride 64, original and mirror) containing ≥1 arc-like run under `eval.expression_profile` `_arcs` (the P1 predicate). It is drawn family-balanced (uniform family, then uniform window). **QA-owned families** (every family in the QA collection states: qa_train, qa_calib, qa_calib2, qa_validate_*, seal, pilot seal) are excluded from the pool. The pool manifest is frozen and committed before training. There is no synthetic arc reward.
- **Evaluation:** same six songs; six E3 seeds on each B0 schedule; the existing matched shipped-flow controls are reused unchanged; the same B0 selector.
- **Acceptance:** unchanged P1–P4 rule against B0 **and** against the matched control. Deltas vs E2 are reported descriptively (not a gate).
- One fit and one evaluation, then stop. No promotion without fresh-generation confirmation.

E3 pool frozen before training: `experiments/expressive-v1/e3/pool.json` sha256 cf3f5f6f46fea72b95a6281414ccea3d7513e3728c1c3f5bc62d9345d4a12a77 (605 families, 18131 windows, 40 QA-owned train-split families excluded).

## Addendum: admission-only ablation on cached E2 candidates , frozen before any admission is recomputed

- **Change:** only the admission flag. A cached attempt is eligible iff it decoded validly (honored schedule, not infeasible) **and** its comparator-v2 machine verdict is `MACHINE_PASS_QUALITY_NOT_EVALUATED` (no structural or model contradiction, no scope/evidence unknowns, ≥90% supported heads, no clustered support gap). SCOPE/SUPPORT unknown means **not eligible** (abstention). The Q1-era ceilings (reposition flags, narrow/converging pairs, 4 s window flags) are no longer used for admission and are reported as diagnostics only.
- **Unchanged:** the selector (`select_arm(..., "b0")`, max of the production key whose admitted bit is now the new flag; its critic − motion-cost component is untouched), every decode (cached; no new ones), and the P1–P4 rule and protections.
- **Matched control:** the shipped-flow control attempts get the same new admission and the same selector, so both arms are compared under identical admission.
- **Report:** per song, which candidates become eligible and which is selected, for E2 and control; the frozen rule vs B0 and vs the re-admitted control. A pass supports designing fresh-generation confirmation only, not promotion.
- One ablation, then stop.

## Addendum: fresh-generation confirmation , frozen before any MI generation

- **Families:** 24, selected metadata-only by `sha256(family + ":expressive-fresh-confirm-v1")` order (`eval/fresh_confirm.py select`). The pool is the held-out test split (73 family representatives, eligible, beatsaver). Excluded: all 43 QA-owned families (2 in test), every family in any QA telemetry state on any side (3 in test, adding fam:4946e, a pilot gen-side family), the Q5 C panel (20 in test), the Q5 D panel and all Q3 MI-generated families (0 in test). E1–E3 trained only on the train split and selected on val, so none are in test. The pool after exclusions is 51, and families lacking corpus audio are skipped before generation (0 skipped). The E1 run's 12 "reserved_confirmation" test families were never generated or evaluated. Three of them (fam:16238, fam:37194, fam:2b1bc) fall in the hash order here, and we disclose it. B0 inherited exposure: the shipped groom/flow were trained on the corpus train split, so test families are unexposed by split construction, which is not re-verified at weight level.
- **Frozen identities** (selection.json sha256 8ab6f9fd7ae61a4d…): fam:47f3, fam:24183, fam:293e3, fam:92ed, fam:2aaff, fam:6475, fam:ed9e, fam:31867, fam:16238, fam:2199f, fam:81c7, fam:2e2bb, fam:3904d, fam:33c66, fam:7106, fam:37194, fam:1fd47, fam:fe30, fam:2b1bc, fam:2dbf2, fam:e7be, fam:385c0, fam:4e2ba, fam:10e8c. An MI failure on any of them stops the packet, with no substitution.
- **Frozen system:** MI v32, difficulty 5.5, seed 20260921, on each family's own corpus audio. The E2 snapshot is u3600 (sha256 47abd57576293cfa…). Shipped groom.pt is 6d26b96dd57cfd3b… and flow.pt is 28e4570f1d2760c9…. Comparator-v2 is frozen at evaluator threshold 7.5153… with the output-equivalent retrieval amendment (8ee85b0) and the qa-train-v2 bank.
- **Arms per song:** B0 is production `convert_groomed` (replay off, calibrate, production timing), with its internal selection unchanged. E2 u3600 and the shipped-flow control each run six seeds (TEMPS ladder) on B0's exact EventSchedule. Shipped-weight injection must replay the B0 winner bit-exactly.
- **Complete exported charts:** every chart (B0 and each candidate) goes through the production export steps: lead shift, intro wall clip, the final `check`, and `.dat` + `Info.dat` via `convert.export_charts`. It is then re-read with `qa.scene.read_scene`, so walls and Info settings (NJS) are in the scene. P1–P4, contradictions and support are all measured on that exported scene.
- **Admission:** a candidate is admitted iff its comparator-v2 verdict is `MACHINE_PASS_QUALITY_NOT_EVALUATED` and its export check passes. The selector is unchanged: `select_arm(..., "b0")` over admitted candidates with key (True, critic − motion.select_cost). No admitted candidate means a fallback, which counts as no improvement. A control fallback means the control comparator is B0.
- **Acceptance** (against both B0 and control):
  - P1, P2 and P3 each improve on ≥20/24 families.
  - The P4 guard and the protections hold on 24/24.
  - For each property, the family-bootstrap 95% CI of the mean improvement-oriented delta has a lower bound > 0.
  - The oriented delta is arm − base for P1/P3 and base − arm for P2. A fallback or unknown property value gives delta 0.
  - The bootstrap is frozen as `eval.quality_eval._boot_ci`: 10,000 resamples of families with replacement, `random.Random(20260921)`, percentile 2.5/97.5.
  - Magnitudes (mean deltas and CIs) are reported alongside counts.
- **Execution:** each song is persisted atomically (per-candidate partials, then a per-song record). Work runs in shards within RAM limits, each operation under one hour. MI generation never runs concurrently with evaluation.
- **Stop after one complete report.** A pass supports a scoped release decision claiming expressive-property improvement, not fun or proven playability. There is no automatic default flip, and B0 stays active.
