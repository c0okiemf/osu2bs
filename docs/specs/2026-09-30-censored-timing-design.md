# Preserve boundary-survival mass in joint timing decoding

Continue the
broader task, preserve production and all frozen experiments.

## Evidence and mechanism

Exact observational v16 validation replay finds294 tiny same-hand intervals from
279 decisions,251 within1/48 beat before a window boundary.266 chosen categories
are modal after masking, but a raw-logit probe finds only4 raw modes. Median raw
chosen probability0.00008713 becomes0.997899 after legality conditioning. Ordinary
positive waits that overshoot the boundary are discarded, concentrating probability
on microscopic legal waits. Reproduce and bind the raw probe before new generation.

Freeze v16 selected checkpoint6000, its45-source prepared data/frontend and all
controls. No new fit. At each timing decision, scale raw logits by the existing
temperature. For each positive timing proposal with target>=next boundary, move
its probability mass into REST using logsumexp with the original REST logit.
Mask the original overflow categories. Other legal logits are unchanged; illegal
zero-time repeat and any nonadvancing positive proposal remain masked. Sample once
from this distribution using the existing RNG. A selected REST emits no note,
advances exactly to the boundary, retains last literal event and recurrent state,
then reconditions. End-of-song overflow also means REST to native audio end.

This is a new decoder interpretation of right-censored waiting time, not a claim
that the original CE training objective already models survival explicitly.
Do not alter residuals, clamp emitted times, merge events, change counts/slots or
hardcode a minimum hand gap. All retained in-window event proposals stay literal.
Tiny genuine source offsets remain representable. Keep native obstacles/settings.

## Fixed comparisons

First reproduce12 v16 selected-checkpoint validation outputs with raw and legal
probability traces; assert exact exported notes and summarize probability inflation.
Then12 new validation outputs (6 songs×seeds0/1,temp1), diagnostic only: checkpoint
and decoder are already fixed, no reranking or validation-based tuning. Preserve
the same conditioning, source measurement audio and approved-train validation scales.
Record boundary mass, completion, tiny gaps and paired rhythm/geometry deltas.

Then48 development outputs on the original8-song panel, seeds0..5 and unchanged
temperatures[.85,1,1.15,.85,1,1.15]. Use actual audio/MI/B0 deployment adapter and
uniform query audio, original-audio measurement, first scoped admitted result or
B0 fallback. Compare to v16 as the matched decoder control, v15 uniform retrieval,
original retrieval and B0. Same unchanged numerical gate, support and missing
opportunity requirements. No QA reuse unless actual chart and Info bytes match.

## Verification and limits

Test probability conservation and exact in-window relative odds, no-overflow
equivalence, boundary equality, illegal zero repeat and terminal behavior. Retain
existing literal-event/BOS/decoder checks. Verify actual serialized exports, scenes,
profiles/errors/selection and repeated report; report all failed and unknown cases.
Audit emitted notes remain the selected raw targets, never a repaired timestamp.
Bind parent checkpoint/data/selection/report, recipes, source/view receipts,
observational probe and all protected artifacts. Namespace
`experiments/joint-phrase-v17/censored-timing/`.45-minute work/55-minute hard limit
per stage,>=2GB headroom. No dependencies, playtests, annotation, judge batches or
push. No further timing/temperature sweep on failure; diagnose residual failure.
No quality guarantee or promotion without full gates and fresh confirmation.
