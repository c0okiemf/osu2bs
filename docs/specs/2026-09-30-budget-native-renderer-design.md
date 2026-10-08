# Render learned phrase budgets with literal native phrases

The budget predictor has
a validation signal; test its effect on actual exported maps before another model
fit. This is one integration of supervised phrase planning with an existing literal
renderer, not a sweep of retrieval feature weights, phrase lengths or frontends.

## Serving contract

Freeze the successful phrase-budget model and the existing v15 approved45 bank
(2,946 full8-beat phrases). Use actual deployment BPM, native audio duration, B0
environment and requested hand-event rate. Predict every window from common-band
audio with the frozen91-feature function. No held-out notes or authored BPM enter
generation. Preserve the three predicted integer budgets as the requested plan.

For each target window, count each donor's left-only/right-only/both timestamp
events. For a partial final window count only the literal events before its end.
Order donors lexicographically by total absolute count error, then the existing
standardized audio descriptor distance, then original bank index. There is no
fitted weight or threshold. Search this order under the existing native temporal
join, directional join and nonterminal-exit contracts. Keep the first six compatible
donors; attempt seed selects its index modulo the available count, exactly as in
the prior renderer. Exclude immediate repetition of the same donor as before.

Append complete literal events, advance to the window end, and retain both hands'
last groups and native exit intervals across empty windows. No note insertion,
deletion inside a full phrase, time warping, boundary epsilon, forced late events,
geometry repair or additional parity policy. The existing partial-tail crop remains.
If no compatible donor exists, record an incomplete attempt. This renderer projects
budgets onto its feasible native phrases; it does **not** promise exact counts.
Store requested/realized counts, absolute error, minimum bank error, compatibility
rejections and both-hand state before every window. Empty chart windows are not
called acoustic silence. Native joins are evidence, not a physical comfort proof.

## Fixed evaluation

No new fit or parameter selection: six attempts on each of the eight opened
development songs. Use the original frozen full-map scales and gate, original audio
for measurement, full environments, actual exports and QA. Select first admitted
attempt or B0. Compare with B0/original R and report paired v15 changes as diagnostic
evidence. Exactly matching chart+Info may reuse bound v15 QA; otherwise one QA writer.
Report all48 attempts, budget projection error, continuity, sub-millisecond intervals
and selected/fallback outcomes. No metric/gate changes or reference-driven selection.

## Verification and limits

Bind predictor, model/report, bank/receipt, query views, deployment proofs, prior
control artifacts, source code/spec/dependencies and protected production hashes.
Tests must cover count identities with stacks, partial-tail accounting, budget-first
ordering, and persistent hand state through an empty window. Independently replay
every attempt, reconstruct literal donor notes and requested budgets, compare actual
exports/full scenes/profiles/errors, re-run first-admitted selection and reproduce
the report without new QA writes. Keep frozen files immutable after execution.

45-minute work/55-minute hard limits and2GB headroom. No dependencies, playtests,
annotations, judge batches, pushes or production replacement. A negative packet
does not finish the task: separate predictor error, renderer projection and QA
fallback costs before deciding the next constructive change. Do not launch a
budget-distance/beam-width sweep merely because this candidate fails.
