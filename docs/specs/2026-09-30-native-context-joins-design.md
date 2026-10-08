# Source-context timing at retrieved phrase joins

## Measured defect

The directional audit traces two joins on fam:2199f to different fit-train donors.
A donor ends0.00186157 beats before a window edge and the next begins exactly on it:
the new hand gap is0.797817ms at140BPM. Its native outgoing and incoming gaps are0.5
and1 beat (minimum214.286ms at the target BPM). Another join collapses to0.565665ms.
The old arrow-family check accepts both because the directions alternate. These
are newly compressed joins, not preserved authored multi-note events. Evidence:
`experiments/joint-continuity-v1/native-seam-audit.json`, with source hashes.

## One join policy, not a universal physical claim

Keep the workload-corrected original retriever and its audio descriptor, literal
notes, shortlist32, first6 compatible options, seeds0..5 and admission/fallback.
Add source-context timing to candidate join compatibility. No generated note moves,
merging, deletion, fitting, parameter sweep or expansion of the search budget.

For every existing bank phrase and hand, derive from its full fit-train source the
interval from the preceding hand event to its first hand event and the interval from
its last hand event to the following hand event. Events mean exact timestamp groups
including arrows/dots, not inferred physical swings. Store gaps in beats, since the
whole donor is transplanted in those units. No source BPM or reference target is
used to invent an output gap. The inherited workload correction remains explicit.

For a newly spliced same-hand boundary, require both native contexts to be known and
positive, then require actual joined gap >= min(native exit gap,native entry gap),
allowing1e-9 beat comparison tolerance for floating arithmetic only. Thus a new join
cannot be faster than BOTH native contexts. This conservative construction policy
is not proof of physical comfort, and a gap permitted by it may still be awkward.
Missing native context rejects that join rather than fabricating support. First-ever
output event on a hand needs no prior context. Empty-hand windows retain the last
hand event and its native exit hint. Keep the existing directional join rule too.

Do not apply a blanket minimum interval/parity ban to authored internal phrases;
copy their complete literal content unchanged. No claims about near-simultaneous
stack intent or biological speed limits follow from this policy. Slow/long-rest joins
remain available. If no donor in the existing shortlist is compatible, record the
failure and count the candidate; do not silently repair or retry extra seeds.

## Frozen comparison and evidence

Namespace: `experiments/joint-phrase-v9/native-context-joins/`. Freeze spec/code,
completed v8 controls/bank, exact source metadata and earlier production/evaluator
identities. No generation until v8 completes and its result is recorded. Native-edge
metadata must come only from the same316 fit-train sources and be hash-verified.

Generate six candidates on the same eight already-opened families with deployment
BPM/duration/environment, first-admitted or exact B0 fallback. Retain all candidates,
rejection reasons, donor ledger and join evidence. QA still reads actual full-scene
exports; measurement still uses reference-time windows. Byte-identical control
chart+Info may reuse its QA. Do not modify previous results.

Report two separate questions: (1) the source-relative timing contract on all complete
nonfallback retrieved candidates, with coverage/fallback and actual seam gaps;
(2) the unchanged whole-generator development rule against B0 and original deployment
retrieval, plus the paired ablation against v8 workload-corrected retrieval. A passed
construction contract does not turn a failed quality gate into a pass. Release
eligibility stays false; no fresh families are opened. Report directional/movement
and missing-opportunity diagnostics alongside the old descriptors.

## Checks and continuation

Reproduce the traced sub-millisecond seam: old directional check accepts, new timing
check rejects. Also test an equal-native-gap join, retained state through an empty
hand/window, first event, unknown native context, both hands and literal preservation.
Reconstruct every emitted chart from its donor ledger and audit joins independently
from the candidate-selection loop, including tail truncation. Repeated decisions
and all protected identities must match. No concurrent QA-cache writers.

45-minute work/55-minute hard bounds,>=2GB headroom, no dependencies/playtests/judge
batches/push. Continue from measured evidence; no automatic seed/fit expansion or
relaxation after failure. Fixing this seam defect is progress, not completion of
perfect-map generation or the broader task.
