# Contiguous32-beat source segments

The quarantined eight-beat control completes all48 attempts but remains negative.
The earlier continuation audit shows why a local tie-break is insufficient:383/384
next source entries exist,365 are eligible, but only39 enter the nearest six; most
native continuations rank far away. Test a larger literal generation unit, not a
weight sweep: four consecutive existing eight-beat source entries, i.e.32 beats.
This preserves authored multi-window timing, hands, geometry and rests together.
It is not a claim that every musical phrase is32 beats or that copying is quality.

Build all full32-beat segments beginning on the existing source eight-beat grid,
using only315 quarantined fit-train families. Recompute the same55-coordinate
mean/std audio descriptor over32 beats plus hand-events/beat, and fit bank mean/std
on these TRAIN segments only, with the unchanged1e-6 floor. No new audio features,
model fit, style labels, source-pool restriction, objective weights or output seeds.
Keep the source-relative interval metadata in the underlying eight-beat bank.

Target segments begin at0,32,64... with literal clipping only at actual audio end.
Rank the full segment bank by the same standardized mean-square distance and choose
seed modulo the first six compatible candidates. Directional entry/exit checks apply
at newly spliced32-beat boundaries; internal source continuations are literal authored
context. Preserve v10/v12's temporal checks at EVERY eight-beat ledger boundary and
its known-exit condition before the final target eight-beat window, including internal
pieces. Preserve empty-hand state. Reject immediate repeated32-beat segment and an
immediate repeated eight-beat piece at a segment boundary. Do not weaken intervals.

Materialize each eight-beat piece directly at its final target start to preserve
bit-exact reconstruction under floating-point addition. Keep both the selected32-beat
segment ledger and its expanded eight-beat donor ledger. The last segment may use
only its prefix; the last eight-beat piece may be clipped. Verify both ledgers,
literal exports, all native temporal checks and provenance. Disclose internal authored
directional transitions rather than mislabeling them as newly created joins.

Namespace `experiments/joint-phrase-v13/contiguous-segments/`. Freeze v12 report,
quarantined bank and data/recipe/code/spec identities before new outputs. Six attempts
per each of eight opened families, first machine-admitted or B0 fallback, same actual
serving inputs/NJS18/full scenes and post-generation reference-time scoring. Main
numeric rule stays against historical B0/original R; paired32-versus8 comparison
separately. Reuse scoped QA only on identical chart+Info bytes. Retain all12 inventory
identities and release eligibility false; inherited model/B0/QA exposure remains.

Report source-continuation share and segment counts separately from quality, full
errors/component coverage, worst movement/continuity diagnostics, failures, internal
search/runtime and donor diversity. Reconstruct/re-profile every actual export and
repeat the report.45-minute work/55-minute hard bound, one QA writer, no dependencies,
new labels, fresh outcomes or push. Continue from measured results without an automatic
segment-length/weight/seed sweep. This single32-beat challenger is a mechanism test.
