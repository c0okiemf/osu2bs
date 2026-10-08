# Lossless phrase budgets and persistent hand state

This is a representation
prerequisite for the next model, not a fit, generated candidate or production change.

## Evidence and intended authority

The retired model's existing candidate pool cannot pass the rhythm gate even with
reference-based oracle selection. Misplaced chart rests and double distribution
dominate the gap. Its recurrent input exposes the most recent global event, but
last events for an inactive hand remain implicit in hidden state. The decoder
already tracks them, without supplying them as explicit model inputs.

Introduce explicit phrase budgets and causal two-hand state, retaining all current
literal event/timing expressivity. A future high-level planner must predict budgets
from serving audio/workload; human budgets here are labels for reachability only.
No future source geometry may enter the low-level history state. No new fit until
the representation and actual source coverage are verified and a separate planning
spec defines serving prediction, decoding feasibility and output acceptance.

## Representation

Keep the existing8-beat half-open windows, native partial tail, exact float beats,
constant BPM and literal events (up to3 notes per hand). For each window store three
nonnegative integer event counts: left-only, right-only, both-hands. A timestamp
with multiple literal notes in one hand still contributes one hand event. Counts
derive coherent totals: events=Lonly+Ronly+both, left=Lonly+both,right=Ronly+both.
An all-zero plan explicitly represents an empty chart window. Eight beats remain
a batching/control interval, not a claim about universal musical phrase length.

Retain existing exact teacher actions including REST, raw target beats, gap category
and residual. Before every action expose current window budget remaining and each
hand's last nonempty literal event (beat plus every literal note), or None before
that hand starts. Store its age from the current cursor. REST and an event using
only the other hand must preserve this history across window boundaries. Update
budgets once per timestamp event; update hand histories only after the entire
literal event. REST is permitted by the source replay only after that window's
budget is consumed. These are oracle replay invariants, not a new serving decoder.

Do not add parity bans, direction-family thresholds, minimum gaps, near-time merging,
rounding, or inferred physical-swing labels. Multiple heads and dots remain literal.
Counts do not specify future cells or directions. The history state is an explicit
record of observed notes, not a physical hand-pose estimate.

## Reachability and evidence

Use the frozen v19 source inventory:336 train plus6 approved validation. Preserve
roles/quarantine and distinguish validation-source auditing from fitting. Encode,
serialize, load and decode every source; recover exactly the original notes, bombs,
walls, BPM/native duration and authored settings. Verify budget identities and all
REST boundaries. Report the actual distributions of counts, empty windows, literal
stack counts, cross-window hand carry and actions where the prior global event
omits a previously active hand. This quantifies availability, not causal benefit.

Tests must cover empty windows, a partial tail, simultaneous hands, multi-note hands,
dot notes, micro-offsets and one hand silent across multiple windows. Mutating future
coordinates/directions without changing presence/timing must not alter earlier
history or any budget. Reject tampered counts/state, missing/extra actions, invalid
REST consumption or changed literal events instead of repairing them.

Freeze code, this spec, source payload/role identities and existing codec dependency
hashes in `experiments/phrase-state-reachability-v1/`. Save authenticated packets and
round-trip receipts, then verify a repeat. No new model, generated maps, QA writers,
dependencies, playtests, annotations, external judges or push. Original metric gates
and B0 remain untouched. Reachability alone proves no generation or quality gain.
