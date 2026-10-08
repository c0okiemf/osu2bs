# Geometry conditioned on explicit persistent hand state

Native-phase refinement retains the rhythm and
directional plan but improves geometry only4.3%, below the10% gate. Retire that
frozen-refiner route. Reuse the successful upstream plan in a new bounded learning
problem: geometry conditional on the known schedule/phase and both hands' state.

## Representation and one model

Use the same336 training families and six approved validation families, the
authenticated source payloads and common14800 audio views. No new songs, donor
bank, feature extractor or training labels. Preserve24227 quarantine. Known
serving inputs are event time, hands/slot counts, native directional phase,
exact lateral cuts/dots, full environment and workload. Only placement and
vertical/diagonal choices within the native phase are learned.

For each event, explicitly supply each hand's last literal group (three padded
tokens per hand), presence and elapsed physical time. State persists through
other-hand events and arbitrary rests; at generation it uses only the model's
emitted prefix. Supply the existing910-dimensional audio/time/environment context
at the target event beat, six planned phase tokens (up/down/left/right/dot/empty),
and log1p elapsed seconds divided by4 plus two known-history flags. No current or
future target positions or within-phase cut choices enter these inputs.

One feedforward geometry model: previous-pose tokens109×16, phase tokens6×8,
two128-wide GELU layers; shared autoregressive slot head with count embedding16×16,
slot embedding6×16 and exclusive-prefix token sums. Slot head176→128→108, GELU.
No recurrent hidden state, timing/count prediction or source-pose injection.
This isolates an explicit-state conditional geometry hypothesis; it does not
claim that the old GRU lacked the capacity to remember both hands.

Reuse the phase-preserving decoder's canonical ordering, distinct-cell and
remaining-slot masks. Use the same legal-token mask during slot training, based
only on the exclusive teacher prefix and known phase/counts. Every human target
must remain reachable before fitting; reject changed/malformed data, never silently
drop examples. Teacher histories are authored prefixes; rollout histories are
generated, an acknowledged distribution shift measured by complete rollouts.

## Fixed fit and validation selection

One initialization, seed20260930;6000 AdamW updates at3e-4,weight decay.01,clip1.
Batch512 independently sampled events: select a training family uniformly then
an event uniformly. Mean legal-slot CE per event, then mean over events. Save
updates0/1000/3000/6000; resume exact model/optimizer/RNG state. No hyperparameter,
data-membership or architecture sweep. Teacher CE is diagnostic only.

Before fitting, construct six validation templates using the frozen audio-budget
planner and approved45 native bank. Use the existing validation proxy: authored
BPM/native audio duration, fixed training-median request, empty environment,
NJS18/offset0. Choose first complete native seed0..5 without reference geometry or
QA-based selection; if any family has no complete template, stop this packet's
fit and report the preparation failure. This proxy differs from deployment's
MI/B0 inputs and admitted-template selection; disclose that limitation.

Evaluate the retired v19 phase refiner as a fixed control on these templates,
seeds0/1 at temperature1. For each new snapshot, export the same12 full rollouts,
read back and measure with the frozen approved45 validation scales. Select by
fewest incomplete outputs, lowest mean geometry error, earliest update. Missing
components are not zero. Qualification requires a nonzero selected update, all12
complete, at least10% lower mean geometry than the fixed control, each component
no worse than1.1×control+.01, and no reduced paired opportunity coverage. Literal
timing/phase/environment/support invariants must hold. Do not use dev for selection.

## Development and audit

Only a validation-qualified model proceeds to48 new development refinements on
the existing eight admitted native templates, same six seeds/temperatures and
first-admitted/native-fallback selector. Reuse unchanged full quality gates,
at least six refined families, phase signatures and directional-category checks.
Report movement, horizontal-pattern exposure, unknowns, fallbacks and costs.
No freshness or release claim follows a development result.

Freeze sources, views, native/planner/model artifacts, scales, code/spec and
protected files. Verify input causality, inactive-hand histories, real-time ages,
source-pose independence, literal reachability, masked loss and exclusive prefixes.
Rebuild tensors/control/validation exports and selection independently; replay
development if reached. Keep one preparation/QA writer and45-minute work/55-minute
hard limits per resumable stage, with2GB system headroom. No new dependencies,
playtests, annotations, judge batches, pushes or production changes. A negative
ends this one-fit hypothesis, without another checkpoint/temperature sweep.
