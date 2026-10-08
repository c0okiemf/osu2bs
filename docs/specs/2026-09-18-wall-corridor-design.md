# Wall corridor diagnostic design



## Goal and scope

Implement the smallest independent A06/D01 subset whose answers can be checked
from construction: which horizontal head-center positions remain clear of
full-height walls, and the necessary lateral travel between two constrained
intervals. It is a diagnostic under a declared model, not a comfort validator.

Global constraints (copied into the execution plan):

- CPU-only, saved artifacts only; no generation, training, or production changes.
- Support standard fixed-BPM v2 full-height rectangular walls first; report every unsupported object/window explicitly.
- Seconds are authoritative; preserve source IDs, chart hashes, and timing provenance.
- Model domain is [0,4] lane units; head half-width sensitivity is 0.15, 0.25, 0.35 lane units.
- All new findings are diagnostic/model-conditional; no new hard rejection, penalty, repair, or physiological threshold.
- A06/D01 coverage is partial; D02–D06, E, and music-dependent F/G remain outside this packet.
- Commit logical units; never push.

## Normalization

Create an evaluation adapter separate from the generation path. Consume raw
v2 obstacles retained by `eval.map_reader.read_dat`, plus an explicit fixed
seconds-per-beat and time origin. For experiment candidates use their sibling
`results.json` BPM; for human references use the frozen benchmark BPM. Preserve
the chart's own time origin; do not align the reference to another recording.

Normalize each supported wall as `(id, start_s, end_s, x0, x1)` with
`x0 = _lineIndex`, `x1 = x0 + _width`. Full-height `_type=0`, integer lane
placement, positive duration/width and bounds within [0,4] are this packet's
supported geometry. Wall time is the nominal map contact interval, not a
simulation of the game's collision box or jump animation.

Reject malformed/nonfinite values from analysis with an explicit reason.
Crouch walls, v3/v4, negative-duration/fake walls, extended/custom geometry,
rotation, and unhandled BPM changes are unsupported here, not silently converted
to full-height v2 walls. Do not infer support from a plausible wall shape when
the map's coordinate/time system is unsupported. Ordinary lighting is irrelevant.
If an unsupported wall has known valid start/end times, mark its interval
unknown; if its timing or the chart coordinate system is unknown, mark the
whole chart unknown. Other supported intervals may still be inspected with
partial-coverage status. Never issue a whole-chart clearance verdict for a
partial chart. Native chains/bombs/notes do not affect this wall-only calculation;
their omission must be explicit in `scope`.

## A06: model corridor occupancy

Use continuous lane coordinates: lane `c` occupies [c,c+1]. For head radius `r`,
allowed centers start in `[r,4-r]`. An active wall [x0,x1] excludes center
positions with distance to the wall less than `r`, i.e. `(x0-r,x1+r)`.
Exact tangency is allowed **by this model**; record that convention. Compute
the complement of the union of these open exclusions, retaining all closed
safe intervals, including degenerate points. Do not choose a preferred side or
replace a union of intervals by its centroid.

Sweep all wall start/end times. Activity is `[start_s,end_s)`: process all events
at one time together, removing ends and adding starts before evaluating the
following interval. Evaluate only positive-duration intervals; exact boundary
contacts do not create an invented duration. Do not copy the existing validator's
one-millisecond end tolerance into this diagnostic. Numerical comparison
tolerance is at most 1e-9 seconds/lane units, not a gameplay grace period.

Emit `A06.corridor_empty_model` when the supported safe set is empty. If an
unknown object overlaps, retain `model_empty=true` but use segment status
`unknown` for overall interpretation. Otherwise status is `empty_model` or
`known`. The result means no position in this **bounded 1D model**,
not impossibility for every real player. A nonempty set is not proof of comfort.

## D01: pairwise travel lower bound

Collapse contiguous intervals having identical supported safe sets and coverage
status, retaining wall-ID provenance; never merge through an unknown span. Compare
successive wall-constrained intervals, including a gap containing no walls.
Skip any pair with unknown intervening geometry or an empty endpoint safe set;
emit a reason rather than a speed. Keep the sets, source wall IDs, and gap.

For two nonempty sets `A` and `B`, compute
`distance = min(abs(a-b) for a in A, b in B)` analytically over interval pairs.
If they intersect, distance is zero even when wall sides/IDs change. Available
time is `max(0, B.start_s - A.end_s)`. For positive distance and positive time,
report `distance / available_time` in lane units/second. For a zero gap and
positive distance, report `discontinuous_model_constraint`, with speed `null`
rather than JSON infinity. Zero distance always gives speed zero.

This is a necessary pairwise lower bound, not a chosen body trajectory or a
global feasibility proof. In particular, pairwise intersections can use different
positions at successive boundaries; zero pairwise distance does not certify the
whole sequence. There is no fixed starting head position, acceleration limit,
note-driven lean, speed cutoff, medical interpretation, or frequency penalty.
Run all three radii; disagreement remains visible and cannot be voted into truth.

## Construction-known validation

These are exact answers for the declared model, not human playability labels.
Nominal radius is 0.25; run radius sensitivity separately.

| Construction | Expected nominal result |
| --- | --- |
| No walls | Safe [0.25,3.75]; no D01 pair |
| Outer-left width 1 | Safe [1.25,3.75] |
| Outer-right width 1 | Safe [0.25,2.75] |
| Both outer width-one walls overlap | Safe [1.25,2.75], not empty |
| Alternate outer-left/right with zero or 0.1s gap | Distance 0; no forced dodge |
| Left half [0,2] and right half [2,4] overlap for 0.2s | Empty corridor for exactly 0.2s |
| Left half ends at 1s, right half starts at 1.1s | Distance 0.5; gap 0.1s; lower bound 5 lane/s |
| Same half-width switch with 1s gap | Distance 0.5; lower bound 0.5 lane/s; no hard-invalid finding |
| Half-width switch at exactly the same time | No positive-duration simultaneous overlap; discontinuous constraint |
| Central wall [1,3] | Safe union [0.25,0.75] and [3.25,3.75]; do not average it |
| Duplicate/coextensive/nested walls | Same geometry as their union; preserve all contributing IDs |
| Supported corridor plus crouch/custom wall | Explicit unknown coverage; not a clearance pass |
| Zero/negative duration, NaN, infinite time, zero width | Explicit invalid/unsupported reason; no fabricated safe interval |

Also require source-order invariance; translation in time; mirrored geometry
with equal distance/speed; identical physical timeline encoded at 130 and 260
BPM; wall-end events within 0.5ms; single-wall onset without an assumed initial
pose; a three-interval case demonstrating that pairwise zero distance does not
prove global zero-motion feasibility. At least one fixture must have overlapping
walls begin/end at the same timestamp, so sequential event handling would fail.

Acceptance: **100% exact model fixture outcomes**, numeric comparisons within
1e-9, **zero false forced-travel results on the counterexamples**, byte-identical
canonical reports on repeat runs, and all unsupported coverage counted. Do not
set a minimum real-chart flag count or tune a comfort threshold on the references.

## Evidence panel and exit

Freeze SHA-256 for the 30 existing `replay-ab-motion/*/e2e-off*.dat` candidates
and their five `results.json` files. Use all six candidates per song, not just
winners. Add the 11 human charts in `eval/benchmark.json` whose IDs start
`human/`, verifying its hashes before analysis. Missing/mismatched files are
reported and make the declared panel incomplete; do not replace or regenerate
them. Do not access sealed-test maps.

Report per map/radius: supported and unsupported object counts, known/unknown
seconds, empty-corridor duration, transition count, positive-distance pairs,
lower-bound distribution, and witness timestamps/IDs. Denominator zero is
`not_applicable`, not a perfect pass. The 714 generated walls should show zero
wall-required travel in this model: this is the expected counterexample, not
proof that their associated note movement is comfortable.

Include three deterministic static wall-time/corridor plots from available
data: a generated outer-wall counterexample, a human wider-wall transition,
and an overlap or unknown-geometry example. If the last case is absent, use a
labeled construction fixture; do not fabricate a real defect. Plots show lane
coordinates and seconds, not an animated human avatar. No audio annotation or
new map generation is required.

Close the packet when exact tests, provenance/coverage, report, and plots exist.
If there is no supported generated defect, state that and stop A06/D expansion.
Then scope E01 contextual visibility against the current center-cell mask as a
separate decision. If there is a finding, first inspect its witness and model
sensitivity; production integration is a separate packet with existing
validity/motion/exposure/rest gates. Neither outcome authorizes automatic repairs.
