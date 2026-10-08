# E01 projected-face visibility diagnostic design

2026-09-20. This is a bounded
new evaluation subsystem, not a generator change.

## Question, alternatives, limits

Does a contextual geometric measurement expose cases that a blanket ban on
middle-row columns 1/2 cannot distinguish? Measure how much of a target note's
face is hidden, for how long, and how long it is exposed before its hit time.
Do not label a fraction of visible surface as player comprehension.

Choose an analytic, flat-face approach model over a full renderer or another
cell/timing heuristic. This intentionally omits jump arcs, note flips/rotation,
cube side faces, lighting, arrow shape, visual acuity, learning and head motion.
Passing its exact fixtures establishes this model's arithmetic, not game-accurate
vision or human readability. No result in this packet authorizes unmasking cells.

Global constraints (copied into the execution plan):

- CPU-only, saved charts only; no generation, training, renderer installation, or production changes.
- Preserve all center-note masks, repairs, candidate scoring, and selection defaults.
- Seconds and metres are authoritative; every chart, setting source, model parameter, and scope exclusion is recorded.
- Unknown settings or unsupported geometry cannot become a clearance pass or an invented historical setting.
- Results are projected-face occlusion proxies, not human readability, reaction thresholds, or difficulty labels.
- Commit logical units; never push.

## Inputs and frozen panel

Create `eval/visibility_panel.json` v1 with exactly nine entries selected from
`eval/wall_panel.json`: five generated paths ending `e2e-off-c00-s0.dat`, and
ExpertPlus references `rap_god_19909`, `reality_check_25f`, `reality_check_3741`,
`spaceman_24e5e`. No winner selection or replacement. Freeze chart and timing
source hashes; record source offsets, Info hashes and binding evidence separately.
Reject incomplete panels for certification, while retaining failure diagnostics.

Support ordinary fixed-time v2 colored notes, including anonymous dot notes as
visible objects. Do not collapse stacks into one sprite. Preserve raw note IDs.
Reject/mark unknown nonfinite/malformed notes, modded coordinates/scales/custom
animation, rotations, unsupported versions and unhandled time systems. Cosmetic
colors and editor bookmarks are not coordinate transforms. Do not use a reader
that silently filters invalid notes to assert complete coverage.

Walls, bombs, chains/arcs, sabers and fragments are outside the **note-to-note**
visibility model. Count them. This is a scoped measurement even where they are
present, never a whole-chart visibility verdict. In particular a hypothetical
fixed camera can lie in a wall; do not call it a feasible body pose. If note
position/time is unknown, affected targets are unknown rather than clear. With
unbounded unknown timing, the whole chart's note occlusion is unknown.

Analyze every chart under the same explicit standard scenario: NJS=16 m/s and
approach lifetime T=0.600 s. This is a hypothetical comparison, not recovered
historical playback. For references with exact Info-to-chart bindings, separately
evaluate authored NJS/spawn offset. Report modes separately; never pool them.

## Jump settings and lifetime

For an authored setting, freeze this **viewer-style** HJD convention:

```text
beat_seconds = 60 / bpm
h0 = 4 beats
while 2 * njs * beat_seconds * h0 > 35.998 metres: h0 = h0 / 2
h = max(0.25 beats, h0 + spawn_offset_beats)
T = h * beat_seconds
half_jump_distance = njs * T
jump_distance = 2 * half_jump_distance
```

This follows the inspected [ArcViewer BeatmapManager](https://github.com/AllPoland/ArcViewer/blob/a7b2d984f91346ade9f25e523afb5ed67cb2cb84/Assets/__Scripts/Previewer/MapControl/BeatmapManager.cs)
revision `a7b2d984f91346ade9f25e523afb5ed67cb2cb84`; the exact 35.998 boundary is
intentional. Pin the convention, not an unversioned library. IEEE-754 double
arithmetic is this diagnostic's convention; values near the halving boundary
need explicit fixtures and are not asserted identical to every game float path.
Validate positive finite NJS/BPM and finite offsets before this calculation.

For standard scenarios T is explicit; derive an equivalent offset only for
reporting, never let BPM recompute a different T. Distinguish duration in beats,
duration in seconds, half distance and full distance in every field name.
No extra pre-jump visibility is included. The [mapping guide's jump-setting
description](https://bsmg.wiki/mapping/intermediate-mapping.html#jump-settings)
provides the terminology; preferences and actual recognition time remain outside
this diagnostic.

## Scene and camera: exact proxy definition

- World axes: x right, y up, z away from the observer. Cut plane z=0.
- Lane centres x=`-0.9 + 0.6*column` metres. Row centres y=`[0.60,1.15,1.65]`
  metres. The vertical origin is a declared model choice, not an inferred player
  height setting. Lane/relative-row spacing is informed by the inspected
  [ArcViewer ObjectManager](https://github.com/AllPoland/ArcViewer/blob/a7b2d984f91346ade9f25e523afb5ed67cb2cb84/Assets/__Scripts/Previewer/MapControl/Objects/ObjectManager.cs).
- A note is an opaque, camera-parallel square of side 0.50 m at its centre.
  Direction/color do not change the square. This is not its game hitbox or mesh.
- For hit time `t_i`, it exists on `[t_i-T, t_i)`, with constant x/y and
  `z_i(t)=njs*(t_i-t)`. Remove it at its nominal hit time. Assume perfect on-time
  cuts; do not retain occluders past the hit, model misses, or predict saber cuts.
- Nominal pinhole eye E=`(0,1.60,-0.65)` m, looking along +z, with normalized
  image plane focal length 1. There is no FOV crop. This reports geometric
  obstruction, not whether a note is inside a headset's view.
- Project the square's x/y bounds by dividing their offsets from E by positive
  depth `z_i-E.z`. Nearer means strictly smaller depth (tolerance 1e-9 m).
  Coplanar distinct notes do not occlude one another in this model. Exact
  same-time same-cell overlapping colored objects are invalid occupancy, not a
  visibility case to resolve by arbitrary ordering.

For target j, clip all nearer projected rectangles to j's projected face and
compute their **union area**, not the sum of pairwise overlaps. Report covered
fraction C_j(t) in [0,1], IDs of contributing occluders, and optionally individual
contributions for witnesses; summed individual contributions must not be used
as total coverage. Compute a second fraction for the central square of side
0.25 m on the target face, labeled `central_marker_proxy`, not arrow recognition.

No blocker means zero occlusion, including for a centre note. This is the key
counterexample to turning note location into a context-free violation.

## Time integration and sensitivity

Measure each target over its complete approach interval, not just at its hit.
Use 10 ms sample cells anchored to target spawn; split cells at every relevant
spawn/hit event, then evaluate midpoints with duration weighting. Never round
hits to an unrelated global sample grid. Initial/final partial cells count.

Record maximum coverage, area-time integral/T, total and longest contiguous time
with C>=0.5 and C>=0.8, and the terminal contiguous duration with C<0.5 before
the hit. Keep early clear time as well: late occlusion does not erase information
previously available. These thresholds are descriptive geometry bins, not
acceptability limits. Zero targets/unknown targets are not perfect scores.

For cost control, run full charts only at nominal settings/camera/size. Freeze
up to 20 witness target IDs per chart from that run: the first five by timestamp
in each of (1) targets hidden by a banned-cell occluder, (2) banned-cell notes
that hide no other target, (3) targets hidden only by non-banned-cell notes,
(4) uniformly spaced valid targets across the chart. Use C>=0.5 at any sample
for categories 1/3. Deduplicate; retain missing strata as absent, do not fill by
searching other maps. For category 2 require no positive-area overlap at any
sample of any other target; label this sampled-model evidence only.

Evaluate those targets with full chart context, including every potentially
active occluder, under one-factor-at-a-time variants: eye x=-0.15/+0.15 m,
eye y=1.40/1.80 m, square side=0.45/0.55 m, NJS=12/20 holding T fixed,
T=0.40/0.80 holding NJS fixed. Central marker side is half the face side.
Other values stay nominal. Also halve the timestep to 5 ms at nominal.
These are hypothetical viewpoints/settings, not measured human variability.
Store disagreements; do not average sensitivity away or claim robustness to
unmodeled jump animation because these variants agree.

## Independent oracle and construction-known fixtures

Production diagnostic math uses projected-rectangle union by x-slab sweep and
merged y intervals. The test oracle must use a different construction: sample
a fixed 512x512 grid of points on the target's physical face; trace eye-to-point
rays; intersect each ray with each blocker plane; test whether the intersection
lies inside its physical square before the target. It must not call the
projection/union functions. This is a CPU geometry oracle, not a renderer.

Exact analytic fixtures use E=(0,0,0) and supplied square/rectangle faces, so
the expected answers can be written by hand:

| Construction | Expected |
| --- | --- |
| Target alone, including banned centre cells | 0 covered; entire lifetime exposed |
| Far target x/y bounds [-1,1], z=4; blocker [-0.5,0.5], z=2 | 1 covered |
| Same target; blocker x=[0,0.5], y=[-0.5,0.5], z=2 | 0.5 covered |
| Same target; blocker x/y=[0,0.5], z=2 | 0.25 covered |
| Same projection but blocker farther away | 0 covered |
| Disjoint projected rectangles or edge-only tangency | 0 positive-area occlusion |
| Two disjoint half blockers | 1 covered |
| Two identical half blockers | 0.5, not 1 |
| Overlapping blockers with known union 0.75 | 0.75 |
| Equal-depth separate faces | No depth-ordered occlusion |
| Blocker hit precedes target spawn | No occlusion; no ghost blocker |
| Blocker removed midway through target approach | Full terminal clear interval retained |
| Early clear, then blocked, then clear | All three intervals reported, not just last visibility |
| Outer-cell blocker aligned from a displaced eye | Positive occlusion despite not being a banned cell |
| Slow aligned stack with enough time after preceding hit | Exposure after removal retained; never a blanket stack violation |
| Off-axis near miss and side-visible target | Partial/zero coverage, not full hiding from same-lane alone |
| Unsupported animation/missing settings/malformed timing | Explicit unknown/scenario-only/invalid, never historical certainty |

Also test positive-depth validation; empty target lists; stable IDs under input
permutation; mirror eye and geometry; rigid translation of scene+eye; timestamp
translation; 130/260 BPM encodings with identical seconds **and T**; contrast
with re-derived authored HJD which need not preserve T; event-boundary endpoints;
sub-10ms lifetimes; projection fractions under uniform scene scaling; NJS and T
changed independently. No expectation that higher NJS always improves visibility.

HJD fixtures: BPM120/NJS16/offset-0.5 gives h=1.5, T=.75s, half distance12m,
full distance24m; exercise either side of the 35.998 threshold and the .25-beat
clamp. Unknown zero/default NJS must not silently become 16 in authored mode.

## Acceptance, report, stop rule

- 100% exact analytic and scope fixtures; numeric tolerance 1e-9 for analytic
  geometry, 1e-6 s for exact event/lifetime fixtures.
- Independent ray oracle within 0.01 absolute coverage on all nondegenerate
  construction scenes; boundaries of zero area have explicit analytic tests.
- On every selected witness, 10ms versus 5ms coverage integrals differ <=0.02
  absolute and interval-duration metrics differ <=20ms. Otherwise report
  unresolved sampling sensitivity and stop; do not retune bins to pass.
- Complete provenance/unknown accounting; repeated canonical reports identical;
  malformed panels fail closed. Every zero denominator is `not_applicable`.
- Publish model values and raw witnesses, not a new pass/fail score for maps.

Report the nominal nine-chart panel, authored subset, four witness strata and
sensitivity separately. Compare the **occluder's** banned-cell predicate with
measured obstruction; do not confuse a hidden target's cell with its blocker.
Call disagreements proxy disagreements, not precision/recall against human truth.
Provide four static projected snapshots plus coverage-vs-time plots: a benign
centre note, centre-caused hiding, non-centre hiding, and a recovery/side-visible
counterexample. Prefer actual witnesses; explicitly labeled fixtures fill absent
categories. Missing real examples remain a result, not permission to generate.

Stop after one implementation/validation pass and necessary correctness repairs.
If fixtures fail, the detector is not usable. If scope/sensitivity dominates, or
no useful real-chart distinctions appear, close inconclusive and keep the mask.
If stable distinctions appear, the next decision is a small trajectory-fidelity
cross-check against a pinned viewer on those witnesses, then a separately
preregistered candidate experiment. **This packet never removes the mask or
adds E01 penalties, even if all its gates pass.** Do not grow a renderer or the
rest of A/D/E to force a useful result.
