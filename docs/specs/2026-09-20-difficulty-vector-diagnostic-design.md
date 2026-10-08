# Cadence, coordination and recovery diagnostic

Date: 2026-09-20.

## Purpose and scope

Test whether aggregate event rate hides substantial differences in per-hand cadence, two-hand coincidence and inter-event gaps on existing maps. This is the first, deliberately small difficulty-vector increment. It supplies descriptive evidence for future calibration decisions, not a learned difficulty score or proof of equal perceived difficulty across genres.

CPU-only existing artifacts and synthetic fixtures; no generation, training, viewer installation or production changes.

No new dependencies. Do not change `ladder.json`, `ladder.py`, `groom.py`, `convert.py`, `motion.py`, existing audit metrics, corpus splits or model/cache files. No sealed-test access. Commit logical units; never push.

## Frozen panel

Freeze `eval/difficulty_panel.json` before computing comparison results. Exactly 45 expected entries:

- All 30 generated `e2e-off` candidates already enumerated in `eval/wall_panel.json`: six each for Still Waiting, numb, rather_be, FENT_TG and if_i_lose_myself. Keep every candidate, seed, selected flag and originating results record; six seeds are not six independent songs.
- All 11 human chart entries in that same panel, preserving known unknown timing/scope. RC's two map versions are one song family for aggregation.
- `latest/08_Still_Waiting` from `eval/benchmark.json`: Expert.dat, ExpertPlus.dat, ExpertPlus2.dat, ExpertPlus3.dat. Cohort `historical_export`; do not pool with the replay-off experiment or imply a paired before/after comparison.

Record chart hashes, source manifest/results hashes, source IDs, BPM, origin, cohort, family, chart label, label provenance, settings provenance and selected-candidate provenance. Validate each source's recorded values at run time, not only its hash. Binding to an existing manifest is explicit; do not guess a path from a song title. A missing chart is a recorded integrity failure, not a replacement opportunity. Expected membership and counts must be required fields.

Info, if present, is hash-bound separately. Authored rank/NJS/offset require exact Standard filename and chart-hash binding using the existing E01 resolver. Missing alias targets in Hit That/SPACEMAN remain unknown. Their benchmark filename labels may be displayed and compared descriptively, explicitly `benchmark_label_unverified`; do not call them verified authored ordering. Historical Still Waiting has verified ranks 7/9/9/9 and custom labels Expert++/+++; label suffixes do not create numeric ranks. The saved replay runner explicitly requested ExpertPlus; record this as the experiment target, not a human-equivalent tier.

Genre is optional annotation with source and `provisional`/`unknown` status. Carry only existing documented labels when identity is bound. Do not infer genre from names, pool tiny genre groups, or publish genre accuracy/equivalence. Song-level comparisons are sufficient for this packet.

## Input and measurement model

Reuse `eval.visibility.load_scene(entry, "standard")` for its fixed-v2 timing/note-scope normalization and unknown propagation. Its hypothetical NJS/lifetime are not inputs to these timing measurements. Unsupported/invalid note scenes remain unmeasured with reasons. Bomb/wall counts are retained as excluded coverage; this is colored-note timing only. Do not import camera conclusions as difficulty features.

Convert beats to absolute seconds with the bound BPM/origin. Preserve all colored notes, including dots. Group same-time notes per hand into **grouped events**, never asserted physical swings. Use a declared 1 microsecond simultaneity tolerance: sort timestamps, anchor each cluster at its earliest timestamp, add only timestamps within that tolerance of the anchor; never use transitive chaining or beat-rounded grouping. Report how many clusters required tolerance rather than exact equality. Duplicate same-cell occupancy retains the existing unknown-scene behavior.

Each event stores timestamp, hand, raw note IDs, number of directional notes and dots. Multiple-direction clusters are counted but marked ambiguous; all-dot clusters are direction-unknown. Neither supplies a trajectory. No endpoint-speed calculation or automatic head selection is introduced here.

For each measured chart use `[first colored event, last colored event]`, duration `D`. Empty charts and `D=0` have counts but undefined rates/quantiles (`null` with reason); never substitute a perfect zero. Do not include unverified audio padding or call this audio duration. Compute the following unrounded quantities; presentation may round separately:

| Field | Exact definition |
|---|---|
| Counts/coverage | Raw colored notes, directional notes, dots; grouped events L/R; occupied instants; multi-direction and dot-only groups; ignored object counts |
| Rate | Grouped events / D; occupied instants / D; L/D and R/D; raw directional notes / D kept distinctly labeled |
| Per-hand cadence | Consecutive grouped-event gaps per hand in ms: minimum, p10, p50, p90; sample count. Missing gaps yield null. Positive gaps only; a grouping failure yielding zero must fail the metric, not be clamped |
| Burst | Maximum grouped count in `(t-2s,t]`, divided by 2, separately L/R/combined; evaluate at every occupied timestamp after ingesting both hands. Fixed 2s denominator even for short charts; return earliest maximizing window and IDs |
| Coordination | Number of occupied instants containing both hands / number of occupied instants. Multiple notes in one hand do not create two-hand coincidence |
| Local imbalance | Maximum `abs(L-R)/(L+R)` in `(t-8s,t]` with at least 8 grouped events, evaluated at every occupied instant after ingesting both hands; return counts/window/IDs. None eligible means null. Mirror swaps L/R, not magnitude |
| Gap structure | Consecutive distinct occupied-instant gaps: p50, p90, maximum; number with gap >=1s, and sum of their full durations / D. Call this **long inter-event-gap share**, not silence, rest-mask correctness, or physiological recovery time |

All quantiles use nearest rank `sorted_values[max(0, ceil(p*n)-1)]`. Keep absolute witness times and IDs but compute invariant features from differences. Window left boundary is excluded, right included, with the same 1 microsecond equality tolerance; events at the left boundary within tolerance are excluded. Use deterministic tie breaking by earliest time then ID.

Do not force every axis to increase with a mapper's nominal difficulty. An easier chart can contain a longer isolated reach or a higher coincidence share. No weighted sum, learned ranking or comfort threshold is part of this packet.

## Comparisons and evidence

1. Per chart: full vector, denominators, coverage and supported/unknown status. Per generated song: median and min/max across six candidates, and the previously selected candidate separately. Never reselect on new metrics.
2. Pair the five selected replay-off charts across different songs. Define rate-close as `abs(a-b)/((a+b)/2) <= .10` on grouped-event rate; zero/undefined rates are ineligible. Show **all** eligible pairs and their component differences, not only the biggest contrast. If there are none, report none; do not relax the threshold.
3. Show human Hit That/SPACEMAN labeled chart vectors side by side with unverified-label caveats, and the historical Still Waiting labels separately. Compare individual chart spans first. An optional common-span table is permitted only within each existing same-audio ladder or same-source candidate set: use the union of first/last event bounds across that group, bind shared audio/time origin, and retain both denominators. No cross-version audio alignment is inferred.
4. At most three timing-raster witnesses, selected deterministically: the eligible selected pair with the largest absolute 2s per-hand peak-rate difference (peak hand = max of L/R), the selected generated chart's largest eligible imbalance, and the selected chart's longest occupied-event gap. Include source IDs, seconds, notes/dots, both hand rows and the exact measurement window. A missing category is `not_applicable`. These are timing rasters, not physical swing visualizations.
5. Keep construction-known equal-rate counterexamples alongside observations. Equal rate with different coordination or cadence proves the metric distinction, not an actual player difficulty difference. No real-data contrast is required for acceptance.

## Construction-known acceptance

All fixtures must pass; the complete frozen panel must be accounted for, hashes/source values verified live, and repeated reports byte-identical. Unsupported entries are explicit unknown rows excluded from numeric denominators, not missing members. No minimum measured count is invented to force the panel to pass. If no meaningful comparison survives, report insufficient coverage and stop.

Mandatory fixtures: time translation; 130/260 BPM re-encoding with identical seconds; hand mirror; note-order permutation; equal-rate alternating singles versus synchronous doubles; equal-rate even versus burst timing; adding same-hand simultaneous dots changes raw counts but not grouped cadence; multi-direction stacks counted/flagged without infinite velocity; exact 2s/8s boundary events; simultaneous hands ingested together; gaps exactly 1s; empty/one-instant charts; all-dot charts; unknown timing/transforms; tampered hashes/BPM/Info; missing membership; repeated rank9 custom labels and unmatched Info aliases. See the plan for numeric examples.

## Stop rule and follow-up eligibility

One implementation/measurement/report cycle. Correctness failures are repaired; observation thresholds are not tuned to manufacture findings. Production remains unchanged regardless of result. A follow-up scheduling/calibration experiment requires a named reproducible contrast on saved valid candidates, stable timing/grouping definitions, and a specific proposed control/guardrail. A scalar difficulty fit, clean rebuild or style model requires its own decision. Null or coverage-limited findings close this packet without panel expansion.
