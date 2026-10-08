# Correct workload units in literal phrase retrieval

The bank currently stores donor hand-events/second at the donor's BPM, but generation
copies those intervals in beats at the target BPM. The compared rate is not the emitted
rate. For H hand-events in an eight-beat donor, emitted rate is H*target_BPM/480.
Matching H*donor_BPM/480 to the target request instead is a coordinate error.
In the already-opened e7be retrieval, requested rate4.714 and mean bank rate4.639
become mean full-phrase emitted rate6.115 after transplantation. Tail truncation is
excluded from that diagnostic; this is not a new quality score.

V7 completed before v8 freezing: ordered retrieval worsens both primary distances
versus original retrieval. Retire temporal ordering; do not extend that failed branch.
Use original retrieval as the v8 control and candidate base.

Single intervention: replace the final scalar in the original retrieval descriptor
with hand-events/beat, H/8. Convert each serving query request from events/second to
requested_rate*60/serving_BPM. Fit this scalar's mean/std on the same fit-train phrases.
Keep all54 original audio descriptor components, joins, literal geometry,
candidate seeds, first-admitted selector and QA unchanged. Temporal features do not
participate in either the original control or corrected candidate; their saved v7
arrays may remain in the bank as unused provenance. This is a correction to units, not stronger density weighting or a new fit.
Verify the first54 standardized bank components remain byte-identical.

Use the v7 deployment inputs and six candidates per family. Both v7 controls remain
available without generation: original retrieval and uncorrected ordered retrieval.
The primary whole-generator positive rule is identical to v7 against B0 and original
retrieval (rhythm/geometry10%, component/coverage/support guards, >=6/8 nonfallback).
The isolated unit ablation is corrected original versus uncorrected original retrieval.
Also report the retired ordered arm's aggregate as historical context, not as the
intervention control or fresh evidence. No thresholds or components change from v7.
Expose all three controls and any tradeoffs. Release eligibility remains false.

Freeze this spec/code, v7 completed report and bank receipt, audit/input identities,
all serving inputs and protected artifacts. Separate namespace:
`experiments/joint-phrase-v8/workload-units/`. No v8 generation until v7 finishes and
its result is recorded. Do not modify frozen v7 outcomes or infer human BPM at serving.
Reuse full-scene QA only if actual chart AND Info bytes match a verified control.
All actual exports are re-profiled through the audited reference-time measurement view.

Required controlled fixture: donors at60 and120BPM each have4 hand-events/second at
source but32 and16 events per8 beats. At120 target BPM and4 events/second requested,
the old scalar ties them; corrected units choose16 events (4/s), not32 (8/s). Also
check request conversion, train-only normalizers, unchanged audio features and independence from retired temporal features, inherited
candidate budgets, unchanged admission/fallback and deterministic decisions.

Record scoped directional-continuity diagnostics alongside profile distances. They
are explanatory, with ambiguity/opportunity limits; no global parity ban or newly
invented quality gate. Geometry descriptor equality is known not to imply arrow-flow
equality. Preserve this limitation and all earlier negative results.

45-minute work/55-minute hard bounds,>=2GB host headroom, no concurrent QA-cache
writers. No fit, seed sweep, dependencies, human judging, playtest or push. Continue
from measured outcomes without ending the broader task at this packet boundary.
