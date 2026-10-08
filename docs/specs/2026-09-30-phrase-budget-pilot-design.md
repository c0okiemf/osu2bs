# Audio/workload budget prediction before another note decoder

No new note generator
or change to B0. This is a small, deterministic prediction pilot for the verified
phrase-budget representation, not another fit of the retired joint GRU recipe.

## Fixed predictor

Use the same336 train/6 approved validation roles and authenticated phrase packets.
Serving feature function accepts only audio features, window start/end, BPM, native
duration and requested hand-event rate; it must never accept target notes/budgets.
Use the existing common-band audio views. Features are the current window's27 mean
and27 standard-deviation channels, requested hand-events/beat, the existing32 ordered
onset/RMS half-beat features, and[60/BPM,window_fraction,start/duration,end/duration].
91 features total. Preserve requested workload even when an audio slice is empty.

For training only, request rate is each source's exact whole-song hand-event rate.
Targets are[left-only,right-only,both] counts divided by requested rate×actual window
seconds. These ratios are finite and bounded in the inspected corpus (maximum1.611).
Each family has equal total weight; its windows have equal within-family weight.
Fit feature means/stds on train only (std floor1e-6), target mean as an unpenalized
intercept, and one weighted ridge solve with penalty0.01 on standardized coefficients.
No hyperparameter, feature or model sweep. Constant control predicts the same
weighted training target mean without audio dependence.

At prediction, clip negative ratio outputs to zero, multiply by requested rate×window
seconds and round with floor(x+0.5) to three nonnegative integers. This parameterizes
the predicted budget; it does not edit or repair notes. Report predicted workloads,
counts, rests and contrast. No hard density cap, source identity, chart geometry,
future labels or held-out statistics enter prediction or normalization.

## Evaluation and claim boundary

Exactly6 deterministic validation plans per arm, using the prior validation proxy:
authored BPM/native duration and the fixed approved-train median request rate.
This is an acknowledged selection/prediction proxy, not deployment confirmation.
Compare to exact reference budgets and the mean of the existing v19 selected-model
seeds0/1 for each family. Keep the same approved45-derived rhythm scales. A budget
directly defines event/left/right density, double share and empty-window status;
it does not define note times, geometry, physical flow or QA admission.

Predefined signal screen: ridge mean rhythm error at least10% below both constant
and cached v19 controls, all5 component errors <=1.1×each control+.01, all6 plans
complete, and each whole-song predicted hand-event rate within20% of its request.
Record every component, empty-window mismatch, double share and activity contrast;
do not treat a constant/boring plan as a generated quality result. Report
PLANNING_SIGNAL_PRESENT or PLANNING_SIGNAL_NOT_ESTABLISHED, never machine pass,
development-positive, production eligible or perfect maps. The existing full-map
quality gate remains unchanged and applies only after actual generation.

## Verification and follow-through

Bind source packet/view/role identities, train-only arrays/weights/model receipts,
cached v19 control records, source-reference payloads, feature/error recipe, code
and this spec in `experiments/phrase-budget-pilot-v1/`. Verify exact budget-to-rhythm
identities, zero self-error, split exclusion, constant-feature ridge behavior,
nonnegative integer predictions, model reconstruction and deterministic report.
No QA writes, full-map generation, dependencies, playtests, annotations, judges or
push.45-minute work/55-minute hard limits,>=2GB headroom.

If signal is absent, do not automatically fit a decoder on these predictions;
inspect the failure and revise the planner hypothesis. If present, separately
specify how a decoder realizes budgets without boundary pile-ups, preserves causal
hand state and is evaluated on actual exports before fitting it. Reachability and
predicted profiles alone are not the requested final quality outcome.
