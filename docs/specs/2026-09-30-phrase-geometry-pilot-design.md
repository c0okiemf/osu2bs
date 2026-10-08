# Test geometry prediction before a candidate selector

The native-budget pool contains a reference-dependent
combination satisfying the full gate, but the serving first-admitted rule does not
find it. Test whether a small train-only predictor offers an alternative before
building a selector or fitting another note generator. Never use the oracle witness
as a label, serving choice, validation family or source of model parameters.

## Fixed experiment

Use the same336 training and6 approved validation families. Keep the frozen budget
predictor, common-band audio, approved45 validation calibration and validation request
rate. Features are the existing91 audio/workload inputs plus the three predicted
integer budgets divided by requested rate times actual window seconds:94 features.
Even during training, these extra counts come from the frozen predictor, not source
budgets. They are available at serving. Training request rate remains each source's
whole-song hand-event rate; validation request is the existing fixed training median.

Targets are the existing five window geometry descriptors: low-vertical share,
arc runs/100 directional heads, double entropy, max4gram share and mean displacement.
Missing target components stay missing, not zero. Fit one separate weighted ridge
head per component (penalty0.01, unpenalized intercept, std floor1e-6). Each family
with supported targets contributes equal total weight per head, uniformly across
its supported windows. Fit means/stds and target means on that head's train rows only.
No feature, regularization, source or model sweep. Constant control uses the same
weighted target means. Clip predicted descriptors to nonnegative values, with shares
(components0 and3) also bounded above by1. These are predicted descriptions, not
edits or mandatory choreography.

## Predefined screen and limits

Predict exactly6 validation geometry profiles per arm. Score only where reference
components exist, using the unchanged approved45 geometry scales. Compare to the
constant profile and mean cached v19 selected-checkpoint seeds0/1 per family. Signal
requires at least10% lower mean error than both controls and every component error
<=1.1 times each control+.01. Require complete6-family predictions and finite outputs.
This is an authored-BPM prediction proxy; reported reference support is not a new
export-coverage claim. No generated maps, new QA, serving selection or production
change occurs in this packet.

Bind source payloads/roles, original feature arrays, frozen budget model/report,
target computation, cached control records, code/spec, arrays and geometry model.
Test masked-target weighting, train-only fit, finite bounded predictions and constant
features; rebuild arrays/models/predictions and reproduce the report. Preserve every
failed component and family. A negative signal blocks automatic selector deployment;
inspect the outcome before another constructive step. A positive signal requires
a separately frozen ranking rule and actual exported-candidate validation. No
universal arc penalty, reference-driven ranking or reinterpretation of the negative
native renderer.45-minute work/55-minute hard limits;2GB headroom; no dependencies,
playtests, annotations, judge batches, push or production replacement.
