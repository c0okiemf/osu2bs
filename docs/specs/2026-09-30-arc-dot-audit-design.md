# Audit dot bridging in the frozen arc descriptor

`eval/expression_profile.py` documents that dots break arc-like runs. `_heads`
removes dots before `_arcs` receives the sequence, so the implementation can bridge
a same-hand dot. A four-directional-head square yields one arc run; inserting a
same-hand dot between the second and third heads still yields one. This is a
specific documentation/implementation mismatch, independent of candidate outcomes.

Before proposing a new measurement version, quantify the affected runs on the
existing source corpus and cached development charts. Reproduce the exact legacy
arc count by enumerating its same-hand directional quadruples in the same8-beat
measurement windows. Mark a run as dot-bridged if a same-hand dot lies between its
first and last head, inclusive for simultaneous ambiguity. Other-hand dots do not
break this hand's sequence. Keep raw counts, per-window values and denominators.
For a scoped counterfactual, subtract those runs with the same legacy denominator;
this is diagnostic only, never an updated quality gate or promotion decision.

Read existing source JSONs and actual candidate chart+Info files; align candidate
measurement windows to reference BPM as usual. Bind source, export, helper and
code hashes. Include original B0/retrieval and v10/v11 selected outputs when those
runs are complete. Report per-family and aggregate affected counts. Freeze the audit
identity before computing corpus outcomes; preserve every old evaluator, report,
scale and production artifact. No new generation, fit, QA-cache write or threshold.

Tests pin the reproduced legacy count, same-hand dot bridge, other-hand-dot
invariance and endpoint/simultaneous cases. A substantial or tiny effect both remain
valid outcomes. Use this evidence to decide whether an experiment-owned corrected
measurement is needed; do not use the correction to retroactively pass candidates.
