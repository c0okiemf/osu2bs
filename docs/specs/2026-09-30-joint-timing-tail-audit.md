# Trace the joint model's small-time proposals



The v16 validation outputs contain many same-hand intervals below1ms; the approved
training corpus has1 such interval in48,043. Its smallest positive timing category
has only16 targets, all residuals below-2.09, including literal near-simultaneity.
Determine whether generated tiny intervals are low-probability sampled categories,
modal choices, or boundary/zero-time effects before changing decoding or fitting.

- [ ] Add `eval/joint_timing_tail_audit.py` in a separate frozen diagnostic namespace.
  Bind v16 selected checkpoint, source roles/payloads, validation records/exports,
  uniform audio views, tracing code and this plan before collecting traces.
- [ ] Replay the12 already-budgeted selected-checkpoint validation rollouts with
  observational hooks only; assert exact source notes and completion match their
  serialized outputs. Record legal probability/rank, chosen timing category and
  same-hand interval below1ms. Preserve rests and literal source timing.
- [ ] Compare the trace with exact approved train actions and teacher-conditioned
  category frequencies; classify sampled-tail versus modal/boundary behavior.
  Save all evidence, make no new candidate or fit, and use it to choose the next
  bounded mechanism. These gaps are diagnostics, not a universal playability gate.
