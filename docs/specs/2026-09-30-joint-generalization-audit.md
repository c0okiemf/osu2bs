# Separate teacher generalization from free-rollout failure



Boundary aggregation removes most microscopic timing failures, but the twelve
validation charts retain substantial geometry error. Before another fit or decoder
variant, determine whether the existing checkpoint predicts held-out literal source
events poorly even with correct history, or mainly deteriorates during rollout.

- [ ] Add `eval/joint_generalization_audit.py`, frozen separately with v16 selected
  checkpoint, prepared/source/view identities, code and this plan. No fitting,
  new generated maps, model/temperature selection or QA-cache writes.
- [ ] Evaluate all45 approved train and6 validation sources with exact authored
  histories. Train uses authored conditioning. Validation reports both authored
  conditioning and the existing generation proxy (median train rate, NJS18,
  empty obstacles). The former is a diagnostic upper bound, never serving input.
- [ ] Report gap/count/slot cross-entropy and top-one accuracy, positive-gap residual
  loss and target counts. Compare continuous teacher history against128-step crops
  with32 warmup and true BOS supervised, covering every target exactly once. This
  tests recurrent-state length mismatch without selecting a new model.
- [ ] Verify crop target coverage, source/action round trips, cached train-context
  equality and unchanged weights. Report equal-family as well as target-weighted
  aggregates, observed differences and limits. Use findings for the next planning
  step; do not automatically launch another fit or infer flow from CE alone.
