# Check simultaneous gate feasibility before another fit

This is a read-only diagnostic on the already opened
native-budget development pool, not a serving selector or a quality claim. Separate
per-axis oracle bounds look promising but do not establish simultaneous feasibility.

- [ ] Add one `eval/phrase_native_pool.py` audit using installed SciPy MILP. Options
  are the24 admitted native-budget outputs plus each family's B0. Exclude options
  that fail per-song support/contradiction requirements. Select exactly one per
  family and at least six nonfallback candidates. Encode both mean margins, every
  component margin and all paired coverage requirements from the unchanged gate.
- [ ] Minimize aggregate geometry error subject to these fixed constraints, using
  binary variables and a60-second solver limit. Independently recompute every gate
  from the selected records; an unchecked/time-limited incumbent is not a pass.
  Preserve no-solution/unknown distinctions. Bind code, this plan, all candidate
  records and the audited parent. Repeat deterministically; save no serving choice.
- [ ] If feasible, investigate a train-only selector before another generator fit.
  If infeasible, do not tune the selector. References are used only in this diagnostic,
  never to choose a real output or promote a candidate. Parent remains negative.
