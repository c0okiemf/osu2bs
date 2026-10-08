# Locate remaining emitted-rhythm failures before redesign



The repaired diverse joint model improves geometry/generalization but rhythm still
fails. Two songs fall back to B0. The current aggregate combines event density,
left/right activity, double share and empty windows, and does not directly explain
what the next representation must control. Retire the current fit recipe; this is
an artifact audit, not another fit, decoder/temperature variant or selector search.

- [ ] Add `eval/joint_rhythm_failure_audit.py`, bind frozen v19, v18, v15, original
  retrieval/B0 artifacts, reference payloads and measurement recipe/scale hashes.
- [ ] Recompute selected profiles/errors from actual exports and original audio;
  decompose mean rhythm error into its five scaled components, paired empty-window
  mismatches, note/double activity and achieved versus requested hand-event rate.
  Show empty-window losses from missed reference activity separately from generated
  activity in reference silence. Preserve original gate/scales and all fallbacks.
- [ ] Quantify fallback contribution using the6 common nonfallback families and
  the two fallback families separately. For each song, report the existing6-candidate
  range and lowest rhythm error among admitted candidates (or B0). This is an oracle
  diagnostic upper bound on any selector from this pool, never a serving selector.
  Also report all-complete-candidate bounds without pretending unknown is admitted.
- [ ] Record worst8-beat passages by rhythm component and directional continuity
  from actual selected exports. Verify self-comparison zero error, complete records,
  artifact hashes and deterministic repeat. No new maps, QA writes, fit or reference
  conditioning at serving. Use findings to plan the next representation; do not
  relax the old gate or claim that lower profile error proves perfect flow.
