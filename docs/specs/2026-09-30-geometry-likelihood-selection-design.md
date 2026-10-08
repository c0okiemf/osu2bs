# Geometry likelihood selection on the frozen typed candidate pool

V4 preserves timing/types and admits43/48 candidates, with nonfallback output on
all8 songs. First-admitted common geometry error6.228 loses to B0's5.679, despite
improving retrieval's7.078. An explicitly nondeployable reference oracle can reach
4.752 using the same admitted pool. This shows potential selection headroom, not
that a serving selector can recover it or that these descriptors establish quality.

Test one fixed reference-free rule, with no fitting, new generation, extra seed or
validation sweep: choose the admitted candidate with highest mean conditional log
probability of its geometry under the fixed v3 checkpoint. Earliest seed breaks ties.
If all candidates fail admission, retain original R exactly. This is a post-v4
hypothesis on exposed development evidence, never fresh confirmation.

Score candidate geometry with its actual previous generated events, exact timing,
hand counts, audio, NJS18 and known template hand-event rate. Use continuous recurrent
state from true BOS including REST actions. For each emitted nonempty canonical slot,
score the actual token under raw108-way log-softmax at temperature1; do not renormalize
by candidate-specific legal support, and do not include ungenerated empty slots or
sampled-gap/count probabilities. Counts/timing/types are identical within a song.
The score API has no reference chart/profile input. A batched forward pass with carried
hidden state is equivalent to incremental decoding; verify this on a multi-event fixture.

Reuse the exact v4 common masks, comparator records, scoped QA, profile scales and
positive rule, replacing only selection among admitted candidates. Preserve v4's
negative report. Report per-candidate NLL, selected seed changes, oracle gap, all48
attempts and inherited cost. Release eligibility is always false: a positive result
is only conditional geometry evidence, still under authored BPM and exposed families.

Freeze this spec, implementation, checkpoint, v4 report and all candidate charts,
Info and records before scoring. Separate namespace:
`experiments/joint-phrase-v6/likelihood-selection/`. Never write v4 reports or models.
Verify all chart identities/signatures and exact repeated decisions. Check likelihood
against raw per-slot sums, deterministic batch/chunk invariance and admission/fallback
filtering. No production changes, fits, new dependencies, playtests, judge batches or
push.45-minute work/55-minute hard bounds;>=2GB headroom.

Continue from the measured result. Independently established next generation issue:
retrieval's mean/std audio descriptor cannot distinguish reordered activity within
an eight-beat phrase. A controlled example changes fixed-note onset alignment1→0
with exactly identical descriptors. Address temporal ordering in a separate frozen
intervention and use deployment timing inputs; do not fold it into this selector test.
