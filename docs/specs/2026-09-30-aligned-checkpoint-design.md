# Align checkpoint selection with the actual boundary decoder

Preserve
production B0 and all earlier frozen evidence; no new fitting.

The v16 checkpoint was selected using the old timing mask. V17 fixes boundary
probability handling while holding that checkpoint fixed for a causal comparison.
Its validation timing improves substantially, but teacher diagnostics also show
poor generalization. Before retiring this fit, apply the original checkpoint
selection rule to the decoder actually used at deployment. This is not a new
teacher-CE criterion, temperature search or expansion of the snapshot inventory.

Exactly the existing v16 snapshots0/1000/3000/6000. Exactly the same6 validation
families, seeds0/1,temperature1, median train rate,NJS18, empty obstacles, uniform
audio and original-audio measurement. Use v17's censored timing rollout unchanged.
Reuse its12 authenticated6000 validation outputs. Generate only the missing36
outputs for the other checkpoints. Rank by incomplete rollouts, mean summed paired
rhythm/geometry error and earliest update, exactly as v16. Do not use development
results, QA, teacher CE or tiny-gap counts to choose a checkpoint.

If6000 wins again, reuse the full v17 development comparison and artifacts, with an
explicit inherited-result receipt and no new development generation/QA. If a
different learned checkpoint wins, generate the same48 development attempts with
fixed existing temperatures/first-admitted selection/B0 fallback, same deployment
adapter, full scenes, unchanged numerical gate and original-audio measurement.
If step0 wins, record that no learned candidate is selected and do not generate a
random-weight development batch or imply success. No production eligibility.

Bind all four checkpoints, parent data/view/source identities, v17 decoder/control
artifacts and reports, code/spec and protected hashes. Verify selected snapshot
and all48 validation outputs/profiles/errors; verify every new development export,
unmodified raw targets, scenes/settings, metrics, selection and report. Inherited
outputs must remain byte-identical and clearly attributed. Failed/unknown outputs
remain visible; no post-generation repair or new fitting.

Namespace `experiments/joint-phrase-v18/aligned-checkpoint/`.45-minute work/55-minute
hard limits per stage,>=2GB headroom; no dependencies, playtests, annotations,
external judges or push. A negative result retires this approved-only fit as a
candidate; use the measured generalization failure to redesign the next training
approach rather than repeatedly changing decoder temperatures or checkpoints.
