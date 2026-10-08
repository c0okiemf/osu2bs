# Comparator and feasibility QA: prospectively validated replacement

Date: 2026-09-24.

## 1. Decision and claim

Build one evaluator version, `comparator-v1`, without training. It separates reproducible contradictions, analogous human execution support, and blinded mapping-quality judgment. This is a bounded replacement contract, not a declaration that the old evaluator passed.

The absolute-kinematics interval-coverage, sharpness, model-NLL, latent-context-selection and model-versus-neighbour disagreement gates cease to be prerequisites **only for this new version after its own development, fresh-validation and sealed-confirmation gates all pass**. Until then neither evaluator has release authority. Historical CALIBRATION_FAILED, EXPANSION_FAILED, RECENTERED_FAILED and DEVELOPMENT_FAILED remain unchanged. Constant-prior latent predictions are diagnostic only: lower development NLL does not establish calibration or fresh generalization. Do not include their values in the decision bundle or use them to select witnesses.

Success means `AUTOMATED_QA_VALIDATED_WITHIN_SCOPE`: the frozen evaluator retains supported human execution controls, detects supported certified violations, preserves benign changes, and reliably distinguishes specified expressive-content destruction from matched controls on fresh families. It does not prove enjoyment, injury safety, or full-body playability. No summed “fun score” or risk-minimizing ranking is introduced.

Alternatives considered:

- Another latent fit or global-factor adjustment: rejected; the bounded branch ended and no new predictive mechanism is justified here.
- NN-only approval: rejected; familiar geometry can be boring or malformed, and distant geometry can be good.
- **Selected:** necessary-condition checks plus a reference-support gate plus an independently tested judge. More explicit abstention and more evidence work, but each conclusion has an identifiable source.

B0 stays active; E1 stays held. This packet ends at evaluator confirmation, not E1 judgment or promotion.

## 2. Roles, freezes and bounded sequence

Use QA-train24 as the only reference bank. Use the eight inspected calibration families as **development evidence**, not fresh generalization. Their charts/replays never become bank entries. Four unread reservations (`11ec7`, `10ff1`, `658e`, `20cae`) become `qa_validate_comparator`, preserving original reservation hashes and prior role lineage. Original six-family seal remains separate.

Family and player exclusions, pseudonymization and restricted originals remain mandatory. Query-family and all known query-family player tokens are excluded from control retrieval. Generator processes cannot read QA roots; the judge gets a bundle-only view, not the repo, original paths, role labels, mutation labels or generator history. Existing generator chart exposure is disclosed in the custodian report, never misrepresented as retroactively clean training.

Two freeze records are required:

1. **Recipe freeze before development measurements:** code, dependencies, geometry assumptions, bank/scaler/descriptor identity, selectors, mutation recipes, denominators, rubric, adjudicator model/version/settings, rendering and audio feature configuration, budgets, reservations, and this spec. Numeric support cutoff is filled by the single formula in §4, not searched. Unit/renderer correctness fixes are allowed before this freeze, not outcome-driven recipe tuning.
2. **Evaluator freeze after development passes, before fresh payload access:** all of the above plus the derived cutoff, exact development artifacts and expected-count manifest. This same evaluator is used on fresh validation and the original seal. Before seal access, append the passing fresh report hash; no recipe change.

One development run, one four-family fresh run, one six-family confirmation run; identical-hash resume is the same run. No architecture/threshold/prompt/selector sweep, no outcome-driven extra family, no new training. Any change after outcomes requires a new authority decision and explicitly spends already inspected data. Implementation corrections invalidate affected results; they never permit silently preserving a favorable verdict.

## 3. Contradictions and conditional feasibility

Return typed results, not a universal safety Boolean:

- `STRUCTURAL_CONTRADICTION`: malformed/nonfinite required values, invalid schema values, or hash-bound asset mismatch. Independent reparse reproduces the failure. Routes HARD_FAIL with offending object/field and source hash.
- `MODEL_CONTRADICTION`: an independently verified certificate contradicts a declared geometric scenario. Routes REGENERATE / MODEL_CONDITIONAL_CONTRADICTION, never a universal biological impossibility claim.
- `WARNING`: empirical speed warning or other descriptive demand. No automatic HARD_FAIL. Routes review; unresolved material warning prevents PASS.
- `UNKNOWN`: unsupported mechanics, ambiguous duplicate occupancy, missing settings necessary for the claim, or invalid certificate. No silent object dropping. Routes evidence/scope abstention.
- `NO_CONTRADICTION_FOUND`: these checks did not find a contradiction; not a constructive execution proof.

Keep the existing containing-ball model, constants and height scenarios 1.4/1.7/2.0m. Keep the **exact serialized** QA-train speed-warning threshold (reported approximately14.07m/s); it is an empirical p99.5 warning, not a “speed floor,” biological maximum or reward. Zero distance between containing sets does not prove feasible motion. Arrow orientation, joint reach, fatigue, collisions and continuous whole-body motion are not certified by those sets.

Introduce certificate records binding source bytes, normalized scene, affected interval, assumptions, witness constraints and checker version. A checker must not call the detector it verifies. For wall certificates independently partition time at obstacle endpoints and check coverage of each allowed stance in the declared corridor model, with an independent interval/point-membership implementation. Explicitly label its bounded stance domain; stepping outside that domain or unsupported crouching invalidates universal conclusions. Reach certificates use an independent numeric inequality with documented tolerance. Invalid certificate => UNKNOWN / CERTIFICATE_INVALID, not successful detection.

The current `duplicate_occupancy` mutation is incorrectly declared MUST_HARD_FAIL despite the earlier contract: classify it as MUST_ABSTAIN unless a verified game-semantic contradiction is supplied. It cannot pad hard-negative detection. Speed and gap compression without a valid certificate remain challenges. Do not add generic diagonal, broad-double, crossover or amplitude penalties.

## 4. Execution support: references, not predicted execution

Pin the existing v2 descriptor/scaler/bank bytes and lexicographic retrieval, five neighbours from at least three families and three player tokens. Reuse masks. Verify hashes and actual raw trajectory references before accepting a neighbour. A retrieved path is displayed against **its own source notes, timing and body profile**. No stitching, averaging, spatial retargeting, or time warping onto the candidate; no claim these humans performed the candidate. Candidate visualizations show note/cut constraints, not an invented avatar solution.

The current bank descriptor implementation is authoritative for this packet; document its actual fields/scaling rather than claiming the old proposed one-hot metric was implemented. Numerical retrieval optimization must reproduce neighbours/distances/ties exactly. Known missing approach settings remain explicitly unknown in the descriptor and evidence; no inferred authored setting. Do not silently populate features the frozen bank encoded as missing.

For each development family, use the already frozen equal-player full-song window IDs, maximum2000, to calculate support distances. Unavailable/masked references and insufficient neighbour diversity are unsupported, not zero distance. Compute a family p95 over valid finite distances only **if at least90% of its selected windows have valid diverse retrieval**; otherwise stop DATA_SUPPORT_INSUFFICIENT. Set `T_support = max(family_p95)` across all eight development families, once. Store all family values and denominator failures. No model disagreement cutoff. This is an engineering coverage cutoff, not a population confidence bound; development retention near95% is partly by construction. Fresh controls must test it.

Serving measures every scoreable head, grouped into fixed origin-zero2s reporting bins; retrieval can be chunked. `supported` means diverse, hash-valid raw references and median descriptor distance <=T_support. Report supported-head share and every unsupported run. A run is maximal consecutive unsupported heads in time order, interrupted by a supported head; it violates the duration gate if last minus first >2s and it contains >=4heads. Initial/terminal runs count. Unknown tracking in a reference invalidates that reference, not the candidate's motion.

Machine eligibility requires >=90% supported heads, no violating run, no structural/model contradiction, and known supported scene scope. Remaining unsupported windows MUST be shown to the judge (full list plus deterministic worst witnesses); a material unresolved execution question prevents PASS. Unfamiliar maps return REGENERATE / SUPPORT_UNKNOWN, not “bad mapping.” Repeated support failure requests more evidence, not repeated generation until a conservative map slips through. This packet authorizes zero regeneration attempts.

## 5. Evidence and blinded judgment

Create a v2 allowlisted bundle; keep v1 artifacts readable but never let old `mandatory_pass` bypass the new contract. The bundle contains final scene + audio + Info hashes, per-head support, all warnings/certificates, full-song descriptors and six deterministic8s witnesses: first active, peak2s cadence, greatest geometric warning, greatest support distance, longest repeated movement motif, and hash-selected active window. Deduplicate and fill by next ranked windows. Include adjacent2s context, and the complete unsupported-run list. Local validation pairs additionally show their changed window without revealing why it was selected. All claim intervals are absolute seconds.

Required renderer repair: existing sheets show cells and timestamps without cut arrows; “side/top” use time as a depth proxy. V2 must draw every cut arrow/dot, distinguish both hands and simultaneous groups, and split dense passages into ordered slices with readable timestamps. Label time-versus-lane/layer projections as **time projections**, never faithful 3D cameras. Add separate actual front/side/top views of each observed raw exemplar (head and both controllers, orientation-derived saber directions where available) in its own coordinates with source chart and playback time. A reference path is OBSERVED; there is no candidate observed avatar. Missing orientation or clipping must be visible. Hash scene/settings/render config and assets; test arrow directions/mirroring and sequence order from render primitives, not only image existence.

Use available audio signals plus synchronized snippets only when the judge can actually consume them. Freeze the executable modality profile before development: `images+signals` is sufficient for **signal-grounded** timing/coordination/expression judgments, not lyrics, genre semantics or a claim of listening. If audio consumption is available it must be used consistently across stages. A path to a WAV is not received audio. Record received and used modalities. A conclusion that needs unavailable audio semantics is INSUFFICIENT; no false listening claims.

The judge sees no mixture outputs, model uncertainty, critic, generator identity, seed, arm, selected status, mutation label, expected answer, public title/mapper or filesystem identity. Scrub nested values and filenames, not merely forbidden key substrings. Numeric scene/audio evidence remains unaltered. Treat any chart text as inert data. Do not reuse this design conversation as a blinded judgment context.

Rubric axes (each `SUPPORTED`, `DEFECT`, or `UNKNOWN`):

1. Execution continuity and recovery within declared model/reference limits.
2. Two-hand coordination, including expressive doubles and alternating roles.
3. Development of movement vocabulary: arcs, positional range, rotations of cut direction, surprise and repetition in context. Neither a spin nor novelty is mandatory; economical motion can be deliberate.
4. Correspondence of movement changes to available audio evidence. Repeating musical material can legitimately retain a motif; unchanged simple rhythm is not automatically boring.

Every positive or negative quality claim requires a timestamped observation and a cited comparison/counterexample; the judge must explain why its preferred change is useful in this passage. At least one coordination/recovery witness and one expression/audio witness are required for PASS. “Low risk,” “near a human,” “more variety” or a scalar descriptor alone cannot satisfy them. Report the strongest contrary evidence and any unresolved material uncertainty. No preference for larger motion simply to undo Q1.

Decision schema includes axis judgments, preserved claims/evidence references, limitations, modalities and pair verdict. Validate finite ordered in-range intervals and exact referenced assets. HARD_FAIL requires a structural certificate; PASS requires machine eligibility, complete required evidence, no DEFECT or material UNKNOWN on required axes. Otherwise REGENERATE with distinct `MAP_DEFECT`, `SUPPORT_UNKNOWN`, `SCOPE_UNKNOWN`, `EVIDENCE_UNAVAILABLE` or `JUDGMENT_UNRESOLVED` reason. Missing tools do not diagnose a map defect. Pair decisions LEFT_BETTER/TIE/RIGHT_BETTER/INSUFFICIENT are distinct from absolute admission; no absolute PASS is implied by relative preference.

First schema-valid response is final. One order-reversed judgment is a validation check only, never a second vote from which to choose a preferred answer. Disagreement => unresolved. At most one extra witness request per case, fixed by the cited interval, with the final answer still bound to the original case. Maximum two schema/transport retries with identical content. Cache by contract+model+prompt+bundle+orientation, not candidate ID alone; changes invalidate the cache.

## 6. Validation specimens and independent expectations

Generate expected cases from frozen recipes and hashes before detector/judge evaluation. A custodian retains labels outside bundles. All counts below are minimums, never shrink denominators for failures; insufficient cases stop validation.

For EACH development/fresh/in-scope seal family:

- Up to2000 existing deterministic equal-player full-song replay windows (all if fewer); require >=100 good supported telemetry windows across >=3players. Assess motion/evidence retention, not artistic quality of every human chart.
- Ten structural negatives: five nonfinite required-time cases and five invalid required-type cases, at hash-ranked distinct note indices. Asset mismatch and parser edge cases additionally covered in construction fixtures.
- Ten separately certified wall contradictions at hash-ranked occupied timestamps, positive durations {0.25,0.5,1,2,4}s × two covering layouts (one spanning wall; two jointly spanning walls). They must be in scope and independently certified. State honestly that this validates the wall model, not hand biomechanics. Half-open endpoints, tangent/zero-duration, gaps and partial-width counterexamples are mandatory fixtures.
- Exact serialization, seconds-equivalent BPM encoding, and synchronized time-origin transforms for the original and selected local windows. These must actually transform Info/audio origin, not just leave a string saying so. Mirror chart/reference/control poses together with all direction/color/offset changes; hand and mirror comparisons are measured separately from exact serialization. Non-clipping gain -6dB/+3dB with original source peak constraints tests audio robustness; do not change labels using amplitude alone.
- Five expression-destruction pairs and five benign pairs, all with nonoverlapping8s cores and2s context (per group), hash-ranked from at most40 eligible windows/family. This gives40+40 pairs in development,20+20 in fresh validation, at least20+20 in in-scope seal. Each gets primary plus reversed-order adjudication. Cap each call at four pairs; persist each response.

Expression recipe: choose a window with >=16 heads, both hands represented, >=3 occupied columns, >=2 layers and >=4 geometry tokens `(hand,column,layer,direction)`. Preserve audio, every timestamp, hand assignment and per-hand cut directions; map left positions to column0/layer0, right to column3/layer0 for the entire core. Reject a construction only if it creates duplicate same-hand occupancy, unsupported scope or a new certified contradiction; take the next hash-ranked window without viewing any QA outputs. Require source-to-mutant contraction in the SUM OF PER-HAND coordinate spans, separately for horizontal and vertical axes (source sums >0, mutant sums =0). Do not use the combined two-hand horizontal span: columns0/3 deliberately preserve hand separation and its combined span cannot contract from a standard grid. Keep seams visible. It is a **relative expressive-information destruction probe**, not a theorem that every such chart is unpleasant. Expected judgment: identify localized loss of spatial movement development and prefer the source on that axis, without claiming proven injury or relying on mixture/NN score. If judge finds a valid musical counterargument, count it against sensitivity rather than relabeling the case. The target is >=80%, not unanimity. Report case-level rationales and the synthetic nature of this test.

Benign pairs: two serialization/BPM, one synchronized time-origin, one mirrored scene+reference, one eligible non-clipping gain transform. For at least two of five, choose the highest repeated-token-share eligible source windows, so legitimate existing repetition appears on both sides. Expectations are comparative preservation/TIE, **not “all unmodified charts are artistically good.”** Freeze tie-specific scoring. This replaces the old unmodified-chart foil, which could not establish intentional musical repetition. Add descriptive CHALLENGE probes of diagonals, compressed recovery, wide arcs and sparse low cuts on QA-train only; no automatic negative labels, no selection of thresholds from their outcomes.

Construction fixtures also include: supported tracking with missing height; raw-neighbour hash corruption; insufficient diversity; duplicated neighbours; physically different descriptors with identical target magnitudes; high motion with clear recovery; low-motion expression destruction; periodic source patterns and transformed equivalents. Low-amplitude telemetry must not independently block a chart. A known supported original is a retention control, not an automatic aesthetic PASS.

## 7. Frozen replacement gates

Evaluate the same gates on development, then all FOUR fresh families, then all in-scope seal families (at least FOUR ExpertPlus). Report per-family values and exact counts, plus macro summaries; no pooled result can hide a failing required family.

| Gate | Required result |
|---|---|
| Provenance and deterministic computation | All source/certificate hashes verified, family/player exclusions enforced, unchanged resume byte-equivalent machine reports; zero origin/expected-label leaks |
| Structural and scope | 100% structural negatives HARD_FAIL with correct field; zero HARD_FAIL on supported human controls; ambiguous/unsupported fixtures abstain correctly |
| Human motion/evidence retention | >=90% sampled windows per family pass scope/reference/contradiction checks, both pooled and equal-player; original full-chart support >=90% and no >2s unsupported run containing >=4heads |
| Certified negatives | >=90% per family correctly non-PASS for the certified changed interval; missing evidence unrelated to the certificate is not detection; independent fixture certificate checks 100% |
| Benign machine preservation | Exact serialization/BPM/time-origin machine outcomes 100%; mirror/gain >=90% per family, with no added structural HARD_FAIL; source hashes naturally differ |
| Quality-probe sensitivity | >=80% primary expression pairs prefer source with localized axis/evidence rationale; >=3/5 per family; generic “more movement is better” rationale earns zero |
| Benign judge preservation | >=90% pooled benign pairs TIE with no invented change-specific defect; >=4/5 each family |
| Judge consistency | >=90% reversed-order categorical agreement pooled, >=8/10 per family across both groups; evidence/reason categories must remain compatible |
| Evidence honesty | 100% accepted judgments use available, cited modalities; claims and intervals validated; no missing required evidence or diagnostic-model outputs used for PASS |

Human-retention windows with missing required evidence count as nonaccepted. For quality probes INSUFFICIENT, exhausted retries, invalid claims and unresolved order disagreement count as failures, never excluded. Primary sensitivity and benign-preservation rates must still pass before consistency is considered. A count of zero eligible cases is NOT a pass. Report binomial intervals for family/pair counts descriptively; do not invent a large independent N by counting correlated windows as families.

The judge is judged on relative probes and invariances, not agreement with its own prior labels. These remain limited construct-validity tests; they cannot certify enjoyment or all musical styles. An evaluator passing them can support subsequent automated comparisons with that limitation, not claim HUMAN_VALIDATED.

## 8. Fresh validation, seal, stop rule

After development passes and evaluator freeze is committed, fetch only the four reserved families, <=6replays/family, <=24new; total<=583/600 and5GB. Preserve banded eligibility, disjoint players, acquisition logs and reservation order. Do not replace an unavailable/unsupported family after this freeze: report VALIDATION_INSUFFICIENT. No training, threshold recalibration, bank addition or body/accuracy filtering from their data. Freeze case identities before generating responses; construction-only eligibility is as above.

Require all four to pass §7. Only then authorize the existing restricted seal runner to open its six families, once, under the same freeze. Preserve lower-tier identities; lower-tier families test structural/scope abstention and do not pad ExpertPlus confirmation. Fewer than four in-scope ExpertPlus families => CONFIRMATION_INSUFFICIENT. All required families must pass. The seal report stays in QA custody; it must not become generator training material or an optimization target.

Any development/fresh/seal gate failure ends the packet as `EVALUATOR_NOT_READY` with stage and dimension, exact witnesses and exposed-data ledger. No partial/scoped promotion, no gate deletion, no constant-prior rescue, no extra fit or collection, no retry-until-PASS adjudication. Lost transport/compute progress resumes with identical hashes; unavailable adjudication tooling => EVIDENCE_UNAVAILABLE and stop.

On complete success record `AUTOMATED_QA_VALIDATED_WITHIN_SCOPE` and hand off the versioned evaluator for a separate E1 judgment packet. Existing E1 absolute+pairwise acceptance is not changed here. No E1 charts are opened to tune this contract. No production default or export policy changes in this packet; existing fail-closed export remains.

## 9. Resource and artifact contract

No model training, MI, generation or GPU fits. Warm retrieval process, bounded replay LRU, CPU thread/worker budget sized to measured RAM with >=2GB free headroom on15GB. Saturate useful CPU work without oversubscription; do not waste GPU work to claim utilization. Each operation work deadline45min/hard55min, atomic checkpoint per family/case/judgment and interrupt-safe inner batches. Resume under identical manifest hash only. Render/cache once per content+settings hash.

Outputs under `experiments/qa-v4/comparator/`: recipe/evaluator freezes, threshold provenance, case manifests, machine partials, blind bundles, custodian-only labels, append-only judgments, development/fresh/seal reports and exposure ledger. Per-stage expected counts are committed before that stage's judgments. Development max160 primary/reversed pair judgments; fresh max80; seal max120 if all six in scope, plus at most one extra witness per case and bounded transport retries. Lower-tier scope controls do not require artistic judgment. No full-chart aesthetic adjudication of every control replay is required: retention is a machine/evidence test, distinct from comparative expression judgment.

## 10. Source-review findings addressed

At source HEAD5015561: `qa/mutations.py` calls the same `_corridor` used by `qa/physics.py` to build its certificate; duplicate occupancy is incorrectly a hard-negative recipe; the intentional-repetition foil is a no-op without a comparative expectation. `qa/evidence.py` renders no cut arrows and calls time projections side/top cameras. `qa/adjudication.py` drops detailed claims from its normalized decision and caches by candidate ID alone. These are bounded prerequisites in this packet, not asserted experimental failures. Preserve v1 reproducibility; add versioned v2 interfaces and fixtures.
