# Ordered audio retrieval under deployment timing

## Evidence and bounded intervention

The original phrase retrieval descriptor contains only frame means/stds plus
requested workload. Reordering frames within a phrase leaves it unchanged. The
saved synthetic diagnostic has identical descriptors but fixed-note mean onset
alignment1 versus0 after swapping phrase halves. It cannot distinguish activity
order even when that order is decisive. Fix this representational limitation in
one no-fit experiment, without changing join logic, geometry, candidate budget or QA.
The v4/v6 learned refiners remain negative and are not part of this intervention.

## Serving and train-only bank

Use the same eight opened supported development families, all12 inventory identities,
316 fit-train families and frozen literal phrase bank. Serving source BPM comes from
cached audio-only MI; duration comes from the original audio file. B0 TimeGrid walls
and its hand-event rate are serving-available conditions. No human-reference values
enter retrieval. Original audio time0 is retained; no snapping or phase correction.
Full TimeGrid is used for B0 walls; the retrieved output has constant MI-BPM units.

Keep the original standardized55-dimensional mean/std/rate descriptor unchanged.
Add16 ordered half-beat bins, each containing mean normalized onset and RMS features
(the final two columns of the existing27-dimensional audio cache). Bins cover a fixed
eight-beat phrase. Past the real tail, leave bins at0 rather than stretch its time.
Fit per-feature mean/std on exactly the existing fit-train phrase entries, floor1e-6.
Search squared distance = mean squared old-feature distance + mean squared ordered
feature distance. Equal group weight is fixed now; no coefficient or bin-size sweep.
A query's temporal descriptor is derived only from audio and serving BPM.

Reuse stable nearest32 shortlist, existing entry/exit compatibility, no immediate
identical donor repeat, first6 compatible donor choices and seed modulo indexing.
Copy full literal donor phrases and truncate only at the actual song tail as before.
No neural refiner, generated-geometry repair, reference reranking or additional fit.
Retain donor provenance and repeated-donor counts.

## Matched comparison and measurement

Arms: original bag-of-frames retrieval versus ordered retrieval, both under identical
MI-derived serving conditions. Six seeds0..5 each; first machine-admitted, otherwise
cached B0. B0 is also reported separately. Reuse original control QA only when actual
exported chart AND Info bytes are identical to the frozen earlier artifact; otherwise
recompute full-scene QA. All candidate profiles are recomputed from actual exports.

Use `joint_deployment.reference_view` only after generation so every profile has the
same authored eight-beat physical-time boundaries as its reference and B0. QA reads
actual serving-BPM files. Keep original train-IQR scales and missing-opportunity counts.
Report audio onset/activity descriptors, admitted/fallback coverage, donor repetition,
physical export identity, latency, all seeds and all12 inventory rows.

Candidate positive rule retains the whole-generator numeric requirements: at least
6/8 nonfallback ordered outputs; rhythm AND geometry error at least10% lower than
B0 and the deployment retrieval control; each component <=1.1*comparator+0.01;
no aggregate supported-opportunity loss on either axis against either; no new scoped
contradictions or per-family supported-share drop>0.05 against either. No learned-
checkpoint prerequisite applies to a no-fit retriever. Family-bootstrap differences
are descriptive. Status is development-positive/negative; release eligibility false.
This does not retroactively alter any previous negative result or establish perfection.

Freeze this spec, code, parent/input-audit identities, audio/MI/B0 inputs and old bank
receipt before generation. Freeze the new bank checksum before first development
attempt. Namespace: `experiments/joint-phrase-v7/ordered-retrieval/`. New outcomes do
not authorize fresh confirmation unless the full predeclared development rule passes.

## Verification and continuation

Synthetic checks must show old descriptor invariance to frame reorder, new descriptor
sensitivity and matching ordered donor preference. Test missing/tail bins, train-only
bank membership, physical-time export scoring and all-rejected fallback. Verify every
saved output/QA binding and repeated decision. Preserve all attempted candidates.
45-minute work/55-minute hard bounds, >=2GB memory headroom, no concurrent QA-cache
writers. No dependencies, playtests, annotations, judge batches or push.

If negative, locate the remaining rhythm/geometry/audio limitation from these saved
outputs before another mechanism. No seed/weight/fit sweeps and no gate relaxation.
If positive, freeze the candidate before auditing fresh-family exposure and promotion
requirements. In either case continue the broader task rather than stopping here.
