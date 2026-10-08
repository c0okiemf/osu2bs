# Cross-role near-duplicate audio audit

A v10 continuation outlier maps development fam:24183 almost entirely from training
fam:24227. Same artist/base song metadata, different TV Size/2022 Remaster suffix;
cached RMS correlation0.995428 at0.36 seconds indicates a likely same-recording
variant. Existing exact-title grouping misses the suffix, and its half-second
fingerprint lag grid misses this alignment. Verify identity before more generation.

Use only the356 already prepared joint-pilot identities (316 fit train,32 validation,
8 opened development). Verify data receipts and original audio/Info hashes. Load
cached raw RMS from each hashed prepared file, resample its first90 seconds at50Hz,
and calculate maximum overlap-normalized Pearson correlation at every20ms lag within
±30s. Require at least30s overlap and nonzero variance. Compare EVERY cross-role
pair, regardless of title, using the existing0.95 correlation screening threshold.
No threshold tuning based on how many pairs are found; retain all scores/peak lags.
This screens for near copies, not remixes/cover versions or every possible leak.

Independently decode each flagged pair's original audio files to verify with a
separate RMS envelope and fine-lag normalized correlation. Record full exact
metadata, audio hashes, role/family and alignments. Identity-level audio duplicates
may join without equal titles; constant/quiet tracks or accidental envelope matches
remain candidates requiring recorded corroboration. Conservative exclusion of a
suspect training donor from a FUTURE bank is allowed, but do not alter old manifests,
reports, models, gates or outputs. No chart-quality inference from correlation.

Freeze input/recipe/code/spec identities before scanning. Keep the suspected pair
in a regression fixture with exact finer-lag arithmetic, plus unrelated and constant
controls. Record connected candidate groups across roles and implications for prior
song-disjointness claims. No new generation, fit, fresh outcomes, labels or QA writes.
If overlaps are confirmed, create an experiment-owned quarantine receipt bound to
this evidence before continuing model/retrieval work. A filtered retrieval bank does
not erase overlap learned by old checkpoints or B0; disclose that distinction.
