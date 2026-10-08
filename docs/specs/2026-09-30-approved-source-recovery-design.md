# Recover approved sources through independently verified local counterparts

The original readiness inventory has88 approved sources, but53 are excluded before
full scene validation because an approved copy named ExpertPlus.dat is not bound by
its Info's sole Standard ExpertPlus entry naming missing ExpertPlusStandard.dat.
All53 pass the raw supported-schema check.27 have local same-family original files
with exact chart-byte hashes and identical parsed Info metadata;13 have identical
audio bytes. Some other approved audios are resampled mono PCM in an .egg filename.

Do NOT rename/stage the unknown source just to bypass exact binding, infer settings
from a difficulty label alone, edit source files, or weaken shared readers. Instead,
search existing corpus members of the SAME frozen family for a chart already bound
by its real sibling Info. Require exact approved chart-byte identity, full parsed
Info equality with the approved copy, verified Standard ExpertPlus/rank9 settings,
and original source hashes. Preserve approved provenance through the original
approved record and these equality proofs, not the new path's name.

Audio must either have identical bytes or corroborated zero-lag physical alignment:
original versus approved first90-second RMS fingerprints at the existing50Hz recipe,
Pearson≥the existing0.95 threshold WITHOUT shifting, and duration difference no larger
than the sum of their native sample periods. Keep the approved audio file itself,
with its original hash and physical timing, even when a resampled representation
passes. This corroborates a transcode; it does not declare decoded PCM bit-identical.
Ambiguous/missing/mismatched counterparts remain unresolved, never guessed.

Create a new composite source record pointing to the independently bound original
chart+Info and the unchanged approved audio. Both chart contents and parsed Info
must equal the approved copy. Then run the UNCHANGED read_source, full event codec
and actual native-NJS/offset export/readback. A binding recovery can still fail other
strict geometry/timing/mechanics checks; report those honestly without repairs.
Save provenance, supported source JSONs, actual exports, hashes and all exclusions
under `experiments/joint-approved-source-recovery-v1/`. Freeze all candidate input
identities, code/spec and original corpus/readiness before outcomes.

No model fitting, generation comparison, source mutation, QA-cache writes or fresh
quality outcomes. Do not automatically assign recovered sources to fit roles: their
original corpus train membership is known, but the future joint train/validation
assignment and cross-role near-copy screening need a separate frozen inventory.
Preserve existing audio quarantine and every historical negative result. Continue
with that source/audio integrity work before another generator packet.
