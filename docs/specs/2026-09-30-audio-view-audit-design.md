# Paired audio-view invariance diagnostic

Metadata census:23/24 original approved fit audios are14800Hz mono, whereas all292
original general fit audios are44100/48000/96000Hz stereo and all8 development audios
are44100/48000Hz stereo. The existing frontend resamples to22050Hz; that does not
restore spectral content missing from low-rate source files. Determine whether this
observed provenance/format imbalance materially changes retrieval descriptors.

Use only the10 recovered TRAIN sources whose approved14800Hz audio was independently
corroborated as a zero-lag transcode of a higher-rate local canonical original. The
five reserved validation groups stay unopened. Freeze identities before measurements.
For each pair, compare queries from (a) the unchanged full-band canonical original,
(b) a deterministic mono14800Hz rendering of that SAME original, then passed through
the unchanged22050Hz audio_features recipe, against the bank's approved source view.
Write the common-band render only under the new audit namespace as float WAV; never
modify originals. Reuse the frozen frontend for extraction, not a reimplemented MFCC.

Use actual approved45 bank entries, fixed v12 moments and native eight-beat source
windows ending within first90 seconds (the corroborated interval). Query workload
is each donor's actual hand-events/beat for this self-retrieval diagnostic only;
this is not a deployable target-derived input. Confirm the existing approved cached
view exactly reconstructs its bank row, then measure source-view descriptor distance,
its own entry's best/worst tied raw ranks and top-six membership for both canonical
views. No generated charts, QA writes, model fits, altered bank, quality labels,
threshold tuning or candidate selection. Aggregate by source as well as by window.

Record source-rate census, original/canonical-render hashes, feature receipts,
normalization/bank identities, every paired result and runtime. A reduction in paired
descriptor error establishes improved view consistency only; it does not prove
better target-song mapping. If material, plan a uniform frontend for both donor and
query extraction as a separate matched experiment. Do not deploy a query-only patch
against a mixed bank or retroactively change v14's running comparison.

Namespace `experiments/joint-audio-view-audit-v1/`. Test tied-rank arithmetic and a
float-WAV render's sample rate/channels/duration bound. Keep earlier reports and
source/quarantine/role identities immutable. Continue from measured evidence.
