# Joint model, retrieval and emitted-output pilot

Prerequisites: readiness `NEW_DECODER_REQUIRED`, all 356 exports round-trip.

## Question and fixed scope

Can one small audio-conditioned joint-event model improve emitted phrase rhythm
and geometry over B0 and a train-only human-phrase retrieval challenger?
A negative result ends this pilot branch honestly. It does not finish the larger
perfect-flow aspiration. No production replacement or fresh confirmation on failure.

Use all 348 supported training families, split deterministically: first 32 by
SHA256(`joint-model-val-v1:` + family) become fit validation, remaining 316 fit train.
No development family is fit/validation. All 12 frozen development IDs remain in
reports; four unsupported source families abstain, eight receive generated paired
comparisons. Results are in-scope eight-family development evidence, not fresh-test
confirmation. No new holdout access or substitution.

## Representation and audio

Reuse literal events and exported scene units. Insert a REST boundary token at
successive eight-beat boundaries and audio end, after events strictly before that
boundary. A boundary event at beat eight is emitted after the REST to eight.
REST advances time and recurrent state; it does not reset either hand's history.

At each prediction, predict REST or a next-event gap. Gap bases in beats are
0, 1/48, 1/32, 1/24, 1/16, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 2/3,
3/4, 1, 1.5, 2, 3, 4, 8. Choose nearest log-space positive base for training and
store residual log(actual_gap/base). Zero-gap is only legal at a boundary/BOS,
not after an emitted event. Inference positive residual clamps to [-16, 8]; a
non-increasing float time or predicted event outside the current window rejects
that attempt, never silently moves a note or converts it to REST. Exact source
codec timing survives as float64; tensor targets use float32 and disclose precision.

Audio features: existing installed librosa STFT, 13 MFCC, 12 chroma, onset strength,
RMS (27 channels), sample rate 22050, hop512. Normalize each channel within its
song, deviation floor 1e-6. Sample at 33 positions from cursor-8 to cursor+8 beats
in half-beat increments; zero outside audio. This is audio-only lookahead, no human
chart targets. Cache per original audio hash and feature recipe. No foundation
model, genre/title feature, or new dependency. Context includes beat/bar phase,
seconds per beat, requested swing rate and NJS, plus 12-cell current wall/bomb
occupancy. Fit-time workload is source mean hand events/second; pilot serve-time
request is B0 mean hand events/second. Report this conditioning distribution shift.

## Model and one fit

One model from scratch, width128 two-layer GRU, input projection128. Inputs include
sampled audio/context and preceding action's six literal-slot embeddings (16 dims
each, empty token108), previous action type and gap. Previous emitted events update
inputs during rollout. The two-hand recurrent state persists for the whole song.

Heads: gap-category logits; one scalar residual per gap category; joint hand-count
logits16 conditioned on sampled/teacher gap; six sequential literal-slot logits108
conditioned on hidden state, joint count, gap, slot position and earlier slot tokens.
Use the existing joint sampler's occupancy/order mask at decode, not the production
parity/cell/axis penalties. Input slots are canonical column/layer ordered.

Loss: gap CE + residual smooth-L1 on positive event gaps + count CE on event actions
+ mean active-slot CE. REST carries only gap CE. Teacher conditioning in training;
free running in evaluation. No risk, entropy or synthetic quality reward.

Seed20260930; AdamW lr3e-4 weight_decay0.01; batch16, sequence128 with initial32
context steps excluded from loss; uniform fit family then uniform crop. No special
approved oversampling in this pilot; report approval provenance separately. One fit
6000 updates maximum, snapshots0/1000/3000/6000. Gradient norm cap1, finite-loss
checks. Persist optimizer, Python/NumPy/Torch CPU/CUDA RNG and update every100 and
at deadline. Continue the same fit on resume; no extra budget.

## Retrieval challenger

Bank = complete eight-beat phrases from fit-train families only, paired with their
own audio feature context and normalized event rate. Retrieve by Euclidean distance
of mean/std audio27 plus requested rate standardized on bank training entries.
Rank deterministically. Exclude already-used exact donor windows from immediate
reuse, preserve a source ledger, report repetition. Adapt donor time by phrase beat
translation only; preserve both hands' joint geometry. Use up to 32 nearest choices
to find a join with no duplicate occupancy or immediate (<1sec) same-family parity
violation on unambiguous boundary heads. This join predicate is a conservative
retrieval heuristic, not a universal hard decoder rule. If no compatible choice,
mark full-song attempt failed; never pad an empty phrase and call it success.
Tail: crop a donor to target remaining beats and preserve context. Candidate seed
chooses among first six compatible donors; no extra search beyond fixed top32.

## Environment, baseline and artifacts

In generation, hold each song's cached B0 bombs/walls fixed across B0, model and
retrieval; model inputs condition on them. Do not copy development human obstacles
or notes as generation inputs. Human environments remain in training and codec
round-trip evidence. This packet does not learn wall/bomb generation.
Use authored constant BPM/audio origin for the new generator. Convert cached B0
notes/walls from their own TimeGrid to absolute audio seconds then into that BPM;
no silent lead shift. Preserve B0 artifacts and record transform/source hashes.
Export all attempted candidates with complete Info/scenes, measure actual files.

## Output evaluation, frozen before fitting

Never label these metrics fun or proven comfort. Compare paired outputs against
same-song human references as one observed mapping, not the sole correct map.

1. Rhythm: per eight-beat window occupied-event rate, each hand's event rate,
   double-event share and empty-window indicator. Mean absolute errors per feature,
   divide by fit-train IQR (floor0.05), then mean. Report every component separately.
2. Geometry: per-window low-vertical share, arc-run frequency, double-token entropy,
   max 4-gram share and mean hand displacement. Use existing expression-profile
   definitions, null opportunities omitted with support counts. Same train-IQR
   scaling and mean error; closeness, not universal increase/decrease.
3. Audio: mean onset strength sampled at event times and correlation of per-window
   activity with audio RMS; compare to human descriptively. Silence, time-shift and
   shuffled-geometry controls must expose the expected change in their relevant
   axis; a low-vertical human control compared with itself must score zero.
4. Machine: existing comparator-v2 on the full exported scene, explicit unknowns,
   contradictions, support share, plus event schema/occupancy and actual density.
   Empty charts are failures. Support unknown never counts as an admitted winner.
5. Report repetition, maximal gap/cadence, worst windows, runtime and fallbacks.

Validate pure arithmetic/invariances/control fixtures before fitting. Do not build
another learned taste evaluator. These measures support a narrow paired-reference
claim; even complete passes do not establish subjective enjoyment.

Snapshot selection: two fixed seeds (0,1; temp1) per each of 32 fit-validation
families, full rollouts. Rank by failure fraction first, then average rhythm+geometry
error; earliest update on ties. If step0 wins, no learned candidate. No development
output influences fitting/checkpoint choice. Evaluation resumable per candidate.

Final development: six seeds0..5, temperatures[0.85,1,1.15,0.85,1,1.15] per arm per
supported song. Choose the first machine-admitted candidate by seed, avoiding
reference-dependent serving ranking. Failed/unknown attempts fall back to B0 with
zero improvement; retain every failure. Matched candidate budgets disclosed.

Pilot positive gate: selected learned outputs must achieve at least10% lower
family-mean rhythm error AND geometry error than each comparator (B0 and retrieval),
with at least6/8 nonfallback families, no new structural/model contradictions and
no per-family support-share drop>0.05 vs B0. Each component mean may regress at most
10% (absolute slack0.01 standardized units) to prevent aggregate hiding. All 12
panel identities remain disclosed; four unsupported families are out of scope.
Report family-bootstrap intervals descriptively; this small pilot is not promotion.
One negative packet stops this branch; no threshold retuning, extra fits or new seeds.
If positive, preregister fresh confirmation before opening untouched families.

## Execution bounds

45-minute work deadline,55-minute hard deadline per process; atomic per-song and
per-candidate artifacts, >=2GB host RAM headroom, serial audio preprocessing cache
writes. No push, playtests or judge batches. Protect production weights and policy.
