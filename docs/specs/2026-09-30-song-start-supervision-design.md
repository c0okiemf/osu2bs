# Song-start supervision experiment

Earlier experiments remain sealed.

## Evidence and question

Legal timing conditioning changes validation completion from0/256 to256/256 without
fitting. Development still has clustered support failures, often near song starts.
In the first24 completed attempts, unsupported-head shares were35.87% in the first
32 generated actions versus2.23% later. This interim association is diagnostic, not
proof that retraining will solve admission or quality.

A separate code fact is conclusive:128-action random crops always excluded their
first32 actions from loss. No source action with global index0..31 can ever appear
after local index31 in such a crop. Thus every training chart's first32 prediction
targets were always unsupervised. GRU warmup gradients do not substitute for direct
prediction supervision at the song start used during serving.

Question: does explicitly supervising real starts improve emitted-song support and
paired quality, retaining the same model architecture, data and legal decoder?

## Single changed training mechanism

Reuse316 fit families,32 validation families, eight supported opened development
families; retain all12 development identities. Use existing authenticated prepared
data, two-layer128-width JointModel, original loss coefficients and initialization
seed20260930.
For each batch16, four crops start at actual source action0. Twelve choose family
uniformly and start uniformly among valid128-action crops, as before. All families
are chosen uniformly in both groups. A random crop that happens to start at0 is
also a real start. No reset/pseudo-BOS is introduced midway through a song.

The loss mask includes every action when crop start==0; otherwise its first32
context actions are excluded. Compute gap CE, residual smooth-L1, count CE and
active-slot CE across the valid positions with the same component coefficients1.
No approved oversampling, architecture/head change, geometry policy or reward.
Log actual BOS crop counts and supervised BOS target steps. Batches remain
16×128; the original32-action context warmup remains for non-BOS crops.

One fresh6000-update fit, AdamW3e-4/weight_decay.01, gradient cap1, snapshots
0/1000/3000/6000. Save optimizer, all RNG states, update, counters and history every100
and at work deadline. Resume the same fit; no seed or learning-rate sweep.
Use legal timing support from the no-fit experiment for all generated candidates.

## Frozen comparisons and limits

Same validation32×two seeds0/1/temp1 for each snapshot; rank by failure count,
mean rhythm+geometry error, then earliest checkpoint. No development-based choice.
Same six model attempts per development family with temperatures
[.85,1,1.15,.85,1,1.15], first machine-admitted selection, B0 fallback on failure.
Reuse authenticated B0/retrieval from v1 and report v2 legal-timing results alongside
v3. Freeze separate identity at `experiments/joint-phrase-v3/start-supervision/`.
Bind input caches, checkpoints for old comparisons, code, this spec and comparator
records. Preserve original/source/production hashes.

Use exactly the same paired metrics, train scales, missing-opportunity protections,
full-scene comparator-v2 and positive gate. Diagnose prefix versus later support,
without using this diagnostic to select serving candidates or relax admission.
A positive result supports this conditional pilot only; fresh confirmation remains
required.

The current authored-BPM input is retained to isolate the training change. The
separate tempo-input audit finds five exact matches to the production audio pipeline
and three differences below0.34% in this panel. This is still a deployment assumption:
before any promotion, test generator inputs derived solely from the production
audio pipeline and score them in reference-time windows. No claim of causal isolation
or production readiness may silently ignore that pending check.

## Verification and resources

Prove the old first32-target omission and new BOS inclusion without a training run.
Test crop boundaries, no invented mid-song BOS, and direct first-action gradients.
Verify the same remaining loss definitions and resumable counters/RNG identity.
Evaluate actual serialized outputs, preserving REST, obstacles and literal stacks.
Same45-minute work/55-minute hard process limits,>=2GB headroom; no dependencies,
playtests, annotations, judge batches or push. Never mark the broader goal complete
merely because this experiment reaches a terminal result.
