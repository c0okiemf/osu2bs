# Legal timing continuation

This is a separate no-fit decoder experiment. Earlier results remain unchanged.

## Hypothesis and minimal change

V1 assigns categorical probability to proposals whose decoded times exceed the
phrase interval, then rejects the entire song. Condition that same distribution
on legal proposals before drawing a category. Reuse all four existing checkpoints;
no optimizer updates, residual retuning, additional seeds or target-chart serving
inputs. This tests support conditioning, not a new learned timing representation.
A later architecture change is justified only by this experiment's evidence.

For each action, compute every category's proposed time using the existing gap
bases and residual clamp[-16,8]. REST0 is always a legal advance to the known
window end. Gap0/category1 is legal only at BOS/after REST. Positive event proposals
must strictly advance cursor and remain strictly before the current eight-beat
boundary or audio tail. Mask illegal categories before softmax/multinomial.
Sample an unchanged count/slot distribution conditioned on the selected category.
Never move a sampled event, replace it with REST, or reset hand/recurrent history.
Nonfinite model output fails explicitly. Record excluded probability mass, action
count and cases where REST is the only legal category. The unchanged20000-action
budget still fails honestly; legal timing does not promise learned song termination.

## Reuse and evidence boundaries

Reuse authenticated356-family data,316/32 fit/validation split, eight supported
opened development families, twelve-entry accounting, B0 rate/environment/audio
origin conversion, literal exporter, comparator-v2 and approved fast path. Keep
all v1 artifacts and production hashes unchanged. Reuse v1 retrieval and B0 records
only with bound record/chart/Info hashes; no regeneration or reference-based choice.

Freeze a separate `experiments/joint-phrase-v2/legal-timing/` identity containing
v1 identity, all four checkpoint hashes, implementation/spec hashes, and comparison
artifact hashes. Same validation32 songs × seeds0/1 × four snapshots, temp1.
Same ordering: completion failure count, rhythm+geometry error, earliest update.
Same eight-song model development seeds0..5/temps[.85,1,1.15,.85,1,1.15]. Select the
first machine-admitted candidate, otherwise B0 fallback. Preserve all attempts.
No development output selects checkpoints or modifies policy.

Retain the v1 metrics, train-IQR scales, opportunity coverage protection and
positive gate verbatim. Existing weaknesses (arc scale floor dominance and no
subjective quality authority) remain disclosed. A positive result unlocks further
planning for fresh confirmation, never automatic promotion. A negative result
closes this experiment and immediately feeds the next evidence-based planning step;
it does not establish overall generation quality.

## Required checks and bounds

Before rollouts: test that overshooting categories cannot be selected; REST remains
explicit; zero-time legality; tiny remaining intervals; NaNs fail; deterministic
seeded sampling; completed scripted songs retain both-hand state across REST.
Validate legal support across the existing356-source action codec, preserving its
lossless source data and disclosing any float32 target-support failures.

Bounded resumable processes:45-minute work,55-minute hard limit,>=2GB RAM headroom.
Use atomic per-candidate artifacts and reject changed identities on resume. Commit
logical units; no push, playtests, annotations, new dependencies or judge batches.
