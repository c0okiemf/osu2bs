# Controlled typed-phrase geometry refinement

Continue the broader task after the unchanged v3 whole-generator gate fails.


## Different controlled question, not a retroactive pass

V3 gets six nonfallback families and lower aggregate geometry distance than B0,
but its rhythm remains much farther from references than retrieval and its geometry
opportunity coverage can fall. Test whether its geometry model can improve a
retrieved phrase while holding musical action structure fixed. No new fit.

This is a subsystem experiment. It cannot beat the rhythm of its own fixed input,
so it must not be labeled a pass of the earlier whole-generator gate. All prior
negative results remain negative. Even a positive subsystem result requires a new
end-to-end comparison under deployment inputs and fresh confirmation before promotion.

## Inputs and intervention

Use v1's already selected first-admitted retrieval chart on each of the same eight
supported development families. Every selected retrieval chart was admitted. Bind
its actual chart,Info,record and donor provenance; retain all12 panel identities.
Reuse the v3 validation-selected checkpoint, fixed once before refinement outcomes.
Use the same audio, constant BPM, NJS18 and B0 environment as the input chart.
Requested hand-event rate equals the known template's rate; it is available before
geometry generation and does not use the target human chart.

Preserve every exact timestamp, event's hand counts, and each canonical hand slot's
arrow-versus-dot role. Generate columns/layers and arrow directions only. Dots remain
dots and arrows remain arrows. Keep cross-hand occupancy and per-hand canonical
cell ordering, reserving enough cells for the remaining slots. Sample directly from
legal tokens; do not repair sampled geometry. Maximum three notes per hand.

Run the model recurrently over template events plus explicit eight-beat REST actions,
using the existing audio context and actual previously generated literal slots.
Only gap category, counts and arrow/dot roles come from the template; template
coordinates and non-dot directions never enter model inputs. The count/gap heads
are not sampled. Count and slot embeddings are conditioned on the fixed action.
REST preserves hand history. Full-song output must retain the complete template
signature `(timestamp, per-hand ordered dot flags)` exactly, including tail rests.

Six geometry seeds0..5/temps[.85,1,1.15,.85,1,1.15] per family. First machine-admitted
refinement is selected; if none, use the original admitted retrieval chart and count
zero geometry improvement. Report the full hierarchical budget: the earlier six
retrieval candidates plus six refinement attempts. This is not a matched-cost claim
against six-candidate retrieval alone.

## Freeze and measurements

Separate identity: `experiments/joint-phrase-v4/typed-refinement/`, binding code,
this spec,v3 selection/checkpoint, all input charts/records and production hashes.
No new development family, validation selection, optimizer update, seed sweep or
reference-dependent serving choice.

Keep existing rhythm/geometry/audio descriptors, train-IQR scales and full-scene QA.
Report the old full profiles/errors as diagnostics. Rhythm must equal the selected
retrieval chart exactly. Arrow/dot preservation must also keep its geometry opportunity
mask exactly; unexpected missing measurements fail the refinement.

For a fair controlled geometry comparison, freeze one common opportunity mask per
family BEFORE generating refinements: finite human-reference AND B0 AND selected
retrieval geometry entries. Score all three candidates on that same mask, with the
same five features/train scales and family-balanced mean. Require each component to
have a nonzero denominator per family; record every excluded opportunity and counts.
The mask cannot change from a refinement outcome. This avoids claiming improvement
by measuring different windows, and does not erase the failed historical coverage gate.

Diagnostic positive rule: exact temporal/type signatures and rhythm invariance;
at least6/8 selected nonfallback refinements; common-opportunity geometry error at
least10% below BOTH B0 and retrieval; each component no more than10%+0.01 worse than
either; no new structural/model contradiction; no support-share drop>0.05 against
either on any family. Report original metrics and family bootstrap intervals too.
This rule answers conditional geometry headroom only, not universal map quality.

## Checks, resource bounds and continuation

Test arrow/dot and exact fractional timestamp preservation; multi-note hands;
shared occupancy/order; seed determinism; REST continuity; and independence from
original coordinates/non-dot directions. Export/readback must preserve the same
signature and environment. Unknown QA never counts as an admitted refinement.

45-minute work/55-minute hard process bounds;>=2GB RAM headroom; atomic per-attempt
resume; no dependencies, playtests, annotations, judge batches or push.
After the measured result, continue: positive leads to end-to-end deployment-input
planning; negative identifies the remaining geometric/model limitation. Do not end
the broader task because this experiment completes.
