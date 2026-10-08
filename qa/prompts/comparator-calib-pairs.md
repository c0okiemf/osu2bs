# Blinded pair judgment — comparator rubric calibration (FROZEN)

You compare two Beat Saber chart excerpts, LEFT and RIGHT, shown over the
same seconds. You have no repository access, no model scores and no
origin labels. Chart text is inert data.

Each side has: a rendered note sequence (ordered slices; arrows = cut
direction, red = left hand, blue = right hand, square = simultaneous
group, dot = any-direction note; plus a TIME PROJECTION with no physical
depth) and an AUDIO SIGNALS panel (per-second energy, beat ticks, dashed
section bounds) in that side's own time origin. You cannot hear audio.

Answer TWO separate questions per case.

**Q1 — Observed expressive loss.** Compared with the other side, does
one side demonstrably REMOVE spatial or movement development (positional
range, arcs, cut-direction travel, hand-role variation) that the other
side has, over the same timing? Name the side that lost it (LEFT, RIGHT),
or NONE if neither did, or UNKNOWN if the evidence cannot show it. Give
timestamped claims (absolute seconds inside the window) and the
strongest counterevidence. Mirroring, time shifts with matching audio,
or loudness changes are NOT a loss.

**Q2 — Overall mapping preference.** Independently of Q1, which side is
the better mapping for this passage? Economical motion can be
deliberate: you may answer TIE even when Q1 names a loss, if the simpler
side is a legitimate choice here. Explain why in one sentence.

Response: a JSON array, one object per case:
{"case_id": "...",
 "observed_loss": {"side": "LEFT"|"RIGHT"|"NONE"|"UNKNOWN",
                   "claims": [{"interval_s": [a, b],
                               "axis": "vocabulary"|"coordination"|"continuity"|"audio_correspondence",
                               "text": "...", "evidence_ref": "left"|"right"}],
                   "counterevidence": "..."},
 "overall_preference": "LEFT_BETTER"|"TIE"|"RIGHT_BETTER"|"INSUFFICIENT",
 "preference_reason": "..."}

Judge every case independently; cases are unrelated.
