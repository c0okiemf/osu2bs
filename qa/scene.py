"""Independent-QA Task 2: independent scene interpretation + monotone
unique alignment (spec §3/§4).

The scene is built from final chart bytes + Info only — no generator
parser as authority. v2 notes (and v3 fixed-timing colorNotes) normalize
to one schema: (time_s, line_index, line_layer, color, cut_direction).
Unsupported mechanics (lane rotation, custom/variable timing, arcs/chains)
are explicit scope failures via the audited eval.map_scope reader — a
blocked scene is never silently cleared.

Alignment: attribute + timing + MONOTONE occurrence matching with
uniqueness; left-handed replays are matched against the mirrored chart
(line 3-li, colors flipped, directions mirrored). Ambiguity is rejected
and counted, never guessed.
"""
import hashlib
import json
from pathlib import Path

ALIGN_TOL_S = 0.2
MIRROR_DIR = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 8}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_scene(dat_path, info_path, audio_path=None):
    from eval.map_scope import fixed_v2_scope_reason
    dat_path, info_path = Path(dat_path), Path(info_path)
    raw = json.loads(dat_path.read_text(encoding="utf-8-sig"))
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    bpm = info.get("_beatsPerMinute") or (info.get("audio") or {}).get("bpm")
    scene = {"chart_sha256": _sha(dat_path), "info_sha256": _sha(info_path),
             "audio_sha256": _sha(audio_path) if audio_path else None,
             "bpm": float(bpm) if bpm else None,
             "notes": [], "bombs": [], "walls": [], "scope": None,
             "settings": {}}
    if not bpm or bpm <= 0:
        scene["scope"] = "invalid_bpm"
        return scene
    beat_s = 60.0 / float(bpm)
    if "_notes" in raw:                              # v2
        scope = fixed_v2_scope_reason(raw)
        if scope is not None:
            scene["scope"] = scope
            return scene
        # fail-closed modchart heuristic : a chart whose
        # scoreable notes are ALL one identical dot note in one cell AND
        # that carries chart-level customData is suspected of defining its
        # real gameplay in a mod — the vanilla notes cannot be aligned.
        # Neither customData nor uniform dots ALONE trigger this.
        sc = [(n.get("_lineIndex"), n.get("_lineLayer"), n.get("_type"),
               n.get("_cutDirection")) for n in raw.get("_notes") or []
              if n.get("_type") in (0, 1)]
        if raw.get("_customData") and len(sc) >= 20 and len(set(sc)) == 1 \
                and sc[0][3] == 8:
            scene["scope"] = "unsupported_suspected_modchart"
            return scene
        import math as _math
        for n in raw.get("_notes", []) or []:
            t = n.get("_time")
            if not isinstance(t, (int, float)) or isinstance(t, bool) \
                    or not _math.isfinite(float(t)):
                scene["scope"] = "invalid_note_time"
                return scene
            rec = (float(t) * beat_s, n.get("_lineIndex"),
                   n.get("_lineLayer"), n.get("_type"),
                   n.get("_cutDirection"))
            if n.get("_type") == 3:
                scene["bombs"].append(rec)
            elif n.get("_type") in (0, 1):
                scene["notes"].append(rec)
            else:
                scene["scope"] = f"unsupported_note_type:{n.get('_type')}"
                return scene
        for w in raw.get("_obstacles", []) or []:
            scene["walls"].append((float(w.get("_time", 0)) * beat_s,
                                   w.get("_lineIndex"), w.get("_type"),
                                   float(w.get("_duration", 0)) * beat_s,
                                   w.get("_width")))
    elif "colorNotes" in raw:                        # v3 fixed timing only
        if raw.get("burstSliders") or raw.get("sliders") \
                or raw.get("rotationEvents") or raw.get("bpmEvents"):
            scene["scope"] = "unsupported_v3_mechanics"
            return scene
        for n in raw.get("colorNotes", []) or []:
            scene["notes"].append((float(n.get("b", 0)) * beat_s,
                                   n.get("x"), n.get("y"), n.get("c"),
                                   n.get("d")))
        for b in raw.get("bombNotes", []) or []:
            scene["bombs"].append((float(b.get("b", 0)) * beat_s,
                                   b.get("x"), b.get("y"), 3, 8))
    else:
        scene["scope"] = "unknown_schema"
        return scene
    scene["notes"].sort()
    # approach settings from Info (recorded; unknown stays unknown)
    for s in info.get("_difficultyBeatmapSets", []) or []:
        for b in s.get("_difficultyBeatmaps", []) or []:
            if b.get("_beatmapFilename") == dat_path.name:
                scene["settings"] = {
                    "njs": b.get("_noteJumpMovementSpeed"),
                    "offset_beats": b.get("_noteJumpStartBeatOffset"),
                    "difficulty": b.get("_difficulty"),
                    "characteristic": s.get("_beatmapCharacteristicName")}
    return scene


def _mirror_note(note):
    t, li, ll, c, d = note
    return (t, 3 - li, ll, 1 - c, MIRROR_DIR.get(d, d))


def align(scene, replay_notes, left_handed=False, tol_s=ALIGN_TOL_S):
    """Monotone unique alignment of replay note events to scene notes.
    Bombs align separately (event_type 3). Ambiguity/ordering violations
    are rejected with counts."""
    chart = scene["notes"]
    view = [_mirror_note(n) for n in chart] if left_handed else chart
    aligned, ambiguous, unmatched, out_of_order = [], 0, 0, 0
    last_ci = -1
    used = set()
    events = sorted(
        ((e.get("event_time", 0.0), i, e) for i, e in enumerate(replay_notes)
         if e.get("event_type") in (0, 1, 2)), key=lambda x: x[0])
    for _t, ei, ev in events:
        cand = [ci for ci, (t, li, ll, c, d) in enumerate(view)
                if ci not in used
                and li == ev.get("line_index") and ll == ev.get("line_layer")
                and c == ev.get("color") and d == ev.get("cut_direction")
                and abs(t - ev["event_time"]) <= tol_s]
        if len(cand) > 1:
            # monotone disambiguation: earliest candidate after last match
            fwd = [ci for ci in cand if ci > last_ci]
            cand = [min(fwd)] if fwd else cand
        if len(cand) == 1:
            ci = cand[0]
            if ci < last_ci:
                out_of_order += 1
                continue
            aligned.append((ci, ei))
            used.add(ci)
            last_ci = ci
        elif len(cand) > 1:
            ambiguous += 1
        else:
            unmatched += 1
    residuals = [abs(view[ci][0] - replay_notes[ei]["event_time"])
                 for ci, ei in aligned]
    return {"aligned": aligned, "chart": view, "ambiguous": ambiguous,
            "unmatched": unmatched, "out_of_order": out_of_order,
            "mirrored": left_handed,
            "timing_residual_p95":
                (sorted(residuals)[int(0.95 * (len(residuals) - 1))]
                 if residuals else None)}
