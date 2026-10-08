"""E01 projected-face visibility diagnostic (scene + jump-settings contract).

Evaluation-only. This is a bounded projected-face occlusion PROXY under a
declared 1-D-eye / flat-face model, NOT a renderer, a game hitbox, human
readability, or a difficulty label. It changes nothing in generation and never
authorizes unmasking centre cells. See
docs/specs/2026-09-20-e01-visibility-design.md (normative).

Task 1: freeze the scene/settings contract (Face/Note, HJD jump settings,
load_scene). Geometry (project/union/oracle) and traces come in later tasks.
"""
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from eval.map_scope import fixed_v2_scope_reason

ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-9

# Standard comparison scenario (design): a hypothetical fixed setting, NOT
# recovered historical playback.
STD_NJS = 16.0
STD_LIFETIME_S = 0.600


@dataclass(frozen=True)
class Face:
    """An opaque camera-parallel square (or projected rectangle) in metres."""
    id: str
    x0: float
    x1: float
    y0: float
    y1: float
    z: float


@dataclass(frozen=True)
class Note:
    """A visible colored note at its hit time and grid cell (no geometry yet)."""
    id: str
    hit_s: float
    col: int
    row: int
    color: int
    direction: int


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if isfinite(v) else None


def jump_settings(bpm, njs, offset_beats):
    """Viewer-style half-jump-duration convention (pinned to ArcViewer
    BeatmapManager rev a7b2d984, incl. the exact 35.998 m boundary and the
    0.25-beat clamp). IEEE-754 double arithmetic is this diagnostic's
    convention, not a claim of bit-identity with every game float path.
    Returns hjd_beats, lifetime_s, half_distance_m, full_distance_m."""
    b, n, off = _num(bpm), _num(njs), _num(offset_beats)
    if b is None or b <= 0 or n is None or n <= 0 or off is None:
        raise ValueError(f"invalid jump inputs bpm={bpm} njs={njs} offset={offset_beats}")
    beat_seconds = 60.0 / b
    h0 = 4.0
    while 2 * n * beat_seconds * h0 > 35.998:
        h0 /= 2.0
    h = max(0.25, h0 + off)
    T = h * beat_seconds
    half = n * T
    return {"hjd_beats": h, "lifetime_s": T,
            "half_distance_m": half, "full_distance_m": 2 * half}


# cosmetic-only note _customData keys; anything else is an executable transform
_COSMETIC_CD = {"_color", "color"}


def _has_transform(cd):
    return isinstance(cd, dict) and any(k not in _COSMETIC_CD for k in cd)


def _sha256(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


# --- scene geometry: note -> physical face ------------------------------

# Lane centre x = -0.9 + 0.6*col metres; row centre y (declared model choice,
# informed by ArcViewer ObjectManager rev a7b2d984).
ROW_Y = (0.60, 1.15, 1.65)


def note_face(note, t, njs, side):
    """The target/blocker's opaque square at evaluation time t: constant x/y at
    its lane/row centre, depth z = njs*(hit-t) (it sits at the cut plane at its
    hit time and is removed then). Not a game hitbox or mesh."""
    cx = -0.9 + 0.6 * note.col
    cy = ROW_Y[note.row]
    h = side / 2.0
    return Face(note.id, cx - h, cx + h, cy - h, cy + h, njs * (note.hit_s - t))


def is_banned_cell(col, row):
    """The production blanket centre-note exclusion: middle row, columns 1/2."""
    return row == 1 and col in (1, 2)


def _cells(spawn, hit, step, events):
    """Sample-cell boundaries: 10 ms grid anchored to target spawn, split at
    every relevant event, clipped to [spawn, hit]. Partial cells retained."""
    bounds = {spawn, hit}
    k = 1
    while spawn + k * step < hit - EPS:
        bounds.add(spawn + k * step)
        k += 1
    for e in events:
        if spawn + EPS < e < hit - EPS:
            bounds.add(e)
    b = sorted(bounds)
    return list(zip(b, b[1:]))


def _runs(samples, key, thr):
    """(total, longest-contiguous) duration where key(sample) >= thr."""
    total = longest = run = 0.0
    for s in samples:
        d = s["end_s"] - s["start_s"]
        if s[key] >= thr - EPS:
            run += d
            total += d
            longest = max(longest, run)
        else:
            run = 0.0
    return total, longest


def trace_target(scene, target_id, config, detail=False):
    """Coverage of one target over its full approach interval, event-split at
    10 ms (anchored to target spawn), midpoints duration-weighted. Returns
    status, duration-weighted metrics, and (detail=True) per-sample samples and
    the contributing-occluder cells. Complete approach is measured, not just the
    hit; late occlusion never erases earlier exposure."""
    # an unknown scene (unsupported timing/rotation, or any invalid/unsupported
    # colored note) cannot be traced as if it were clear
    if scene.get("chart_unknown"):
        return {"target_id": target_id, "status": "chart_unknown"}
    njs, T = scene["njs"], scene["lifetime_s"]
    if njs is None or T is None:
        return {"target_id": target_id, "status": "settings_unknown"}
    eye, side, step = config["eye"], config["face_side_m"], config["step_s"]
    notes = scene["notes"]
    tgt = next(n for n in notes if n.id == target_id)
    spawn, hit = tgt.hit_s - T, tgt.hit_s
    # nearer, temporally-overlapping blockers: earlier hit within (spawn, hit)
    blockers = [n for n in notes if n.id != target_id
                and spawn < n.hit_s < hit]
    cell_of = {n.id: (n.col, n.row) for n in blockers}
    events = [n.hit_s for n in blockers] + [n.hit_s - T for n in blockers]
    samples = []
    contributing = set()          # any positive-area overlap, any sample
    hidden_occ = set()            # occluders at samples with coverage >= 0.5
    for a, b in _cells(spawn, hit, step, events):
        mid = (a + b) / 2.0
        active = [n for n in blockers if n.hit_s - T <= mid < n.hit_s]
        tf = note_face(tgt, mid, njs, side)
        faces = [note_face(n, mid, njs, side) for n in active]
        cov, ids = covered_detail(tf, faces, eye)
        contributing.update(ids)
        if cov >= 0.5 - EPS:
            hidden_occ.update(ids)
        rec = {"start_s": a, "end_s": b, "coverage": cov}
        if detail:
            mf = Face(tgt.id, (tf.x0 + tf.x1) / 2 - side / 4,
                      (tf.x0 + tf.x1) / 2 + side / 4,
                      (tf.y0 + tf.y1) / 2 - side / 4,
                      (tf.y0 + tf.y1) / 2 + side / 4, tf.z)
            rec["marker_coverage"] = covered_fraction(mf, faces, eye)
            rec["blocker_ids"] = ids
        samples.append(rec)

    dur = sum(s["end_s"] - s["start_s"] for s in samples) or T
    max_cov = max((s["coverage"] for s in samples), default=0.0)
    t50, l50 = _runs(samples, "coverage", 0.5)
    t80, l80 = _runs(samples, "coverage", 0.8)
    terminal = 0.0                # contiguous C<0.5 ending at the hit
    for s in reversed(samples):
        if s["coverage"] < 0.5 - EPS:
            terminal += s["end_s"] - s["start_s"]
        else:
            break
    early = 0.0                   # contiguous C<0.5 from spawn
    for s in samples:
        if s["coverage"] < 0.5 - EPS:
            early += s["end_s"] - s["start_s"]
        else:
            break
    occ_cells = sorted({(cell_of[i][0], cell_of[i][1]) for i in hidden_occ})
    out = {"target_id": target_id, "status": "ok",
           "cell": {"col": tgt.col, "row": tgt.row,
                    "banned": is_banned_cell(tgt.col, tgt.row)},
           "hit_s": tgt.hit_s, "max_coverage": max_cov,
           "area_time_integral": sum(
               s["coverage"] * (s["end_s"] - s["start_s"]) for s in samples) / dur,
           "time_ge_0.5_s": t50, "longest_ge_0.5_s": l50,
           "time_ge_0.8_s": t80, "longest_ge_0.8_s": l80,
           "terminal_clear_s": terminal, "early_clear_s": early,
           "n_blockers": len(blockers),
           "contributing_ids": sorted(contributing),
           "hidden_occluder_ids": sorted(hidden_occ),
           "hidden_occluder_cells": [{"col": c, "row": r,
                                      "banned": is_banned_cell(c, r)}
                                     for c, r in occ_cells]}
    if detail:
        out["samples"] = samples
    return out


# --- Task 2: analytic projected-face occlusion ---------------------------

def project_face(face, eye):
    """Pinhole projection (focal length 1, looking +z) of an axis-aligned face
    at constant z. Returns (u0,u1,v0,v1,depth). Nonpositive depth (at/behind the
    eye) raises ValueError. This is geometric obstruction, not a headset FOV."""
    depth = face.z - eye[2]
    if depth <= 0:
        raise ValueError(f"nonpositive depth {depth} for face {face.id}")
    u0 = (face.x0 - eye[0]) / depth
    u1 = (face.x1 - eye[0]) / depth
    v0 = (face.y0 - eye[1]) / depth
    v1 = (face.y1 - eye[1]) / depth
    return (min(u0, u1), max(u0, u1), min(v0, v1), max(v0, v1), depth)


def _clip(r, t):
    """Intersect projected rect r with target rect t; None if no positive area."""
    x0, x1 = max(r[0], t[0]), min(r[1], t[1])
    y0, y1 = max(r[2], t[2]), min(r[3], t[3])
    if x1 - x0 > EPS and y1 - y0 > EPS:
        return (x0, x1, y0, y1)
    return None


def _union_area(rects):
    """Area of the union of axis-aligned rects: x-slab sweep, merged y intervals
    (not a sum of pairwise overlaps)."""
    if not rects:
        return 0.0
    xs = sorted({r[0] for r in rects} | {r[1] for r in rects})
    area = 0.0
    for xa, xb in zip(xs, xs[1:]):
        w = xb - xa
        if w <= EPS:
            continue
        ys = sorted((r[2], r[3]) for r in rects
                    if r[0] <= xa + EPS and r[1] >= xb - EPS)
        cur, ylen = None, 0.0
        for y0, y1 in ys:
            if cur is None:
                cur = [y0, y1]
            elif y0 <= cur[1] + EPS:
                cur[1] = max(cur[1], y1)
            else:
                ylen += cur[1] - cur[0]
                cur = [y0, y1]
        if cur:
            ylen += cur[1] - cur[0]
        area += w * ylen
    return area


def covered_detail(target, blockers, eye):
    """(fraction, contributing_ids): fraction of `target`'s projected face
    hidden by strictly-nearer blockers (union area / target area, in [0,1]), and
    the ids of blockers with positive clipped area. Coplanar/equal-depth notes
    do not occlude; the target is ignored by id; a repeated blocker is not
    double charged; a duplicate id with different geometry is invalid occupancy."""
    tu0, tu1, tv0, tv1, td = project_face(target, eye)
    tarea = (tu1 - tu0) * (tv1 - tv0)
    if tarea <= EPS:
        return 0.0, []
    seen, clipped, ids = {}, [], []
    for b in blockers:
        if b.id == target.id:
            continue
        geom = (b.x0, b.x1, b.y0, b.y1, b.z)
        if b.id in seen and seen[b.id] != geom:
            raise ValueError(f"duplicate blocker id {b.id} with different geometry")
        seen[b.id] = geom
        bu0, bu1, bv0, bv1, bd = project_face(b, eye)
        if not (bd < td - EPS):            # strictly nearer only
            continue
        c = _clip((bu0, bu1, bv0, bv1), (tu0, tu1, tv0, tv1))
        if c:
            clipped.append(c)
            ids.append(b.id)
    return _union_area(clipped) / tarea, ids


def covered_fraction(target, blockers, eye):
    """Fraction of `target`'s projected face hidden by strictly-nearer blockers."""
    return covered_detail(target, blockers, eye)[0]


def resolve_authored(chart_path, chart_sha256):
    """Bind authored NJS/offset from the sibling Info.dat by EXACT beatmap
    filename + content hash. A title/difficulty-label match is insufficient;
    zero, multiple, or hash-mismatched bindings return an explicit unknown, and
    conflicts are never resolved by taking the first matching label."""
    chart = Path(chart_path)
    info_p = next((f for f in chart.parent.iterdir()
                   if f.name.lower() == "info.dat"), None)
    if info_p is None:
        return {"status": "unknown", "reason": "no_info_dat"}
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    # only the Standard characteristic binds; a OneSaber/other set with the same
    # filename must not authenticate (review difficulty closeout)
    matches = [(s, b) for s in info.get("_difficultyBeatmapSets", [])
               if s.get("_beatmapCharacteristicName") == "Standard"
               for b in s.get("_difficultyBeatmaps", [])
               if b.get("_beatmapFilename") == chart.name]
    if not matches:
        return {"status": "unknown", "reason": "no_info_entry_binds_this_filename"}
    if len(matches) > 1:
        return {"status": "unknown", "reason": "ambiguous_info_entries"}
    s, b = matches[0]
    ref_file = chart.parent / b["_beatmapFilename"]
    if not ref_file.exists() or _sha256(ref_file) != chart_sha256:
        return {"status": "unknown", "reason": "info_filename_hash_mismatch"}
    njs = _num(b.get("_noteJumpMovementSpeed"))
    off = _num(b.get("_noteJumpStartBeatOffset", 0.0))
    if njs is None or njs <= 0 or off is None:
        return {"status": "unknown", "reason": "absent_or_invalid_njs_offset"}
    rank = b.get("_difficultyRank")
    rank = int(rank) if isinstance(rank, int) and not isinstance(rank, bool) else None
    label = (b.get("_customData") or {}).get("_difficultyLabel")
    return {"status": "verified", "njs": njs, "offset_beats": off,
            "rank": rank, "label": label, "difficulty": b.get("_difficulty"),
            "characteristic": s.get("_beatmapCharacteristicName"),
            "info": str(info_p), "info_sha256": _sha256(info_p),
            "binding": "beatmapFilename==chart and sha256 match"}


def load_scene(entry, mode):
    """Load a chart's visible colored notes + jump settings for `mode`
    ("standard" or "authored"). Returns notes, bpm, njs, lifetime_s,
    settings_status, source_hashes, mode, unsupported, invalid, chart_unknown.
    Unavailable authored settings yield an explicit unknown status, never a
    silent fallback to the standard scenario. Walls/bombs/chains/arcs are
    counted as ignored, never silently dropped to assert coverage."""
    chart = _resolve(entry["chart"])
    raw = json.loads(chart.read_text(encoding="utf-8-sig"))
    bpm = float(entry["bpm"])
    origin_s = float(entry.get("origin_s", 0))
    source_hashes = {"chart": entry.get("sha256"),
                     "timing_source": entry.get("timing_source_sha256")}

    scope = fixed_v2_scope_reason(raw)
    base = {"bpm": bpm, "mode": mode, "source_hashes": source_hashes,
            "notes": [], "unsupported": [], "invalid": [], "ignored": {}}

    # jump settings first; authored-unknown fails closed to an unknown scene
    if mode == "standard":
        njs, lifetime_s, settings_status = STD_NJS, STD_LIFETIME_S, "standard"
    elif mode == "authored":
        a = entry.get("authored") or {}
        # verify the binding LIVE from Info, never trust the stored status string
        live = resolve_authored(chart, entry.get("sha256"))
        if live["status"] != "verified":
            return {**base, "njs": None, "lifetime_s": None,
                    "settings_status": "unknown",
                    "settings_reason": live.get("reason", "authored_settings_unverified"),
                    "chart_unknown": True}
        # the panel's recorded authored record must match the live binding
        if (a.get("status") != "verified" or _num(a.get("njs")) != live["njs"]
                or _num(a.get("offset_beats")) != live["offset_beats"]
                or a.get("info_sha256") != live["info_sha256"]):
            return {**base, "njs": None, "lifetime_s": None,
                    "settings_status": "unknown",
                    "settings_reason": "authored_record_altered",
                    "chart_unknown": True}
        try:
            js = jump_settings(bpm, live["njs"], live["offset_beats"])
        except (ValueError, KeyError):
            return {**base, "njs": None, "lifetime_s": None,
                    "settings_status": "unknown",
                    "settings_reason": "invalid_authored_settings",
                    "chart_unknown": True}
        njs, lifetime_s, settings_status = live["njs"], js["lifetime_s"], "verified"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if scope is not None:
        return {**base, "njs": njs, "lifetime_s": lifetime_s,
                "settings_status": settings_status, "chart_unknown": True,
                "scope": scope}

    beat_seconds = 60.0 / bpm
    notes, unsupported, invalid = [], [], []
    ignored = {"walls": len(raw.get("_obstacles", []) or []),
               "bombs": 0, "chains": 0, "arcs": 0}
    for i, n in enumerate(raw.get("_notes", []) or []):
        nid = f"note:{i}"
        if not isinstance(n, dict):
            invalid.append({"id": nid, "reason": "note_not_object"})
            continue
        typ = n.get("_type")
        if typ == 3:                      # bomb: counted, not a visible target
            ignored["bombs"] += 1
            continue
        if isinstance(typ, bool) or typ not in (0, 1):
            invalid.append({"id": nid, "reason": f"noncolor_type:{typ}"})
            continue
        t = _num(n.get("_time"))
        col = n.get("_lineIndex")
        row = n.get("_lineLayer")
        d = n.get("_cutDirection")
        if t is None or not isinstance(col, int) or isinstance(col, bool) \
                or not isinstance(row, int) or isinstance(row, bool):
            invalid.append({"id": nid, "reason": "nonfinite_or_missing_field"})
            continue
        hit_s = origin_s + t * beat_seconds
        if not isfinite(hit_s):
            invalid.append({"id": nid, "reason": "nonfinite_after_conversion"})
            continue
        if not (0 <= col <= 3 and 0 <= row <= 2):
            invalid.append({"id": nid, "reason": "out_of_grid"})
            continue
        # a malformed cut direction must not quietly become an ordinary dot
        if isinstance(d, bool) or not isinstance(d, int) or not (0 <= d <= 8):
            invalid.append({"id": nid, "reason": "bad_cut_direction"})
            continue
        if _has_transform(n.get("_customData")):
            unsupported.append({"id": nid, "reason": "custom_geometry_transform"})
            continue
        notes.append(Note(nid, hit_s, col, row, typ, d))

    # duplicate same-time same-cell colored occupancy is invalid (two coplanar
    # notes would otherwise both read as clear); flag rather than swallow it
    seen, dup = set(), False
    for n in notes:
        key = (round(n.hit_s, 9), n.col, n.row)
        if key in seen:
            dup = True
            invalid.append({"id": n.id, "reason": "duplicate_cell_occupancy"})
        seen.add(key)

    # any invalid/unsupported colored note (or duplicate occupancy) makes the
    # whole note-occlusion analysis unknown for this packet -- a dropped blocker
    # must never let a target read as clear (review R1)
    chart_unknown = bool(invalid or unsupported or dup)
    return {**base, "notes": notes, "njs": njs, "lifetime_s": lifetime_s,
            "settings_status": settings_status, "unsupported": unsupported,
            "invalid": invalid, "ignored": ignored,
            "chart_unknown": chart_unknown,
            "scope": ("note_scope_unknown" if chart_unknown else "note_to_note")}
