"""Model-conditional wall-corridor geometry (A06 / D01 diagnostic).

Evaluation-only. Consumes raw v2 obstacles and a declared fixed seconds-per-beat
+ time origin, and produces a normalized interval representation. A separate pure
interval solver (`safe_intervals`/`analyze_walls`, Task 2) answers, under a
declared 1-D head model, which head-center positions stay clear of full-height
walls and the necessary lateral travel between constrained intervals.

Nothing here changes generation. Wall time is the nominal map contact interval,
NOT a simulation of the game's collision box, jump animation, or a human limit.
See docs/specs/2026-09-18-wall-corridor-design.md (normative).
"""
from dataclasses import dataclass
from math import isfinite

from eval.map_scope import fixed_v2_scope_reason


@dataclass(frozen=True)
class Wall:
    """A supported full-height wall as a seconds/lane-units interval.
    `x0 = _lineIndex`, `x1 = _lineIndex + _width`, lane domain [0,4]."""
    id: str
    start_s: float
    end_s: float
    x0: float
    x1: float


def _num(v):
    """finite int/float only (reject bool, str, None, NaN, inf)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if isfinite(v) else None


def normalize_v2(raw, seconds_per_beat, origin_s=0.0):
    """Normalize raw v2 obstacles to supported walls + explicit scope.

    Returns {walls, unknown, invalid, chart_unknown, scope}. `walls` are the
    supported full-height (_type=0) integer-lane rectangles inside [0,4] with
    positive duration/width. `unknown` are well-formed obstacles whose geometry
    this packet does not model (crouch, extended/custom lane, out-of-domain) —
    their time interval is known, so overlapping analysis segments are `unknown`.
    `invalid` are malformed obstacles (nonfinite, zero/negative duration/width,
    missing/wrong-typed fields). `chart_unknown` is True when the chart's
    version / coordinate-time system itself is unsupported, in which case no
    walls are trusted. Never substitute a supported-empty chart for unsupported
    input.
    """
    spb = _num(seconds_per_beat)
    org = _num(origin_s)
    if spb is None or spb <= 0 or org is None:
        return {"walls": [], "unknown": [], "invalid": [], "chart_unknown": True,
                "scope": "unsupported_timing"}

    ver = str(raw.get("_version") or raw.get("version") or "")
    obs = raw.get("_obstacles")
    if not ver.startswith("2") or not isinstance(obs, list):
        # coordinate/time system not this packet's supported v2: trust nothing
        return {"walls": [], "unknown": [], "invalid": [], "chart_unknown": True,
                "scope": f"unsupported_schema:{ver or 'none'}"}

    # Executable transforms / ambiguous timing / any lane rotation must not be
    # silently accepted as vanilla fixed-time geometry (review review 2026-09-20).
    # One shared predicate keeps the wall and visibility audits in lockstep.
    scope = fixed_v2_scope_reason(raw)
    if scope is not None:
        return {"walls": [], "unknown": [], "invalid": [],
                "chart_unknown": True, "scope": scope}

    walls, unknown, invalid = [], [], []
    unresolvable = False
    for i, o in enumerate(obs):
        wid = f"wall:{i}"
        if not isinstance(o, dict):
            invalid.append({"id": wid, "reason": "obstacle_not_object",
                            "start_s": None, "end_s": None})
            unresolvable = True
            continue
        t = _num(o.get("_time"))
        dur = _num(o.get("_duration"))
        li = _num(o.get("_lineIndex"))
        w = _num(o.get("_width"))
        typ = o.get("_type")
        # nonfinite / missing / wrong-typed numeric fields -> invalid (no interval)
        if t is None or dur is None or li is None or w is None:
            invalid.append({"id": wid, "reason": "nonfinite_or_missing_field",
                            "start_s": None, "end_s": None})
            unresolvable = True
            continue
        start_s = org + t * spb
        end_s = org + (t + dur) * spb
        # post-conversion overflow (finite fields, nonfinite product) -> invalid
        if not (isfinite(start_s) and isfinite(end_s)):
            invalid.append({"id": wid, "reason": "nonfinite_after_conversion",
                            "start_s": None, "end_s": None})
            unresolvable = True
            continue
        # _type must be a genuine integer (bool is not; False must NOT read as
        # full-height _type=0)
        if isinstance(typ, bool) or not isinstance(typ, int):
            invalid.append({"id": wid, "reason": "noninteger_type",
                            "start_s": start_s, "end_s": end_s})
            continue
        # zero/negative duration is malformed regardless of geometry type
        if dur <= 0:
            invalid.append({"id": wid, "reason": "nonpositive_duration",
                            "start_s": start_s, "end_s": end_s})
            continue
        # executable per-object transform (Noodle/Chroma position/scale/etc.):
        # geometry is not the vanilla lane rectangle -> unknown coverage
        if _has_transform(o.get("_customData")):
            unknown.append({"id": wid, "reason": "custom_geometry_transform",
                            "start_s": start_s, "end_s": end_s})
            continue
        # unsupported obstacle TYPE (crouch=1, or anything not full-height=0):
        # well-formed time interval, geometry not modeled -> unknown coverage
        if typ != 0:
            unknown.append({"id": wid, "reason": f"unsupported_type:{typ}",
                            "start_s": start_s, "end_s": end_s})
            continue
        if w <= 0:
            invalid.append({"id": wid, "reason": "nonpositive_width",
                            "start_s": start_s, "end_s": end_s})
            continue
        # extended/custom geometry (fractional or out-of-domain lanes): known
        # time, geometry outside this packet's [0,4] integer model -> unknown
        if li != int(li) or w != int(w) or li < 0 or li + w > 4:
            unknown.append({"id": wid, "reason": "extended_or_out_of_domain_lane",
                            "start_s": start_s, "end_s": end_s})
            continue
        walls.append(Wall(wid, start_s, end_s, float(li), float(li + w)))

    # an invalid object with an unresolvable interval contaminates the whole
    # chart; invalid objects with a known interval contaminate that interval
    # (folded into the unknown-coverage sweep by analyze_walls).
    return {"walls": walls, "unknown": unknown, "invalid": invalid,
            "chart_unknown": unresolvable, "scope": "wall_only"}


# cosmetic-only obstacle _customData keys (color) do not move geometry; any
# other key is treated as an executable coordinate/time transform.
_COSMETIC_CD = {"_color", "color"}


def _has_transform(cd):
    return isinstance(cd, dict) and any(k not in _COSMETIC_CD for k in cd)


# --- Task 2: exact corridor occupancy (A06) and pairwise travel (D01) ---

# Numerical comparison tolerance (design: at most 1e-9; NOT a gameplay grace).
EPS = 1e-9


def safe_intervals(walls, radius):
    """Closed 1-D head-center positions clear of the given active full-height
    walls, under the declared model. Lane `c` occupies [c,c+1]; a head of radius
    `r` centered at `p` is allowed in [r,4-r] and is excluded by wall [x0,x1]
    over the OPEN interval (x0-r, x1+r) — exact tangency is allowed by this
    model. Returns sorted closed safe intervals, singleton points included. This
    is a bounded 1-D model, not a proof of comfort for any real player.
    """
    r = float(radius)
    lo, hi = r, 4.0 - r
    if lo > hi:
        return []
    excl = sorted((w.x0 - r, w.x1 + r) for w in walls)
    # Merge only STRICTLY overlapping open exclusions. Two exclusions that only
    # touch at a point leave that point safe (open intervals keep their
    # endpoints), so they must NOT merge — that is how a central wall yields two
    # safe intervals and how singleton tangencies survive.
    merged = []
    for a, b in excl:
        if merged and a < merged[-1][1] - EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    safe = []
    cursor = lo
    for a, b in merged:
        if a >= cursor - EPS:           # closed safe gap [cursor, min(a,hi)]
            end = min(a, hi)
            if end >= cursor - EPS:
                safe.append((cursor, end))
        if b > cursor:
            cursor = b
        if cursor > hi + EPS:
            break
    if cursor <= hi + EPS:
        safe.append((cursor, hi))
    return safe


def _set_distance(A, B):
    """Analytic minimum lane distance between two interval sets (0 if they
    intersect). = min over pairs of max(0, b_lo-a_hi, a_lo-b_hi)."""
    best = float("inf")
    for a_lo, a_hi in A:
        for b_lo, b_hi in B:
            d = max(0.0, b_lo - a_hi, a_lo - b_hi)
            if d < best:
                best = d
    return best


def _active(intervals, t0, get):
    """Objects active over [t0, t1): start<=t0<end (events are grouped, so no
    boundary lies strictly inside a segment)."""
    out = []
    for o in intervals:
        s, e = get(o)
        if s is not None and e is not None and s <= t0 + EPS and e > t0 + EPS:
            out.append(o)
    return out


def analyze_walls(normalized, radius):
    """Model-conditional A06 corridor occupancy + D01 pairwise travel over a
    normalized chart. Returns {scope, radius, segments, transitions, counts,
    chart_unknown}. Non-numeric/unavailable distances and speeds are null,
    never Infinity/NaN. A necessary pairwise lower bound only — pairwise
    intersections may use different positions at successive boundaries, so zero
    pairwise distance does not certify a whole sequence.
    """
    r = float(radius)
    walls = normalized["walls"]
    invalid = normalized.get("invalid", [])
    # invalid objects with a resolvable interval contaminate that interval as
    # uncertain coverage, exactly like unmodeled (unknown) geometry
    n_unknown = len(normalized["unknown"])   # true unknown-object count
    unknowns = list(normalized["unknown"]) + [
        x for x in invalid if x.get("start_s") is not None
        and x.get("end_s") is not None]
    scope = normalized.get("scope", "")
    if normalized.get("chart_unknown"):
        return {"scope": scope, "radius": r, "segments": [], "transitions": [],
                "chart_unknown": True,
                "counts": {"supported_walls": 0, "unknown_objects": n_unknown,
                           "invalid_objects": len(invalid), "known_s": 0.0,
                           "unknown_s": 0.0, "empty_model_s": 0.0, "segments": 0,
                           "transitions": 0, "positive_distance_pairs": 0}}

    times = set()
    for w in walls:
        times.add(w.start_s)
        times.add(w.end_s)
    for u in unknowns:
        if u["start_s"] is not None:
            times.add(u["start_s"])
        if u["end_s"] is not None:
            times.add(u["end_s"])
    times = sorted(times)

    raw = []
    for t0, t1 in zip(times, times[1:]):
        if t1 <= t0 + EPS:
            continue
        act = _active(walls, t0, lambda w: (w.start_s, w.end_s))
        aun = _active(unknowns, t0, lambda u: (u["start_s"], u["end_s"]))
        safe = safe_intervals(act, r)
        model_empty = len(safe) == 0
        status = "unknown" if aun else ("empty_model" if model_empty else "known")
        raw.append({"start_s": t0, "end_s": t1,
                    "wall_ids": [w.id for w in act],
                    "safe_intervals": safe,
                    "unknown_ids": [u["id"] for u in aun],
                    "model_empty": model_empty, "status": status})

    # collapse contiguous segments with identical safe set AND status; never
    # merge through an unknown span; retain wall-ID provenance.
    collapsed = []
    for s in raw:
        if collapsed:
            p = collapsed[-1]
            if (p["status"] == s["status"] and s["status"] != "unknown"
                    and p["safe_intervals"] == s["safe_intervals"]
                    and abs(p["end_s"] - s["start_s"]) <= EPS):
                p["end_s"] = s["end_s"]
                for wid in s["wall_ids"]:
                    if wid not in p["wall_ids"]:
                        p["wall_ids"].append(wid)
                continue
        collapsed.append(dict(s))

    # D01: compare successive WALL-CONSTRAINED segments; a no-wall known gap
    # contributes available time but is not an endpoint; unknown geometry
    # between (or on) endpoints is a barrier.
    transitions = []
    prev = None
    unknown_between = False
    for s in collapsed:
        if s["status"] == "unknown":
            unknown_between = True
            continue
        if not s["wall_ids"]:            # no-wall gap: available time only
            continue
        if prev is not None:
            tr = {"from_ids": list(prev["wall_ids"]), "to_ids": list(s["wall_ids"]),
                  "from_end_s": prev["end_s"], "to_start_s": s["start_s"],
                  "gap_s": s["start_s"] - prev["end_s"]}
            if unknown_between:
                tr.update(distance_lane=None, speed_lane_s=None,
                          status="skipped", reason="unknown_geometry")
            elif not prev["safe_intervals"] or not s["safe_intervals"]:
                tr.update(distance_lane=None, speed_lane_s=None,
                          status="skipped", reason="empty_endpoint")
            else:
                dist = _set_distance(prev["safe_intervals"], s["safe_intervals"])
                avail = max(0.0, s["start_s"] - prev["end_s"])
                if dist <= EPS:
                    tr.update(distance_lane=0.0, speed_lane_s=0.0,
                              status="ok", reason="zero_distance")
                elif avail > EPS:
                    tr.update(distance_lane=dist, speed_lane_s=dist / avail,
                              status="ok", reason="lower_bound")
                else:
                    tr.update(distance_lane=dist, speed_lane_s=None,
                              status="discontinuous",
                              reason="discontinuous_model_constraint")
            transitions.append(tr)
        prev = s
        unknown_between = False

    def _dur(seg):
        return seg["end_s"] - seg["start_s"]

    counts = {
        "supported_walls": len(walls),
        "unknown_objects": n_unknown,
        "invalid_objects": len(invalid),
        "known_s": sum(_dur(s) for s in collapsed if s["status"] == "known"),
        "unknown_s": sum(_dur(s) for s in collapsed if s["status"] == "unknown"),
        "empty_model_s": sum(_dur(s) for s in collapsed
                             if s["status"] == "empty_model"),
        "segments": len(collapsed),
        "transitions": sum(1 for t in transitions if t["status"] != "skipped"),
        "positive_distance_pairs": sum(1 for t in transitions
                                       if t["distance_lane"] not in (None, 0.0)),
    }
    return {"scope": scope, "radius": r, "segments": collapsed,
            "transitions": transitions, "counts": counts, "chart_unknown": False}
