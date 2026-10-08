"""Independent-QA Task 3: conservative geometric checks (spec §4).

A declared geometric MODEL, never a biomechanics simulator. Everything is
relaxed into containing balls so that every reported bound is a sound
LOWER bound and every infeasibility is a witness FOR THE SCENARIO ONLY:

- pinned grid convention: x = (line-1.5)*0.6, y = 0.6+layer*0.6, z = 0;
  note half-size 0.25 -> containing ball r = 0.25*sqrt(3);
- admissible grip volume per note = ball(note, note_r + saber 1.0) ∩
  ball(shoulder, 0.40h + torso ball 0.47) per height scenario
  {1.4, 1.7, 2.0} and hand (shoulder x = ±0.12h, y = 0.82h);
- grip-speed lower bound between successive same-hand notes =
  ball-set distance / dt (zero when the relaxation is too loose — that is
  the honest answer, not a fabricated demand);
- full-height wall corridor: intervals where every stance column is
  simultaneously blocked are empty-corridor witnesses; crouch/custom walls
  contaminate their intervals as UNKNOWN;
- speed warnings compare bounds against the QA-train p99.5 filtered
  controller speed (an empirical warning threshold, not a biological
  maximum) and route to review, never a universal HARD_FAIL;
- no unavoidable-collision claims in v1 beyond structural overlap: one IK
  solution intersecting a body is not a certificate.

No motion-minimization reward. No Q1 repair. Scenario constants are design
decisions, recorded, not inferred anatomy.
"""
import math

NOTE_SPACING = 0.6
NOTE_BALL_R = 0.25 * math.sqrt(3)
SABER_LEN = 1.0
TORSO_BALL_R = 0.47                # contains ±0.30 horiz / ±0.20 vert box
SCENARIOS = ("1.4", "1.7", "2.0")
DEFAULT_SPEED_WARN = 12.0          # replaced by frozen QA-train p99.5


def note_center(line_index, line_layer):
    return ((line_index - 1.5) * NOTE_SPACING,
            NOTE_SPACING + line_layer * NOTE_SPACING, 0.0)


def _shoulder(height, hand):
    h = float(height)
    return ((-0.12 * h) if hand == "left" else (0.12 * h), 0.82 * h, 0.0)


def ball_set_distance(c1, r1, c2, r2):
    return max(0.0, math.dist(c1, c2) - r1 - r2)


def reach_feasible(center, scenario, hand):
    """Nonempty admissible grip volume under the relaxation — NOT proof of
    playability; emptiness is an infeasibility witness for the scenario."""
    h = float(scenario)
    sh = _shoulder(h, hand)
    grip_r = NOTE_BALL_R + SABER_LEN
    reach_r = 0.40 * h + TORSO_BALL_R
    return math.dist(center, sh) <= grip_r + reach_r


def grip_speed_lower_bound(c1, c2, dt):
    """Sound lower bound on grip speed between two note constraints."""
    d = ball_set_distance(c1, NOTE_BALL_R + SABER_LEN,
                          c2, NOTE_BALL_R + SABER_LEN)
    return d / max(dt, 1e-6)


def _corridor(walls, t_lo, t_hi):
    """Full-height wall occupancy sweep. walls: (t, line, type, dur, width).
    type 0 = full height (supported); anything else contaminates."""
    empty, unknown = [], []
    events = []
    for t, li, typ, dur, w in walls:
        if typ != 0:
            unknown.append([t, t + dur])
            continue
        try:
            cols = set(range(int(li), int(li) + int(w)))
        except (TypeError, ValueError):
            unknown.append([t, t + dur])
            continue
        events.append((t, t + dur, cols))
    ts = sorted({e for w in events for e in (w[0], w[1])})
    for a, b in zip(ts, ts[1:]):
        mid = (a + b) / 2
        blocked = set()
        for t0, t1, cols in events:
            if t0 <= mid < t1:
                blocked |= cols
        if blocked >= {0, 1, 2, 3}:
            empty.append([a, b])
    return {"empty_intervals": empty, "unknown_intervals": unknown}


def physics(scene, speed_warn=DEFAULT_SPEED_WARN):
    """PhysicsReport for one supported scene."""
    if scene.get("scope") is not None:
        return {"status": "unknown", "reason": scene["scope"]}
    notes = sorted(scene["notes"])
    per_scenario = {}
    for sc in SCENARIOS:
        infeasible = []
        for i, (t, li, ll, c, d) in enumerate(notes):
            hand = "right" if c == 1 else "left"
            if not reach_feasible(note_center(li, ll), sc, hand):
                infeasible.append({"index": i, "t": t, "hand": hand})
        per_scenario[sc] = {"reach_infeasible": infeasible}
    # grip-speed lower bounds per hand (relative times: translation-safe)
    last = {}
    bounds, violations = [], []
    for i, (t, li, ll, c, d) in enumerate(notes):
        p = last.get(c)
        if p is not None:
            pt, pc = p
            dt = round(t - pt, 9)
            lb = grip_speed_lower_bound(pc, note_center(li, ll), dt)
            if lb > 0:
                bounds.append({"index": i, "dt": dt,
                               "lower_bound": round(lb, 3)})
            if lb > speed_warn:
                violations.append({"index": i, "dt": dt,
                                   "lower_bound": round(lb, 3),
                                   "threshold": speed_warn,
                                   "status": "model_conditional_warning"})
        last[c] = (t, note_center(li, ll))
    ts = [t for t, *_ in notes]
    corridor = _corridor(scene.get("walls") or [],
                         min(ts, default=0), max(ts, default=0))
    return {"status": "ok", "scenarios": per_scenario,
            "grip_speed": {
                "max_lower_bound": max((b["lower_bound"] for b in bounds),
                                       default=0.0),
                "nonzero_bounds": bounds, "violations": violations,
                "threshold": speed_warn,
                "threshold_kind": "empirical_warning_not_biological_max"},
            "corridor": corridor,
            "unavoidable_collision": False,
            "limits": "containing-ball relaxation; scenario-conditional; "
                      "no orientation demand from arrows; no fatigue score"}


def train_speed_threshold():
    """Frozen empirical warning threshold: QA-train p99.5 filtered peak
    speed over supported good windows, per height stratum + pooled."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    speeds = {"lt160": [], "160to180": [], "gt180": [], "unknown": []}
    for wp in (root / "experiments" / "qa-v1-train").rglob(
            "*.windows_v2.json"):
        rec = json.loads(wp.read_text())
        for w in rec["windows"]:
            if not w.get("supported") or w.get("outcome") != "good":
                continue
            v = w.get("filtered_peak_speed")
            if v is None:
                continue
            speeds["unknown"].append(v)     # stratum via replay height TODO
    pooled = sorted(speeds["unknown"])
    if not pooled:
        return None
    p995 = pooled[min(len(pooled) - 1, int(0.995 * len(pooled)))]
    out = {"pooled_p995": p995, "n": len(pooled),
           "source": "qa-train supported good windows, filtered peaks"}
    p = root / "experiments" / "qa-v1" / "speed_threshold.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(p)
    print(f"speed threshold p99.5 = {p995:.2f} m/s over {len(pooled)} windows")
    return out


if __name__ == "__main__":
    train_speed_threshold()
