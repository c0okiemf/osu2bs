"""Two-saber parity state machine: forehand/backhand alternation per hand.

Standalone so it can later double as a logit mask at sampling time (Phase 4).

BS cut directions: 0=up 1=down 2=left 3=right 4=upLeft 5=upRight 6=downLeft 7=downRight 8=dot
"""

FOREHAND = frozenset({1, 6, 7})   # downward swings
BACKHAND = frozenset({0, 4, 5})   # upward swings
LATERAL = frozenset({2, 3})       # legal under either parity

# fixed reset window; tune if slow sections feel stiff
RESET_MS = 1000  # gap after which a hand may start with either parity

# direction -> angle (degrees, screen coords: 0=right, 90=down)
_ANGLE = {3: 0, 7: 45, 1: 90, 6: 135, 2: 180, 4: 225, 0: 270, 5: 315}

# direction -> grid vector (col axis, layer axis)
DIR_VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0),
           4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1), 8: (0, 0)}


def is_hammer(a_col, a_lay, b_col, b_lay, d):
    """Same-time red+blue, equal direction d, positions in line with d
    (like '>>'): one saber swings through the other's note — unhittable."""
    if d == 8:
        return False
    vx, vy = DIR_VEC[d]
    dx, dy = b_col - a_col, b_lay - a_lay
    return dx * vy == dy * vx


def ang_dist(a, b):
    """Angular distance in degrees between two cut directions (dots -> 0)."""
    if a == 8 or b == 8:
        return 0
    d = abs(_ANGLE[a] - _ANGLE[b]) % 360
    return min(d, 360 - d)


_ang_dist = ang_dist


def family(direction):
    if direction in FOREHAND:
        return "fore"
    if direction in BACKHAND:
        return "back"
    return "lateral"


class HandParity:
    """Feed desired cut directions in time order; get parity-legal ones back."""

    def __init__(self):
        self.last_family = None  # "fore" | "back" | None
        self.last_time = None

    def required(self, time_ms):
        """Parity the next swing must have: 'fore', 'back', or None (reset)."""
        if self.last_family is not None and time_ms - self.last_time < RESET_MS:
            return "back" if self.last_family == "fore" else "fore"
        return None

    def commit(self, direction, time_ms):
        """Record an emitted swing. Laterals/dots flip state like any swing."""
        fam = family(direction)
        if fam == "lateral":
            fam = (self.required(time_ms)
                   or ("back" if self.last_family == "fore" else "fore"))
        self.last_family = fam
        self.last_time = time_ms

    def next_direction(self, desired, time_ms):
        if desired == 8:
            desired = 1  # dots resolved to down; we don't emit dots
        req = self.required(time_ms)
        fam = family(desired)
        if req is None or fam == req or fam == "lateral":
            out = desired
        else:
            pool = FOREHAND if req == "fore" else BACKHAND
            out = min(pool, key=lambda d: _ang_dist(d, desired))
        self.commit(out, time_ms)
        return out


def violations(notes):
    """Count parity violations in emitted notes: [(time_ms, hand, direction), ...].
    Two consecutive same-family (fore/back) cuts on one hand within RESET_MS = violation.
    """
    state = {}  # hand -> (family, time)
    bad = 0
    for t, hand, d in sorted(notes):
        if d == 8:
            continue  # dots are chain followers / ambiguous: not a swing
        fam = family(d)
        prev = state.get(hand)
        if prev and fam != "lateral" and t - prev[1] < RESET_MS and fam == prev[0]:
            bad += 1
        if fam != "lateral":
            state[hand] = (fam, t)
        elif prev:
            state[hand] = ("back" if prev[0] == "fore" else "fore", t)
        else:
            state[hand] = ("fore", t)  # matches HandParity's first-lateral state
    return bad
