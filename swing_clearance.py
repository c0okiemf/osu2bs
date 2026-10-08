"""Simultaneous wrong-color approach obstruction, in grid coordinates.

A style constraint, not a physical reachability certificate: each directional
cut's approach ray extends behind its center across the grid; wrong-color blocks
are disks of radius half a grid spacing. Exact shared timestamps only. Dots may
block a directional cut, but their own approach is unknown. No inter-note-time,
arm, saber-to-saber, visibility or follow-through model is implied.
"""
from collections import defaultdict
from itertools import product
from math import hypot

from parity import DIR_VEC


def blocked_approaches(notes):
    """Return (time, target index, blocker index) for (time, hand, col, row, dir).

    Units of time are irrelevant. Check every cross-color pair, including chords
    with more than two heads, rather than permitting extra notes to bypass a ban.
    """
    groups = defaultdict(list)
    for i, note in enumerate(notes):
        groups[note[0]].append((i, note))
    blocked = []
    for t, group in groups.items():
        for i, (_, hand, col, row, direction) in group:
            if direction == 8:
                continue
            dx, dy = DIR_VEC[direction]
            norm = hypot(dx, dy)
            dx, dy = -dx / norm, -dy / norm
            for j, (_, other, x, y, _) in group:
                if other == hand:
                    continue
                x, y = x - col, y - row
                along = max(0., x * dx + y * dy)
                if (x - along * dx) ** 2 + (y - along * dy) ** 2 <= .25:
                    blocked.append((t, i, j))
    return blocked


def repair_approaches(notes, walls=()):
    """Translate obstructed simultaneous hand groups without changing their cuts.

    Keep times, colors, directions and complete chain shapes. Use the smallest
    squared displacement that respects the generator's grid/center/head-side
    restrictions and existing walls. Unplaceable groups remain invalid so the
    caller's normal candidate checks reject them; never drop notes to get a pass.
    Walls use (start, duration, column), in the same time units as notes.
    """
    out = list(notes); groups = defaultdict(list)
    stats = dict(events_changed=0, notes_moved=0, unresolved_events=0)
    for i, note in enumerate(out):
        groups[note[0]].append(i)
    for t, indices in groups.items():
        group = [out[i] for i in indices]
        if not blocked_approaches(group):
            continue
        forbidden = {col for start, duration, col in walls if start <= t < start + duration}
        options = []
        for hand in (0, 1):
            hand_notes = [n for n in group if n[1] == hand]
            choices = []
            for dx, dy in product(range(-3, 4), range(-2, 3)):
                moved = [(s, h, x+dx, y+dy, d) for s, h, x, y, d in hand_notes]
                if all(0 <= x <= 3 and 0 <= y <= 2 and x not in forbidden
                       and not (y == 1 and x in (1, 2))
                       and not (d != 8 and x == (3 if h == 0 else 0))
                       for _, h, x, y, d in moved):
                    choices.append(((dx*dx+dy*dy)*len(moved), dx, dy, moved))
            options.append(choices)
        best = None
        for left, right in product(*options):
            proposal = left[3] + right[3]
            if len({(n[2], n[3]) for n in proposal}) != len(proposal) or blocked_approaches(proposal):
                continue
            key = (left[0]+right[0], left[0], left[1:3], right[1:3])
            if best is None or key < best[0]:
                best = (key, (left[1:3], right[1:3]))
        if best is None:
            stats['unresolved_events'] += 1
            continue
        stats['events_changed'] += 1
        for i in indices:
            s, h, x, y, d = out[i]; dx, dy = best[1][h]
            out[i] = (s, h, x+dx, y+dy, d)
            stats['notes_moved'] += bool(dx or dy)
    return out, stats
