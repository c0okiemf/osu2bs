"""Experimental literal joint-event state and structural sampling masks.

No production positional/parity policy. A valid event is not proof of good flow.
"""
import math

from eval.joint_phrase import UnsupportedSource, _number, decode_events, encode_events


class JointState:
    def __init__(self, duration_beats):
        self.duration = _number(duration_beats, "duration", 0)
        self.cursor = 0.0
        self.events = []
        self.last_by_hand = [None, None]

    def rest(self, beat):
        _number(beat, "rest_time", 0)
        if not self.cursor <= beat <= self.duration:
            raise UnsupportedSource("invalid_rest_advance")
        self.cursor = beat

    def append(self, event):
        # Validate before mutating state; copy nested lists from caller.
        canonical = encode_events(decode_events([event]))[0]
        beat = canonical["beat"]
        if beat < self.cursor or beat >= self.duration \
                or (self.events and beat <= self.events[-1]["beat"]):
            raise UnsupportedSource("invalid_event_advance")
        self.events.append(canonical)
        self.cursor = beat
        for hand, notes in enumerate(canonical["hands"]):
            if notes:
                self.last_by_hand[hand] = {"beat": beat, "notes": tuple(notes)}
        return canonical

    def notes(self):
        return decode_events(self.events)


def sample_event(beat, count_logits, slot_logits, generator, temperature=1.0):
    """Sample a timestamp event with joint counts and conditional literal slots.

    slot_logits(hand, slot, emitted_hands) sees earlier sampled notes, including
    the other hand. It must not mutate emitted_hands. No shortening or repair.
    """
    import torch
    _number(beat, "event_time", 0)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    def choose(logits, allowed, n):
        logits = torch.as_tensor(logits).detach().float().cpu()
        if logits.shape != (n,) or not torch.isfinite(logits).all():
            raise UnsupportedSource("invalid_logits")
        if not allowed:
            raise UnsupportedSource("infeasible_event")
        indices = torch.tensor(allowed)
        weights = torch.softmax(logits[indices] / temperature, 0)
        return int(indices[torch.multinomial(weights, 1, generator=generator)])

    count = choose(count_logits, list(range(1, 16)), 16)
    counts = [count // 4, count % 4]
    hands, occupied = [[], []], set()
    for hand, total in enumerate(counts):
        last_cell = -1
        for slot in range(total):
            allowed = [cell * 9 + d for cell in range(last_cell + 1, 12)
                       if cell not in occupied for d in range(9)]
            # Reserve enough higher cells to realize the sampled count.
            cells = sorted({token // 9 for token in allowed})
            viable = set(cells[:len(cells) - (total - slot - 1)]) \
                if len(cells) >= total - slot else set()
            allowed = [token for token in allowed if token // 9 in viable]
            frozen = tuple(tuple(notes) for notes in hands)
            token = choose(slot_logits(hand, slot, frozen), allowed, 108)
            cell, direction = divmod(token, 9)
            col, layer = divmod(cell, 3)
            hands[hand].append((col, layer, direction))
            occupied.add(cell)
            last_cell = cell
    event = {"beat": beat, "hands": hands}
    decode_events([event])
    return event
