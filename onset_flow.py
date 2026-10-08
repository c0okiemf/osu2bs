"""Osu owns onset times/density; human Beat Saber maps teach hands and geometry."""
import bisect
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from groom import CTX, Flow, MAX_CHAIN
from parity import HandParity, family
from learned_geometry import sample_geometry

SCHEMA = 'onset-hands-v1-exact-ms-density2s-absolute-geometry'
HAND_SIZE = 20                         # dir9 + col4 + layer3 + count3 + known1
INPUT_SIZE = 2 * HAND_SIZE + 4 + 8 + 3


def onehot(value, size):
    return [float(i == value) for i in range(size)]


def hand_features(cut):
    if cut is None:
        return [0.] * HAND_SIZE
    d, c, l, k = cut
    return onehot(d, 9) + onehot(c, 4) + onehot(l, 3) + onehot(k, 3) + [1.]


def source_onsets(objects, timing, bpm):
    """Exact heads and positive-duration slider ends; spinners are rests.

    Simultaneous source objects share a timestamp, but their identities remain
    in the ledger. Multiplicity/hand choice belongs to the Beat Saber model.
    """
    points = sorted(timing or [(0., 60000. / bpm)])
    if any(not math.isfinite(t) or not math.isfinite(bl) or bl <= 0 for t, bl in points):
        raise ValueError('invalid source tempo')
    starts = [t for t, _ in points]
    grouped = {}
    for i, obj in enumerate(objects):
        if obj['kind'] == 'spinner':
            continue
        if obj['kind'] not in ('circle', 'slider'):
            raise ValueError('unsupported source object')
        events = [('head', obj['t'])]
        if obj['kind'] == 'slider':
            if obj['end_t'] < obj['t']:
                raise ValueError('slider ends before its head')
            if obj['end_t'] > obj['t']:
                events.append(('tail', obj['end_t']))
        for role, t in events:
            if not math.isfinite(t) or t < 0:
                raise ValueError('invalid source timestamp')
            grouped.setdefault(float(t), []).append(dict(object=i, role=role))
    result = []
    for t, sources in sorted(grouped.items()):
        start, beat_ms = points[max(0, bisect.bisect_right(starts, t)-1)]
        result.append(dict(t=t, beat=(t-start)/beat_ms, beat_ms=beat_ms, sources=sources))
    return result


def onset_context(onsets):
    """Read-only upstream features, identical for training and generation."""
    times = [r['t'] for r in onsets]
    if any(not math.isfinite(t) or (i and t <= times[i-1]) for i, t in enumerate(times)):
        raise ValueError('onsets must have distinct, increasing, finite times')
    result = []
    for i, r in enumerate(onsets):
        t, beat, period = r['t'], r['beat'], r['beat_ms']
        if not math.isfinite(beat) or not math.isfinite(period) or period <= 0:
            raise ValueError('invalid onset beat context')
        # Density uses occupied times, never target hands, stacks or audio RMS.
        count = bisect.bisect_right(times, t+1000.) - bisect.bisect_left(times, t-1000.)
        gap = times[i+1]-t if i+1 < len(times) else 0.
        result.append([math.log1p(gap/250.), float(i+1 < len(times)),
                       math.log1p(period/250.), math.log1p(count/2.),
                       math.sin(2*math.pi*beat), math.cos(2*math.pi*beat),
                       math.sin(math.pi*beat/2), math.cos(math.pi*beat/2)])
    return result


def token(context, time, state, last_times, previous_mode, previous_time):
    gaps = [0. if t is None else math.log1p((time-t)/250.)
            for t in (*last_times, previous_time)]
    return (hand_features(state[0]) + hand_features(state[1])
            + onehot(previous_mode, 4) + context + gaps)


def training_sequence(events, grid):
    groups = {}
    for s, h, d, c, l, k in events:
        if h in groups.setdefault(s, {}):
            raise ValueError('multiple heads for one hand at an onset')
        if not (h in (0, 1) and 0 <= d < 9 and 0 <= c < 4 and 0 <= l < 3 and 1 <= k <= MAX_CHAIN):
            raise ValueError('invalid human-map target')
        groups[s][h] = (d, c, l, k-1)
    onsets = [dict(t=grid.time(s), beat=s/4., beat_ms=4*grid.local_dt(s)) for s in sorted(groups)]
    contexts = onset_context(onsets)
    state = [None, None]; last_times = [None, None]; previous_mode = 3; previous_time = None
    xs, ys = [], []
    for r, ctx, s in zip(onsets, contexts, sorted(groups)):
        group = groups[s]
        mode = 2 if len(group) == 2 else next(iter(group))
        xs.append(token(ctx, r['t'], state, last_times, previous_mode, previous_time))
        ys.append([mode] + list(group.get(0, (-100,)*4)) + list(group.get(1, (-100,)*4)))
        for h, cut in group.items():
            state[h] = cut; last_times[h] = r['t']
        previous_mode, previous_time = mode, r['t']
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


class OnsetFlow(Flow):
    """One causal token per onset; hand choice is an output, never an input label."""
    def __init__(self, width=256):
        super().__init__(width=width)
        self.inp = nn.Linear(INPUT_SIZE, width)
        self.mode_head = nn.Linear(width, 3)
        n = width + 3 + 2 + HAND_SIZE
        self.dir_head = nn.Linear(n, 9)
        self.col_head = nn.Linear(n+9, 4)
        self.lay_head = nn.Linear(n+9+4, 3)
        self.chain_head = nn.Linear(n+9, MAX_CHAIN)

    def geometry_hidden(self, hidden, mode, hand, other=None):
        extra = onehot(mode, 3) + onehot(hand, 2) + hand_features(other)
        return torch.cat([hidden, hidden.new_tensor(extra)], -1)

    def forward(self, x, y):
        h = self.hidden(x)
        mode = F.one_hot(y[..., 0].clamp_min(0), 3).float()
        cuts = y[..., 1:].reshape(*y.shape[:-1], 2, 4)
        left = cuts[..., 0, :].clamp_min(0)
        other = torch.cat([F.one_hot(left[..., j], size).float()
                           for j, size in enumerate((9, 4, 3, 3))]
                          + [torch.ones_like(left[..., :1], dtype=h.dtype)], -1)
        other = other * (y[..., :1] == 2)
        logits = []
        for hand in (0, 1):
            hand_id = h.new_tensor(onehot(hand, 2)).expand(*h.shape[:-1], 2)
            gh = torch.cat([h, mode, hand_id, other if hand else torch.zeros_like(other)], -1)
            d = F.one_hot(cuts[..., hand, 0].clamp_min(0), 9).float()
            c = F.one_hot(cuts[..., hand, 1].clamp_min(0), 4).float()
            logits.append((self.dir_head(gh), self.col_head(torch.cat([gh, d], -1)),
                           self.lay_head(torch.cat([gh, d, c], -1)),
                           self.chain_head(torch.cat([gh, d], -1))))
        return self.mode_head(h), [torch.stack([a[i] for a in logits], -2) for i in range(4)]


def loss(model, x, y):
    modes, cuts = model(x, y)
    result = F.cross_entropy(modes.reshape(-1, 3), y[..., 0].reshape(-1), ignore_index=-100)
    targets = y[..., 1:].reshape(*y.shape[:-1], 2, 4)
    for j, logits in enumerate(cuts):
        result = result + F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                         targets[..., j].reshape(-1), ignore_index=-100)
    return result


def save_model(path, model):
    torch.save(dict(schema=SCHEMA, width=model.width, state=model.state_dict()), path)


def load_model(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get('schema') != SCHEMA:
        raise ValueError('incompatible onset-flow checkpoint')
    model = OnsetFlow(checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['state']); model.eval()
    return model


@torch.inference_mode()
def decode(onsets, model, seed=0, temperature=1.):
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError('invalid temperature')
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    contexts = onset_context(onsets)
    state = [None, None]; last_times = [None, None]; parity = [HandParity(), HandParity()]
    previous_mode = 3; previous_time = None; tokens = []; notes = []
    for r, ctx in zip(onsets, contexts):
        t = r['t']
        tokens.append(token(ctx, t, state, last_times, previous_mode, previous_time))
        x = torch.tensor(tokens[-CTX:], dtype=torch.float32, device=device)
        hidden = model.hidden(x[None])[0, -1]
        mode = int(torch.multinomial(torch.softmax(model.mode_head(hidden)/temperature, -1), 1, generator=gen))
        hands = (0, 1) if mode == 2 else (mode,)
        group = []; first = None
        for hand in hands:
            required = parity[hand].required(t)
            allowed = [d for d in range(8) if required is None or family(d) in (required, 'lateral')]
            gh = model.geometry_hidden(hidden, mode, hand, first)
            d, c, l, k = sample_geometry(model, gh, hand, t, (0, 0), allowed,
                                         group, set(), gen, temperature, relative=False)
            from parity import DIR_VEC
            dx, dy = DIR_VEC[d]
            group.extend((t, hand, c+j*dx, l+j*dy, d if j == 0 else 8) for j in range(k))
            state[hand] = (d, c, l, k-1); last_times[hand] = t
            parity[hand].commit(d, t)
            first = state[hand]
        notes.extend(dict(t=t, hand=h, col=c, layer=l, dir=d) for t, h, c, l, d in group)
        previous_mode, previous_time = mode, t
    if {n['t'] for n in notes} != {r['t'] for r in onsets}:
        raise ValueError('source onset times changed')
    return notes
