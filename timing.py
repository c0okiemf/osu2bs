"""Explicit timestamp grid for the 1/4-beat schedule (Packet C, phase 2A).

The decoder indexes notes by integer step. A single global BPM misgrids any
song whose tempo changes (SPACEMAN 178->160, Reality Check 130->260): hits
snap to the wrong lines and beat-phase features drift against the music.

TimeGrid owns, per integer step:
- `times[s]`  absolute ms (authoritative; nps/motion/replay/gaps all read it)
- `pos[s]`    position-in-beat 0..3  (local to the step's tempo segment)
- `bar[s]`    beat-in-bar 0..3       (local, resets at a segment boundary)

Within one tempo segment the grid is uniform 1/4-beat, so a single-tempo
song is byte-identical to the old `offset + s*step_ms` / `s%4` arithmetic
(asserted in the self-check). Only tempo boundaries differ — and there the
old global grid was simply wrong. The model's input featurization is
unchanged for single-tempo maps (all training data), so no retraining is
implied; multi-tempo inference just gets correct LOCAL phase.

Export stays single-BPM (seconds encoded as export beats); the analyzed
segment map is preserved in generation metadata, never in the game file.

Self-check: .venv/bin/python timing.py
"""
import bisect

STEPS_PER_BEAT = 4
TEMPO_TOL = 0.08  # Mapperatorinator's auto-timing wobbles a few % between
                  # sections (numb 110/110.05/112, SW 182-197 around 192);
                  # those are noise, not tempo changes. Only a shift past
                  # this ratio (SPACEMAN ~10%, Reality Check 130->260 = 2x)
                  # starts a new grid segment. Below it, one grid — identical
                  # to the old single-global-BPM behavior.
PHASE_EPS_MS = 5.0  # a point that re-anchors the grid off the previous
                    # segment's 1/4 lines by more than this is a genuine
                    # phase reset (an offset change even at the same BPM),
                    # kept as a candidate; an on-grid redundant point is not.


def merge_tempo(uninherited, tol=TEMPO_TOL):
    """Collapse consecutive uninherited points that neither change tempo past
    tol NOR re-anchor the grid phase; keep genuine tempo AND offset changes.
    The snap-error adoption gate is the backstop, so a kept candidate that is
    actually noise is simply rejected downstream. First point always kept."""
    out = []
    for t, bl in sorted(uninherited):
        if not out:
            out.append((t, bl))
            continue
        pt, pbl = out[-1]
        step = pbl / STEPS_PER_BEAT
        # distance from t to the nearest previous-segment 1/4 line
        phase_off = abs((((t - pt) / step + 0.5) % 1.0) - 0.5) * step
        if abs(bl / pbl - 1.0) > tol or phase_off > PHASE_EPS_MS:
            out.append((t, bl))
    return out


class TimeGrid:
    def __init__(self, times, pos, bar, segments=None):
        assert len(times) == len(pos) == len(bar)
        self.times = times          # absolute ms per step
        self.pos = pos              # 0..3 position in beat (local)
        self.bar = bar              # 0..3 beat in bar (local)
        self.segments = segments or []  # [(start_ms, beat_len_ms)] analyzed
        self.n = len(times)

    def time(self, s):
        if s < 0:
            return self.times[0] + s * self._dt0()
        if s >= self.n:
            return self.times[-1] + (s - self.n + 1) * self._dt_last()
        return self.times[s]

    def _dt0(self):
        return self.times[1] - self.times[0] if self.n > 1 else 0.0

    def _dt_last(self):
        return self.times[-1] - self.times[-2] if self.n > 1 else 0.0

    def gap_ms(self, a, b):
        """Absolute ms between two step indices (b later)."""
        return self.time(b) - self.time(a)

    def local_dt(self, s):
        """Local 1/4-beat spacing (ms) around step s — for physical-time
        thresholds expressed as 'one step'."""
        s = max(0, min(s, self.n - 1))
        if s + 1 < self.n:
            return self.times[s + 1] - self.times[s]
        return self._dt_last()

    def snap(self, ms):
        """Nearest step to an absolute time; (step, abs_error_ms)."""
        i = bisect.bisect_left(self.times, ms)
        best, err = 0, float("inf")
        for j in (i - 1, i, i + 1):
            if 0 <= j < self.n:
                e = abs(self.times[j] - ms)
                if e < err:
                    best, err = j, e
        return best, err

    @classmethod
    def from_beatmap(cls, bpm, raw, shift, T):
        """Clock for supported v2/v3 Beat Saber notes, including native BPM events.

        Editor-only _customData BPM markers do not change gameplay time. Native
        v2 type-100 and v3 bpmEvents do (BSMG beatmap-format BPM Events).
        shift is the loader's constant phase in quarter-beat steps.
        """
        import math
        points = [(0.0, float(bpm))]
        changes = ([(e.get('b', 0), e['m']) for e in raw.get('bpmEvents', [])]
                   + [(e['_time'], e['_floatValue']) for e in raw.get('_events', [])
                      if e.get('_type') == 100])
        by_beat = {}
        for beat, rate in changes:
            beat, rate = float(beat), float(rate)
            if beat in by_beat and by_beat[beat] != rate:
                raise ValueError("conflicting beatmap BPM events")
            by_beat[beat] = rate
        if 0.0 in by_beat:
            points[0] = (0.0, by_beat.pop(0.0))
        points.extend(sorted(by_beat.items()))
        if any(not math.isfinite(b) or not math.isfinite(r) or b < 0 or r <= 0
               for b, r in points):
            raise ValueError("invalid beatmap BPM")
        starts = [b for b, _ in points]
        elapsed = [0.0]
        for (a, rate), (b, _) in zip(points, points[1:]):
            elapsed.append(elapsed[-1] + (b-a)*60000.0/rate)
        def at(beat):
            i = max(0, bisect.bisect_right(starts, beat)-1)
            return elapsed[i] + (beat-starts[i])*60000.0/points[i][1]
        return cls([at((s+shift)/STEPS_PER_BEAT) for s in range(T)],
                   [s % 4 for s in range(T)], [(s//4) % 4 for s in range(T)])

    @classmethod
    def uniform(cls, step_ms, offset, T):
        """The legacy single-tempo grid — kept identical for self-checks and
        every single-BPM song."""
        times = [offset + s * step_ms for s in range(T)]
        pos = [s % 4 for s in range(T)]
        bar = [(s // 4) % 4 for s in range(T)]
        return cls(times, pos, bar, [(offset, step_ms * 4)])

    @classmethod
    def covering(cls, uninherited, offset0, end_ms, step0_ms):
        """Build enough steps to cover [offset0, end_ms] (+1 beat margin).
        Single/no tempo point -> uniform. Used by the converter where the
        step count follows the audio, not a fixed T."""
        if len(uninherited) <= 1:
            n = int((end_ms - offset0) / step0_ms) + 5
            return cls.uniform(step0_ms, offset0, max(n, 1))
        segs = [(t, bl / STEPS_PER_BEAT) for t, bl in uninherited]
        times, pos, bar = [], [], []
        t, seg_i, beat_ct, sub = offset0, 0, 0, 0
        limit = end_ms + segs[-1][1] * STEPS_PER_BEAT  # one beat past the end
        while t <= limit:
            while seg_i + 1 < len(segs) and t >= segs[seg_i + 1][0] - 1e-6:
                t = segs[seg_i + 1][0]
                seg_i += 1
                beat_ct, sub = 0, 0
            times.append(t)
            pos.append(sub)
            bar.append(beat_ct % 4)
            t += segs[seg_i][1]
            sub = (sub + 1) % STEPS_PER_BEAT
            if sub == 0:
                beat_ct += 1
        return cls(times, pos, bar, segs)

    @classmethod
    def from_timing(cls, uninherited, offset0, T, step0_ms):
        """uninherited: sorted [(time_ms, beat_len_ms)] tempo points (osu
        uninherited timing). offset0: ms of grid step 0 (may be < first
        point after the lead extension). step0_ms: 1/4 spacing of the first
        segment (fallback for steps before the first point)."""
        if len(uninherited) <= 1:
            return cls.uniform(step0_ms, offset0, T)
        # segment boundaries in ms, each with its 1/4 spacing
        segs = [(t, bl / STEPS_PER_BEAT) for t, bl in uninherited]
        times, pos, bar = [], [], []
        t = offset0
        seg_i = 0
        # local phase counters, reset when a boundary is crossed
        beat_ct = 0
        sub = 0
        for s in range(T):
            # advance segment if we've reached/passed the next boundary
            while seg_i + 1 < len(segs) and t >= segs[seg_i + 1][0] - 1e-6:
                # re-anchor exactly to the boundary and reset local phase
                t = segs[seg_i + 1][0]
                seg_i += 1
                beat_ct, sub = 0, 0
            times.append(t)
            pos.append(sub)
            bar.append(beat_ct % 4)
            t += segs[seg_i][1]
            sub += 1
            if sub == STEPS_PER_BEAT:
                sub = 0
                beat_ct += 1
        return cls(times, pos, bar, segs)


def _selfcheck():
    # uniform grid reproduces the old arithmetic exactly
    g = TimeGrid.uniform(125.0, 40.0, 20)
    assert all(g.time(s) == 40.0 + s * 125.0 for s in range(20))
    assert [g.pos[s] for s in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert [g.bar[s] for s in range(20)] == [(s // 4) % 4 for s in range(20)]
    # from_timing with a single point == uniform
    g1 = TimeGrid.from_timing([(0.0, 500.0)], 0.0, 12, 125.0)
    assert all(g1.time(s) == s * 125.0 for s in range(12))
    # two segments: 120bpm (500ms/beat -> 125ms/step) then 240bpm at 1000ms
    # (250ms/beat -> 62.5ms/step). Boundary re-anchors and phase resets.
    g2 = TimeGrid.from_timing([(0.0, 500.0), (1000.0, 250.0)], 0.0, 14, 125.0)
    assert g2.time(0) == 0.0 and g2.time(8) == 1000.0
    assert g2.pos[8] == 0 and g2.bar[8] == 0, (g2.pos[8], g2.bar[8])
    assert abs(g2.time(9) - 1062.5) < 1e-9, g2.time(9)
    # snapping reports sub-ms error on-grid, real error off-grid
    st, err = g2.snap(1000.0)
    assert st == 8 and err < 1e-6, (st, err)
    st, err = g2.snap(1030.0)
    assert st == 8 and abs(err - 30.0) < 1e-6, (st, err)
    # 130 vs 260 BPM quarter grids are physically different spacings
    a = TimeGrid.uniform(60000 / 130 / 4, 0.0, 8)
    b = TimeGrid.uniform(60000 / 260 / 4, 0.0, 8)
    assert abs(a.local_dt(0) - 2 * b.local_dt(0)) < 1e-9
    # export round-trip: absolute ms -> export beats -> ms is < 1ms for any
    # note (the fixed-export-BPM encoding of seconds Packet C keeps)
    beat = 60000 / 174.0
    for t in (0.0, 1234.5, 98765.4321):
        assert abs(round(t / beat, 5) * beat - t) < 1.0
    # merge_tempo: auto-timing wobble collapses, a real 2x split survives
    assert len(merge_tempo([(0, 500), (1000, 495), (2000, 505)])) == 1
    assert len(merge_tempo([(0, 500), (4000, 250)])) == 2
    # on-grid redundant point drops; an off-grid same-BPM phase reset is kept
    assert len(merge_tempo([(0, 500), (4000, 500)])) == 1     # 4000 on 125 grid
    assert len(merge_tempo([(0, 500), (4050, 500)])) == 2     # +50ms reset kept
    print("timing selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
