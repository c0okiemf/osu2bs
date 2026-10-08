"""Q2: complete-schedule geometry decoding on the schedule_in seam.

`EventSchedule` materializes the FINAL production winner's events — absolute
step/time, hand, head/follower counts, walls — bound to its source identity
(the Q1 immutable signature) and grid. Unlike threshold vectors, a schedule
fixes the actual events: two plans with equal density are not interchangeable.

`decode_geometry` re-runs groom_notes' geometry verbatim on that fixed
schedule with any flow checkpoint/seed (groom_notes(schedule_in=...)): every
scheduling decision is bypassed, chain draws are burned so the RNG stream
matches an unpinned decode, pinned follower counts are enforced, and a
proposal that cannot place them legally is REJECTED, never shortened.
Supplying the baseline's own model+seed reproduces the baseline notes
bit-identically (proven in eval/test_quality_policy fixtures).
"""
import json
from dataclasses import dataclass, field

from eval.quality_metrics import immutable_signature


@dataclass
class EventSchedule:
    """The complete event schedule of one decoded chart."""
    entries: dict                      # step -> {"hands": (h..), "k": {h: n}}
    walls: list                        # (s0, len, col) step runs
    T: int
    step_ms: float
    source_signature: str              # immutable signature of the source raw
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw, wall_runs, grid, T, step_ms, meta=None):
        entries = {}
        for s, h, _c, _l, d in raw:
            ent = entries.setdefault(int(s), {"hands": set(), "k": {}})
            if d != 8:
                ent["hands"].add(int(h))
                ent["k"].setdefault(int(h), 0)
            else:
                ent["k"][int(h)] = ent["k"].get(int(h), 0) + 1
        for ent in entries.values():
            # ascending hand order matches the production scheduler's emission
            ent["hands"] = tuple(sorted(ent["hands"]))
        return cls(entries=entries, walls=[tuple(w) for w in wall_runs],
                   T=int(T), step_ms=float(step_ms),
                   source_signature=immutable_signature(
                       raw, wall_runs, grid).hex(),
                   meta=dict(meta or {}))

    def to_json(self):
        return json.dumps({
            "entries": {str(s): {"hands": list(e["hands"]),
                                 "k": {str(h): n for h, n in e["k"].items()}}
                        for s, e in sorted(self.entries.items())},
            "walls": [list(w) for w in self.walls], "T": self.T,
            "step_ms": self.step_ms,
            "source_signature": self.source_signature,
            "meta": self.meta}, sort_keys=True)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls(entries={int(k): {"hands": tuple(v["hands"]),
                                     "k": {int(h): n
                                           for h, n in v["k"].items()}}
                            for k, v in d["entries"].items()},
                   walls=[tuple(w) for w in d["walls"]], T=d["T"],
                   step_ms=d["step_ms"],
                   source_signature=d["source_signature"], meta=d["meta"])

    def signature_of(self, raw, wall_runs, grid):
        """A decode honored this schedule iff the emitted chart's immutable
        signature equals the source's (same events/times/hands/roles/walls)."""
        return immutable_signature(raw, wall_runs, grid).hex() \
            == self.source_signature


@dataclass
class GeometryProposal:
    raw: list
    walls: list
    checkpoint: str
    seed: int
    infeasible: int                    # pinned followers that could not place
    honored: bool                      # signature matched the schedule source
    trace: dict


def decode_geometry(schedule, steps, T, step_ms, offset, grid, afeat, spec,
                    *, flow_model=None, rhythm_model=None, seed=0, temp=None,
                    checkpoint="shipped", mask=True):
    """One full-song geometry rollout on the fixed schedule. Returns a
    GeometryProposal; `honored=False` or `infeasible>0` proposals must be
    rejected by the caller, never exported or shortened."""
    import groom
    tr = {}
    sched_in = schedule.entries
    raw, walls = groom.groom_notes(
        steps, T, step_ms, offset, model=rhythm_model, flow=flow_model,
        afeat=afeat, seed=seed,
        temp=groom.TEMP if temp is None else temp,
        drate=spec["cond"], band_scale=spec["scale"], band=spec["band"],
        replay_mode="off", trace=tr, grid=grid, calibrate=True,
        schedule_in=sched_in, schedule_mask=mask)
    infeasible = int(tr.get("schedule_infeasible", 0))
    honored = schedule.signature_of(raw, walls, grid) and infeasible == 0
    return GeometryProposal(raw=raw, walls=walls, checkpoint=checkpoint,
                            seed=seed, infeasible=infeasible,
                            honored=honored, trace=tr)
