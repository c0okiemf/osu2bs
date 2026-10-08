"""Q2: explicit quality-first admission and selection over complete-schedule
geometry proposals.

Candidates: the original B0 winner, the Q1-repaired B0 winner, and geometry
rollouts from decode_geometry (each already Q1-repaired when the policy has Q1
enabled). Admission is gate-based and reported for EVERY candidate; selection
minimizes the spec J among admitted candidates, using the critic/preference
score only as a tie-break after motion equivalence (within 1 flag/1000 and
identical pair counts). An originally valid B0 always remains available, so no
new neutral infeasibility. No learned weights live in this module.
"""
from dataclasses import dataclass, field

import motion

POLICY_VERSION = "q2-v1"


@dataclass
class Candidate:
    cid: str                     # e.g. "b0", "b0+q1", "shipped:0+q1"
    raw: list
    walls: list
    rv: dict                     # quality_repair.risk_vector
    score: float = 0.0           # critic scale, tie-break only
    honored: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class SelectionResult:
    selected: Candidate
    admitted_ids: list
    rejections: dict             # cid -> [reasons] for every non-admitted
    accounting: dict


def _j(rv):
    from quality_repair import _j as jj
    return jj(rv)


def gates_vs_baseline(rv, base_rv, orig_rv=None):
    """Per-candidate admission gates against the B0 winner's risk vector.
    Returns failing gate names (empty = admissible)."""
    o = orig_rv or base_rv
    fails = []
    for e in motion.EXT_VARIANTS:
        if rv["flags"][e] > base_rv["flags"][e]:
            fails.append(f"flags@{e}")
    for k in ("narrow", "broad", "converging", "opposite",
              "offaxis", "reverse", "unknown_dots"):
        if rv[k] > base_rv[k]:
            fails.append(k)
    if rv["p95"] > 1.05 * o["p95"] + 1e-9:
        fails.append("p95")
    st, so = rv["style"], o["style"]
    if st["vert"] > max(0.80, so["vert"] + 1e-9) \
            or st["longrun"] > max(0.55, so["longrun"] + 1e-9) \
            or st["lat"] > max(0.30, so["lat"] + 1e-9):
        fails.append("style")
    return fails


def _motion_equivalent(a, b):
    """Tie-break eligibility: within 1 flag/1000 at every extent and identical
    pair counts."""
    return all(abs(a["flags_p1000"][str(e)] - b["flags_p1000"][str(e)])
               <= 1.0 + 1e-9 for e in motion.EXT_VARIANTS) \
        and all(a[k] == b[k] for k in ("narrow", "broad", "converging",
                                      "opposite"))


def admit_and_select(baseline, candidates, policy=None):
    """baseline: the B0 Candidate (always a valid fallback). candidates: every
    other Candidate (Q1 output, rollouts). Admission gates run against the
    baseline; risk-first ranking by J; preference score breaks ties only among
    motion-equivalent leaders. Every rejection is reported."""
    admitted, rejections = [baseline], {}
    for c in candidates:
        fails = [] if c.honored else ["not_honored"]
        fails += gates_vs_baseline(c.rv, baseline.rv)
        if fails:
            rejections[c.cid] = fails
        else:
            admitted.append(c)
    ranked = sorted(admitted, key=lambda c: (_j(c.rv), c.cid))
    leader = ranked[0]
    peers = [c for c in ranked if _motion_equivalent(c.rv, leader.rv)]
    selected = max(peers, key=lambda c: (c.score, c.cid)) if len(peers) > 1 \
        else leader
    return SelectionResult(
        selected=selected,
        admitted_ids=sorted(c.cid for c in admitted),
        rejections=rejections,
        accounting={"policy": POLICY_VERSION, "n_candidates": len(candidates),
                    "n_admitted": len(admitted)})
