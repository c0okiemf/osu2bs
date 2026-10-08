"""Feature-parity repair Task 2 (review spec 2026-09-24 §2): deterministic
full-song player-balanced sampling + complete coverage denominators.

select_windows replaces the historical prefix caps (calibration stopped at
the first 2000 usable windows in replay-enumeration order; the diagnostic
took an early prefix per replay). Selection is order-invariant: eligible
windows group by player token, each player gets an equal share of the
family budget by integer water filling, remaining slots and within-player
choice follow SHA256("<seed>:<window_id>") — never enumeration order, no
accuracy filtering. Selection happens BEFORE NN retrieval so a comparator
failure can never censor the sample.

coverage_table keeps every expected family visible (zero support is
INSUFFICIENT, never a silent omission), counts NN-unsupported windows in
the predictive denominator, and reports pooled, per-component, per-player
and primary equal-player coverage (player -> component mean -> component
average -> player average).
"""
import hashlib

from qa.model import TARGET_NAMES

DEFAULT_SEED = "qa-parity-v2"


def _h(seed, s):
    return hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()


def select_windows(records, cap=2000, seed=DEFAULT_SEED):
    """Equal-player integer water filling over ONE family's eligible
    records ({player_token, window_id}); returns hash-ordered window ids."""
    by_p = {}
    for r in records:
        by_p.setdefault(r["player_token"], []).append(r["window_id"])
    for p in by_p:
        by_p[p] = sorted(set(by_p[p]), key=lambda w: _h(seed, w))
    players = sorted(by_p, key=lambda p: _h(seed, "player:" + p))
    take = {p: 0 for p in players}
    rem = min(cap, sum(len(v) for v in by_p.values()))
    while rem > 0:
        active = [p for p in players if take[p] < len(by_p[p])]
        if not active:
            break
        share = max(1, rem // len(active))
        for p in active:                     # hash order, incl. remainder
            t = min(share, len(by_p[p]) - take[p], rem)
            take[p] += t
            rem -= t
            if rem == 0:
                break
    out = []
    for p in players:
        out += by_p[p][:take[p]]
    return sorted(out, key=lambda w: _h(seed, w))


def coverage_table(observations, expected_families=()):
    """Aggregate coverage observations ({family, player, window_id,
    inside[10], observed[10], nn_supported, event_time?})."""
    fams = {}
    for o in observations:
        fams.setdefault(o["family"], []).append(o)
    out = {"predictive_observation_count": len(observations),
           "nn_supported_count": sum(1 for o in observations
                                     if o.get("nn_supported")),
           "families": {}}
    for fam in sorted(set(fams) | set(expected_families)):
        obs = fams.get(fam, [])
        rec = {"n_windows": len(obs),
               "predictive_observation_count": len(obs),
               "nn_supported_count": sum(1 for o in obs
                                         if o.get("nn_supported"))}
        ins = ob = 0
        per_comp = {}
        by_player = {}
        for o in obs:
            for i, name in enumerate(TARGET_NAMES):
                if o["observed"][i]:
                    ob += 1
                    ins += int(o["inside"][i])
                    c = per_comp.setdefault(name, [0, 0])
                    c[0] += int(o["inside"][i])
                    c[1] += 1
                    p = by_player.setdefault(o["player"], {})
                    pc = p.setdefault(name, [0, 0])
                    pc[0] += int(o["inside"][i])
                    pc[1] += 1
        rec["pooled_coverage"] = round(ins / ob, 4) if ob else None
        rec["per_component"] = {n: {"coverage": round(c[0] / c[1], 4),
                                    "n": c[1]}
                                for n, c in per_comp.items()}
        rec["components_insufficient"] = [n for n in TARGET_NAMES
                                          if n not in per_comp]
        rec["per_player"] = {}
        p_means = []
        for p, comps in sorted(by_player.items()):
            comp_covs = [c[0] / c[1] for c in comps.values()]
            m = sum(comp_covs) / len(comp_covs) if comp_covs else None
            rec["per_player"][p] = {
                "n_observed_components": sum(c[1] for c in comps.values()),
                "coverage": round(m, 4) if m is not None else None}
            if m is not None:
                p_means.append(m)
        rec["equal_player_coverage"] = (round(sum(p_means) / len(p_means), 4)
                                        if p_means else None)
        # distinct replay windows overlapping in time (±0.4 s windows) are
        # not independent observations — reported, never hidden
        times = sorted(o["event_time"] for o in obs
                       if o.get("event_time") is not None)
        rec["overlapping_window_count"] = sum(
            1 for i, t in enumerate(times)
            if (i > 0 and t - times[i - 1] < 0.8)
            or (i + 1 < len(times) and times[i + 1] - t < 0.8))
        rec["status"] = "OK" if ob else "INSUFFICIENT"
        out["families"][fam] = rec
    return out
