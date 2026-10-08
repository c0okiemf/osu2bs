"""Time-only cadence / coordination / gap vector over colored notes.

Grouped events are same-hand simultaneous notes clustered at their earliest
timestamp; they are NOT validated physical swings, and the long inter-event-gap
share is NOT silence or a rest-mask diagnosis. No trajectory, endpoint speed,
head selection, weighted score or learned ranking is computed here. See
docs/specs/2026-09-20-difficulty-vector-diagnostic-design.md.
"""
from math import ceil

TOL = 1e-6          # 1 microsecond simultaneity tolerance (anchored, no chaining)


def nearest_rank(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[max(0, ceil(p * len(v)) - 1)]


def grouped_events(notes):
    """Cluster same-hand notes within TOL of the cluster's earliest timestamp
    (never transitive), hand = note color. Returns events, occupied instants and
    the number of clusters that actually needed the tolerance."""
    events, tol_clusters = [], 0
    for hand in (0, 1):
        hs = sorted((n for n in notes if n.color == hand),
                    key=lambda n: (n.hit_s, n.id))
        i = 0
        while i < len(hs):
            anchor = hs[i].hit_s
            j, members = i, []
            while j < len(hs) and hs[j].hit_s <= anchor + TOL:
                members.append(hs[j])
                j += 1
            if any(m.hit_s != anchor for m in members):
                tol_clusters += 1
            events.append({"t_s": anchor, "hand": hand,
                           "ids": [m.id for m in members],
                           "n_directional": sum(1 for m in members if m.direction != 8),
                           "n_dots": sum(1 for m in members if m.direction == 8)})
            i = j
    events.sort(key=lambda e: (e["t_s"], e["hand"]))
    # occupied instants: cluster events of both hands by timestamp within TOL
    instants, es, i = [], events, 0
    while i < len(es):
        anchor = es[i]["t_s"]
        j, hands, ids = i, set(), []
        while j < len(es) and es[j]["t_s"] <= anchor + TOL:
            hands.add(es[j]["hand"])
            ids += es[j]["ids"]
            j += 1
        instants.append({"t_s": anchor, "hands": sorted(hands), "ids": ids})
        i = j
    return {"events": events, "instants": instants,
            "tolerance_cluster_count": tol_clusters}


def _in_win(tp, t, w):
    """left-open, right-closed (t-w, t] with TOL equality; left boundary excluded."""
    return tp <= t + TOL and tp > t - w + TOL


def _hand_gaps_ms(events, hand):
    ts = [e["t_s"] for e in events if e["hand"] == hand]
    if len(ts) < 2:
        return None
    gaps = [(ts[k] - ts[k - 1]) * 1000.0 for k in range(1, len(ts))]
    if any(g <= 0 for g in gaps):        # a grouping failure, never clamp to 0
        return {"failed": "nonpositive_gap"}
    return {"min": min(gaps), "p10": nearest_rank(gaps, 0.10),
            "p50": nearest_rank(gaps, 0.50), "p90": nearest_rank(gaps, 0.90),
            "count": len(gaps)}


def _burst(events, instants, w=2.0):
    """Max grouped count in (t-w, t] over occupied instants, /w, per hand and
    combined; earliest maximizing window + its ids."""
    out = {}
    for key, pred in (("left", lambda e: e["hand"] == 0),
                      ("right", lambda e: e["hand"] == 1),
                      ("combined", lambda e: True)):
        best_n, best_t, best_ids = -1, None, []
        for inst in instants:
            t = inst["t_s"]
            win = [e for e in events if pred(e) and _in_win(e["t_s"], t, w)]
            if len(win) > best_n:
                best_n = len(win)
                best_t = t
                best_ids = [i for e in win for i in e["ids"]]
        out[key] = {"count": max(best_n, 0), "rate": max(best_n, 0) / w,
                    "window_end_s": best_t, "ids": best_ids} if instants \
            else {"count": 0, "rate": None, "window_end_s": None, "ids": []}
    return out


def _imbalance(events, instants, w=8.0, min_events=8):
    """Max |L-R|/(L+R) over (t-w,t] windows with >=min_events grouped events."""
    best = None
    for inst in instants:
        t = inst["t_s"]
        win = [e for e in events if _in_win(e["t_s"], t, w)]
        if len(win) < min_events:
            continue
        left = sum(1 for e in win if e["hand"] == 0)
        right = len(win) - left
        val = abs(left - right) / (left + right)
        if best is None or val > best["value"]:
            best = {"value": val, "left": left, "right": right,
                    "window_end_s": t, "n": len(win),
                    "ids": [i for e in win for i in e["ids"]]}
    return best


def _occupied_gaps(instants, D):
    ts = [i["t_s"] for i in instants]
    if len(ts) < 2 or D <= 0:
        return {"p50": None, "p90": None, "max": None, "count_ge_1s": 0,
                "long_gap_share": None}
    gaps = [ts[k] - ts[k - 1] for k in range(1, len(ts))]
    longs = [g for g in gaps if g >= 1.0 - TOL]
    kmax = max(range(len(gaps)), key=lambda k: gaps[k])
    return {"p50": nearest_rank(gaps, 0.50), "p90": nearest_rank(gaps, 0.90),
            "max": max(gaps), "count_ge_1s": len(longs),
            "long_gap_share": sum(longs) / D,
            "max_gap_start_s": ts[kmax], "max_gap_end_s": ts[kmax + 1]}


def measure_scene(scene):
    """Full time-only vector for a supported scene from load_scene(.,'standard').
    An unknown/invalid scene returns status unknown with a reason and ignored
    counts, never a fabricated low rate. Empty / zero-span charts have counts but
    null rates/quantiles."""
    ignored = scene.get("ignored", {})
    if scene.get("chart_unknown"):
        return {"status": "unknown", "reason": scene.get("scope", "chart_unknown"),
                "ignored": ignored}
    notes = scene["notes"]
    g = grouped_events(notes)
    events, instants = g["events"], g["instants"]
    n_dir = sum(1 for n in notes if n.direction != 8)
    n_dot = sum(1 for n in notes if n.direction == 8)
    lg = [e for e in events if e["hand"] == 0]
    rg = [e for e in events if e["hand"] == 1]
    counts = {"raw_colored_notes": len(notes), "directional_notes": n_dir,
              "dots": n_dot, "grouped_events_L": len(lg), "grouped_events_R": len(rg),
              "grouped_events": len(events), "occupied_instants": len(instants),
              "multi_direction_groups": sum(1 for e in events if e["n_directional"] >= 2),
              "dot_only_groups": sum(1 for e in events
                                     if e["n_directional"] == 0 and e["n_dots"] > 0),
              "tolerance_cluster_count": g["tolerance_cluster_count"],
              "ignored": ignored}
    if not events:
        return {"status": "empty", "reason": "no_colored_events", "counts": counts,
                "span_s": 0.0, "rates": None}
    t0, t1 = events[0]["t_s"], events[-1]["t_s"]
    D = t1 - t0
    span = {"first_s": t0, "last_s": t1, "duration_s": D}
    if D <= 0:
        rates = None
    else:
        rates = {"grouped_per_s": len(events) / D,
                 "occupied_per_s": len(instants) / D,
                 "L_per_s": len(lg) / D, "R_per_s": len(rg) / D,
                 "raw_directional_per_s": n_dir / D}
    both = sum(1 for inst in instants if len(inst["hands"]) == 2)
    coincidence = both / len(instants) if instants else None
    return {"status": "ok", "counts": counts, "span_s": span, "rates": rates,
            "hand_gaps_ms": {"left": _hand_gaps_ms(events, 0),
                             "right": _hand_gaps_ms(events, 1)},
            "burst_2s": _burst(events, instants),
            "coincidence_share": coincidence,
            "imbalance_8s": _imbalance(events, instants),
            "occupied_gaps": _occupied_gaps(instants, D if D > 0 else 0)}
