"""Motion trajectory normalization + per-swing extraction (QA arc).

Operates on parsed BSOR replays. v1 scope (recorded limits): head/controller
TRAJECTORIES only — no elbow/shoulder/IK claims; no yaw/room calibration
(not present in telemetry); height normalization only when the replay
reports a positive height, otherwise scale stays unknown and untouched.

Canonicalization:
- left-handed replays are mirrored (x -> -x) to a right-handed canonical
  frame, flagged `mirrored`;
- with height h > 0, positions scale by 1.7/h to a reference stature
  (recorded; raw values remain in the stored replay files).

Per-swing extraction around each aligned note: the +-400 ms controller
window resampled to N_SAMPLES time-normalized points, plus scalars (path
length, peak/mean speed, straightness, approach/exit headings, vertical
extent). MOTION3D_VERSION participates in every cache identity.
"""
import bisect
import json
import math
from pathlib import Path

MOTION3D_VERSION = 1
REF_HEIGHT = 1.7
WINDOW_S = 0.4
N_SAMPLES = 24


def canonicalize(parsed):
    """Mirrored/height-normalized copies of the frame arrays (positions
    only; rotations retained raw alongside)."""
    info = parsed["info"]
    h = info.get("height") or 0.0
    scale = (REF_HEIGHT / h) if h > 0.2 else None
    mirrored = bool(info.get("leftHanded"))
    frames = []
    for t, fps, head, left, right in parsed["frames"]:
        def _p(p):
            x, y, z = p
            if mirrored:
                x = -x
            if scale:
                x, y, z = x * scale, y * scale, z * scale
            return (x, y, z)
        # a left-handed player's dominant hand maps to canonical right
        l, r = (right, left) if mirrored else (left, right)
        frames.append((t, _p(head[0]), _p(l[0]), _p(r[0])))
    return {"frames": frames, "mirrored": mirrored,
            "height_scale": scale, "height_known": scale is not None}


def _resample(pts, times, t0, t1, n=N_SAMPLES):
    """Linear time-resample of a position track to n points on [t0, t1]."""
    out = []
    for i in range(n):
        t = t0 + (t1 - t0) * i / (n - 1)
        j = bisect.bisect_left(times, t)
        if j <= 0:
            out.append(pts[0])
        elif j >= len(pts):
            out.append(pts[-1])
        else:
            ta, tb = times[j - 1], times[j]
            u = (t - ta) / max(tb - ta, 1e-9)
            out.append(tuple(a + u * (b - a)
                             for a, b in zip(pts[j - 1], pts[j])))
    return out


def _dist(a, b):
    return math.dist(a, b)


def swing_trajectories(parsed, aligned):
    """Per aligned note: canonical-frame swing window features. Returns
    (records, meta). Windows with <6 frames are unsupported (null), never
    fabricated."""
    canon = canonicalize(parsed)
    frames = canon["frames"]
    times = [f[0] for f in frames]
    out = []
    for ci, ei in aligned:
        ev = parsed["notes"][ei]
        t = ev["event_time"]
        lo = bisect.bisect_left(times, t - WINDOW_S)
        hi = bisect.bisect_right(times, t + WINDOW_S)
        if hi - lo < 6:
            out.append({"chart_index": ci, "supported": False})
            continue
        saber = (ev.get("cut") or {}).get("saberType", 1)
        # canonical: dominant hand is index 3 (right) after mirroring
        idx = 3 if saber == 1 else 2
        if canon["mirrored"]:
            idx = 5 - idx                     # note colors mirror too
        win_t = times[lo:hi]
        track = [frames[k][idx] for k in range(lo, hi)]
        head = [frames[k][1] for k in range(lo, hi)]
        rs = _resample(track, win_t, t - WINDOW_S, t + WINDOW_S)
        path = sum(_dist(a, b) for a, b in zip(track, track[1:]))
        speeds = [_dist(a, b) / max(tb - ta, 1e-6)
                  for (a, b), (ta, tb) in zip(zip(track, track[1:]),
                                              zip(win_t, win_t[1:]))]
        chord = _dist(track[0], track[-1])
        appr = tuple(b - a for a, b in zip(rs[0], rs[N_SAMPLES // 2]))
        exit_ = tuple(b - a for a, b in zip(rs[N_SAMPLES // 2], rs[-1]))
        ys = [p[1] for p in track]
        out.append({
            "chart_index": ci, "supported": True,
            "event_type": ev["event_type"],
            "resampled": [[round(v, 4) for v in p] for p in rs],
            "path_m": round(path, 4),
            "peak_speed": round(max(speeds), 3) if speeds else None,
            "mean_speed": round(sum(speeds) / len(speeds), 3)
            if speeds else None,
            "straightness": round(chord / path, 4) if path > 1e-6 else None,
            "vertical_extent": round(max(ys) - min(ys), 4),
            "approach": [round(v, 4) for v in appr],
            "exit": [round(v, 4) for v in exit_],
            "head_sway": round(sum(_dist(a, b) for a, b in
                                   zip(head, head[1:])), 4)})
    meta = {"motion3d_version": MOTION3D_VERSION,
            "mirrored": canon["mirrored"],
            "height_known": canon["height_known"],
            "height_scale": canon["height_scale"],
            "limits": "head/controller trajectories only; no IK/"
                      "biomechanics; no room-yaw calibration"}
    return out, meta


def derive_all():
    """Chunk-safe motion3d derivation for every kinematics-bearing side
    (gen / qa_train / qa_calib — NEVER the seal): resolve each family's
    chart from the corpus manifest, re-align, extract, write atomically."""
    from eval import corpus
    from qa.bsor import parse_bsor
    from qa.replays import align_notes, _chart_notes_for
    ROOT = Path(__file__).resolve().parent.parent
    states = []
    for name in ("pilot_state.json", "collect2_state.json"):
        p = ROOT / "experiments" / "qa-v1" / name
        if p.exists():
            states.append(json.loads(p.read_text()))
    m = corpus._load_validated()
    fam_dir = {}
    for r in m["maps"]:
        if r.get("family_rep") and "/beatsaver/" in r["dir"]:
            fam_dir.setdefault(r["family"], r["dir"])
    n_new = n_done = n_skip = 0
    for state in states:
        for fam, rec in sorted(state["charts"].items()):
            if rec.get("side") in ("qa", "seal") or not rec.get("replays"):
                continue                     # seal stays untouched
            dp = fam_dir.get(fam)
            if not dp:
                n_skip += len(rec["replays"])
                continue
            chart_notes, _bpm = _chart_notes_for(dp, rec["difficulty"])
            if not chart_notes:
                n_skip += len(rec["replays"])
                continue
            for entry in rec["replays"]:
                bsor_p = ROOT / entry["file"]
                m3_p = bsor_p.with_suffix(".motion3d.json")
                if m3_p.exists():
                    n_done += 1
                    continue
                parsed = parse_bsor(bsor_p.read_bytes())
                al = align_notes(chart_notes, parsed["notes"], tol_s=0.2)
                recs, meta = swing_trajectories(parsed, al["aligned"])
                tmp = m3_p.with_suffix(".tmp")
                tmp.write_text(json.dumps({"meta": meta, "swings": recs}))
                tmp.replace(m3_p)
                n_new += 1
    print(f"motion3d: {n_new} derived, {n_done} cached, {n_skip} skipped")
    return {"new": n_new, "cached": n_done, "skipped": n_skip}


if __name__ == "__main__":
    derive_all()
