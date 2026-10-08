"""Independent-QA Task 2: validated telemetry records (spec §3).

Strict window hygiene around each aligned note (±400 ms):
- reject when either endpoint is unobserved, any tracking gap >50 ms,
  timestamps nonmonotone, any pose nonfinite, quaternion norm outside
  [0.9, 1.1], or fewer than 16 frames — with the reason recorded, never a
  fabricated zero;
- accepted quaternions are normalized, sign-aligned and SLERP-interpolated;
  positions resample at 100 Hz; a symmetric 5-sample moving average feeds
  DERIVATIVE descriptors only (endpoints trimmed); raw and filtered peaks
  stored separately;
- coordinates stay raw SI; the head anchor is the head position at window
  CENTER (a constant), not the moving head; mirroring reflects positions
  AND rotations via matrix reflection conjugation AND chart hand/direction
  semantics; unknown height stays unknown;
- miss events take the CHART hand/color — never a default saberType.

The v1 qa.motion3d caches are exploratory and are not read here.
TELEMETRY_VERSION participates in every cache identity.
"""
import bisect
import json
import math
from pathlib import Path

TELEMETRY_VERSION = 2
WINDOW_S = 0.4
MAX_GAP_S = 0.05
MIN_FRAMES = 16
RESAMPLE_HZ = 100
MA_HALF = 2                       # symmetric 5-sample moving average


def _norm(q):
    n = math.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q)


def slerp(a, b, u):
    """Shortest-path spherical interpolation of unit quaternions."""
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b = tuple(-x for x in b)
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        return _norm(tuple(x + u * (y - x) for x, y in zip(a, b)))
    th = math.acos(dot)
    sa, sb = math.sin((1 - u) * th) / math.sin(th), \
        math.sin(u * th) / math.sin(th)
    return _norm(tuple(sa * x + sb * y for x, y in zip(a, b)))


def mirror_pose(pos, quat):
    """Reflection across the x=0 plane via matrix conjugation M R M (M =
    diag(-1,1,1)): position x flips; quaternion (x,y,z,w) -> (x,-y,-z,w)."""
    x, y, z = pos
    qx, qy, qz, qw = quat
    return (-x, y, z), _norm((qx, -qy, -qz, qw))


def _quat_angle(a, b):
    """Relative rotation angle between two unit quaternions."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, dot))


def window_record(replay, alignment, index, mirrored=False):
    """One validated swing-window record for aligned pair `index`."""
    ci, ei = alignment["aligned"][index]
    chart = alignment["chart"][ci]
    ev = replay["notes"][ei]
    t = ev["event_time"]
    out = {"telemetry_version": TELEMETRY_VERSION, "chart_index": ci,
           "event_time": t}
    etype = ev.get("event_type")
    out["outcome"] = {0: "good", 1: "bad", 2: "miss", 3: "bomb"}.get(
        etype, "unknown")
    # hand: cut saberType when present; misses use the CHART color
    cut = ev.get("cut")
    if cut is not None:
        hand = "right" if cut.get("saberType") == 1 else "left"
    else:
        hand = "right" if chart[3] == 1 else "left"
    if mirrored:
        hand = "left" if hand == "right" else "right"
    out["hand"] = hand

    frames = replay["frames"]
    times = [f[0] for f in frames]
    lo = bisect.bisect_left(times, t - WINDOW_S)
    hi = bisect.bisect_right(times, t + WINDOW_S)
    win = frames[lo:hi]
    if not win or win[0][0] > t - WINDOW_S + 1e-9 \
            or win[-1][0] < t + WINDOW_S - 1e-9:
        # both endpoints must be OBSERVED (a frame at/beyond each edge)
        edge_lo = lo > 0 or (win and win[0][0] <= t - WINDOW_S + 1e-9)
        edge_hi = hi < len(frames) or (win and win[-1][0]
                                       >= t + WINDOW_S - 1e-9)
        if not (edge_lo and edge_hi):
            out.update(supported=False, reason="unobserved_endpoint")
            return out
        win = frames[max(lo - 1, 0):min(hi + 1, len(frames))]
    wt = [f[0] for f in win]
    if len(win) < MIN_FRAMES:
        out.update(supported=False, reason="too_few_frames")
        return out
    if any(b - a <= 0 for a, b in zip(wt, wt[1:])):
        out.update(supported=False, reason="nonmonotone_timestamps")
        return out
    if any(b - a > MAX_GAP_S for a, b in zip(wt, wt[1:])):
        out.update(supported=False, reason="tracking_gap")
        return out
    idx = 4 if hand == "right" else 3
    other = 3 if hand == "right" else 4

    def _pose(f, k):
        return f[k][0], f[k][1]
    for f in win:
        for k in (2, 3, 4):
            p, q = _pose(f, k)
            if not all(math.isfinite(v) for v in p + q):
                out.update(supported=False, reason="nonfinite_pose")
                return out
            qn = math.sqrt(sum(v * v for v in q))
            if not (0.9 <= qn <= 1.1):
                out.update(supported=False, reason="quaternion_norm")
                return out

    poses = [( _pose(f, idx)[0], _norm(_pose(f, idx)[1])) for f in win]
    other_pos = [_pose(f, other)[0] for f in win]
    head_pos = [_pose(f, 2)[0] for f in win]
    if mirrored:
        poses = [mirror_pose(p, q) for p, q in poses]
        other_pos = [mirror_pose(p, (0, 0, 0, 1))[0] for p in other_pos]
        head_pos = [mirror_pose(p, (0, 0, 0, 1))[0] for p in head_pos]
    # sign-align successive quaternions
    qs = [poses[0][1]]
    for _p, q in poses[1:]:
        prev = qs[-1]
        if sum(a * b for a, b in zip(prev, q)) < 0:
            q = tuple(-x for x in q)
        qs.append(q)
    # constant head anchor: head position at window center
    c_idx = bisect.bisect_left(wt, t)
    c_idx = min(max(c_idx, 0), len(head_pos) - 1)
    anchor = head_pos[c_idx]

    # resample positions at 100 Hz over [t-W, t+W]
    n_rs = int(2 * WINDOW_S * RESAMPLE_HZ) + 1

    def _resample(track):
        outp = []
        for i in range(n_rs):
            tt = t - WINDOW_S + 2 * WINDOW_S * i / (n_rs - 1)
            j = bisect.bisect_left(wt, tt)
            if j <= 0:
                outp.append(track[0])
            elif j >= len(track):
                outp.append(track[-1])
            else:
                u = (tt - wt[j - 1]) / max(wt[j] - wt[j - 1], 1e-9)
                outp.append(tuple(a + u * (b - a)
                                  for a, b in zip(track[j - 1], track[j])))
        return outp
    rs = _resample([p for p, _q in poses])
    rs_other = _resample(other_pos)
    rs_head = _resample(head_pos)
    rel = [tuple(a - b for a, b in zip(p, anchor)) for p in rs]

    def _ma(track):
        sm = []
        for i in range(MA_HALF, len(track) - MA_HALF):
            seg = track[i - MA_HALF:i + MA_HALF + 1]
            sm.append(tuple(sum(v[k] for v in seg) / len(seg)
                            for k in range(3)))
        return sm
    dt_rs = 1.0 / RESAMPLE_HZ
    sm = _ma(rs)
    speeds_f = [math.dist(a, b) / dt_rs for a, b in zip(sm, sm[1:])]
    speeds_raw = [math.dist(a, b) / dt_rs for a, b in zip(rs, rs[1:])]
    accs = [abs(b - a) / dt_rs for a, b in zip(speeds_f, speeds_f[1:])]

    def _p90(v):
        s = sorted(v)
        return s[min(len(s) - 1, int(0.9 * len(s)))] if s else None
    rot = sum(_quat_angle(a, b) for a, b in zip(qs, qs[1:]))
    dur = wt[-1] - wt[0]
    ang_speeds = [_quat_angle(a, b) / max(tb - ta, 1e-6)
                  for (a, b), (ta, tb) in zip(zip(qs, qs[1:]),
                                              zip(wt, wt[1:]))]
    xs = [p[0] for p in rs]
    ys = [p[1] for p in rs]
    min_sep = min(math.dist(a, b) for a, b in zip(rs, rs_other))
    out.update(supported=True, n_frames=len(win),
               anchor=[round(v, 4) for v in anchor],
               rel_path_24=[[round(v, 4) for v in rel[i]]
                            for i in range(0, n_rs, max(1, n_rs // 24))][:24],
               targets={
                   "path_m": sum(math.dist(a, b)
                                 for a, b in zip(rs, rs[1:])),
                   "speed_p90": _p90(speeds_f),
                   "accel_p90": _p90(accs),
                   "lateral_extent": max(xs) - min(xs),
                   "vertical_extent": max(ys) - min(ys),
                   "rotation_rad": rot,
                   "angular_speed_p90": _p90(ang_speeds),
                   "other_path_m": sum(math.dist(a, b) for a, b in
                                       zip(rs_other, rs_other[1:])),
                   "head_path_m": sum(math.dist(a, b) for a, b in
                                      zip(rs_head, rs_head[1:])),
                   "min_hand_separation": min_sep},
               raw_peak_speed=max(speeds_raw) if speeds_raw else None,
               filtered_peak_speed=max(speeds_f) if speeds_f else None,
               window_duration_s=round(dur, 4))
    return out


# ---------------- role derivation with support gates ----------------

ROOT = Path(__file__).resolve().parent.parent
ROLE_STATE = {"qa_train": ("collect2_state.json", "qa_train"),
              "qa_calib": ("collect2_state.json", "qa_calib"),
              "qa_calib2": ("collect2_state.json", "qa_calib2"),
              "qa_validate_latent": ("collect2_state.json",
                                     "qa_validate_latent"),
              "qa_validate_comparator": ("collect2_state.json",
                                         "qa_validate_comparator")}
MIN_GOOD_WINDOWS = 100
MIN_PLAYERS = 3


def _family_chart(manifest, family, difficulty):
    """Chart+Info paths for a family's rep dir at a difficulty — direct
    manifest JSON read; no generator-importing corpus module."""
    for r in manifest["maps"]:
        if r["family"] == family and r.get("family_rep") \
                and "/beatsaver/" in r["dir"]:
            d = Path(r["dir"])
            info_p = next((p for p in d.iterdir()
                           if p.name.lower() == "info.dat"), None)
            if info_p is None:
                return None, None
            info = json.loads(info_p.read_text(encoding="utf-8-sig"))
            for s in info.get("_difficultyBeatmapSets", []):
                if s.get("_beatmapCharacteristicName") != "Standard":
                    continue
                for b in s.get("_difficultyBeatmaps", []):
                    if b.get("_difficulty") == difficulty:
                        return d / b["_beatmapFilename"], info_p
    return None, None


def derive_role(role):
    """Window records for every sanitized replay of a role, chunk-safe.
    Writes <replay>.windows_v2.json beside each sanitized record and a role
    census; applies the spec's support gates at the end."""
    import torch
    from qa.scene import align, read_scene
    state_name, side = ROLE_STATE[role]
    state = json.loads((ROOT / "experiments" / "qa-v1" / state_name)
                       .read_text())
    manifest = json.loads((ROOT / "eval" / "corpus_manifest.json")
                          .read_text())
    census = {"role": role, "families": {}, "telemetry_version":
              TELEMETRY_VERSION}
    for fam, chart_rec in sorted(state["charts"].items()):
        if chart_rec.get("side") != side or not chart_rec.get("replays"):
            continue
        dat_p, info_p = _family_chart(manifest, fam,
                                      chart_rec["difficulty"])
        fam_out = {"good_windows": 0, "players": set(), "replays": 0,
                   "rejections": {}}
        if dat_p is None:
            fam_out["rejections"]["no_chart"] = len(chart_rec["replays"])
            census["families"][fam] = fam_out
            continue
        scene = read_scene(dat_p, info_p)
        if scene["scope"] is not None:
            fam_out["rejections"][f"scope:{scene['scope']}"] = \
                len(chart_rec["replays"])
            census["families"][fam] = fam_out
            continue
        for entry in chart_rec["replays"]:
            sp = ROOT / entry["file"].replace(".bsor", ".sanitized.pt")
            if not sp.exists():
                fam_out["rejections"]["missing_sanitized"] = \
                    fam_out["rejections"].get("missing_sanitized", 0) + 1
                continue
            wp = sp.with_suffix(".windows_v2.json")
            if wp.exists():
                recs = json.loads(wp.read_text())
            else:
                san = torch.load(sp)
                lh = bool(san["info"].get("leftHanded"))
                al = align(scene, san["notes"], left_handed=lh)
                recs = {"alignment": {k: al[k] for k in
                                      ("ambiguous", "unmatched",
                                       "out_of_order",
                                       "timing_residual_p95")},
                        "player_token": entry["player_token"],
                        "windows": [window_record(san, al, i, mirrored=lh)
                                    for i in range(len(al["aligned"]))]}
                tmp = wp.with_suffix(".tmp")
                tmp.write_text(json.dumps(recs))
                tmp.replace(wp)
            fam_out["replays"] += 1
            fam_out["players"].add(recs["player_token"])
            for w in recs["windows"]:
                if w.get("supported") and w.get("outcome") == "good":
                    fam_out["good_windows"] += 1
                elif not w.get("supported"):
                    r = w.get("reason", "?")
                    fam_out["rejections"][r] = \
                        fam_out["rejections"].get(r, 0) + 1
        census["families"][fam] = fam_out
    for fam, rec in census["families"].items():
        rec["players"] = sorted(rec["players"]) \
            if isinstance(rec["players"], set) else rec["players"]
        rec["supported"] = (rec["good_windows"] >= MIN_GOOD_WINDOWS
                            and len(rec["players"]) >= MIN_PLAYERS)
    n_sup = sum(1 for r in census["families"].values() if r["supported"])
    need = 12 if role == "qa_train" else 4
    census["n_supported"] = n_sup
    census["gate"] = "OK" if n_sup >= need else "DATA_SUPPORT_INSUFFICIENT"
    out = ROOT / "experiments" / "qa-v1" / f"telemetry_census_{role}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(census, indent=1, sort_keys=True))
    tmp.replace(out)
    print(f"[{role}] {n_sup} supported families (need {need}) -> "
          f"{census['gate']}")
    return census


if __name__ == "__main__":
    import sys
    derive_role(sys.argv[1] if len(sys.argv) > 1 else "qa_train")
