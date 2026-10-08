"""Version-aware .dat reader + swing grouping + flow-audit metrics.

One shared reader so the
audit, the replay A/B experiment, and future evaluators measure identically.
Supported: v2 (_notes) and v3 (colorNotes) beatmaps. v4 and unknown schemas
are rejected with a reason — never silently mis-parsed.

Metrics come in two families:
- raw_*: directional colored notes counted individually — reproduces the
  flow-audit table in the plan (human stacks inflate these; that is the
  documented head-count proxy, kept for continuity).
- grouped_*: same-time same-hand clusters grouped into ONE event; clusters
  with >1 directional note are flagged ambiguous, not guessed. These are
  grouped-event counts, NOT validated swing/trajectory semantics (R6) —
  geometry-aware interpretation is future phase-3 work.

Self-check: .venv/bin/python -m eval.map_reader
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from convert import peak_nps

TS_ROUND = 3       # beat-timestamp grouping precision (matches .dat's 5dp)
DUP_BEATS = 8      # duplicate-detection window, beats
DUP_MIN_NOTES = 4  # windows with fewer directional notes don't count
BOUND_EPS = 1e-6   # block-boundary tolerance, beats: float subtraction can
                   # land a boundary note ~1e-14 short of its block (e.g.
                   # 34.33333-18.33333 = 15.999999999999996), making the
                   # metric depend on song offset (review R1). Matches the
                   # original audit's tolerance; reproduces all 52
                   # historical numbered-batch dup8 scores.
METRIC_VERSION = 2  # v1 = no boundary tolerance (pre-R1; undercounts)


class UnsupportedMap(Exception):
    pass


def read_dat(path):
    """-> dict(notes=[(beat, hand, col, layer, dir)], bombs, walls, chains,
    arcs, version, plus *_raw lists kept verbatim for inspection).
    Raises UnsupportedMap with a reason otherwise. Version dispatch runs
    BEFORE structural fallback: a realistic v4 file also carries a
    `colorNotes` key (indices into colorNotesData) and must never parse as
    an apparently valid empty v3 chart (review R2)."""
    d = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    ver = str(d.get("_version") or d.get("version") or "")
    if ver.startswith("4") or "colorNotesData" in d:
        raise UnsupportedMap(f"{path}: v4 beatmap schema not supported")
    # declared version wins; structural keys are only a versionless fallback
    if ver.startswith("3") or (not ver.startswith("2") and "_notes" not in d
                               and "colorNotes" in d):
        notes = [(n.get("b", 0), n.get("c"), n.get("x", -1), n.get("y", -1),
                  n.get("d", -1)) for n in d.get("colorNotes", [])
                 if n.get("c") in (0, 1)]
        angles = [(n.get("b", 0), n["a"]) for n in d.get("colorNotes", [])
                  if n.get("a")]  # angle offsets: retained, not interpreted
        bombs = len(d.get("bombNotes", []))
        walls_raw = d.get("obstacles", [])
        chains_raw = d.get("burstSliders", [])
        arcs_raw = d.get("sliders", [])
    elif ver.startswith("2") or "_notes" in d:
        raw = d.get("_notes", [])
        notes = [(n["_time"], n["_type"], n["_lineIndex"], n["_lineLayer"],
                  n["_cutDirection"]) for n in raw if n.get("_type") in (0, 1)]
        bombs = sum(1 for n in raw if n.get("_type") == 3)
        walls_raw, chains_raw, arcs_raw, angles = d.get("_obstacles", []), [], [], []
    else:
        raise UnsupportedMap(f"{path}: unknown beatmap schema (version {ver!r})")
    if not notes and (d.get("_notes") or d.get("colorNotes")):
        raise UnsupportedMap(
            f"{path}: version {ver!r} parsed 0 colored notes from non-empty "
            "note data — mismatched or unsupported schema")
    bad = sum(1 for _, _, c, l, dd in notes
              if not (0 <= c <= 3 and 0 <= l <= 2 and 0 <= dd <= 8))
    notes = sorted((b, h, c, l, dd) for b, h, c, l, dd in notes
                   if 0 <= c <= 3 and 0 <= l <= 2 and 0 <= dd <= 8)
    return {"notes": notes, "bombs": bombs, "walls": len(walls_raw),
            "chains": len(chains_raw), "arcs": len(arcs_raw),
            "walls_raw": walls_raw, "chains_raw": chains_raw,
            "arcs_raw": arcs_raw, "angles": angles,
            "version": ver or "2(implied)", "out_of_range": bad}


def swings(notes):
    """Group same-timestamp same-hand notes into swings.
    -> (swing list [(ts, hand, head_dir|None, n_directional, n_dots)],
        n_ambiguous). head_dir is None when the cluster has no directional
    note or several (ambiguous human stack — flagged, not guessed)."""
    groups = {}
    for b, h, c, l, d in notes:
        groups.setdefault((round(b, TS_ROUND), h), []).append((c, l, d))
    out, ambiguous = [], 0
    for (ts, h), g in sorted(groups.items()):
        dirs = [d for _, _, d in g if d != 8]
        head = dirs[0] if len(dirs) == 1 else None
        ambiguous += len(dirs) > 1
        out.append((ts, h, head, len(dirs), len(g) - len(dirs)))
    return out, ambiguous


def _dup_pct(heads, phase):
    t0 = heads[0][0] + phase
    wins = {}
    for b, h, c, l, d in heads:
        if b - t0 < -BOUND_EPS:
            continue
        w = int((b - t0 + BOUND_EPS) // DUP_BEATS)
        wins.setdefault(w, []).append(
            (round(b - t0 - w * DUP_BEATS, 3) + 0.0, h, c, l, d))
    sigs = [tuple(sorted(v)) for v in wins.values() if len(v) >= DUP_MIN_NOTES]
    return 100 * (len(sigs) - len(set(sigs))) / max(1, len(sigs))


def metrics(notes, beat_ms):
    """notes: [(beat, hand, col, layer, dir)] sorted; beat_ms: 60000/bpm.
    -> flat metric dict (raw_* reproduce the flow-audit table columns)."""
    heads = [n for n in notes if n[4] != 8]
    m = {"notes": len(notes), "raw_heads": len(heads)}
    if len(heads) < 2:
        return m
    span_s = (heads[-1][0] - heads[0][0]) * beat_ms / 1000
    m["span_s"] = round(span_s, 1)
    m["raw_sps"] = round(len(heads) / max(1e-9, span_s), 2)
    m["raw_peak_nps"] = round(peak_nps([b * beat_ms for b, *_ in heads]), 2)
    # exact duplicate 8-beat blocks: anchored at the first directional note
    # (the plan's table) plus the max over all 1-beat phase shifts, which
    # catches duplication the single anchor happens to split
    m["raw_dup8_pct"] = round(_dup_pct(heads, 0), 1)
    m["raw_dup8_maxphase_pct"] = round(
        max(_dup_pct(heads, p) for p in range(DUP_BEATS)), 1)
    # occupied timestamps with exactly two directional notes
    by_ts = {}
    for b, h, c, l, d in heads:
        by_ts.setdefault(round(b, TS_ROUND), []).append((h, c, l, d))
    m["raw_double_pct"] = round(
        100 * sum(1 for v in by_ts.values() if len(v) == 2) / len(by_ts), 1)
    # same-hand consecutive notes in the same cell (no time cutoff)
    trans = same = 0
    lastc = {}
    for b, h, c, l, d in heads:
        if h in lastc:
            trans += 1
            same += lastc[h] == (c, l)
        lastc[h] = (c, l)
    m["raw_same_cell_pct"] = round(100 * same / max(1, trans), 1)
    nl = sum(1 for n in heads if n[1] == 0)
    nr = len(heads) - nl
    m["raw_hand_ratio"] = round(max(nl, nr) / max(1, min(nl, nr)), 2)
    m["raw_hand_lr"] = f"{nl}:{nr}"
    # 2s bins over the mapped span (final partial bin included), <=1 head
    nbins = int(span_s / 2) + 1
    counts = [0] * nbins
    for b, *_ in heads:
        counts[min(nbins - 1, int((b - heads[0][0]) * beat_ms / 2000))] += 1
    m["raw_rest2s_pct"] = round(
        100 * sum(1 for c in counts if c <= 1) / nbins, 1)
    # lateral doubles: same-row outward '<>' (red-left cuts left, blue-right
    # cuts right) exactly as counted in the plan, plus the broader
    # any-height outward-horizontal pair family
    exact, broad = [], 0
    for ts, v in sorted(by_ts.items()):
        if len(v) != 2:
            continue
        a, b2 = sorted(v, key=lambda n: n[1])  # by column
        if a[3] == 2 and b2[3] == 3 and a[1] != b2[1]:
            broad += 1
            if a[2] == b2[2] and a[0] == 0 and b2[0] == 1:
                exact.append(round(ts * beat_ms / 1000, 3))
    m["lr_outward_row_doubles"] = len(exact)
    m["lr_outward_row_times_s"] = exact[:10]
    m["lr_outward_broad_doubles"] = broad
    # grouped-event stats: stacks/chains count once (NOT validated swings)
    sw, amb = swings(notes)
    m["grouped_events"] = len(sw)
    m["grouped_ambiguous"] = amb
    m["grouped_sps"] = round(len(sw) / max(1e-9, span_s), 2)
    ts_hands = {}
    for ts, h, *_ in sw:
        ts_hands.setdefault(ts, set()).add(h)
    m["grouped_two_hand_pct"] = round(
        100 * sum(1 for v in ts_hands.values() if len(v) == 2)
        / max(1, len(ts_hands)), 1)
    return m


def _selfcheck():
    import tempfile
    v2 = {"_version": "2.0.0", "_notes": [
        {"_time": 0.0, "_type": 0, "_lineIndex": 1, "_lineLayer": 0, "_cutDirection": 2},
        {"_time": 0.0, "_type": 1, "_lineIndex": 2, "_lineLayer": 0, "_cutDirection": 3},
        {"_time": 1.0, "_type": 0, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 1},
        {"_time": 1.0, "_type": 0, "_lineIndex": 0, "_lineLayer": 1, "_cutDirection": 1},
        {"_time": 2.0, "_type": 0, "_lineIndex": 0, "_lineLayer": 0, "_cutDirection": 1},
        {"_time": 2.0, "_type": 0, "_lineIndex": 0, "_lineLayer": 1, "_cutDirection": 8},
        {"_time": 3.0, "_type": 3, "_lineIndex": 2, "_lineLayer": 0, "_cutDirection": 0}]}
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
        json.dump(v2, f)
    r = read_dat(f.name)
    assert len(r["notes"]) == 6 and r["bombs"] == 1, r
    sw, amb = swings(r["notes"])
    assert len(sw) == 4, sw          # stack and dot-follower group once each
    assert amb == 1, amb             # the 2-directional stack is flagged
    m = metrics(r["notes"], 500.0)
    assert m["lr_outward_row_doubles"] == 1 and m["lr_outward_row_times_s"] == [0.0], m
    assert m["raw_heads"] == 5 and m["grouped_events"] == 4, m
    v3 = {"version": "3.2.0", "colorNotes": [
        {"b": 0, "x": 1, "y": 0, "c": 0, "d": 1}, {"b": 1, "x": 2, "y": 0, "c": 1, "d": 0}],
        "burstSliders": [{}], "bombNotes": [{}]}
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
        json.dump(v3, f)
    r = read_dat(f.name)
    assert len(r["notes"]) == 2 and r["chains"] == 1 and r["bombs"] == 1, r

    def rejected(obj, why):
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
            json.dump(obj, f)
        try:
            read_dat(f.name)
            raise AssertionError(why + " not rejected")
        except UnsupportedMap:
            pass
    rejected({"version": "4.0.0"}, "bare v4")
    # realistic v4 (R2): colorNotes carries INDICES into colorNotesData —
    # must reject, never parse as a valid empty chart
    rejected({"version": "4.0.0",
              "colorNotes": [{"b": 1, "r": 0, "i": 0}],
              "colorNotesData": [{"x": 1, "y": 0, "c": 0, "d": 1}]},
             "realistic v4")
    rejected({"version": "3.2.0", "_notes": [
        {"_time": 0, "_type": 0, "_lineIndex": 1, "_lineLayer": 0,
         "_cutDirection": 1}]}, "version/key mismatch")
    # R1: block duplication must be translation-invariant. Six identical
    # 8-beat blocks, four notes each -> 5/6 duplicated = 83.3%, at origin 0
    # AND shifted to the review's boundary-hostile offset serialized to 5dp
    base = [(w * 8.0 + r, 0, 1, 0, 1) for w in range(6) for r in (0, 1, 2, 3)]
    shifted = [(round(b + 18.33333, 5), h, c, l, d) for b, h, c, l, d in base]
    assert round(_dup_pct(base, 0), 1) == 83.3, _dup_pct(base, 0)
    assert round(_dup_pct(shifted, 0), 1) == 83.3, _dup_pct(shifted, 0)
    print("map_reader selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # ad-hoc: python -m eval.map_reader [bpm] file.dat...
        args = sys.argv[1:]
        bpm = float(args.pop(0)) if args and args[0].replace(".", "", 1).isdigit() else 120.0
        for p in args:
            print(p, json.dumps(metrics(read_dat(p)["notes"], 60000.0 / bpm)))
    else:
        _selfcheck()
