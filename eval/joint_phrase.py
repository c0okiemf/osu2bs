"""Lossless joint phrase source inventory; experiment-only, no generation.

python -m eval.joint_phrase audit [--root PATH] [--deadline-seconds 2700]
Beats are authored floats. Simultaneous arrows/dots stay literal notes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/joint-phrase-v1/readiness"
SALT = "joint-phrase-v1:"
MAX_SLOTS = 3
PROTECTED = ("groom.pt", "flow.pt", "critic.pt", "ladder.json",
             "quality_policy.json")


class UnsupportedSource(ValueError):
    pass


def _number(x, name, minimum=None):
    if isinstance(x, bool) or not isinstance(x, (int, float)) \
            or not math.isfinite(x) or (minimum is not None and x < minimum):
        raise UnsupportedSource(f"invalid_{name}")
    return x


def _integer(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, int) or not lo <= x <= hi:
        raise UnsupportedSource(f"invalid_{name}")
    return x


def encode_events(notes):
    """Validate and group (beat, hand, col, layer, direction), without loss."""
    by_time = {}
    occupied = set()
    for b, h, c, l, d in notes:
        _number(b, "note_time", 0)
        _integer(h, "hand", 0, 1)
        _integer(c, "column", 0, 3)
        _integer(l, "layer", 0, 2)
        _integer(d, "direction", 0, 8)
        if (b, c, l) in occupied:
            raise UnsupportedSource("duplicate_occupancy")
        occupied.add((b, c, l))
        hands = by_time.setdefault(b, [[], []])
        hands[h].append((c, l, d))
        if len(hands[h]) > MAX_SLOTS:
            raise UnsupportedSource("slot_capacity_exceeded")
    return [{"beat": b, "hands": [sorted(h) for h in by_time[b]]}
            for b in sorted(by_time)]


def decode_events(events):
    notes = []
    previous = -1
    for event in events:
        b = _number(event["beat"], "event_time", 0)
        if b <= previous or len(event["hands"]) != 2:
            raise UnsupportedSource("invalid_event_order_or_hands")
        previous = b
        if not any(event["hands"]):
            raise UnsupportedSource("empty_timestamp_event")
        notes.extend((b, h, *n) for h, ns in enumerate(event["hands"]) for n in ns)
    encode_events(notes)  # Same validation at the decoding boundary.
    return sorted(notes)


def phrase_windows(notes, bombs, walls, duration_beats, beats=8):
    """Half-open windows retain empty rests, tail and literal source objects.

    A boundary note belongs to the later window. Source notes at/beyond audio end
    fail explicitly. Obstacles keep their original intervals and stable indices.
    """
    _number(duration_beats, "duration", 0)
    _number(beats, "window_length", 0)
    if beats == 0:
        raise UnsupportedSource("invalid_window_length")
    events = encode_events(notes)
    for b, c, l in bombs:
        _number(b, "bomb_time", 0)
        _integer(c, "bomb_column", 0, 3)
        _integer(l, "bomb_layer", 0, 2)
    for b, dur, c, width, y, height in walls:
        _number(b, "wall_time")  # A wall may start before the audio origin.
        _number(dur, "wall_duration", 0)
        _integer(c, "wall_column", 0, 3)
        _integer(width, "wall_width", 1, 4)
        _integer(y, "wall_y", 0, 4)
        _integer(height, "wall_height", 1, 5)
        if dur == 0 or c + width > 4 or y + height > 5:
            raise UnsupportedSource("invalid_wall_geometry")
    if any(n[0] >= duration_beats for n in list(notes) + list(bombs)):
        raise UnsupportedSource("notes_outside_audio")
    result, i = [], 0
    for w in range(math.ceil(duration_beats / beats)):
        start, end = w * beats, min((w + 1) * beats, duration_beats)
        previous = events[i - 1] if i else None
        current = []
        while i < len(events) and events[i]["beat"] < end:
            current.append(events[i])
            i += 1
        result.append({"start": start, "end": end, "previous": previous,
                       "events": current,
                       "bomb_ids": [j for j, n in enumerate(bombs)
                                    if start <= n[0] < end],
                       "wall_ids": [j for j, wall in enumerate(walls)
                                    if wall[0] < end and wall[0] + wall[1] > start]})
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path, obj):
    from eval.expressive_manifest import _atomic_write
    _atomic_write(Path(path), json.dumps(obj, indent=1, sort_keys=True))


def _validate_raw(raw):
    """Scope first; the shared reader must not silently filter invalid fields."""
    from eval.map_scope import fixed_v2_scope_reason
    version = str(raw.get("_version") or raw.get("version") or "")
    for key in ("_notes", "_obstacles", "colorNotes", "bombNotes", "obstacles"):
        if key in raw and (not isinstance(raw[key], list)
                           or any(not isinstance(x, dict) for x in raw[key])):
            raise UnsupportedSource(f"invalid_object_list:{key}")
    if version.startswith("2"):
        reason = fixed_v2_scope_reason(raw)
        if reason:
            raise UnsupportedSource(reason)
        ns = raw.get("_notes", [])
        if any(n.get("_type") not in (0, 1, 3) for n in ns):
            raise UnsupportedSource("unsupported_note_type")
        for n in ns:
            _number(n.get("_time"), "note_time", 0)
            _integer(n.get("_type"), "note_type", 0, 3)
            _integer(n.get("_lineIndex"), "column", 0, 3)
            _integer(n.get("_lineLayer"), "layer", 0, 2)
            _integer(n.get("_cutDirection"), "direction", 0, 8)
        for w in raw.get("_obstacles", []):
            for k in ("_time", "_duration", "_lineIndex", "_width", "_type"):
                if k not in w:
                    raise UnsupportedSource(f"missing_wall_field:{k}")
    elif version.startswith("3"):
        if any(raw.get(k) for k in ("sliders", "burstSliders", "bpmEvents",
                                    "rotationEvents")):
            raise UnsupportedSource("unsupported_v3_mechanics")
        ns = raw.get("colorNotes", []) + raw.get("bombNotes", [])
        for n in ns:
            _number(n.get("b"), "note_time", 0)
            _integer(n.get("x"), "column", 0, 3)
            _integer(n.get("y"), "layer", 0, 2)
        for n in raw.get("colorNotes", []):
            _integer(n.get("c"), "hand", 0, 1)
            _integer(n.get("d"), "direction", 0, 8)
            if n.get("a", 0) != 0:
                raise UnsupportedSource("unsupported_angle_offset")
        for w in raw.get("obstacles", []):
            for k in ("b", "d", "x", "w", "y", "h"):
                if k not in w:
                    raise UnsupportedSource(f"missing_wall_field:{k}")
    else:
        raise UnsupportedSource(f"unsupported_schema:{version or 'none'}")
    objects = ns + (raw.get("_obstacles", []) if version.startswith("2")
                    else raw.get("obstacles", []))
    if any(o.get("_customData") or o.get("customData") for o in objects):
        raise UnsupportedSource("unsupported_object_custom_data")
    cd = raw.get("_customData") or raw.get("customData") or {}
    if not isinstance(cd, dict):
        raise UnsupportedSource("invalid_custom_data")
    if any(cd.get(k) for k in ("_customEvents", "customEvents", "fakeColorNotes",
                               "fakeBombNotes", "fakeObstacles", "_environment",
                               "environment", "_pointDefinitions", "pointDefinitions")):
        raise UnsupportedSource("unsupported_chart_custom_gameplay")
    return version


def verify_sources(record):
    for name, src in record["sources"].items():
        if not Path(src["path"]).is_file() or _sha(src["path"]) != src["sha256"]:
            raise ValueError(f"source identity changed: {record['family']} {name}")


def read_source(record):
    import soundfile as sf
    from eval.map_reader import read_dat, UnsupportedMap
    from eval.visibility import resolve_authored
    verify_sources(record)
    paths = {k: Path(v["path"]) for k, v in record["sources"].items()}
    raw = json.loads(paths["chart"].read_text(encoding="utf-8-sig"))
    version = _validate_raw(raw)
    binding = resolve_authored(paths["chart"], record["sources"]["chart"]["sha256"])
    if binding["status"] != "verified" or binding["difficulty"] != "ExpertPlus":
        raise UnsupportedSource("authored_binding:" + binding.get("reason", "wrong_difficulty"))
    info = json.loads(paths["info"].read_text(encoding="utf-8-sig"))
    bpm = _number(info.get("_beatsPerMinute"), "bpm", 0)
    if bpm == 0 or info.get("_songTimeOffset", 0) != 0 \
            or info.get("_shuffle", 0) != 0:
        raise UnsupportedSource("unsupported_audio_timing")
    try:
        parsed = read_dat(paths["chart"])
    except UnsupportedMap as exc:
        raise UnsupportedSource(str(exc)) from exc
    if parsed["out_of_range"]:
        raise UnsupportedSource("out_of_range_notes")
    if version.startswith("2"):
        bombs = [(n["_time"], n["_lineIndex"], n["_lineLayer"])
                 for n in raw.get("_notes", []) if n["_type"] == 3]
        walls = []
        for w in raw.get("_obstacles", []):
            typ = _integer(w.get("_type"), "wall_type", 0, 1)
            walls.append((w["_time"], w["_duration"], w["_lineIndex"],
                          w["_width"], 0 if typ == 0 else 2, 5 if typ == 0 else 3))
    else:
        bombs = [(n["b"], n["x"], n["y"]) for n in raw.get("bombNotes", [])]
        walls = [(w["b"], w["d"], w["x"], w["w"], w["y"], w["h"])
                 for w in raw.get("obstacles", [])]
    duration = sf.info(paths["audio"]).duration * bpm / 60
    notes = parsed["notes"]
    events = encode_events(notes)
    windows = phrase_windows(notes, bombs, walls, duration)
    assert decode_events(events) == notes
    assert decode_events([e for w in windows for e in w["events"]]) == notes
    assert sorted(j for w in windows for j in w["bomb_ids"]) == list(range(len(bombs)))
    return {"notes": notes, "bombs": bombs, "walls": walls, "events": events,
            "windows": windows, "duration_beats": duration, "bpm": bpm,
            "authored": binding}


def decoder_restrictions(notes, bpm, walls=(), bombs=()):
    from parity import BACKHAND, DIR_VEC, FOREHAND, HandParity, LATERAL, ang_dist
    from groom import FAST_MIN_ANG, FAST_STEPS, grid_shift
    counts, witnesses = Counter(), defaultdict(list)

    def hit(reason, beat, detail=None, n=1):
        counts[reason] += n
        if len(witnesses[reason]) < 3:
            witnesses[reason].append({"beat": beat, "detail": detail})

    events = encode_events(notes)
    shift = grid_shift([n[0] * 4 for n in notes]) if notes else 0
    errors = [abs(n[0] * 4 - shift - round(n[0] * 4 - shift)) / 4 for n in notes]
    for note, error in zip(notes, errors):
        if error > 1e-6:
            hit("off_quarter_grid", note[0], {"error_beats": error})
        if (note[1] == 0 and note[2] == 3) or (note[1] == 1 and note[2] == 0):
            hit("cross_body_outer_lane", note[0], list(note[1:]))
        if note[2] in (1, 2) and note[3] == 1:
            hit("central_middle_cell", note[0], list(note[1:]))
    machines, last = [HandParity(), HandParity()], [None, None]
    for event in events:
        b = event["beat"]
        for hand, ns in enumerate(event["hands"]):
            if not ns:
                continue
            heads = [n for n in ns if n[2] != 8]
            dots = [n for n in ns if n[2] == 8]
            if len(heads) != 1:
                hit("multiple_directional_heads" if heads else "standalone_dots", b,
                    {"hand": hand, "notes": ns})
                machines[hand], last[hand] = HandParity(), None
                continue
            c, l, d = heads[0]
            vx, vy = DIR_VEC[d]
            expected = sorted((c + i * vx, l + i * vy, 8)
                              for i in range(1, len(dots) + 1))
            if len(dots) > 2 or sorted(dots) != expected:
                hit("non_unit_chain", b, {"hand": hand, "notes": ns})
            t = b * 60000 / bpm
            req = machines[hand].required(t)
            allowed = set(range(8)) if req is None else set(
                (FOREHAND if req == "fore" else BACKHAND) | LATERAL)
            if d not in allowed:
                hit("parity_machine", b, {"hand": hand, "dir": d, "required": req})
            step = round(b * 4 - shift)
            if last[hand] is not None and step - last[hand][0] <= FAST_STEPS:
                fast = {v for v in allowed if ang_dist(v, last[hand][1]) >= FAST_MIN_ANG}
                if fast and d not in fast:
                    hit("fast_angle_policy", b, {"hand": hand, "dir": d})
            machines[hand].commit(d, t)
            last[hand] = (step, d)
    for b, dur, c, width, y, height in walls:
        if not (c in (0, 3) and width == 1 and y == 0 and height == 5):
            hit("wall_not_exportable", b, [dur, c, width, y, height])
    for b, c, l in bombs:
        hit("bomb_not_exportable", b, [c, l])
    return {"counts": dict(counts), "witnesses": dict(witnesses),
            "max_quarter_grid_error_beats": max(errors, default=0),
            "note": "Existing decoder restrictions, not human quality failures; "
                    "axis-run intervention and soft penalties not audited here."}


def inventory_records(manifest, exposed, protected):
    """Freeze selection before reading outcomes; no replacements after exclusions."""
    from eval.expressive_manifest import family_strata
    strata = family_strata(manifest)
    rejected, approved = set(strata["rejected"]), set(strata["approved"])
    dev = sorted({r["family"] for r in exposed["families"]},
                 key=lambda f: hashlib.sha256((SALT + f).encode()).hexdigest())[:12]
    if len(dev) != 12:
        raise ValueError("development inventory must contain 12 families")
    reps = {r["family"]: r for r in manifest["maps"] if r.get("family_rep")}
    train = sorted(f for f, r in reps.items() if r["split"] == "train"
                   and r["eligible"] == "ok" and "ExpertPlus" in r["charts"]
                   and f not in rejected)
    if set(train) & set(dev):
        raise ValueError("train/development family overlap")
    records = []
    for role, families in (("train", train), ("development", dev)):
        for family in families:
            r = reps[family]
            record = {"family": family, "role": role, "source_split": r["split"],
                      "approved": family in approved, "sources": {}}
            d = Path(r["dir"])
            ip = next((p for p in d.iterdir() if p.name.lower() == "info.dat"), None)
            chart = r["charts"].get("ExpertPlus")
            audio = d / r["audio_file"] if r.get("audio_file") else None
            cp = d / chart["file"] if chart else None
            if not all(p is not None and p.is_file() for p in (ip, cp, audio)):
                record["inventory_error"] = "missing_source"
            else:
                for key, p in (("chart", cp), ("info", ip), ("audio", audio)):
                    record["sources"][key] = {"path": str(p), "sha256": _sha(p)}
                if record["sources"]["chart"]["sha256"] != chart["sha256"]:
                    raise ValueError(f"corpus chart identity changed: {family}")
            records.append(record)
    return {"version": 1, "salt": SALT, "max_slots": MAX_SLOTS, "window_beats": 8,
            "protected": protected, "records": records}


def freeze_inventory(root=OUT):
    from eval.expressive_manifest import freeze_run
    root = Path(root)
    paths = {"manifest": ROOT / "eval/corpus_manifest.json",
             "exposed": ROOT / "experiments/expressive-v1/fresh/selection.json",
             "recipe": Path(__file__)}
    sources = {k: {"path": str(p), "sha256": _sha(p)} for k, p in paths.items()}
    protected = {p: _sha(ROOT / p) for p in PROTECTED}
    prior = root / "run.json"
    if prior.exists():
        run = json.loads(prior.read_text())
        if run["config"]["inputs"] != sources or run["config"]["protected"] != protected:
            raise ValueError("frozen inventory/recipe/protected identity changed")
        return run
    config = inventory_records(json.loads(paths["manifest"].read_text()),
                               json.loads(paths["exposed"].read_text()), protected)
    config["inputs"] = sources
    return freeze_run(root, config)


def run_audit(root=OUT, deadline_seconds=2700):
    root = Path(root)
    run = freeze_inventory(root)
    deadline = time.monotonic() + min(deadline_seconds, 2700)
    rows = []
    for r in run["config"]["records"]:
        if time.monotonic() >= deadline:
            return {"status": "INCOMPLETE", "completed": len(rows),
                    "required": len(run["config"]["records"])}
        verify_sources(r)
        path = root / "partial" / (r["family"].replace(":", "_") + ".json")
        if path.exists():
            rec = json.loads(path.read_text())
            if rec["identity"] != run["identity"] or rec["sources"] != r["sources"]:
                raise ValueError("partial identity mismatch")
            if rec["status"] == "supported" and _sha(root / "sources" / path.name) != rec["artifact_sha256"]:
                raise ValueError("source artifact identity changed")
        else:
            rec = {"family": r["family"], "role": r["role"], "sources": r["sources"],
                   "identity": run["identity"], "status": "unsupported"}
            try:
                if r.get("inventory_error"):
                    raise UnsupportedSource(r["inventory_error"])
                source = read_source(r)
                rec.update(status="supported", roundtrip=True,
                           n_notes=len(source["notes"]), n_events=len(source["events"]),
                           n_windows=len(source["windows"]),
                           empty_windows=sum(not w["events"] for w in source["windows"]),
                           restrictions=decoder_restrictions(source["notes"], source["bpm"],
                                                             source["walls"], source["bombs"]))
                _atomic(root / "sources" / path.name, source)
                rec["artifact_sha256"] = _sha(root / "sources" / path.name)
            except UnsupportedSource as exc:
                rec["reason"] = str(exc)
            _atomic(path, rec)
            print(r["family"], rec["status"], rec.get("reason", ""), flush=True)
        rows.append(rec)
    summary = {}
    for role in ("train", "development"):
        chosen = [r for r in rows if r["role"] == role]
        ok = [r for r in chosen if r["status"] == "supported"]
        restrictions, affected = Counter(), Counter()
        for r in ok:
            restrictions.update(r["restrictions"]["counts"])
            affected.update(r["restrictions"]["counts"].keys())
        summary[role] = {"total": len(chosen), "supported": len(ok),
                         "excluded": dict(Counter(r["reason"] for r in chosen
                                                  if r["status"] != "supported")),
                         "restriction_counts": dict(restrictions),
                         "restriction_families": dict(affected),
                         "notes": sum(r["n_notes"] for r in ok),
                         "windows": sum(r["n_windows"] for r in ok),
                         "empty_windows": sum(r["empty_windows"] for r in ok)}
    enough = summary["train"]["supported"] >= 100 and summary["development"]["supported"] >= 6
    status = ("NEW_DECODER_REQUIRED" if any(s["restriction_counts"] for s in summary.values())
              else "REPRESENTATION_READY") if enough else "INSUFFICIENT_SUPPORTED_DATA"
    report = {"status": status, "identity": run["identity"], "summary": summary,
              "families": [{k: r[k] for k in ("family", "role", "status")} for r in rows],
              "claim": "source fidelity and decoder compatibility only"}
    if {p: _sha(ROOT / p) for p in PROTECTED} != run["config"]["protected"]:
        raise ValueError("production artifacts changed during audit")
    _atomic(root / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit"])
    parser.add_argument("--root", type=Path, default=OUT)
    parser.add_argument("--deadline-seconds", type=float, default=2700)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.root, args.deadline_seconds), indent=1))
