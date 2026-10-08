"""Experiment-owned v3 full-scene export/readback; frozen QA stays unchanged.

python -m eval.joint_export verify
"""
import argparse
import json
from pathlib import Path
import time

from eval.joint_phrase import (OUT as READINESS, ROOT, PROTECTED, UnsupportedSource,
                               _atomic, _number, _sha, _validate_raw, encode_events,
                               phrase_windows, verify_sources)

OUT = ROOT / "experiments/joint-phrase-v1/export-check"


def export_chart(source, outdir, njs=18, offset=0):
    """Emit unrounded beats/BPM, literal notes and every bomb/obstacle."""
    bpm = _number(source["bpm"], "bpm", 0)
    _number(njs, "njs", 0)
    _number(offset, "offset")
    if bpm == 0 or njs == 0:
        raise UnsupportedSource("invalid_bpm_or_njs")
    phrase_windows(source["notes"], source["bombs"], source["walls"],
                   source["duration_beats"])
    dat = {"version": "3.0.0", "colorNotes": [
        {"b": b, "c": h, "x": c, "y": l, "d": d, "a": 0}
        for b, h, c, l, d in source["notes"]],
        "bombNotes": [{"b": b, "x": c, "y": l} for b, c, l in source["bombs"]],
        "obstacles": [{"b": b, "d": dur, "x": c, "w": w, "y": y, "h": h}
                      for b, dur, c, w, y, h in source["walls"]],
        "sliders": [], "burstSliders": [], "bpmEvents": [], "rotationEvents": [],
        "basicBeatmapEvents": []}
    info = {"_version": "2.1.0", "_songName": source.get("title", "Joint phrase pilot"),
            "_songSubName": "", "_songAuthorName": source.get("artist", ""),
            "_levelAuthorName": "osu2bs experimental", "_beatsPerMinute": bpm,
            "_songFilename": source.get("audio_filename", "song.egg"),
            "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5,
            "_previewStartTime": 0, "_previewDuration": 10, "_coverImageFilename": "",
            "_environmentName": "DefaultEnvironment", "_difficultyBeatmapSets": [{
                "_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [{
                    "_difficulty": "ExpertPlus", "_difficultyRank": 9,
                    "_beatmapFilename": "ExpertPlus.dat", "_noteJumpMovementSpeed": njs,
                    "_noteJumpStartBeatOffset": offset}]}]}
    outdir = Path(outdir)
    _atomic(outdir / "ExpertPlus.dat", dat)
    _atomic(outdir / "Info.dat", info)
    return read_chart(outdir / "ExpertPlus.dat", outdir / "Info.dat")


def read_chart(chart_path, info_path):
    """Read actual serialized bytes, including v3 obstacles missing in QA reader."""
    from eval.map_reader import read_dat
    chart_path, info_path = Path(chart_path), Path(info_path)
    dat = json.loads(chart_path.read_text(encoding="utf-8-sig"))
    info = json.loads(info_path.read_text(encoding="utf-8-sig"))
    if not _validate_raw(dat).startswith("3"):
        raise UnsupportedSource("experimental_readback_requires_v3")
    matches = [b for s in info.get("_difficultyBeatmapSets", [])
               if s.get("_beatmapCharacteristicName") == "Standard"
               for b in s.get("_difficultyBeatmaps", [])
               if b.get("_beatmapFilename") == chart_path.name]
    if len(matches) != 1 or matches[0].get("_difficulty") != "ExpertPlus" \
            or matches[0].get("_difficultyRank") != 9:
        raise UnsupportedSource("invalid_info_binding")
    binding = matches[0]
    bpm = _number(info.get("_beatsPerMinute"), "bpm", 0)
    njs = _number(binding.get("_noteJumpMovementSpeed"), "njs", 0)
    offset = _number(binding.get("_noteJumpStartBeatOffset"), "offset")
    if bpm == 0 or njs == 0 or info.get("_songTimeOffset", 0) != 0 \
            or info.get("_shuffle", 0) != 0:
        raise UnsupportedSource("unsupported_info_timing")
    parsed = read_dat(chart_path)
    if parsed["out_of_range"]:
        raise UnsupportedSource("invalid_note_geometry")
    notes = parsed["notes"]
    bombs = sorted((n["b"], n["x"], n["y"]) for n in dat.get("bombNotes", []))
    walls = sorted((w["b"], w["d"], w["x"], w["w"], w["y"], w["h"])
                   for w in dat.get("obstacles", []))
    # Duration is not encoded in a beatmap; validate geometry with a containing
    # interval, never report it as the actual audio duration.
    bound = max([0] + [n[0] for n in notes + bombs]) + 1
    phrase_windows(notes, bombs, walls, bound)
    return {"notes": notes, "bombs": bombs, "walls": walls, "bpm": bpm,
            "settings": {"njs": njs, "offset_beats": offset,
                         "difficulty": "ExpertPlus", "characteristic": "Standard"},
            "chart_sha256": _sha(chart_path), "info_sha256": _sha(info_path)}


def qa_scene(readback):
    """Map every serialized object to frozen QA's scoped units, never drop walls."""
    factor = 60.0 / readback["bpm"]
    walls = []
    for b, dur, c, width, y, height in readback["walls"]:
        typ = 0 if (y, height) == (0, 5) else 1 if (y, height) == (2, 3) else 2
        walls.append((b * factor, c, typ, dur * factor, width))
    return {"scope": None, "chart_sha256": readback["chart_sha256"],
            "info_sha256": readback["info_sha256"], "settings": readback["settings"],
            "bpm": readback["bpm"],
            "notes": sorted((b * factor, c, l, h, d)
                            for b, h, c, l, d in readback["notes"]),
            "bombs": [(b * factor, c, l, 3, 8) for b, c, l in readback["bombs"]],
            "walls": walls, "wall_geometry_beats": readback["walls"],
            "adapter": "joint-export-v1; non-full-height shapes stay unknown"}


def assert_same(source, exported, njs, offset):
    for name in ("notes", "bombs", "walls"):
        if sorted(map(tuple, source[name])) != sorted(map(tuple, exported[name])):
            raise ValueError(f"export roundtrip failed: {name}")
    if source["bpm"] != exported["bpm"] or exported["settings"]["njs"] != njs \
            or exported["settings"]["offset_beats"] != offset:
        raise ValueError("export settings changed")


def verify_corpus(deadline_seconds=2700, root=OUT):
    from eval.expressive_manifest import freeze_run
    root = Path(root)
    run = json.loads((READINESS / "run.json").read_text())
    report = json.loads((READINESS / "report.json").read_text())
    if report["identity"] != run["identity"] or report["status"] != "NEW_DECODER_REQUIRED":
        raise ValueError("readiness identity/status mismatch")
    protected = {p: _sha(ROOT / p) for p in PROTECTED}
    if protected != run["config"]["protected"]:
        raise ValueError("protected production artifacts changed")
    config = {"readiness": run["identity"], "report_sha256": _sha(READINESS / "report.json"),
              "recipe": {str(p.relative_to(ROOT)): _sha(p) for p in
                         (Path(__file__), ROOT / "eval/joint_phrase.py",
                          ROOT / "eval/joint_decode.py", ROOT / "eval/map_reader.py")},
              "protected": protected}
    frozen = freeze_run(root, config)
    deadline, receipts = time.monotonic() + min(deadline_seconds, 2700), []
    supported = {r["family"] for r in report["families"] if r["status"] == "supported"}
    for r in run["config"]["records"]:
        if r["family"] not in supported:
            continue
        if time.monotonic() >= deadline:
            return {"status": "INCOMPLETE", "completed": len(receipts), "required": len(supported)}
        verify_sources(r)
        name = r["family"].replace(":", "_")
        partial = json.loads((READINESS / "partial" / (name + ".json")).read_text())
        source_path = READINESS / "sources" / (name + ".json")
        if _sha(source_path) != partial["artifact_sha256"]:
            raise ValueError("readiness source artifact changed")
        source = json.loads(source_path.read_text())
        njs, offset = source["authored"]["njs"], source["authored"]["offset_beats"]
        destination, receipt_path = root / "charts" / name, root / "partial" / (name + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["identity"] != frozen["identity"] \
                    or receipt["source_sha256"] != partial["artifact_sha256"]:
                raise ValueError("export receipt identity mismatch")
            exported = read_chart(destination / "ExpertPlus.dat", destination / "Info.dat")
            if any(exported[k] != receipt[k] for k in ("chart_sha256", "info_sha256")):
                raise ValueError("cached export bytes changed")
        else:
            exported = export_chart(source, destination, njs, offset)
            assert_same(source, exported, njs, offset)
            receipt = {"identity": frozen["identity"], "family": r["family"], "role": r["role"],
                       "source_sha256": partial["artifact_sha256"], "roundtrip": True,
                       **{k: exported[k] for k in ("chart_sha256", "info_sha256")}}
            _atomic(receipt_path, receipt)
        assert_same(source, exported, njs, offset)
        scene = qa_scene(exported)
        assert len(scene["walls"]) == len(source["walls"])
        assert len(scene["bombs"]) == len(source["bombs"])
        receipts.append(receipt)
    if {p: _sha(ROOT / p) for p in PROTECTED} != protected:
        raise ValueError("production artifacts changed during export verification")
    result = {"status": "ROUNDTRIP_VERIFIED", "identity": frozen["identity"],
              "families": len(receipts), "all_roundtrip": all(r["roundtrip"] for r in receipts),
              "claim": "serialized source fidelity only; not QA admission or generated quality"}
    _atomic(root / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify"])
    parser.add_argument("--deadline-seconds", type=float, default=2700)
    args = parser.parse_args()
    print(json.dumps(verify_corpus(args.deadline_seconds), indent=1))
