"""mp3/ogg in -> Beat Saber Expert+ map folder out.

Usage: .venv/bin/python run.py song.mp3 [outdir] [difficulty] [--playlist NAME]
           [--diffs ExpertPlus,Expert,Hard,Normal,Easy | ExpertPlus2 (=Expert++)]
       .venv/bin/python run.py folder/  [out_parent] [difficulty] [--playlist NAME] [--force]
A folder processes every audio file in it (resumable: done songs skipped
unless --force, which re-generates everything).
--playlist adds finished maps to <out_parent>/<NAME>.bplist (created if missing).
Defaults to pinned36k onset flow. Each selected tier gets its own upstream osu
chart; difficulty is the Expert+ star anchor, with one star per tier step.
--legacy-flow explicitly selects the previous procedural pipeline.
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import convert

RSS_KILL_KB = 12 * 1024 * 1024  # WSL box has 15GB; runaway RAM killed it before


def _watchdog(proc):
    while proc.poll() is None:
        try:
            rss = int(next(l for l in open(f"/proc/{proc.pid}/status")
                           if l.startswith("VmRSS")).split()[1])
            if rss > RSS_KILL_KB:
                print(f"WATCHDOG: killing inference at {rss // 1024}MB RSS")
                proc.kill()
                return
        except (FileNotFoundError, StopIteration):
            return
        time.sleep(2)

HERE = Path(__file__).parent
MAPPER = HERE / "Mapperatorinator"
PY = Path(sys.executable)  # whatever interpreter runs run.py runs inference
ONSET_PT = HERE / "onset_flow.pt"  # pinned 36k checkpoint


AUDIO_EXTS = {".mp3", ".ogg", ".egg", ".wav", ".m4a", ".flac"}


def _clean(s):
    # hydra's override lexer rejects ()= and even exotic whitespace (\xa0 etc.)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", "", s)).strip()


def _slug(s):
    return _clean(s).replace(" ", "_")


def _hydra_str(s):
    """Quote any string as a hydra override value, verbatim. Unquoted values
    only lex ASCII, so non-English titles or a non-ASCII Windows username in
    the temp path killed inference before it started."""
    s = re.sub(r"(\\*)('|$)", lambda m: m[1] * 2 + ("\\'" if m[2] else ""), str(s))
    return "'" + s.replace("${", "\\${") + "'"


def add_to_playlist(name, mapdir, parent):
    """Append the map in mapdir to <parent>/<slug(name)>.bplist (deduped by hash)."""
    mapdir = Path(mapdir)
    info_bytes = (mapdir / "Info.dat").read_bytes()
    info = json.loads(info_bytes)
    h = hashlib.sha1(info_bytes)  # SongCore level hash: Info.dat + each diff .dat
    for bs in info["_difficultyBeatmapSets"]:
        for d in bs["_difficultyBeatmaps"]:
            h.update((mapdir / d["_beatmapFilename"]).read_bytes())
    digest = h.hexdigest()
    path = Path(parent) / (_slug(name) + ".bplist")
    pl = (json.loads(path.read_text()) if path.exists() else
          {"playlistTitle": name, "playlistAuthor": "osu2bs", "image": "",
           "songs": []})
    if not any(s.get("hash") == digest for s in pl["songs"]):
        pl["songs"].append({"songName": info["_songName"], "hash": digest,
                            "levelid": "custom_level_" + digest.upper()})
        path.write_text(json.dumps(pl, indent=2))
    return path


def upstream_stars(name, expert_plus_stars):
    """Request difficulty from osu; never manufacture tiers by thinning BS notes."""
    index = list(convert.DIFFS).index(name) if name in convert.DIFFS else 4 + int(name[10:]) - 1
    stars = float(expert_plus_stars) + index - 4
    if not math.isfinite(stars) or not 0 < stars <= 12:
        raise ValueError(f"{name} requires {stars:g} osu stars; adjust the Expert+ stars setting (0–12 per tier)")
    return stars


# Upstream timing sometimes finds no beat (crash: "No timing points") or a
# few stray objects across the whole song; super_timing crashes the same way
# when it hears no steady beat (rubato ballads). Escalate per tier until the
# chart covers the song (convert.MIN_COVERAGE): plain -> super_timing -> map
# against a librosa beat-tracked timing reference, which always has timing.
ATTEMPTS = ("plain", "super", "beat", "beat")


def beat_reference(song, path):
    """Constant-BPM timing-only .osu from a classical beat tracker."""
    import librosa
    import numpy as np
    if path.is_file():
        return path
    y, sr = librosa.load(str(song), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0]) or 120.
    offset = round(float(beats[0]) * 1000) if len(beats) else 0
    path.write_text(
        "osu file format v14\n\n[General]\nAudioFilename: song.mp3\nMode: 0\n\n"
        "[Metadata]\nTitle:ref\nArtist:ref\nCreator:osu2bs\nVersion:timing\n\n"
        "[Difficulty]\nHPDrainRate:5\nCircleSize:4\nOverallDifficulty:8\n"
        "ApproachRate:9\nSliderMultiplier:1.4\nSliderTickRate:1\n\n"
        f"[TimingPoints]\n{offset},{60000 / bpm},4,2,0,100,1,0\n\n[HitObjects]\n")
    return path


def coverage(osu, duration_s):
    _, objects, _, _ = convert.parse_osu(osu)
    return convert.coverage([t for o in objects for t in (o["t"], o["end_t"])], duration_s)


def generate_osu(song, work, stars, hydra_dev, env, title):
    """Run the upstream mapper for one tier until its chart covers the song.
    Gives up with an error when no attempt does; the app retries the song on
    the next run (finished tiers are kept)."""
    duration = convert._audio_info(song)[1]
    best = 0.
    for i, mode in enumerate(ATTEMPTS):
        out = work / f"try{i}"
        out.mkdir(parents=True)
        extra = {"plain": [], "super": ["super_timing=true"]}.get(mode) or [
            "beatmap_path=" + _hydra_str(beat_reference(song, work / "timing.osu")),
            "in_context=[TIMING]"]
        proc = subprocess.Popen(
            [str(PY), str(HERE / "mapper.py"), "-cn", "v32",
             "audio_path=" + _hydra_str(song), "output_path=" + _hydra_str(out),
             "gamemode=0", f"difficulty={stars}", "export_osz=false",
             hydra_dev, "title=" + _hydra_str(title), *extra,
             "hydra.run.dir=" + _hydra_str(out / "logs")], cwd=MAPPER, env=env)
        threading.Thread(target=_watchdog, args=(proc,), daemon=True).start()
        rc = proc.wait()
        osus = sorted(out.glob("*.osu"), key=lambda p: p.stat().st_mtime)
        if rc != 0 or not osus:
            print(f"upstream attempt {i + 1} ({mode}) failed", flush=True)
            continue
        cov = coverage(osus[-1], duration)
        if cov >= convert.MIN_COVERAGE:
            return osus[-1]
        best = max(best, cov)
        print(f"upstream attempt {i + 1} ({mode}): chart covers {cov:.0%} of the song",
              flush=True)
    raise RuntimeError(f"upstream inference failed: {len(ATTEMPTS)} attempts, best chart "
                       f"covered {best:.0%} of the song (need {convert.MIN_COVERAGE:.0%})")


def main(audio, outdir=None, difficulty="5.5", playlist=None, force=False,
         diffs=None, onset_checkpoint=ONSET_PT):
    names = convert.selected_difficulties(diffs)
    if onset_checkpoint is not None:
        onset_checkpoint = Path(onset_checkpoint).resolve()
        if not onset_checkpoint.is_file():
            raise ValueError(f"onset-flow checkpoint missing: {onset_checkpoint}")
        for name in names:
            upstream_stars(name, difficulty)
    audio = Path(audio).resolve()
    if audio.is_dir():
        parent = Path(outdir).resolve() if outdir else HERE / "out"
        songs = sorted(p for p in audio.iterdir()
                       if p.suffix.lower() in AUDIO_EXTS)
        fails = 0
        for i, p in enumerate(songs, 1):
            dest = parent / _slug(p.stem)
            if not force and dest.with_suffix(".zip").exists():  # zip written last = done
                print(f"[{i}/{len(songs)}] {p.name}: already done, skipping")
            else:
                print(f"[{i}/{len(songs)}] {p.name}")
                try:
                    main(p, dest, difficulty, diffs=diffs, onset_checkpoint=onset_checkpoint)
                except (Exception, SystemExit) as e:
                    print(f"  FAILED: {e}")
                    fails += 1
                    shutil.rmtree(dest, ignore_errors=True)  # no half-written maps
                    continue
            if playlist:
                add_to_playlist(playlist, dest, parent)
        if playlist:
            print(f"playlist: {parent / (_slug(playlist) + '.bplist')}")
        print(f"batch done: {len(songs) - fails}/{len(songs)} ok -> {parent}")
        return 1 if fails else 0
    outdir = Path(outdir or audio.with_suffix("")).resolve()
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        ext = audio.suffix.lower()  # SONG.EGG is still ogg
        song = work / ("song" + (".ogg" if ext == ".egg" else ext))
        try:
            song.symlink_to(audio)
        except OSError:  # Windows without dev-mode: no symlinks
            shutil.copyfile(audio, song)
        # OSU2BS_DEVICE: "cuda", "cuda:N" or "cpu" (default first GPU)
        dev = os.environ.get("OSU2BS_DEVICE", "cuda:0")
        env = {**os.environ}
        if dev.startswith("cuda"):
            # the app's GPU list is nvidia-smi (PCI) order; CUDA defaults to
            # fastest-first, so cuda:1 = 5080 in the UI ran on the 5070
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = dev.split(":")[1] if ":" in dev else "0"
            hydra_dev = "device=cuda"
        else:
            hydra_dev = "device=cpu"
        sources, requested_stars = {}, {}
        for name in names if onset_checkpoint else names[:1]:
            tier_work = work / name
            stars = upstream_stars(name, difficulty) if onset_checkpoint else float(difficulty)
            print(f"-- {name}: requesting osu timing at {stars:g} stars", flush=True)
            # a paused (killed) run keeps finished tiers; resume reuses them
            source = outdir / f"source-{name}-{stars:g}.osu"
            if not source.is_file():
                generated = generate_osu(song, tier_work, stars, hydra_dev, env, audio.stem)
                outdir.mkdir(parents=True, exist_ok=True)
                tmp = source.with_suffix(".part")
                tmp.write_bytes(generated.read_bytes())
                tmp.replace(source)
            sources[name], requested_stars[name] = source, stars
        rc = 1
        try:
            rc = convert.main(sources if onset_checkpoint else next(iter(sources.values())),
                              audio, outdir, ",".join(names),
                              **({"onset_checkpoint": onset_checkpoint} if onset_checkpoint else {}))
        finally:
            if rc:  # a source that broke conversion must not be reused by the retry
                for src in sources.values():
                    src.unlink(missing_ok=True)
        if onset_checkpoint:
            metadata_path = outdir / "generation.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["requested_osu_stars"] = requested_stars
            metadata_path.write_text(json.dumps(metadata, indent=1))
    if playlist:
        add_to_playlist(playlist, outdir, outdir.parent)
    return rc


if __name__ == "__main__":
    argv = sys.argv[1:]
    pl = None
    if "--playlist" in argv:
        i = argv.index("--playlist")
        pl = argv[i + 1]
        del argv[i:i + 2]
    force = "--force" in argv
    if force:
        argv.remove("--force")
    dfs = None
    if "--diffs" in argv:
        i = argv.index("--diffs")
        dfs = argv[i + 1]
        del argv[i:i + 2]
    onset = ONSET_PT
    if "--onset-flow" in argv:
        i = argv.index("--onset-flow")
        onset = argv[i + 1]
        del argv[i:i + 2]
    if "--legacy-flow" in argv:
        onset = None
        argv.remove("--legacy-flow")
    sys.exit(main(*argv[:3], playlist=pl, force=force, diffs=dfs, onset_checkpoint=onset))
