"""Staged Mapperatorinator inference over the frozen pilot subset (Packet D
premise check, review-approved <=20 calls).

Each chosen family's OWN corpus audio is the MI input, so the paired human
target (the chart in the same dir) shares the exact audio file — clock
alignment is by construction (same audio sha), not by warping onsets.

  .venv/bin/python -m eval.pilot_mi --n 4     # first 4 (one/group) smoke
  .venv/bin/python -m eval.pilot_mi --n 20    # continue up to 20

Writes experiments/pilot-mi/<family>/gen.osu + provenance.json. One frozen
seed, production v32 config, no best-of-N/retries. Skips families already
done (idempotent). Records runtime and failures; never hides cost.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

HERE = Path(__file__).parent.parent
MAPPER = HERE / "Mapperatorinator"
PY = HERE / ".venv/bin/python"
SUBSET = HERE / "experiments" / "pilot-subset.json"
OUTROOT = HERE / "experiments" / "pilot-mi"
SEED = 20260918          # frozen, recorded
DEVICE = os.environ.get("OSU2BS_DEVICE", "1")  # 5080 by default


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def audio_of(d):
    return next((q for q in Path(d).iterdir()
                 if q.suffix.lower() in (".egg", ".ogg")), None)


def run_one(fam):
    dst = OUTROOT / fam["family"].replace(":", "_")
    if (dst / "gen.osu").exists():
        return "done", 0.0
    aud = audio_of(fam["dir"])
    if aud is None:
        return "no_audio", 0.0
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        song = work / ("song" + (".ogg" if aud.suffix == ".egg" else aud.suffix))
        try:
            song.symlink_to(aud)
        except OSError:
            import shutil
            shutil.copyfile(aud, song)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": DEVICE}
        r = subprocess.run(
            [str(PY), "inference.py", "-cn", "v32", f"audio_path={song}",
             f"output_path={work}", "gamemode=0", "difficulty=5.5",
             "export_osz=false", "device=cuda", f"seed={SEED}",
             "title=" + fam["family"], "hydra.run.dir=" + str(work / "logs")],
            cwd=MAPPER, env=env, capture_output=True, text=True)
        dur = time.time() - t0
        osus = sorted(work.glob("*.osu"), key=lambda p: p.stat().st_mtime)
        if r.returncode != 0 or not osus:
            (dst / "FAILED.txt").write_text(r.stderr[-4000:])
            return "failed", dur
        (dst / "gen.osu").write_bytes(osus[-1].read_bytes())
    (dst / "provenance.json").write_text(json.dumps({
        "family": fam["family"], "split": fam["split"], "genre": fam["genre"],
        "song": fam["song"], "author": fam["author"],
        "corpus_dir": fam["dir"], "audio_file": aud.name,
        "audio_sha256_16": sha(aud), "gen_osu_sha256_16": sha(dst / "gen.osu"),
        "mi_config": "v32", "difficulty": 5.5, "seed": SEED,
        "device": "cuda", "runtime_s": round(dur, 1),
        "clock_alignment": "human target shares this exact audio file "
                           "(same sha) — aligned by construction",
    }, indent=1))
    return "ok", dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="max families to run")
    a = ap.parse_args()
    subset = json.loads(SUBSET.read_text())
    chosen = subset["chosen"]
    # first pass covers one per genre group (smoke), then fills up to --n
    seen, ordered = set(), []
    for g in ("punk_rock", "electronic", "pop", "rap_hiphop"):
        for c in chosen:
            if c["genre"] == g and g not in seen:
                ordered.append(c)
                seen.add(g)
    ordered += [c for c in chosen if c not in ordered]
    OUTROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for c in ordered[:a.n]:
        status, dur = run_one(c)
        results.append({"family": c["family"], "genre": c["genre"],
                        "split": c["split"], "status": status,
                        "runtime_s": round(dur, 1)})
        print(f"  [{status:7s}] {c['genre']:11s} {c['split']:5s} "
              f"{str(c['song'])[:32]:32s} {dur:5.0f}s")
        if status == "failed":
            print(f"    STOP: MI failed on {c['family']} — see FAILED.txt")
            break
    (OUTROOT / "run_log.json").write_text(json.dumps(results, indent=1))
    ok = sum(1 for r in results if r["status"] in ("ok", "done"))
    print(f"MI run: {ok}/{len(results)} ok, "
          f"{sum(r['runtime_s'] for r in results):.0f}s total -> {OUTROOT}")


if __name__ == "__main__":
    main()
