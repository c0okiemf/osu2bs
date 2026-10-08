"""Q3.1: hash-select and generate the paired-MI evidence corpus.

Selection: up to 240 train + 32 val families from the corpus manifest,
deterministic sha256(family + salt) order INSIDE strata (BPM tercile x
multi-tier availability, filled round-robin so no stratum dominates), eligible
maps only (eligible=="ok", dir + audio present). Original clean split
membership preserved; C/test families excluded by construction. Genre labels
exist for only the 18 pilot families — recorded where known, "unknown"
otherwise, never invented.

Generation: Mapperatorinator v32, difficulty 5.5, frozen seed 20260921, one
call per family on the family's OWN corpus audio (paired human target shares
the exact audio file — clock alignment by construction). Idempotent; failures
recorded, never hidden. Output: experiments/quality-v1/q3-mi/<family>/gen.osu
+ provenance.json + selection.json.
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

ROOT = Path(__file__).resolve().parent.parent
MAPPER = ROOT / "Mapperatorinator"
PY = ROOT / ".venv/bin/python"
MANIFEST = Path(__file__).parent / "corpus_manifest.json"
OUTROOT = ROOT / "experiments" / "quality-v1" / "q3-mi"
SALT = "quality-v1-q3"
SEED = 20260921
N_TRAIN, N_VAL = 240, 32
DEVICE = os.environ.get("OSU2BS_DEVICE", "1")


def _hash(f):
    return hashlib.sha256(f"{f}:{SALT}".encode()).hexdigest()


def _pilot_genres():
    p = ROOT / "experiments" / "pilot-subset.json"
    if not p.exists():
        return {}
    return {c["family"]: c.get("genre", "unknown")
            for c in json.loads(p.read_text())["chosen"]}


def select_families():
    man = json.loads(MANIFEST.read_text())
    genres = _pilot_genres()
    fams = {}
    for m in man["maps"]:
        if m["split"] not in ("train", "val") or m.get("eligible") != "ok":
            continue
        d = Path(m["dir"])
        if not d.exists() or not (d / m["audio_file"]).exists():
            continue
        e = fams.setdefault(m["family"], {
            "family": m["family"], "split": m["split"], "bpm": m["bpm"],
            "dirs": [], "tiers": set(), "genre": genres.get(m["family"],
                                                            "unknown")})
        e["dirs"].append(m["dir"])
        e["tiers"].update(c.get("difficulty") for c in m.get("charts", [])
                          if isinstance(c, dict))
    for e in fams.values():
        e["dirs"].sort()
        e["dir"] = e["dirs"][0]
        e["multi_tier"] = len(e["tiers"]) > 1
        e["tiers"] = sorted(str(t) for t in e["tiers"])
    picked = {"train": [], "val": []}
    for split, cap in (("train", N_TRAIN), ("val", N_VAL)):
        pool = [e for e in fams.values() if e["split"] == split]
        bpms = sorted(e["bpm"] for e in pool)
        t1 = bpms[len(bpms) // 3] if bpms else 0
        t2 = bpms[2 * len(bpms) // 3] if bpms else 0
        strata = {}
        for e in pool:
            b = 0 if e["bpm"] <= t1 else (1 if e["bpm"] <= t2 else 2)
            strata.setdefault((b, e["multi_tier"]), []).append(e)
        for s in strata.values():
            s.sort(key=lambda e: _hash(e["family"]))
        order = sorted(strata)                     # deterministic round-robin
        while len(picked[split]) < cap and any(strata.values()):
            for key in order:
                if strata[key] and len(picked[split]) < cap:
                    picked[split].append(strata[key].pop(0))
        picked[split] = picked[split][:cap]
    return picked


def run_one(e):
    dst = OUTROOT / e["family"].replace(":", "_")
    if (dst / "gen.osu").exists():
        return "done", 0.0
    aud = next((q for q in Path(e["dir"]).iterdir()
                if q.suffix.lower() in (".egg", ".ogg")), None)
    if aud is None:
        return "no_audio", 0.0
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        song = work / ("song" + (".ogg" if aud.suffix == ".egg"
                                 else aud.suffix))
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
             "title=" + e["family"], "hydra.run.dir=" + str(work / "logs")],
            cwd=MAPPER, env=env, capture_output=True, text=True)
        dur = time.time() - t0
        osus = sorted(work.glob("*.osu"), key=lambda p: p.stat().st_mtime)
        if r.returncode != 0 or not osus:
            (dst / "FAILED.txt").write_text(r.stderr[-4000:])
            return "failed", dur
        (dst / "gen.osu").write_bytes(osus[-1].read_bytes())
    (dst / "provenance.json").write_text(json.dumps({
        "family": e["family"], "split": e["split"], "genre": e["genre"],
        "corpus_dir": e["dir"], "audio_file": aud.name,
        "audio_sha256_16": hashlib.sha256(
            aud.read_bytes()).hexdigest()[:16],
        "mi_config": "v32", "difficulty": 5.5, "seed": SEED,
        "runtime_s": round(dur, 1), "salt": SALT,
        "clock_alignment": "human target shares this exact audio file"},
        indent=1))
    return "ok", dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10 ** 6)
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    sel_path = OUTROOT / "selection.json"
    if sel_path.exists():                          # frozen once, never reshuffled
        picked = json.loads(sel_path.read_text())
    else:
        picked = select_families()
        sel_path.write_text(json.dumps(picked, indent=1, sort_keys=True))
    print(f"selection: {len(picked['train'])} train, {len(picked['val'])} val "
          f"(salt {SALT}, seed {SEED})")
    if a.select_only:
        return
    results, n = [], 0
    consecutive_fail = 0
    for e in picked["train"] + picked["val"]:
        if n >= a.n:
            break
        status, dur = run_one(e)
        if status != "done":
            n += 1
        results.append({"family": e["family"], "split": e["split"],
                        "status": status, "runtime_s": round(dur, 1)})
        print(f"  [{status:7s}] {e['split']:5s} {e['family']:14s} {dur:5.0f}s",
              flush=True)
        consecutive_fail = consecutive_fail + 1 if status == "failed" else 0
        if consecutive_fail >= 3:
            print("STOP: 3 consecutive MI failures")
            break
    (OUTROOT / "run_log.json").write_text(json.dumps(results, indent=1))
    ok = sum(1 for r in results if r["status"] in ("ok", "done"))
    print(f"MI corpus: {ok}/{len(results)} ok")


if __name__ == "__main__":
    main()
