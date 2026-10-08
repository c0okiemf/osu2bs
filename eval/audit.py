"""Read-only benchmark audit (plan phase 0C).

  .venv/bin/python -m eval.audit --manifest eval/benchmark.json \
      --out experiments/flow-v1/baseline

Measures every difficulty file in the manifest with eval/map_reader metrics,
verifies frozen hashes, and writes metrics.json + metrics.csv + report.txt
into --out. Never generates, regenerates, or overwrites maps, and touches no
feature cache. Deterministic: two runs on the same inputs are identical.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.map_reader import read_dat, metrics, UnsupportedMap, METRIC_VERSION

CSV_COLS = ["kind", "song", "difficulty", "bpm", "raw_heads", "span_s",
            "raw_sps", "raw_peak_nps", "raw_dup8_pct",
            "raw_dup8_maxphase_pct", "raw_same_cell_pct", "raw_double_pct",
            "raw_hand_ratio", "raw_rest2s_pct", "lr_outward_row_doubles",
            "grouped_events", "grouped_ambiguous", "grouped_sps",
            "grouped_two_hand_pct"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_state():
    """Executing-source digests + git rev/dirty — audit provenance (R4)."""
    root = Path(__file__).parent.parent
    run = lambda *a: subprocess.run(a, text=True, capture_output=True,
                                    cwd=root).stdout.strip()
    return {"code_rev": run("git", "rev-parse", "HEAD") or "unknown",
            "git_dirty": bool(run("git", "status", "--porcelain")),
            "code_sha256": {p: sha(root / p) for p in
                            ("eval/map_reader.py", "eval/audit.py")},
            "metric_version": METRIC_VERSION}


def main(manifest_path, out_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, exclusions, hash_mismatches = [], [], []

    def verify(song, rel_or_abs, frozen, what):
        if frozen is None or rel_or_abs is None:
            return
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = Path(song["dir"]) / p
        if not p.exists():
            hash_mismatches.append(f"{song['id']}/{what}: file missing ({p})")
        elif sha(p) != frozen:
            hash_mismatches.append(
                f"{song['id']}/{what}: {sha(p)[:12]} != frozen {frozen[:12]}")

    for song in manifest["songs"]:
        d = Path(song["dir"])
        verify(song, song.get("audio"), song.get("audio_sha256"), "audio")
        verify(song, song.get("original_audio"),
               song.get("original_audio_sha256"), "original_audio")
        verify(song, song.get("source_osu"),
               song.get("source_osu_sha256"), "source_osu")
        for fname, frozen in sorted(song["difficulties"].items()):
            p = d / fname
            if not p.exists():
                exclusions.append(f"{song['id']}/{fname}: file missing")
                continue
            actual = sha(p)
            if actual != frozen:
                hash_mismatches.append(
                    f"{song['id']}/{fname}: {actual[:12]} != frozen {frozen[:12]}")
            try:
                r = read_dat(p)
            except UnsupportedMap as e:
                exclusions.append(f"{song['id']}/{fname}: {e}")
                continue
            m = metrics(r["notes"], 60000.0 / song["bpm"])
            m.update(kind=song["kind"], song=song["id"], difficulty=p.stem,
                     bpm=song["bpm"], sha256=actual,
                     hash_ok=actual == frozen, bombs=r["bombs"],
                     walls_in_file=r["walls"], chains=r["chains"],
                     version=r["version"], out_of_range=r["out_of_range"])
            if r["out_of_range"]:
                exclusions.append(f"{song['id']}/{fname}: "
                                  f"{r['out_of_range']} out-of-range notes dropped")
            rows.append(m)
    certified = not hash_mismatches
    result = {"manifest": str(manifest_path),
              "manifest_sha256": sha(manifest_path), **code_state(),
              "certified": certified,
              "exclusions": exclusions, "hash_mismatches": hash_mismatches,
              "maps": rows}
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    csv = [",".join(CSV_COLS)]
    for m in rows:
        csv.append(",".join(str(m.get(c, "")) for c in CSV_COLS))
    (out / "metrics.csv").write_text("\n".join(csv) + "\n")
    lines = [f"{'CERTIFIED' if certified else 'UNCERTIFIED'}: "
             f"{len(rows)} difficulty files measured, "
             f"{len(exclusions)} exclusions, "
             f"{len(hash_mismatches)} hash mismatches "
             f"(metric v{METRIC_VERSION})"]
    lines += ["EXCLUDED: " + e for e in exclusions]
    lines += ["HASH MISMATCH: " + h for h in hash_mismatches]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {out}/metrics.{{json,csv}}")
    return 0 if certified else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval/benchmark.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(main(a.manifest, a.out))
