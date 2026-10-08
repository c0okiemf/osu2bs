"""Corpus inventory, song/audio-family grouping, and frozen splits (phase 5A).

One authoritative manifest for every learned artifact (rhythm, flow, critic,
ladder, difficulty calibration, preprocessing stats, future style
clustering). A family (a song and all its audio-identical copies / alternate
versions / difficulties) lives entirely in ONE split, so nothing leaks
across train/dev/test.

  .venv/bin/python -m eval.corpus build   # writes eval/corpus_manifest.json
  .venv/bin/python -m eval.corpus check   # re-asserts overlap + prints report

Families are connected components (union-find) over: same map id, same chart
sha at any tier, and same normalized (title+artist). Cross-split title joins
are confirmed with a low-rate waveform fingerprint and the evidence stored.
A family lives entirely in ONE split.

Splits (priority order):
- dev: reference gems (HOLDOUT_IDS) + tuned benchmark refs — never a clean test.
- mapper_eval: every family touched by a MAPPER_SLICE credit — a REAL
  mapper-disjoint slice, fully removed from train.
- test: a frozen ~10% family slice, SEALED and disclosed contaminated for the
  current weights (trained on all but HOLDOUT_IDS); post-retrain eval only.
- train: the rest.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import groom

MANIFEST = Path(__file__).parent / "corpus_manifest.json"
MANIFEST_VERSION = 2
TEST_FRACTION = 0.10
FP_CORR = 0.95     # waveform-fingerprint correlation confirming a same-song join
# target mappers held fully OUT of train for a mapper-generalization slice
# (review 5A finding 4: a real disjoint split, not a tag). WITHDRAWN for now
# (empty): the style-generalization claim is a phase-6 (personalities) concern,
# and carving the two most prolific mappers (Ryger 158, Joetastic 97 — many
# APPROVED, the style authority) out of the phase-5B rhythm pilot would gut
# training for no pilot benefit. Re-introduce by listing target mappers here
# when the generalization claim is actually made; the split machinery is ready.
MAPPER_SLICE = set()
# families kept in dev: the held-out gems plus benchmark refs already used for
# tuning (numb 14059, Hit That 12fbf) — all development material
DEV_IDS = set(groom.HOLDOUT_IDS) | {"14059", "12fbf"}
DIFF_NAMES = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
MIN_NOTES = 100    # load_map_all's usable-chart floor (parser-only census)
OFFGRID_SKIP = groom.OFFGRID_SKIP
MIN_SPAN_S = 30


def sha(b):
    return hashlib.sha256(b).hexdigest()


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def map_id(dirname):
    return dirname.split(" ")[0].lower()


def inventory():
    """Walk MAPS_DIRS -> (records, exclusions). One record per usable dir."""
    approved = {Path(p).resolve() for p in groom.APPROVED_DIRS}
    records, exclusions = [], []
    for base in groom.MAPS_DIRS:
        base = Path(base)
        if not base.exists():
            continue
        src = "approved" if base.resolve() in approved else "beatsaver"
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            ip = next((p for p in d.iterdir() if p.name.lower() == "info.dat"), None)
            if not ip:
                exclusions.append({"dir": d.name, "reason": "no Info.dat"})
                continue
            try:
                info = json.loads(ip.read_text(encoding="utf-8-sig"))
            except Exception as e:
                exclusions.append({"dir": d.name, "reason": f"info parse: {e}"})
                continue
            files = {}
            for s in info.get("_difficultyBeatmapSets", []):  # v2 info
                if s.get("_beatmapCharacteristicName", "Standard") != "Standard":
                    continue
                for dm in s.get("_difficultyBeatmaps", []):
                    if dm.get("_difficulty") in DIFF_NAMES:
                        files[dm["_difficulty"]] = dm.get("_beatmapFilename")
            for dm in info.get("difficultyBeatmaps", []):  # v4 info
                if (dm.get("characteristic") == "Standard"
                        and dm.get("difficulty") in DIFF_NAMES):
                    files[dm["difficulty"]] = dm.get("beatmapDataFilename")
            charts = {}
            for name, f in files.items():
                p = d / f if f else None
                if p is None or not p.exists():
                    # Info often names ExpertPlusStandard.dat while the file is
                    # ExpertPlus.dat (load_map_all handles the same mismatch)
                    p = next((q for q in d.iterdir() if q.suffix.lower() == ".dat"
                              and q.name.lower().startswith(name.lower())), None)
                if p is None or not p.exists():
                    continue
                b = p.read_bytes()
                try:
                    dat = json.loads(b.decode("utf-8-sig"))
                    n = len(dat.get("_notes") or dat.get("colorNotes") or [])
                except Exception:
                    n = -1
                charts[name] = {"sha256": sha(b), "notes": n, "file": p.name}
            if not charts:
                exclusions.append({"dir": d.name, "reason": "no Standard charts"})
                continue
            song = next((q for q in d.iterdir()
                         if q.suffix.lower() in (".egg", ".ogg")), None)
            records.append({
                "dir": str(d), "map_id": map_id(d.name), "source": src,
                "weight": groom.APPROVED_W if src == "approved" else 1.0,
                "song_name": info.get("_songName"),
                "song_author": info.get("_songAuthorName"),
                "level_author": info.get("_levelAuthorName"),
                "bpm": info.get("_beatsPerMinute"),
                "held_out": groom.held_out(d),
                "title_key": _norm(info.get("_songName")) + "|"
                             + _norm(info.get("_songAuthorName")),
                "audio_file": song.name if song else None,
                "audio_size": song.stat().st_size if song else None,
                "charts": charts,
            })
    return records, exclusions


def _fingerprint(path, secs=90, sr=4000):
    """Low-rate mono RMS-envelope of the first `secs` — cheap same-song
    similarity probe (review used aligned waveform correlation)."""
    import numpy as np
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=secs)
    env = librosa.feature.rms(y=y, hop_length=sr // 50)[0]  # ~50 Hz envelope
    return (env - env.mean()) / (env.std() + 1e-9)


def _corr(a, b, max_lag=1500):
    """Best normalized cross-correlation of two envelopes within +/- max_lag
    (~30s at 50 Hz), tolerating a lead/trim offset."""
    import numpy as np
    n = min(len(a), len(b))
    if n < 100:
        return 0.0
    a, b = a[:n], b[:n]
    best = 0.0
    for lag in range(-max_lag, max_lag + 1, 25):
        if lag >= 0:
            x, y = a[lag:], b[:n - lag]
        else:
            x, y = a[:n + lag], b[-lag:]
        if len(x) > 100:
            best = max(best, float(np.dot(x, y) / len(x)))
    return best


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i, j):
        self.p[self.find(i)] = self.find(j)


def _families(records):
    """Connected components (union-find) over: same map id, same chart sha at
    any tier, same normalized title+artist (non-empty). Cross-split title
    joins are fingerprint-confirmed and the evidence recorded. Empty metadata
    never joins (would collapse unrelated songs)."""
    uf = _UF(len(records))
    edges = []  # (i, j, reason) for the audit trail

    def link(groups, reason, fp_check=False):
        for members in groups.values():
            for j in members[1:]:
                uf.union(members[0], j)
                edges.append((members[0], j, reason))

    by_id, by_chart, by_title = {}, {}, {}
    for i, r in enumerate(records):
        by_id.setdefault(r["map_id"], []).append(i)
        for c in r["charts"].values():
            by_chart.setdefault(c["sha256"], []).append(i)
        if r["title_key"].strip("|"):
            by_title.setdefault(r["title_key"], []).append(i)
    link(by_id, "map_id")
    link(by_chart, "chart_sha")
    link(by_title, "title+artist")

    # fingerprint-confirm title joins that unite different map ids: record the
    # waveform correlation as evidence (a low value is flagged, not un-merged
    # — same title+artist is grouped conservatively to avoid any leakage).
    fp_evidence = []
    cache = {}
    for members in by_title.values():
        if len({records[i]["map_id"] for i in members}) < 2:
            continue
        reps = {}
        for i in members:
            reps.setdefault(records[i]["map_id"], i)
        ids = list(reps)
        for a, b in zip(ids, ids[1:]):
            ia, ib = reps[a], reps[b]
            try:
                for i in (ia, ib):
                    if i not in cache and records[i]["audio_file"]:
                        cache[i] = _fingerprint(
                            Path(records[i]["dir"]) / records[i]["audio_file"])
                corr = round(_corr(cache[ia], cache[ib]), 3) \
                    if ia in cache and ib in cache else None
            except Exception:
                corr = None
            fp_evidence.append({"title": records[ia]["title_key"],
                                "ids": [a, b], "corr": corr,
                                "confirmed": corr is not None and corr >= FP_CORR})
    fam_of = {i: "fam:" + records[uf.find(i)]["map_id"] for i in range(len(records))}
    return fam_of, edges, fp_evidence


def _split_for(fam_key, recs):
    """Family split, priority: dev (gems + benchmark refs) > mapper_eval
    (any target-mapper credit, fully out of train) > sealed test slice >
    song-disjoint val (model selection) > train. All by whole family."""
    if any(r["map_id"] in DEV_IDS or r["held_out"] for r in recs):
        return "dev"
    if any(_norm(r["level_author"]) in MAPPER_SLICE for r in recs):
        return "mapper_eval"
    h = int(hashlib.sha256(fam_key.encode()).hexdigest(), 16) % 100
    if h < TEST_FRACTION * 100:          # [0,10) sealed test
        return "test"
    if h < TEST_FRACTION * 100 + 8:      # [10,18) validation (selection)
        return "val"
    return "train"


def build():
    records, exclusions = inventory()
    fam_of, edges, fp_evidence = _families(records)
    fams = {}
    for i, r in enumerate(records):
        fams.setdefault(fam_of[i], []).append(i)
    split_of = {fk: _split_for(fk, [records[i] for i in idxs])
                for fk, idxs in fams.items()}
    # one representative dir per family for training sampling — duplicate
    # folders must not multiply a song's gradient mass (review finding 6)
    fam_rep = {fk: min(idxs, key=lambda i: records[i]["dir"])
               for fk, idxs in fams.items()}
    for i, r in enumerate(records):
        r["family"] = fam_of[i]
        r["split"] = split_of[fam_of[i]]
        r["mapper_slice"] = _norm(r["level_author"]) in MAPPER_SLICE
        r["family_rep"] = (i == fam_rep[fam_of[i]])
    manifest = {
        "version": MANIFEST_VERSION,
        "note": "Connected-component families; no family spans two splits. "
                "test is SEALED and contaminated for current weights — "
                "post-retrain eval only. mapper_eval is fully out of train.",
        "counts": _counts(records, fams, split_of),
        "join_edges": len(edges),
        "audio_fingerprints": fp_evidence,
        "eligibility": _eligibility(records),
        "maps": records, "exclusions": exclusions,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    _report(manifest)
    check()  # assert no leakage immediately


def _eligible_reason(rec):
    """Parser-only training eligibility for a record's ExpertPlus chart,
    replicating load_map_all's filters WITHOUT audio features (review 7).
    -> reason string ('ok' or why it would be skipped)."""
    if not rec["song_name"] and not rec["level_author"] and not rec["bpm"]:
        return "no_metadata"
    if not rec["audio_file"]:
        return "no_audio"
    bpm = rec["bpm"]
    ep = rec["charts"].get("ExpertPlus")
    if not ep:
        return "no_expertplus"
    p = Path(rec["dir"])
    dp = next((q for q in p.iterdir() if q.suffix.lower() == ".dat"
               and q.name.lower().startswith("expertplus")), None)
    if dp is None:
        return "no_expertplus"
    try:
        dat = json.loads(dp.read_text(encoding="utf-8-sig"))
    except Exception:
        return "parse_error"
    notes = dat.get("_notes") or dat.get("colorNotes") or []
    ts = [n.get("_time", n.get("b", 0)) for n in notes
          if n.get("_type", n.get("c")) in (0, 1)]
    if len(ts) < MIN_NOTES:
        return "too_few_notes"
    if not bpm:
        return "no_bpm"
    # SAME off-grid test as the loader: 1/4-step times with the baked-in
    # constant offset removed first (review v2-3 — the naive test mislabeled
    # 81 loadable maps as offgrid)
    steps = [t * 4 for t in ts]
    steps = [t - groom.grid_shift(steps) for t in steps]
    if groom.offgrid_fraction(steps) > OFFGRID_SKIP:
        return "offgrid_swing_triplet"
    if (max(ts) - min(ts)) * 60 / bpm < MIN_SPAN_S:
        return "too_short"
    return "ok"


def _eligibility(records):
    from collections import Counter
    by_reason = Counter()
    ok_by_split = Counter()
    for r in records:
        reason = _eligible_reason(r)
        r["eligible"] = reason
        by_reason[reason] += 1
        if reason == "ok":
            ok_by_split[r["split"]] += 1
    return {"by_reason": dict(by_reason),
            "eligible_ok_by_split": dict(ok_by_split)}


def _counts(records, fams, split_of):
    from collections import Counter
    c = {"maps": len(records), "families": len(fams)}
    c["by_split_maps"] = dict(Counter(r["split"] for r in records))
    c["by_split_families"] = dict(Counter(split_of.values()))
    c["by_source"] = dict(Counter(r["source"] for r in records))
    c["mapper_slice_maps"] = sum(1 for r in records if r["mapper_slice"])
    c["unique_mappers"] = len({_norm(r["level_author"]) for r in records
                               if r["level_author"]})
    # tier coverage (chart-level) per split — expose thin tiers before training
    tier = {}
    for r in records:
        for name in r["charts"]:
            tier.setdefault(r["split"], Counter())[name] += 1
    c["tier_charts_by_split"] = {k: dict(v) for k, v in tier.items()}
    # effective sampling mass: ONE representative dir per family (duplicate
    # folders are dropped, not summed), so the number reflects real gradient
    # influence per song, not folder/difficulty multiplicity (review 6)
    trainreps = [r for r in records if r["split"] == "train" and r["family_rep"]]
    c["train_effective_mass"] = {
        "approved_x4": round(sum(r["weight"] for r in trainreps
                                 if r["source"] == "approved"), 1),
        "beatsaver_x1": sum(1 for r in trainreps if r["source"] == "beatsaver"),
        "families": len(trainreps),
        "raw_dirs_before_dedup": sum(1 for r in records if r["split"] == "train")}
    c["genre_labels"] = "ABSENT — Info has no genre field; genre coverage " \
        "cannot be reported until labels are added (dev-panel labels TODO)"
    return c


def _report(m):
    c = m["counts"]
    print(f"corpus_manifest v{m['version']}: {c['maps']} maps, "
          f"{c['families']} families ({m['join_edges']} join edges), "
          f"{len(m['exclusions'])} excluded")
    print(f"  split maps:     {c['by_split_maps']}")
    print(f"  split families: {c['by_split_families']}")
    print(f"  source: {c['by_source']}  unique mappers: {c['unique_mappers']}")
    print(f"  train tiers: {c['tier_charts_by_split'].get('train', {})}")
    print(f"  train effective mass: {c['train_effective_mass']}")
    print(f"  eligibility: {m['eligibility']['by_reason']}")
    fp = [f for f in m["audio_fingerprints"] if f["corr"] is not None]
    lo = [f for f in fp if not f["confirmed"]]
    print(f"  audio fingerprints: {len(fp)} title-join pairs probed, "
          f"{len(lo)} below {FP_CORR} (grouped conservatively): "
          f"{[(f['title'][:24], f['corr']) for f in lo][:4]}")
    print(f"  genre: {c['genre_labels']}")


def _load_validated():
    """Read the manifest with fail-closed validation — the ONE gate every
    fitting entry point passes through (review findings 5 + v2-1). A missing,
    unreadable, or version-drifted manifest raises rather than returning a
    leaky/stale set."""
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "eval/corpus_manifest.json missing — run `python -m eval.corpus "
            "build` before any split-restricted fitting")
    m = json.loads(MANIFEST.read_text())
    if m.get("version") != MANIFEST_VERSION:
        raise ValueError(f"corpus manifest v{m.get('version')} != expected "
                         f"v{MANIFEST_VERSION}; rebuild it")
    return m


def split_dirs(split, require=True):
    """Dir paths in a given split — authoritative, fail-closed."""
    m = _load_validated()
    dirs = [r["dir"] for r in m["maps"] if r["split"] == split]
    if require and not dirs:
        raise ValueError(f"no maps in split {split!r}: "
                         f"{sorted(set(r['split'] for r in m['maps']))}")
    return dirs


def train_families_rep(split="train"):
    """Family-representative dirs for a split (one dir per family), through
    the SAME validated reader as split_dirs — no direct JSON bypass
    (review v2-1). Deduplicated training sample set (review finding 6)."""
    m = _load_validated()
    dirs = [r["dir"] for r in m["maps"] if r["split"] == split and r["family_rep"]]
    if not dirs:
        raise ValueError(f"no family reps in split {split!r}")
    return dirs


def check():
    global MANIFEST
    m = json.loads(MANIFEST.read_text())
    assert m.get("version") == MANIFEST_VERSION, "manifest version drift"
    fam_split = {}
    for r in m["maps"]:
        prev = fam_split.setdefault(r["family"], r["split"])
        assert prev == r["split"], \
            f"family {r['family']} spans {prev} and {r['split']}"
    # every audio-fingerprint-joined pair is in one split (no cross-split leak)
    fam_of = {r["map_id"]: r["family"] for r in m["maps"]}
    # mapper_eval must be FULLY out of train (a real disjoint slice)
    train_mappers = {_norm(r["level_author"]) for r in m["maps"]
                     if r["split"] == "train"}
    assert not (train_mappers & MAPPER_SLICE), \
        f"target mappers leaked into train: {train_mappers & MAPPER_SLICE}"
    for hid in DEV_IDS:
        present = [r for r in m["maps"] if r["map_id"] == hid]
        assert all(r["split"] == "dev" for r in present), \
            f"dev id {hid} not all in dev"
    # fail-closed manifest fixture: a bad version must raise, not fall back
    orig = MANIFEST
    try:
        bad = _P_tmp(m)
        MANIFEST = bad
        for fn in (lambda: split_dirs("train"),
                   lambda: train_families_rep("train")):  # both consumers
            try:
                fn()
                raise AssertionError("stale manifest did not fail closed")
            except ValueError:
                pass
    finally:
        MANIFEST = orig
    # reverify a sample of chart content hashes — a manifest label alone is
    # insufficient (review finding 2). Full re-hash on `check --hashes`.
    full = "--hashes" in sys.argv
    sample = m["maps"] if full else m["maps"][::37]
    bad = 0
    for r in sample:
        for c in r["charts"].values():
            dp = Path(r["dir"]) / c["file"]  # exact resolved filename
            if not dp.is_file() or sha(dp.read_bytes()) != c["sha256"]:
                bad += 1
    assert bad == 0, f"{bad} chart hashes no longer match the manifest"
    print(f"check ok: {len(fam_split)} families, none span two splits; "
          f"mapper_eval fully out of train; dev ids all in dev; "
          f"split_dirs fails closed on version drift; "
          f"{'all' if full else 'sampled'} chart hashes reverified "
          f"({len(sample)} maps)")


def _P_tmp(m):
    import tempfile
    d = json.loads(json.dumps(m))
    d["version"] = -1
    p = Path(tempfile.mkstemp(suffix=".json")[1])
    p.write_text(json.dumps(d))
    return p


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "check": check}[cmd]()
