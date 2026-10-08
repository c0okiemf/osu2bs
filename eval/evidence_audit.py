"""Paired upstream/human evidence audit (Packet D pre-pilot, CPU-only).

review's approved scope: audit ONLY existing frozen upstream (Mapperatorinator)
outputs paired with human targets on the exact same audio. Missing MI outputs
are reported as a coverage/budget item — this tool never launches MI inference.

The only existing paired data is the three held-out gems (eval/<name>/gen.osu
is the MI upstream; the corpus holds the human chart of the same song on the
same audio). For each, the evidence pipeline is dissected into distinct STAGES

  raw source onsets -> unthinned expanded hits -> legacy-thinned hits
  -> quarter-grid cells -> human target notes

with per-stage counts, collisions, drops, snap-error and rest/accent structure,
plus an audio-clock alignment check (offset from provenance, not by warping
onsets onto notes). For the training corpus it reports how many families LACK
MI upstream (the inference cost the pilot must budget), never hiding it.

  .venv/bin/python -m eval.evidence_audit   # writes experiments/evidence-audit/
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import convert
import groom
from eval.map_reader import read_dat
from eval.eval_gen import SONGS

OUT = Path(__file__).parent.parent / "experiments" / "evidence-audit"
# gem name -> (MI gen.osu, human chart dir) — same song, same audio
HUMAN = {
    "rap_god": Path.home() / "app/beat-saber-map-gen/input/bytrius/"
               "19909 (Rap God V2 - Ryger)",
    "reality_check": Path.home() / "app/beat-saber-map-gen/input/input/"
                     "25f (Reality Check Through The Skull - DM DOKURO)",
    "spaceman": Path.home() / "app/beat-saber-map-gen/input/input/"
                "24e5e (SPACEMAN - oegoe)",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _bins(times_ms, span_ms, win=2000):
    n = int(span_ms / win) + 1
    b = [0] * n
    for t in times_ms:
        if 0 <= t < span_ms:
            b[int(t / win)] += 1
    return b


def _human_ep(d):
    p = next((q for q in Path(d).iterdir() if q.suffix.lower() == ".dat"
              and q.name.lower().startswith("expertplus")), None)
    return p


def _audio_of(name):
    """The egg used to make gen.osu (eval_gen.SONGS) — and, by provenance,
    the same audio the human chart was mapped to. Returns (path, sha)."""
    a = Path(SONGS[name])
    return a, (sha(a) if a.exists() else None)


def audit_gem(name):
    osu = Path(__file__).parent / name / "gen.osu"
    meta, objects, bpm, offset = convert.parse_osu(osu)
    unthinned_hits = convert._hits(objects, thin=False)
    thinned_hits = convert._hits(objects, thin=True)
    raw = sorted(o["t"] for o in objects if o["kind"] != "spinner")
    thinned = sorted(h[0] for h in thinned_hits)
    unthinned = sorted(h[0] for h in unthinned_hits)
    steps, T, step_ms, off2, grid = convert.grid_steps(objects, bpm, offset,
                                                        meta.get("_timing"))
    # human target (same audio by provenance)
    hp = _human_ep(HUMAN[name])
    hdat = read_dat(hp)
    hbpm = json.loads((Path(HUMAN[name]) / next(
        q.name for q in Path(HUMAN[name]).iterdir()
        if q.name.lower() == "info.dat")).read_text(encoding="utf-8-sig")
        ).get("_beatsPerMinute")
    hbeat = 60000.0 / hbpm
    hnotes = [(b * hbeat, h) for b, h, c, l, d in hdat["notes"] if d != 8]
    human_heads = len(hnotes)                                  # arrow notes
    human_distinct = len({round(t, 1) for t, _ in hnotes})     # merged times
    # grouped per-hand swings (same time+hand = one swing)
    human_swings = len({(round(t, 1), h) for t, h in hnotes})
    human_ms = sorted(t for t, _ in hnotes)

    # STAGE 4: occupied grid cells for BOTH streams on the SAME adopted grid —
    # the sharper "does thinning change the model's input" measure
    def occupied(hits):
        return {grid.snap(h[0])[0] for h in hits}
    cells_un, cells_th = occupied(unthinned_hits), occupied(thinned_hits)
    changed = len(cells_un ^ cells_th)
    union = len(cells_un | cells_th)
    # per-10s-window changed-cell fraction (review: whole-song totals hide it)
    span = max((raw[-1] if raw else 0), (human_ms[-1] if human_ms else 0))
    win = 10000.0
    wchg = [0] * (int(span / win) + 1)
    wtot = [0] * (int(span / win) + 1)
    for s in cells_un | cells_th:
        w = int(grid.time(s) / win) if grid.time(s) < span else len(wchg) - 1
        wtot[min(w, len(wtot) - 1)] += 1
        if (s in cells_un) != (s in cells_th):
            wchg[min(w, len(wchg) - 1)] += 1
    max_win_frac = max((c / t for c, t in zip(wchg, wtot) if t >= 10), default=0.0)

    snap_err = sorted(grid.snap(h[0])[1] for h in thinned_hits)
    # nearest human-note displacement — a MAPPING-DISAGREEMENT statistic, NOT
    # clock alignment (review correction 2); clock alignment is by provenance
    import bisect
    disagree = []
    for t in thinned:
        i = bisect.bisect_left(human_ms, t)
        near = min((human_ms[j] for j in (i - 1, i) if 0 <= j < len(human_ms)),
                   key=lambda x: abs(x - t), default=None)
        if near is not None:
            disagree.append(abs(t - near))
    disagree.sort()
    _, aud_sha = _audio_of(name)

    def bins(times):
        b = _bins(times, span)
        return {"zero": sum(1 for c in b if c == 0),
                "le1": sum(1 for c in b if c <= 1), "total": len(b)}
    return {
        "gem": name, "bpm": bpm, "human_bpm": hbpm,
        "osu_sha": sha(osu), "human_chart_sha": sha(hp),
        "audio_sha": aud_sha,
        "clock_alignment": "same audio file by eval_gen provenance"
                           if aud_sha else "audio missing — unverified",
        "grid": grid.decision,
        "stages": {
            "raw_onsets": len(raw),
            "unthinned_hits": len(unthinned),
            "legacy_thinned": len(thinned),
            "thin_dropped_pct": round(100 * (len(unthinned) - len(thinned))
                                      / max(1, len(unthinned)), 3),
        },
        # comparable units on one clock (review correction 1)
        "counts": {"mi_distinct_times": len({round(t, 1) for t in thinned}),
                   "human_heads": human_heads,
                   "human_distinct_times": human_distinct,
                   "human_grouped_swings": human_swings},
        "density_vs_human_distinct": round(
            len({round(t, 1) for t in thinned}) / max(1, human_distinct), 3),
        "grid_thinning_effect": {
            "unthinned_cells": len(cells_un), "thinned_cells": len(cells_th),
            "changed_cells": changed,
            "changed_pct_of_union": round(100 * changed / max(1, union), 3),
            "max_10s_window_changed_pct": round(100 * max_win_frac, 1)},
        "snap_error_ms": {"p50": round(snap_err[len(snap_err) // 2], 1),
                          "p95": round(snap_err[int(0.95 * (len(snap_err) - 1))], 1),
                          "max": round(snap_err[-1], 1)},
        "mapping_disagreement_ms": {
            "median_nearest_note": round(disagree[len(disagree) // 2], 1),
            "note": "nearest human-note distance = mapping choice, NOT clock "
                    "offset; clock alignment is by audio provenance above"},
        "rest_bins_2s": {"mi": bins(thinned), "human": bins(human_ms)},
    }


def _norm(s):
    import re
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def corpus_coverage():
    """Train/val MI-upstream coverage + a search for EXISTING frozen MI
    outputs (out/, experiments/) whose song matches a corpus family, so
    reusable pairs reduce new inference (review correction 3). Beside-chart
    .osu search alone does not establish only 3 pairs exist."""
    root = Path(__file__).parent.parent
    m = json.loads((Path(__file__).parent / "corpus_manifest.json").read_text())
    beside = {r["dir"] for r in m["maps"]
              if any(q.suffix.lower() == ".osu" for q in Path(r["dir"]).iterdir())}
    train = [r for r in m["maps"] if r["split"] in ("train", "val")]
    # index frozen MI .osu outputs and match by normalized song title against
    # corpus family metadata (audio/version compat still to be verified)
    ext_osu = list((root / "out").glob("*/*.osu")) + \
        list((root / "experiments").glob("**/*.osu"))
    # corpus song-title norms (len>=4 to avoid trivial matches)
    corpus_titles = [(r, _norm(r["song_name"])) for r in m["maps"]
                     if len(_norm(r["song_name"])) >= 4]
    candidates = []
    for o in ext_osu:
        try:
            meta, *_ = convert.parse_osu(o)
        except Exception:
            continue
        key = _norm(meta.get("Title"))
        if len(key) < 4:
            continue
        # bidirectional containment = candidate counterpart (compat unverified)
        hits = [r for r, t in corpus_titles if t in key or key in t]
        if hits:
            candidates.append({"osu": str(o.relative_to(root)),
                               "title": meta.get("Title"),
                               "corpus_families": sorted({r["family"] for r in hits}),
                               "splits": sorted({r["split"] for r in hits}),
                               "corpus_song_names": sorted({r["song_name"] for r in hits}),
                               "audio_version_verified": False})
    return {
        "train_val_families": len({r["family"] for r in train}),
        "train_val_maps": len(train),
        "maps_with_beside_chart_osu": sum(1 for r in train if r["dir"] in beside),
        "external_mi_osu_indexed": len(ext_osu),
        "reusable_candidate_pairs": candidates,
        "note": "corpus = human BeatSaver maps, no beside-chart MI upstream. "
                "Frozen MI outputs in out/experiments are indexed and title-"
                "matched to corpus families below; audio/version compat NOT yet "
                "verified. Paired dense-evidence at corpus scale still needs MI "
                "inference per family (GPU budget) — NOT run here.",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gems = [audit_gem(n) for n in SONGS]
    cov = corpus_coverage()
    result = {"gems": gems, "corpus_coverage": cov}
    (OUT / "evidence_audit.json").write_text(json.dumps(result, indent=1))
    print("=== paired evidence audit (existing MI upstream: 3 gems) ===")
    for g in gems:
        c, e = g["counts"], g["grid_thinning_effect"]
        print(f"{g['gem']:14s} MI distinct {c['mi_distinct_times']:4d} vs human "
              f"distinct {c['human_distinct_times']:4d} (heads {c['human_heads']}, "
              f"swings {c['human_grouped_swings']}) density "
              f"{g['density_vs_human_distinct']}x")
        print(f"{'':14s} thinning on fixed grid: {e['changed_cells']} cells "
              f"changed ({e['changed_pct_of_union']}% union, max 10s window "
              f"{e['max_10s_window_changed_pct']}%)  snap p95 {g['snap_error_ms']['p95']}ms  "
              f"mapping-disagree {g['mapping_disagreement_ms']['median_nearest_note']}ms")
    print(f"\ncorpus coverage: {cov['train_val_families']} train/val families, "
          f"{cov['maps_with_beside_chart_osu']} with beside-chart MI osu; "
          f"{cov['external_mi_osu_indexed']} external MI osu indexed, "
          f"{len(cov['reusable_candidate_pairs'])} title-matched to corpus")
    for c in cov["reusable_candidate_pairs"]:
        print(f"  candidate: {c['title']!r} -> {c['splits']} "
              f"{c['corpus_families']} (audio compat unverified)")
    print(f"  -> {cov['note']}")
    print(f"-> {OUT}/evidence_audit.json")


if __name__ == "__main__":
    main()
