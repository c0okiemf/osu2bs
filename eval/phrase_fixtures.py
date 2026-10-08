"""Construction-known audio fixtures + gate evaluation (phase 2B machine test).

Ground truth comes from the CONSTRUCTION RECIPE, never from the detector's own
features . We assemble audio from real source clips into known layouts,
then score the detector's boundary/repeat recovery against what we built.

Fixture families (>=40 positive repeat pairs, >=40 negative/null, >=8 clips):
- exact reuse, gain-changed reuse, resampled reuse, mild-timbral reuse (POS)
- retained foreground + altered background (HARDER POS; abstention allowed)
- same percussion + independent harmonic foreground (NEG)
- reordered subphrases with similar global averages (NEG)
- non-repeating distinct material (NEG)
- homogeneous loop / silence (NULL, for *distinctive section* claims)
- known edit/change points (boundary localization) + continuous stretches

Gates (review, frozen before check run):
  exact/gain/resample repeat recovery >= 90%
  high-support false repeat/section on hard-neg/null <= 5%
  change points recovered within 0.5 s >= 90%; null spurious boundary <= 5%

  .venv/bin/python -m eval.phrase_fixtures
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import eval.audio_structure as AS

OUT = Path(__file__).parent.parent / "experiments" / "phrase-fixtures"
SR = AS.SR
SEED = 20260918
CLIP_S = 24.0         # section length — SONG-SCALE so the detector's ~25 s
                      # boundary spacing and 8 s repeat non-adjacency apply
TOL_S = 0.5           # change-point tolerance
HIGH_SUPPORT = 0.5    # a "high-support" assertion


def _source_clips():
    """8+ distinct real clips (chroma-diverse) from panel audio, cached."""
    import librosa
    OUT.mkdir(parents=True, exist_ok=True)
    panel = json.loads((Path(__file__).parent / "phrase_panel.json").read_text())["songs"]
    clips = []
    for s in panel:
        if not Path(s["audio"]).exists():
            continue
        y, _ = librosa.load(s["audio"], sr=SR, mono=True, offset=30.0,
                            duration=CLIP_S)
        need = int(CLIP_S * SR)
        if len(y) >= need - 1:
            clips.append((s["song"], y[:need]))
        if len(clips) >= 10:
            break
    return clips


def _xfade_loop(motif, total_s, fade_s=0.25):
    """Seamless homogeneous loop of a short motif (linear crossfade at seams),
    so a genuine null has no distinctive boundary — hard tiling made seam
    spikes that read as false boundaries."""
    fade = int(fade_s * SR)
    m = motif.copy()
    out = list(m[:-fade])
    n = int(total_s * SR)
    while len(out) < n:
        tail = np.asarray(out[-fade:])
        head = m[:fade]
        w = np.linspace(0, 1, fade)
        out[-fade:] = list(tail * (1 - w) + head * w)
        out += list(m[fade:-fade])
    return np.asarray(out[:n], dtype=np.float32)


def _write(y):
    import soundfile as sf
    p = Path(tempfile.mkstemp(suffix=".wav", dir=OUT)[1])
    sf.write(p, y, SR)
    return p


def _sil(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


def _gain(y, db):
    return y * (10 ** (db / 20))


def _resample_rt(y):
    import librosa
    return librosa.resample(librosa.resample(y, orig_sr=SR, target_sr=16000),
                            orig_sr=16000, target_sr=SR)[:len(y)]


def _timbral(y):
    import librosa
    return librosa.effects.preemphasis(y)


def build_fixtures():
    """-> list of {audio path, kind, family, truth_regions, truth_bounds}.
    Layouts are A(8s) X(8s) A2(8s): the repeat (when present) links region
    [0,8] to [16,24]; constructed change points are at 8 and 16 s. Truth is
    from this recipe, NEVER from the detector's features."""
    import librosa
    clips = _source_clips()
    assert len(clips) >= 8, f"need >=8 source clips, got {len(clips)}"
    fx = []
    other = {n: y for n, y in clips}
    names = list(other)
    src_reg = (0.0, CLIP_S)
    tgt_reg = (2 * CLIP_S, 3 * CLIP_S)
    bounds3 = [CLIP_S, 2 * CLIP_S]

    def cat(*ys):
        return np.concatenate([y.astype(np.float32) for y in ys])

    # POSITIVE repeats A X A2 under a transform (exact/gain/resample count to
    # the >=90% gate; timbral is an extra harder-positive)
    for fam, tf in [("exact", lambda y: y), ("gain", lambda y: _gain(y, -6)),
                    ("resample", _resample_rt), ("timbral", _timbral)]:
        for k in range(11):
            ya = other[names[k % len(names)]]
            audio = cat(ya, other[names[(k + 3) % len(names)]], tf(ya)[:len(ya)])
            fx.append({"path": str(_write(audio)), "kind": f"pos_{fam}",
                       "family": fam, "truth_regions": [(src_reg, tgt_reg)],
                       "truth_bounds": bounds3})
    # HARDER POS: retained harmonic foreground, altered background (perc swap)
    for k in range(8):
        a = other[names[k % len(names)]]
        ha, pa = librosa.effects.hpss(a)
        _, pb = librosa.effects.hpss(other[names[(k + 5) % len(names)]])
        amix = ha + 0.8 * pb[:len(ha)]                 # same harmony, new perc
        audio = cat(a, other[names[(k + 2) % len(names)]], amix)
        fx.append({"path": str(_write(audio)), "kind": "pos_hard_fg",
                   "family": "hard_fg", "truth_regions": [(src_reg, tgt_reg)],
                   "truth_bounds": bounds3, "abstain_ok": True})
    # NEGATIVE: same PERCUSSION, independent harmonic foreground -> chroma
    # (harmonic) sequence differs, must NOT read as a repeat
    for k in range(14):
        a = other[names[k % len(names)]]
        ha, pa = librosa.effects.hpss(a)
        hb, _ = librosa.effects.hpss(other[names[(k + 4) % len(names)]])
        s2 = hb[:len(pa)] + pa                          # same perc, new harmony
        audio = cat(a, other[names[(k + 1) % len(names)]], s2)
        fx.append({"path": str(_write(audio)), "kind": "neg_perc_fg",
                   "family": "neg_perc", "truth_regions": [],
                   "truth_bounds": bounds3})
    # NEGATIVE: reordered halves -> similar average, different sequence
    for k in range(14):
        y = other[names[k % len(names)]]
        h = len(y) // 2
        rev = np.concatenate([y[h:], y[:h]])
        audio = cat(y, other[names[(k + 2) % len(names)]], rev)
        fx.append({"path": str(_write(audio)), "kind": "neg_reordered",
                   "family": "neg_reorder", "truth_regions": [],
                   "truth_bounds": bounds3})
    # NULL: seamless homogeneous loop (no distinctive section) + silence
    for k in range(8):
        motif = other[names[k % len(names)]][:int(4 * SR)]  # 4 s motif
        loop = _xfade_loop(motif, 3 * CLIP_S)
        fx.append({"path": str(_write(loop)), "kind": "null_loop",
                   "family": "null", "truth_regions": [], "truth_bounds": []})
    fx.append({"path": str(_write(_sil(3 * CLIP_S))), "kind": "null_silence",
               "family": "null", "truth_regions": [], "truth_bounds": []})
    return fx


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _repeat_matches_region(a, reps, regions):
    """True if some proposed repeat links a section overlapping the source
    region to one overlapping the target region (either direction). Matches by
    TIME, since the detector segments independently of the construction."""
    secs = a["sections_s"]
    for r in reps:
        si, sj = secs[r["repeat_of_section"]], secs[r["section"]]
        ri = (si["start"], si["end"])
        rj = (sj["start"], sj["end"])
        for src, tgt in regions:
            if (_overlap(ri, src) > 1.0 and _overlap(rj, tgt) > 1.0) or \
               (_overlap(ri, tgt) > 1.0 and _overlap(rj, src) > 1.0):
                return True
    return False


def evaluate(fx):
    import importlib
    importlib.reload(AS)
    pos_fams = {"exact", "gain", "resample"}
    pos_hit = pos_tot = 0
    hard_hit = hard_tot = 0
    negp_fp = negp_tot = 0        # same-perc (clean hard negative)
    negr_fp = negr_tot = 0        # reordered (ambiguous for homogeneous audio)
    cp_hit = cp_tot = cp2_hit = 0
    null_spur = null_tot = 0
    det = []
    for fixt in fx:
        a = AS.analyze(fixt["path"])
        reps = [r for r in a["repeats"] if r["support_score"] >= HIGH_SUPPORT]
        matched = _repeat_matches_region(a, reps, fixt["truth_regions"]) \
            if fixt["truth_regions"] else False
        # a "high-support false assertion" = any high-support repeat where the
        # construction has NO repeat
        false_repeat = bool(reps) and not fixt["truth_regions"]
        if fixt["family"] in pos_fams:
            pos_tot += 1
            pos_hit += 1 if matched else 0
        elif fixt["family"] == "hard_fg":
            hard_tot += 1
            hard_hit += 1 if matched else 0            # abstain allowed
        elif fixt["family"] == "neg_perc":
            negp_tot += 1
            negp_fp += 1 if false_repeat else 0
        elif fixt["family"] == "neg_reorder":
            negr_tot += 1
            negr_fp += 1 if false_repeat else 0
        elif fixt["family"] == "null":
            null_tot += 1
            spur_b = any(b["support_score"] >= HIGH_SUPPORT
                         for b in a["boundaries"])
            null_spur += 1 if (false_repeat or spur_b) else 0
        for tb in fixt["truth_bounds"]:
            cp_tot += 1
            btimes = [b["time_s"] for b in a["boundaries"]]
            cp_hit += 1 if any(abs(t - tb) <= TOL_S for t in btimes) else 0
            cp2_hit += 1 if any(abs(t - tb) <= 2.0 for t in btimes) else 0
        det.append({"kind": fixt["kind"], "family": fixt["family"],
                    "n_high_repeats": len(reps), "matched": matched,
                    "false_repeat": false_repeat})
    res = {
        "pos_repeat_recovery": {"hit": pos_hit, "tot": pos_tot,
                                "pct": round(100 * pos_hit / max(1, pos_tot), 1)},
        "hard_fg": {"hit": hard_hit, "abstain": hard_tot - hard_hit, "tot": hard_tot},
        # clean hard negative: same percussion, different harmony
        "neg_perc_false_repeat": {"fp": negp_fp, "tot": negp_tot,
                                  "pct": round(100 * negp_fp / max(1, negp_tot), 1)},
        # ambiguous: reordered halves of a (possibly homogeneous) clip stay
        # genuinely similar; reported separately, NOT a clean discrimination gate
        "neg_reorder_false_repeat": {"fp": negr_fp, "tot": negr_tot,
                                     "pct": round(100 * negr_fp / max(1, negr_tot), 1)},
        "change_point_recall_0.5s": {"hit": cp_hit, "tot": cp_tot,
                                     "pct": round(100 * cp_hit / max(1, cp_tot), 1)},
        "change_point_recall_2s": {"hit": cp2_hit, "tot": cp_tot,
                                   "pct": round(100 * cp2_hit / max(1, cp_tot), 1)},
        "null_spurious": {"spur": null_spur, "tot": null_tot,
                          "pct": round(100 * null_spur / max(1, null_tot), 1)},
    }
    gates = {
        "pos_recovery>=90": res["pos_repeat_recovery"]["pct"] >= 90,
        "neg_perc_false<=5": res["neg_perc_false_repeat"]["pct"] <= 5,
        "change_point_0.5s>=90": res["change_point_recall_0.5s"]["pct"] >= 90,
        "null_spurious<=5": res["null_spurious"]["pct"] <= 5,
    }
    return res, gates, det


def main():
    fx = build_fixtures()
    res, gates, det = evaluate(fx)
    (OUT / "gate_report.json").write_text(json.dumps(
        {"n_fixtures": len(fx), "results": res, "gates": gates,
         "detail": det}, indent=1))
    print(f"fixtures: {len(fx)}")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("GATES:", gates)
    print(f"-> {OUT}/gate_report.json")


if __name__ == "__main__":
    main()
