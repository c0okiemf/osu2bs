"""Audio-structure & phrase diagnostics (phase 2B, machine-only contract).

Explains WHERE the music changes / repeats so later phrase planning and F/G
catalog penalties have musical context. Read-only DIAGNOSTIC: proposes section
boundaries and repeats with UNCALIBRATED support scores and emits a phrase
DRAFT; it never feeds generation here, and it makes no real-song accuracy claim
(no human annotation exists — validation is on construction-known fixtures, see
eval/phrase_fixtures.py).

Repeats use TEMPORAL SEQUENCE matching of beat-synchronous chroma :
section averages match reordered music incorrectly, so we align time-resolved
sequences and report the duration ratio. Competing matches are preserved and
ambiguous cases abstain. Cache keys include audio + feature + proposal params
+ version; feature and proposal caches are separate.

  .venv/bin/python -m eval.audio_structure           # panel
  .venv/bin/python -m eval.audio_structure --audio X.egg   # one file
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PANEL = Path(__file__).parent / "phrase_panel.json"
OUT = Path(__file__).parent.parent / "experiments" / "phrase-diag"
FEAT_CACHE = OUT / "feat"
PROP_CACHE = OUT / "prop"
FEAT_VERSION = 2
PROP_VERSION = 2
SR = 22050
HOP = 512
# proposal params (in the proposal cache key so a retune invalidates it)
PARAMS = {"boundary_delta": 0.06, "boundary_gap_s": 10, "sec_per_boundary": 20,
          "boundary_edge_s": 4.0, "boundary_abs_floor": 0.08,
          "repeat_min_sep_s": 8.0, "repeat_seq_thr": 0.80,
          "repeat_margin_sd": 1.0, "dur_ratio": (0.7, 1.43),
          "accent_delta": 0.3, "accents_per_section": 8}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def _load_features(audio):
    """Beat-synchronous chroma/mfcc + frame novelty + energy baseline. Cached
    by audio content + feature version (separate from proposal cache)."""
    import numpy as np
    import librosa
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    key = FEAT_CACHE / f"v{FEAT_VERSION}_{sha(audio)}.npz"
    if key.exists():
        z = np.load(key, allow_pickle=True)
        return {k: z[k] for k in z.files} | {"dur": float(z["dur"])}
    y, sr = librosa.load(audio, sr=SR, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=sr, hop_length=HOP, n_mels=64))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    flux = np.concatenate([[0], np.maximum(0, np.diff(mel, axis=1)).sum(0)])
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr,
                                           hop_length=HOP)
    beats = np.asarray(beats)
    beat_t = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    bchroma = librosa.util.sync(chroma, beats, aggregate=np.mean) \
        if len(beats) else chroma
    out = {"times": times, "chroma": chroma, "mfcc": mfcc, "onset_env": onset_env,
           "flux": flux, "beat_t": beat_t, "bchroma": bchroma,
           "tempo": np.array(float(tempo)), "dur": len(y) / sr}
    np.savez(key, **out)
    return out | {"dur": len(y) / sr}


def _boundaries(f):
    import numpy as np
    import librosa
    feat = np.vstack([librosa.util.normalize(f["chroma"], axis=0),
                      librosa.util.normalize(f["mfcc"], axis=0)])
    win = max(1, int(2.0 * SR / HOP))
    nov = np.zeros(feat.shape[1])
    for i in range(1, feat.shape[1]):
        a, b = feat[:, max(0, i - win):i].mean(1), feat[:, i:i + win].mean(1)
        nov[i] = 1 - float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
    # RAW novelty (0..~1) is the support score — NOT normalized to max, so a
    # homogeneous/silent clip stays low and yields no boundary (normalizing to
    # max gave flat audio a spurious 1.0). Absolute floor + edge exclusion.
    gap = int(PARAMS["boundary_gap_s"] * SR / HOP)
    pk = librosa.util.peak_pick(nov, pre_max=win, post_max=win, pre_avg=win,
                                post_avg=win, delta=PARAMS["boundary_delta"], wait=gap)
    edge = PARAMS["boundary_edge_s"]
    pk = [p for p in pk if nov[p] >= PARAMS["boundary_abs_floor"]
          and edge <= f["times"][p] <= f["dur"] - edge]
    budget = max(0, min(12, round(f["dur"] / PARAMS["sec_per_boundary"])))
    pk = sorted(pk, key=lambda p: -nov[p])[:budget]  # budget, may be 0
    # localize: a real section change lands on a strong onset — snap each
    # coarse novelty peak to the nearest onset-envelope max within +/-1.5 s
    oe = f["onset_env"]
    r = int(1.5 * SR / HOP)
    out = []
    for p in pk:
        lo, hi = max(0, p - r), min(len(oe), p + r)
        p2 = lo + int(np.argmax(oe[lo:hi])) if hi > lo else p
        out.append({"time_s": round(float(f["times"][p2]), 2),
                    "support_score": round(float(nov[p]), 3)})
    return sorted(out, key=lambda b: b["time_s"])


def _sections(bounds, dur):
    edges = sorted(set([0.0] + [round(b["time_s"], 2) for b in bounds] + [round(dur, 2)]))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def _seq_match(bchroma, beat_t, s0, s1, t0, t1):
    """Sequence cosine of two beat-chroma spans, aligned by resampling the
    shorter onto the longer's length (order-preserving; a reordered phrase
    scores low even with matching averages). Returns (seq_cos, dur_ratio)."""
    import numpy as np
    ia = np.searchsorted(beat_t, s0), np.searchsorted(beat_t, s1)
    ib = np.searchsorted(beat_t, t0), np.searchsorted(beat_t, t1)
    A = bchroma[:, ia[0]:ia[1]]
    B = bchroma[:, ib[0]:ib[1]]
    if A.shape[1] < 2 or B.shape[1] < 2:
        return 0.0, 0.0
    L = min(A.shape[1], B.shape[1])
    idxA = np.linspace(0, A.shape[1] - 1, L).round().astype(int)
    idxB = np.linspace(0, B.shape[1] - 1, L).round().astype(int)
    a, b = A[:, idxA], B[:, idxB]
    a = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-9)
    seq = float((a * b).sum(0).mean())
    ratio = (A.shape[1] / B.shape[1]) if B.shape[1] else 0.0
    return seq, ratio


def _repeats(f, sections):
    """Temporal-sequence repeat proposals. Threshold is DISTINCTIVE over the
    song's own baseline (median+SD of all candidate pairs) AND above an
    absolute floor. Competing matches kept; abstain (empty) when nothing is
    distinctive. Duration ratio must be near 1 (no arbitrary warp)."""
    import numpy as np
    lo, hi = PARAMS["dur_ratio"]
    cand = []
    for j in range(len(sections)):
        for i in range(j - 1):
            if sections[j][0] - sections[i][1] <= PARAMS["repeat_min_sep_s"]:
                continue
            seq, ratio = _seq_match(f["bchroma"], f["beat_t"],
                                    *sections[i], *sections[j])
            if lo <= ratio <= hi:
                cand.append((i, j, seq, ratio))
    if not cand:
        return []
    seqs = np.array([c[2] for c in cand])
    base, sd = float(np.median(seqs)), float(np.std(seqs))
    thr = max(PARAMS["repeat_seq_thr"], base + PARAMS["repeat_margin_sd"] * sd)
    props = []
    for i, j, seq, ratio in cand:
        if seq >= thr:
            props.append({"repeat_of_section": i, "section": j,
                          "seq_cosine": round(seq, 3),
                          "duration_ratio": round(ratio, 3),
                          "distinctiveness": round(min(1.0, (seq - base)
                                                       / (1 - base + 1e-9)), 3),
                          # UNCALIBRATED support = the raw sequence agreement;
                          # distinctiveness (margin over the song's baseline) is
                          # reported separately, not multiplied in
                          "support_score": round(seq, 3)})
    return props  # competing matches preserved (a section may have several)


def _accents(f, sections):
    """Flux peaks, FULL-song coverage with a per-section budget (no first-N
    bias)."""
    import numpy as np
    import librosa
    pk = librosa.util.peak_pick(librosa.util.normalize(f["flux"]), pre_max=10,
                                post_max=10, pre_avg=20, post_avg=20,
                                delta=PARAMS["accent_delta"], wait=20)
    times = f["times"]
    per = {}
    for p in pk:
        t = float(times[p])
        for k, (a, b) in enumerate(sections):
            if a <= t < b:
                per.setdefault(k, []).append((round(t, 2), float(f["flux"][p])))
    out = []
    for k, lst in per.items():
        lst.sort(key=lambda x: -x[1])
        out += [t for t, _ in lst[:PARAMS["accents_per_section"]]]
    return sorted(out)


def analyze(audio):
    import numpy as np
    PROP_CACHE.mkdir(parents=True, exist_ok=True)
    pk = hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:8]
    key = PROP_CACHE / f"v{PROP_VERSION}_{pk}_{sha(audio)}.json"
    if key.exists():
        return json.loads(key.read_text())
    t0 = time.time()
    f = _load_features(audio)
    bounds = _boundaries(f)
    secs = _sections(bounds, f["dur"])
    reps = _repeats(f, secs)
    accents = _accents(f, secs)

    def act(a, b):
        i0, i1 = int(np.searchsorted(f["times"], a)), int(np.searchsorted(f["times"], b))
        return round(float(f["onset_env"][i0:max(i0 + 1, i1)].mean()), 3)
    r = {"audio_sha": sha(audio), "feat_version": FEAT_VERSION,
         "prop_version": PROP_VERSION, "params_hash": pk, "params": PARAMS,
         "duration_s": round(f["dur"], 1),
         "tempo_bpm_est": round(float(f["tempo"]), 1),
         "tempo_uncertainty": "librosa global estimate; approximate, not authoritative",
         "n_sections": len(secs),
         "sections_s": [{"start": a, "end": b, "activity": act(a, b)} for a, b in secs],
         "boundaries": bounds, "repeats": reps, "accent_candidates_s": accents,
         "support_semantics": "support_score is an UNCALIBRATED heuristic, not a "
                              "probability of correctness; no human ground truth",
         "runtime_s": round(time.time() - t0, 1)}
    key.write_text(json.dumps(r, indent=1))
    return r


def phrase_draft(a, song):
    rep = {}
    for r in a["repeats"]:  # competing matches: keep the strongest per section
        if r["section"] not in rep or r["support_score"] > rep[r["section"]]["support_score"]:
            rep[r["section"]] = r
    phrases = []
    for i, s in enumerate(a["sections_s"]):
        phrases.append({
            "section_id": f"S{i}", "start_s": s["start"], "end_s": s["end"],
            "activity_envelope": s["activity"],
            "repeat_of": f"S{rep[i]['repeat_of_section']}" if i in rep else None,
            "repeat_support": rep[i]["support_score"] if i in rep else None,
            "accent_candidates_s": [t for t in a["accent_candidates_s"]
                                    if s["start"] <= t < s["end"]],
            "musical_focus": "unknown", "motif_id": None, "body_motion_intent": None,
            "evidence": "novelty boundary + beat-sync sequence repeat (machine)",
            "uncertainty": "uncalibrated support; unvalidated vs human annotation"})
    return {"song": song, "seconds_authoritative": True,
            "tempo_bpm_est": a["tempo_bpm_est"], "no_semantic_labels": True,
            "phrases": phrases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--song")
    args = ap.parse_args()
    if args.audio:
        panel = [{"song": Path(args.audio).stem, "audio": args.audio,
                  "group": "?", "role": "?"}]
    else:
        panel = json.loads(PANEL.read_text())["songs"]
        if args.song:
            panel = [s for s in panel if s["song"] == args.song]
    for s in panel:
        if not Path(s["audio"]).exists():
            print(f"  UNSUPPORTED (missing audio): {s['song']}")
            continue
        a = analyze(s["audio"])
        d = phrase_draft(a, s["song"])
        (OUT / (s["song"].replace(" ", "_").replace("!", "").replace("/", "-")
                + ".draft.json")).write_text(json.dumps(d, indent=1))
        print(f"  {s.get('group','?'):11s} {s.get('role','?'):5s} "
              f"{s['song'][:24]:24s} {a['duration_s']:5.0f}s "
              f"{a['n_sections']:2d} sections {len(a['repeats'])} repeats "
              f"{len(a['accent_candidates_s'])} accents ({a['runtime_s']}s)")


if __name__ == "__main__":
    main()
