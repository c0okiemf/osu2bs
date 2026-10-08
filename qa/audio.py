"""QA audio evidence (architecture point 5): audio-derived features the
adjudicator consumes INSTEAD of listening — onset envelope, harmonic/
percussive energy, beat grid, chroma, spectral change, intensity and
section boundaries. Generator-independent: nothing here reads groom/critic
state or the four-channel groom.audio_features (explicitly too impoverished
for QA). Cached per audio content hash + AUDIO_EVIDENCE_VERSION.

Recorded limits: no true downbeat tracking (beat phase only), section
boundaries are unsupervised agglomerative estimates, and none of this
claims musical semantics — it is evidence for a judge, not ground truth.
"""
import hashlib
import json
from pathlib import Path

AUDIO_EVIDENCE_VERSION = 1
SR = 22050
HOP = 512


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audio_cache_key(audio_path, cache_dir, n_sections=12):
    """Complete cache identity: audio content hash + every numerical
    option + library version. Different bytes or config never collide."""
    import librosa as _lb
    cfg = hashlib.sha256(
        f"v{AUDIO_EVIDENCE_VERSION}:{SR}:{HOP}:{n_sections}:"
        f"{_lb.__version__}".encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{_sha(audio_path)[:16]}-{cfg}.json"


def evidence(audio_path, cache_dir=None, n_sections=12):
    """Per-second evidence arrays + onset/beat/section times (seconds)."""
    audio_path = Path(audio_path)
    key = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = audio_cache_key(audio_path, cache_dir, n_sections)
        if key.exists():
            return json.loads(key.read_text())
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import librosa
        import numpy as np
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
        dur = len(y) / sr
        if dur < 2.0 or float(np.max(np.abs(y)) if len(y) else 0.0) < 1e-5:
            return {"version": AUDIO_EVIDENCE_VERSION,
                    "duration_s": round(dur, 2), "supported": False,
                    "reason": "too_short_or_silent"}
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                            hop_length=HOP, units="time")
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP, units="time")
        y_h, y_p = librosa.effects.hpss(y)

        def per_second(x, agg=np.mean):
            n = max(1, int(dur))
            frames_per_s = len(x) / dur
            return [round(float(agg(
                x[int(i * frames_per_s):
                  max(int((i + 1) * frames_per_s), int(i * frames_per_s) + 1)]
            )), 5) for i in range(n)]
        rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
        rms_h = librosa.feature.rms(y=y_h, hop_length=HOP)[0]
        rms_p = librosa.feature.rms(y=y_p, hop_length=HOP)[0]
        cent = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                 hop_length=HOP)[0]
        flux = np.concatenate([[0.0], np.diff(onset_env)])
        chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
        feats = np.vstack([librosa.util.normalize(chroma, axis=1),
                           librosa.util.normalize(mfcc, axis=1)])
        k = min(n_sections, max(2, feats.shape[1] // (sr // HOP * 8)))
        bounds = librosa.segment.agglomerative(feats, k)
        bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP)
        frame_t = librosa.frames_to_time(
            range(len(onset_env)), sr=sr, hop_length=HOP)
        out = {
            "version": AUDIO_EVIDENCE_VERSION, "duration_s": round(dur, 2),
            "supported": True,
            "frames": {"times": [round(float(t), 4) for t in frame_t],
                       "onset_strength": [round(float(v), 5)
                                          for v in onset_env]},
            "tempo_bpm": round(float(np.atleast_1d(tempo)[0]), 2),
            "onset_times": [round(float(t), 3) for t in onsets],
            "beat_times": [round(float(t), 3) for t in beats],
            "section_bounds": [round(float(t), 3) for t in bound_times],
            "per_second": {
                "intensity_rms": per_second(rms),
                "harmonic_rms": per_second(rms_h),
                "percussive_rms": per_second(rms_p),
                "onset_strength": per_second(onset_env),
                "spectral_centroid": per_second(cent),
                "spectral_flux": per_second(np.abs(flux))},
            "limits": "beat phase only (no downbeats); sections are "
                      "unsupervised estimates; evidence, not semantics"}
    if key is not None:
        tmp = key.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(key)
    return out
