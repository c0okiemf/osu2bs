"""Chart-latent Task 2 (spec §3): small, identity-free chart context.

21 numeric fields for the latent prior: ten observables per
nonoverlapping 8 s window over the declared audio duration (rests
included, final partial window weighted by its true duration), each
summarized as duration-weighted mean and duration-weighted p90, plus
the explicit audio-available bit. Computed on the CANONICAL unmirrored
chart only. No mapper/title/family ID, file hash, duration, raw audio
fingerprint, telemetry target, accuracy, player statistic or NN result
enters the numeric tensor; hashes live only in provenance.

Head-group schema: same-hand notes at the same timestamp form ONE head;
head position = mean col/row of the group; head direction = the first
group note in (col,row) order. Coincidence uses the existing anchored
1e-3 s tolerance (no transitive clustering). Displacements spanning a
gap >2 s are excluded; transitions and occupied-gap pairs attribute to
the LATER head's window. Empty denominators yield 0 with support counts
retained in the audit.

Known caveat (documented, not hidden): the ordered four-head-token
concentration (field 7) orders same-time two-hand pairs by hand, so a
mirrored VIEW can permute tokens inside such pairs; production context
never sees mirrored scenes (canonical chart only) and the mirror
fixtures assert exact invariance for every field on pair-free charts
and for all non-sequential fields on paired charts.
"""
import bisect
import hashlib
import math

import torch

CONTEXT_VERSION = 1
WINDOW_S = 8.0
COINC_TOL = 1e-3
GAP_EXCLUDE_S = 2.0
LONG_GAP_S = 0.5
CLIP = 4.0
STD_FLOOR = 1e-3
OBS_NAMES = ("head_rate", "coincidence", "dx", "dy", "vertical_share",
             "dot_share", "gram4_max", "long_gap_share", "audio_rms_rel",
             "harmonic_share")
FIELD_ORDER = tuple(f"{n}_{s}" for n in OBS_NAMES
                    for s in ("mean", "p90")) + ("audio_available",)


def schema_sha():
    return hashlib.sha256(
        (f"v{CONTEXT_VERSION}:" + ",".join(FIELD_ORDER)).encode()
    ).hexdigest()


class ContextRecord:
    def __init__(self, fields, audit, provenance):
        self.fields = fields
        self.audit = audit
        self.provenance = provenance
        self.schema_sha256 = schema_sha()

    def tensor(self):
        return torch.tensor([self.fields[k] for k in FIELD_ORDER],
                            dtype=torch.float32)


def _heads(scene):
    """Same-hand same-time grouping on the canonical chart."""
    groups = {}
    for t, li, ll, c, d in scene["notes"]:
        groups.setdefault((round(t, 6), c), []).append((li, ll, d))
    heads = []
    for (t, hand), notes in groups.items():
        notes.sort()
        heads.append({"t": t, "hand": hand,
                      "col": sum(n[0] for n in notes) / len(notes),
                      "row": sum(n[1] for n in notes) / len(notes),
                      "d": notes[0][2]})
    heads.sort(key=lambda h: (h["t"], h["hand"]))
    return heads


def _weighted_mean(vals, weights):
    tot = sum(weights)
    return sum(v * w for v, w in zip(vals, weights)) / tot if tot else 0.0


def _weighted_p90(vals, weights):
    tot = sum(weights)
    if not tot:
        return 0.0
    acc = 0.0
    for v, w in sorted(zip(vals, weights)):
        acc += w
        if acc >= 0.9 * tot:
            return v
    return max(vals)


def chart_context(scene, audio_evidence):
    """ContextRecord for one canonical chart + its audio evidence."""
    heads = _heads(scene)
    aud = (audio_evidence or {}).get("audio") \
        if isinstance(audio_evidence, dict) and "policy" in \
        (audio_evidence or {}) else audio_evidence
    audio_ok = bool(aud and aud.get("supported"))
    if audio_ok:
        duration = float(aud["duration_s"])
    else:
        duration = (max(h["t"] for h in heads) if heads else 0.0) + 1e-6
    n_w = max(1, math.ceil(duration / WINDOW_S))
    w_dur = [min(WINDOW_S, duration - i * WINDOW_S) for i in range(n_w)]
    w_dur = [max(d, 1e-6) for d in w_dur]

    def win_of(t):
        return min(int(t // WINDOW_S), n_w - 1)

    # global sequences with later-window attribution
    occupied = sorted({h["t"] for h in heads})
    coincident = set()
    by_hand_t = {0: sorted(h["t"] for h in heads if h["hand"] == 0),
                 1: sorted(h["t"] for h in heads if h["hand"] == 1)}

    def _near(ts, t):
        i = bisect.bisect_left(ts, t - COINC_TOL)
        return i < len(ts) and ts[i] <= t + COINC_TOL
    for t in occupied:
        if _near(by_hand_t[0], t) and _near(by_hand_t[1], t):
            coincident.add(t)
    trans = {0: [], 1: []}                     # (later_t, |dx|, |dy|)
    for hand in (0, 1):
        hs = sorted((h for h in heads if h["hand"] == hand),
                    key=lambda h: h["t"])
        for a, b in zip(hs, hs[1:]):
            if b["t"] - a["t"] > GAP_EXCLUDE_S:
                continue
            trans[hand].append((b["t"], abs(b["col"] - a["col"]) / 3.0,
                                abs(b["row"] - a["row"]) / 2.0))
    gap_pairs = [(b, b - a > LONG_GAP_S)
                 for a, b in zip(occupied, occupied[1:])]

    per_w = {n: [0.0] * n_w for n in OBS_NAMES}
    audit = {"n_heads": len(heads), "n_windows": n_w,
             "windows_under4_heads": 0, "empty_windows": 0,
             "audio_available": audio_ok}
    heads_by_w = {}
    for h in heads:
        heads_by_w.setdefault(win_of(h["t"]), []).append(h)
    occ_by_w = {}
    for t in occupied:
        occ_by_w.setdefault(win_of(t), []).append(t)
    trans_by_w, gaps_by_w = {}, {}
    for hand in (0, 1):
        for t, dx, dy in trans[hand]:
            trans_by_w.setdefault(win_of(t), []).append((dx, dy))
    for t, is_long in gap_pairs:
        gaps_by_w.setdefault(win_of(t), []).append(is_long)
    ps = (aud or {}).get("per_second") or {}
    rms = ps.get("intensity_rms") or []
    harm = ps.get("harmonic_rms") or []
    perc = ps.get("percussive_rms") or []
    rms_p90 = 0.0
    if rms:
        s = sorted(rms)
        rms_p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
    for i in range(n_w):
        hs = heads_by_w.get(i, [])
        occ = occ_by_w.get(i, [])
        if not hs:
            audit["empty_windows"] += 1
        per_w["head_rate"][i] = len(hs) / w_dur[i]
        per_w["coincidence"][i] = (sum(1 for t in occ if t in coincident)
                                   / len(occ)) if occ else 0.0
        tr = trans_by_w.get(i, [])
        per_w["dx"][i] = (sum(d[0] for d in tr) / len(tr)) if tr else 0.0
        per_w["dy"][i] = (sum(d[1] for d in tr) / len(tr)) if tr else 0.0
        directional = [h for h in hs if h["d"] != 8]
        per_w["vertical_share"][i] = (
            sum(1 for h in directional if h["d"] in (0, 1))
            / len(directional)) if directional else 0.0
        per_w["dot_share"][i] = (sum(1 for h in hs if h["d"] == 8)
                                 / len(hs)) if hs else 0.0
        if len(hs) < 4:
            audit["windows_under4_heads"] += 1
            per_w["gram4_max"][i] = 0.0
        else:
            toks = [(h["hand"], h["col"], h["row"], h["d"],
                     h["t"] in coincident) for h in hs]
            grams = {}
            for k in range(len(toks) - 3):
                g = tuple(toks[k:k + 4])
                grams[g] = grams.get(g, 0) + 1
            per_w["gram4_max"][i] = max(grams.values()) / (len(toks) - 3)
        gp = gaps_by_w.get(i, [])
        per_w["long_gap_share"][i] = (sum(gp) / len(gp)) if gp else 0.0
        if audio_ok and rms:
            lo, hi = int(i * WINDOW_S), int(min(duration,
                                                (i + 1) * WINDOW_S))
            seg = rms[lo:max(hi, lo + 1)]
            per_w["audio_rms_rel"][i] = ((sum(seg) / len(seg))
                                         / max(rms_p90, 1e-6)) if seg \
                else 0.0
            sh = harm[lo:max(hi, lo + 1)]
            sp = perc[lo:max(hi, lo + 1)]
            if sh and sp:
                h_m, p_m = sum(sh) / len(sh), sum(sp) / len(sp)
                per_w["harmonic_share"][i] = h_m / (h_m + p_m + 1e-6)
    fields = {}
    for n in OBS_NAMES:
        fields[f"{n}_mean"] = _weighted_mean(per_w[n], w_dur)
        fields[f"{n}_p90"] = _weighted_p90(per_w[n], w_dur)
    fields["audio_available"] = 1.0 if audio_ok else 0.0
    provenance = {"chart_sha256": scene.get("chart_sha256"),
                  "info_sha256": scene.get("info_sha256"),
                  "context_version": CONTEXT_VERSION,
                  "duration_s": round(duration, 3)}
    return ContextRecord(fields, audit, provenance)


def fit_context_scaler(train_contexts):
    """One row per QA-train family; mean/std with the spec floor."""
    m = torch.stack([c.tensor() if isinstance(c, ContextRecord) else c
                     for c in train_contexts])
    return {"mu": m.mean(dim=0),
            "sd": m.std(dim=0).clamp(min=STD_FLOOR),
            "n_families": len(train_contexts),
            "schema_sha256": schema_sha()}


def standardize(context_tensor, scaler):
    """Standardized, clipped context + which dims clipped (diagnostic)."""
    z = (context_tensor - scaler["mu"]) / scaler["sd"]
    clipped = [FIELD_ORDER[i] for i in range(len(FIELD_ORDER))
               if abs(float(z[i])) > CLIP]
    return z.clamp(-CLIP, CLIP), clipped
