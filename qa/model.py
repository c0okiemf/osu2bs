"""Independent-QA Task 4: conditional kinematics mixture (spec §5).

Predicts a DISTRIBUTION of measured kinematic descriptors per note from the
final scene + audio evidence + optional body profile — never a single
optimal swing, never an accept label. 2x128 GELU MLP -> 4-component
diagonal Gaussian mixture over 10 log1p-transformed SI targets; mixture
log-scales clamped [-4, 2]; masked NLL divides by observed component
count and missing components are ABSENT from the likelihood, not imputed
zero. Serving has no target trajectory: `evaluate_map` returns quantiles/
samples/support and no observed_nll (that function exists only for replay
validation). No outcomes, poses, player/HMD identifiers or generator
features ever enter the inputs.
"""
import bisect
import math

import torch
import torch.nn.functional as F
from torch import nn

N_TARGETS = 10
N_COMPONENTS = 4
TARGET_NAMES = ("path_m", "speed_p90", "accel_p90", "lateral_extent",
                "vertical_extent", "rotation_rad", "angular_speed_p90",
                "other_path_m", "head_path_m", "min_hand_separation")
CTX_EACH = 4                      # prev/next events per hand
AUDIO_BINS = 8


def _ctx_names(prefix):
    out = []
    for i in range(1, CTX_EACH + 1):
        out += [f"{prefix}{i}_ok", f"{prefix}{i}_dt", f"{prefix}{i}_dx",
                f"{prefix}{i}_dy", f"{prefix}{i}_dir_sin",
                f"{prefix}{i}_dir_cos"]
    return out


FEATURE_NAMES = tuple(
    ["note_x", "note_y", "note_dir_sin", "note_dir_cos", "note_is_dot",
     "hand_right"]
    + _ctx_names("prev") + _ctx_names("next")
    + _ctx_names("oprev")[:6] + _ctx_names("onext")[:6]
    + ["simul_ok", "simul_dx", "simul_dy",
       "cad_2s", "cad_8s", "rest_2s", "doubles_2s", "doubles_8s",
       "njs", "njs_ok", "offset_beats", "offset_ok",
       "body_height", "body_height_known", "body_left_handed"]
    + [f"audio_onset_{i}" for i in range(AUDIO_BINS)]
    + [f"audio_harm_{i}" for i in range(AUDIO_BINS)]
    + [f"audio_perc_{i}" for i in range(AUDIO_BINS)])

_DIR_ANGLE = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}


def target_from_window(window):
    """(y[10], mask[10]) with log1p on nonnegative targets; None components
    are masked out, never zero-imputed."""
    t = window["targets"]
    y, mask = [], []
    for name in TARGET_NAMES:
        v = t.get(name)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            y.append(0.0)
            mask.append(False)
        else:
            y.append(math.log1p(max(0.0, float(v))))
            mask.append(True)
    return torch.tensor(y, dtype=torch.float32), \
        torch.tensor(mask, dtype=torch.bool)


def _dir_feat(d):
    if d == 8 or d not in _DIR_ANGLE:
        return 0.0, 0.0, 1.0
    a = math.radians(_DIR_ANGLE[d])
    return math.sin(a), math.cos(a), 0.0


def map_features(scene, audio, note_index, body_profile):
    """Feature vector for one scene note. Final scene + audio evidence +
    body profile only; padding uses explicit *_ok masks."""
    notes = scene["notes"]
    t, li, ll, color, d = notes[note_index]
    ds, dc, dot = _dir_feat(d)
    feats = {"note_x": (li - 1.5) / 1.5, "note_y": (ll - 1.0),
             "note_dir_sin": ds, "note_dir_cos": dc, "note_is_dot": dot,
             "hand_right": 1.0 if color == 1 else 0.0}
    same = [(i, n) for i, n in enumerate(notes) if n[3] == color]
    other = [(i, n) for i, n in enumerate(notes) if n[3] != color]
    pos_same = next(k for k, (i, _n) in enumerate(same) if i == note_index)

    def _fill(prefix, seq, pos, direction):
        for k in range(1, CTX_EACH + 1):
            j = pos + k * direction
            ok = 0 <= j < len(seq)
            n2 = seq[j][1] if ok else None
            feats[f"{prefix}{k}_ok"] = 1.0 if ok else 0.0
            feats[f"{prefix}{k}_dt"] = (
                max(-2.0, min(2.0, (n2[0] - t))) if ok else 0.0)
            feats[f"{prefix}{k}_dx"] = ((n2[1] - li) / 3.0) if ok else 0.0
            feats[f"{prefix}{k}_dy"] = ((n2[2] - ll) / 2.0) if ok else 0.0
            if ok:
                s2, c2, _ = _dir_feat(n2[4])
            else:
                s2 = c2 = 0.0
            feats[f"{prefix}{k}_dir_sin"] = s2
            feats[f"{prefix}{k}_dir_cos"] = c2
    _fill("prev", same, pos_same, -1)
    _fill("next", same, pos_same, +1)
    # nearest other-hand prev/next (single each)
    ot_prev = max((n for n in other if n[1][0] <= t),
                  key=lambda x: x[1][0], default=None)
    ot_next = min((n for n in other if n[1][0] > t),
                  key=lambda x: x[1][0], default=None)
    for pre, nb in (("oprev1", ot_prev), ("onext1", ot_next)):
        ok = nb is not None
        n2 = nb[1] if ok else None
        feats[f"{pre}_ok"] = 1.0 if ok else 0.0
        feats[f"{pre}_dt"] = (max(-2.0, min(2.0, n2[0] - t))) if ok else 0.0
        feats[f"{pre}_dx"] = ((n2[1] - li) / 3.0) if ok else 0.0
        feats[f"{pre}_dy"] = ((n2[2] - ll) / 2.0) if ok else 0.0
        if ok:
            s2, c2, _ = _dir_feat(n2[4])
        else:
            s2 = c2 = 0.0
        feats[f"{pre}_dir_sin"] = s2
        feats[f"{pre}_dir_cos"] = c2
    simul = next((n for i, n in other
                  if abs(n[0] - t) <= 1e-3), None)
    feats["simul_ok"] = 1.0 if simul else 0.0
    feats["simul_dx"] = ((simul[1] - li) / 3.0) if simul else 0.0
    feats["simul_dy"] = ((simul[2] - ll) / 2.0) if simul else 0.0
    ts = [n[0] for n in notes]
    lo2 = bisect.bisect_left(ts, t - 2.0)
    hi2 = bisect.bisect_right(ts, t + 2.0)
    lo8 = bisect.bisect_left(ts, t - 8.0)
    hi8 = bisect.bisect_right(ts, t + 8.0)
    n2s, n8s = hi2 - lo2, hi8 - lo8
    by_t = {}
    for tt, *_r in notes[lo8:hi8]:
        by_t[round(tt, 6)] = by_t.get(round(tt, 6), 0) + 1
    dbl8 = sum(1 for v in by_t.values() if v >= 2) / max(1, len(by_t))
    by_t2 = {}
    for tt, *_r in notes[lo2:hi2]:
        by_t2[round(tt, 6)] = by_t2.get(round(tt, 6), 0) + 1
    feats["cad_2s"] = n2s / 4.0 / 4.0
    feats["cad_8s"] = n8s / 16.0 / 4.0
    feats["rest_2s"] = 1.0 if n2s <= 1 else 0.0
    feats["doubles_2s"] = (sum(1 for v in by_t2.values() if v >= 2)
                           / max(1, len(by_t2)))
    feats["doubles_8s"] = dbl8
    st = scene.get("settings") or {}
    njs = st.get("njs")
    off = st.get("offset_beats")
    feats["njs"] = (float(njs) / 20.0) if isinstance(njs, (int, float)) \
        else 0.0
    feats["njs_ok"] = 1.0 if isinstance(njs, (int, float)) else 0.0
    feats["offset_beats"] = float(off) if isinstance(off, (int, float)) \
        else 0.0
    feats["offset_ok"] = 1.0 if isinstance(off, (int, float)) else 0.0
    h = body_profile.get("height")
    known = bool(body_profile.get("height_known")) and h
    feats["body_height"] = (float(h) - 1.7) if known else 0.0
    feats["body_height_known"] = 1.0 if known else 0.0
    feats["body_left_handed"] = 1.0 if body_profile.get("left_handed") \
        else 0.0
    # audio bins over ±1 s
    fr = (audio or {}).get("frames") or {}
    ftimes, fons = fr.get("times") or [], fr.get("onset_strength") or []
    for i in range(AUDIO_BINS):
        a = t - 1.0 + 2.0 * i / AUDIO_BINS
        b = a + 2.0 / AUDIO_BINS
        lo = bisect.bisect_left(ftimes, a)
        hi = bisect.bisect_right(ftimes, b)
        seg = fons[lo:hi]
        feats[f"audio_onset_{i}"] = (sum(seg) / len(seg)) if seg else 0.0
    ps = (audio or {}).get("per_second") or {}
    for name, key in (("audio_harm", "harmonic_rms"),
                      ("audio_perc", "percussive_rms")):
        arr = ps.get(key) or []
        for i in range(AUDIO_BINS):
            sec = int(t - 1.0 + 2.0 * i / AUDIO_BINS)
            feats[f"{name}_{i}"] = float(arr[sec]) \
                if 0 <= sec < len(arr) else 0.0
    return torch.tensor([feats[k] for k in FEATURE_NAMES],
                        dtype=torch.float32)


class MotionMixture(nn.Module):
    def __init__(self, n_in=len(FEATURE_NAMES)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.GELU(),
                                 nn.Linear(128, 128), nn.GELU())
        self.w_head = nn.Linear(128, N_COMPONENTS)
        self.mu_head = nn.Linear(128, N_COMPONENTS * N_TARGETS)
        self.ls_head = nn.Linear(128, N_COMPONENTS * N_TARGETS)

    def forward(self, x):
        h = self.net(x)
        w = torch.log_softmax(self.w_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, N_COMPONENTS, N_TARGETS)
        ls = self.ls_head(h).view(-1, N_COMPONENTS, N_TARGETS)
        return w, mu, ls.clamp(-4.0, 2.0)


def mixture_quantiles(log_weights, means, log_scales, probs=(0.1, 0.5, 0.9)):
    """Deterministic marginal quantiles of the diagonal Gaussian mixture
    (parity-repair spec §3): float64 CDF inversion by bracketed bisection.
    F_j(t) = sum_k w_k Phi((t - mu_kj)/sigma_kj); shapes [B,K], [B,K,T],
    [B,K,T] -> [B, len(probs), T] float64. No RNG anywhere; explicit
    errors on nonfinite parameters or nonconvergence (CDF error > 1e-7
    after bracket expansion + 80 bisection iterations)."""
    w = log_weights.detach().double().exp()
    mu = means.detach().double()
    sd = log_scales.detach().double().exp()
    if not (torch.isfinite(w).all() and torch.isfinite(mu).all()
            and torch.isfinite(sd).all()) or bool((sd <= 0).any()):
        raise ValueError("nonfinite mixture parameters")
    n_b, _n_k, n_t = mu.shape
    p = torch.tensor(probs, dtype=torch.float64)
    n_p = len(probs)

    def cdf(t):                              # t: [B,P,T]
        z = (t[:, :, None, :] - mu[:, None, :, :]) / sd[:, None, :, :]
        return (w[:, None, :, None] * torch.special.ndtr(z)).sum(dim=2)

    lo = (mu - 12 * sd).amin(dim=1)[:, None, :].repeat(1, n_p, 1)
    hi = (mu + 12 * sd).amax(dim=1)[:, None, :].repeat(1, n_p, 1)
    pt = p[None, :, None].expand(n_b, n_p, n_t)
    for _ in range(8):                       # validated, expand if needed
        bad_lo = cdf(lo) > pt
        bad_hi = cdf(hi) < pt
        if not bool(bad_lo.any() or bad_hi.any()):
            break
        span = hi - lo
        lo = torch.where(bad_lo, lo - span, lo)
        hi = torch.where(bad_hi, hi + span, hi)
    else:
        raise ValueError("quantile bracket validation failed")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = cdf(mid) < pt
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    q = 0.5 * (lo + hi)
    err = (cdf(q) - pt).abs().max()
    if not bool(err <= 1e-7):
        raise ValueError(f"quantile nonconvergence: CDF error "
                         f"{float(err):.3e}")
    return q


def mixture_nll(log_w, mu, log_scale, y, mask):
    """Masked mixture NLL: per-sample log-sum-exp over components with
    missing target components absent from the likelihood; normalized by
    the observed component count; fully-masked rows contribute nothing."""
    y_ = y[:, None, :].expand_as(mu)
    m_ = mask[:, None, :].expand_as(mu)
    lp = -0.5 * (((y_ - mu) / log_scale.exp()) ** 2) \
        - log_scale - 0.5 * math.log(2 * math.pi)
    lp = torch.where(m_, lp, torch.zeros_like(lp))
    comp = lp.sum(-1) + log_w
    obs = mask.sum(-1).clamp(min=1).float()
    row = -torch.logsumexp(comp, dim=-1) / obs
    keep = mask.any(-1)
    if not keep.any():
        return torch.tensor(0.0)
    return row[keep].mean()


def evaluate_map(scene, audio, model, body_profile, scaler=None,
                 n_samples=32, seed=20260924):
    """Map-only serving: per-note ANALYTIC predictive quantiles (the same
    deterministic routine calibration/diagnostics use) + fixed-seed samples
    kept for illustration only. NO observed_nll exists here by
    construction."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for i in range(len(scene["notes"])):
            x = map_features(scene, audio, i, body_profile)
            if scaler is not None:
                x = (x - scaler[0]) / scaler[1]
            w, mu, ls = model(x[None])
            probs = w.exp()[0]
            comp = torch.multinomial(probs.expand(n_samples, -1), 1,
                                     generator=g).squeeze(-1)
            eps = torch.randn(n_samples, N_TARGETS, generator=g)
            samp = mu[0][comp] + eps * ls[0][comp].exp()
            qs = mixture_quantiles(w, mu, ls)[0].float()
            out.append({"note_index": i,
                        "quantiles": {"p10": qs[0].tolist(),
                                      "p50": qs[1].tolist(),
                                      "p90": qs[2].tolist()},
                        "samples": samp.tolist(),
                        "mixture_entropy": float(
                            -(probs * probs.clamp(min=1e-9).log()).sum()),
                        "targets_space": "log1p_SI"})
    return out
