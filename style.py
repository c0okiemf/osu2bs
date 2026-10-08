"""Q4.1: supported style vocabulary (spec §7, first bullet).

Phrase/geometry descriptors per TRAIN family from the human window-target
dataset (phrase_planner windows — human charts only, train split), raw kept;
residualized against cadence, coincidence, tempo and event count with
train-only OLS; K-means at K=4/6/10 (fixed seed) plus neutral K=1.

Support rules: exposed style needs >=25 unique train families; >=2 residual
descriptors must differ >=0.5 train SD from the pooled rest (matched
workload holds because workload confounds are regressed out); bootstrap
assignment stability >=0.75 mean adjusted Rand over 20 family resamples.
Smallest supported K wins by VAL stability, never D quality. Unsupported
clusters merge to neutral. Artist/title/genre strings are never inputs;
corpus genre labels are incomplete and no genre-support claim is made.

Artifacts: experiments/quality-v1/q4-style/style_vocab.json (cluster ->
stable ID map, residualization + normalization coefficients, versioned).
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WINDOWS = ROOT / "experiments" / "quality-v1" / "q3-planner" / "windows"
OUT = ROOT / "experiments" / "quality-v1" / "q4-style"
SEED = 20260921
KS = (4, 6, 10)
MIN_FAMS = 25
MIN_ARI = 0.75
N_RESAMPLE = 20
SEP_SD = 0.5
MIN_SEP_DIMS = 2

# y layout (phrase_planner TARGETS + BINARY)
_T = ["occ_rate", "coincidence", "cad_l", "cad_r", "burst", "travel",
      "axis_vert", "axis_horiz", "axis_diag", "motif_conc", "quiet",
      "accent_occupied"]

DESCRIPTORS = ["axis_vert", "axis_horiz", "axis_diag", "travel_mean",
               "motif_mean", "quiet_share", "accent_share", "occ_std",
               "cad_asym", "burst_std", "coinc_std", "motif_std"]
CONFOUNDS = ["cad_mean", "coinc_mean", "tempo", "log_events"]


def _row_from_y(y, bpm):
    c = {k: y[:, _T.index(k)] for k in _T}
    desc = [c["axis_vert"].mean(), c["axis_horiz"].mean(),
            c["axis_diag"].mean(), c["travel"].mean(),
            c["motif_conc"].mean(), c["quiet"].mean(),
            c["accent_occupied"].mean(), c["occ_rate"].std(),
            np.abs(c["cad_l"] - c["cad_r"]).mean(), c["burst"].std(),
            c["coincidence"].std(), c["motif_conc"].std()]
    conf = [(c["cad_l"] + c["cad_r"]).mean(), c["coincidence"].mean(),
            bpm / 300.0,
            float(np.log(max(1.0, c["occ_rate"].sum() * 32)))]
    return (np.array(desc, dtype=np.float64), np.array(conf, dtype=np.float64))


def family_rows_full(split, limit=10 ** 6):
    """(dir, descriptor, confound) per FAMILY-REP dir of the whole split —
    human window targets only (no MI needed), idempotent per-dir JSON cache.
    Numeric only; no artist/title/genre string ever enters."""
    import hashlib
    from phrase_planner import human_window_targets
    from eval import corpus
    cache = OUT / "desc"
    cache.mkdir(parents=True, exist_ok=True)
    rows, n_new = [], 0
    for dp in sorted(corpus.train_families_rep(split=split)):
        key = hashlib.sha256(dp.encode()).hexdigest()[:16]
        cp = cache / f"{key}.json"
        if cp.exists():
            rec = json.loads(cp.read_text())
        else:
            if n_new >= limit:
                continue
            y, bpm = human_window_targets(dp)
            rec = ({"dir": dp, "split": split, "status": "skip", "why": bpm}
                   if y is None else
                   {"dir": dp, "split": split, "status": "ok",
                    "y": y.tolist(), "bpm": bpm})
            tmp = cp.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec))
            tmp.replace(cp)
            n_new += 1
        if rec["status"] != "ok":
            continue
        d, c = _row_from_y(np.array(rec["y"]), rec["bpm"])
        rows.append((rec["dir"], d, c))
    return rows


def family_rows(split):
    """(family, descriptor vector, confound vector) per family, numeric only
    — no artist/title/genre string ever enters."""
    rows = []
    for p in sorted(WINDOWS.glob("*.pt")):
        d = torch.load(p)
        if d["meta"]["split"] != split:
            continue
        y = d["y"].numpy()
        c = {k: y[:, _T.index(k)] for k in _T}
        desc = [c["axis_vert"].mean(), c["axis_horiz"].mean(),
                c["axis_diag"].mean(), c["travel"].mean(),
                c["motif_conc"].mean(), c["quiet"].mean(),
                c["accent_occupied"].mean(), c["occ_rate"].std(),
                np.abs(c["cad_l"] - c["cad_r"]).mean(), c["burst"].std(),
                c["coincidence"].std(), c["motif_conc"].std()]
        conf = [(c["cad_l"] + c["cad_r"]).mean(), c["coincidence"].mean(),
                d["meta"]["bpm"] / 300.0,
                float(np.log(max(1.0, c["occ_rate"].sum() * 32)))]
        rows.append((d["meta"]["family"], np.array(desc, dtype=np.float64),
                     np.array(conf, dtype=np.float64)))
    return rows


def fit_residualizer(desc, conf):
    """Train-only OLS of each descriptor on [1, confounds]; returns beta
    [n_conf+1, n_desc]."""
    X = np.hstack([np.ones((len(conf), 1)), conf])
    beta, *_ = np.linalg.lstsq(X, desc, rcond=None)
    return beta


def residualize(desc, conf, beta):
    X = np.hstack([np.ones((len(conf), 1)), conf])
    return desc - X @ beta


def adjusted_rand(a, b):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def _kmeans(z, k, seed=SEED):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(z)


def support_table(z, labels, k):
    """Per-cluster support: family count and separated residual dims vs the
    pooled rest (train SD units; z is already train-standardized)."""
    out = {}
    for c in range(k):
        m = labels == c
        if m.sum() == 0:
            out[c] = {"n": 0, "sep_dims": [], "exposed": False}
            continue
        sep = np.abs(z[m].mean(axis=0) - z[~m].mean(axis=0)) if (~m).any() \
            else np.zeros(z.shape[1])
        dims = [DESCRIPTORS[i] for i in np.where(sep >= SEP_SD)[0]]
        out[c] = {"n": int(m.sum()), "sep_dims": dims,
                  "exposed": bool(m.sum() >= MIN_FAMS
                                  and len(dims) >= MIN_SEP_DIMS)}
    return out


def stability(z_train, z_val, k, ref_labels_train, ref_labels_val):
    """20 family bootstrap resamples: refit, relabel by best centroid match,
    ARI vs the reference assignment on train and on val."""
    rng = np.random.RandomState(SEED)
    tr_ari, va_ari = [], []
    for _ in range(N_RESAMPLE):
        idx = rng.randint(0, len(z_train), len(z_train))
        km = _kmeans(z_train[idx], k, seed=int(rng.randint(0, 2 ** 31)))
        tr_ari.append(adjusted_rand(ref_labels_train, km.predict(z_train)))
        va_ari.append(adjusted_rand(ref_labels_val, km.predict(z_val)))
    return float(np.mean(tr_ari)), float(np.mean(va_ari))


def fit_vocab():
    tr = family_rows_full("train")
    va = family_rows_full("val")
    assert tr and va, "descriptor rows missing"
    fams_t = [f for f, _d, _c in tr]
    D_t = np.stack([d for _f, d, _c in tr])
    C_t = np.stack([c for _f, _d, c in tr])
    D_v = np.stack([d for _f, d, _c in va])
    C_v = np.stack([c for _f, _d, c in va])
    beta = fit_residualizer(D_t, C_t)
    R_t, R_v = residualize(D_t, C_t, beta), residualize(D_v, C_v, beta)
    mu, sd = R_t.mean(axis=0), np.maximum(R_t.std(axis=0), 1e-9)
    z_t, z_v = (R_t - mu) / sd, (R_v - mu) / sd
    results = {}
    for k in KS:
        km = _kmeans(z_t, k)
        lab_t, lab_v = km.labels_, km.predict(z_v)
        sup = support_table(z_t, lab_t, k)
        tr_ari, va_ari = stability(z_t, z_v, k, lab_t, lab_v)
        exposed = [c for c, s in sup.items() if s["exposed"]]
        results[k] = {"support": sup, "train_ari": tr_ari, "val_ari": va_ari,
                      "exposed": exposed,
                      "supported": bool(tr_ari >= MIN_ARI
                                        and len(exposed) >= 2),
                      "centroids": km.cluster_centers_.tolist()}
        print(f"K={k}: ARI train {tr_ari:.3f} val {va_ari:.3f}, exposed "
              f"{len(exposed)}/{k} "
              f"({[sup[c]['n'] for c in exposed]}) -> "
              f"{'SUPPORTED' if results[k]['supported'] else 'unsupported'}")
    # smallest supported K by VAL stability rule: among supported Ks, take
    # the smallest; neutral K=1 if none is supported
    chosen = min((k for k in KS if results[k]["supported"]), default=1)
    vocab = {"version": 1, "seed": SEED, "descriptors": DESCRIPTORS,
             "confounds": CONFOUNDS, "beta": beta.tolist(),
             "mu": mu.tolist(), "sd": sd.tolist(),
             "n_train_families": len(fams_t), "k": chosen,
             "results_per_k": {str(k): {kk: vv for kk, vv in r.items()
                                        if kk != "centroids"}
                               for k, r in results.items()},
             "note": ("corpus genre labels are incomplete; no genre-support "
                      "claim; descriptors are numeric only, no artist/title/"
                      "genre input; unsupported clusters merge to neutral"),
             "styles": {}}
    if chosen != 1:
        r = results[chosen]
        # stable IDs: exposed clusters ordered by size desc then centroid
        order = sorted(r["exposed"],
                       key=lambda c: (-r["support"][c]["n"], c))
        for i, c in enumerate(order):
            vocab["styles"][f"s{i + 1}"] = {
                "cluster": int(c), "n_train_families": r["support"][c]["n"],
                "sep_dims": r["support"][c]["sep_dims"],
                "centroid": r["centroids"][c]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "style_vocab.json").write_text(json.dumps(vocab, indent=1))
    print(f"chosen K={chosen} "
          f"({'neutral only' if chosen == 1 else str(len(vocab['styles'])) + ' exposed styles'}); "
          f"vocab -> {OUT / 'style_vocab.json'}")
    return vocab


def assign_style(desc_vec, conf_vec, vocab):
    """Nearest exposed style (or 'neutral') for one family's raw descriptor +
    confound vectors under a fitted vocabulary."""
    if not vocab["styles"]:
        return "neutral"
    beta = np.array(vocab["beta"])
    x = np.concatenate([[1.0], conf_vec])
    r = desc_vec - x @ beta
    z = (r - np.array(vocab["mu"])) / np.array(vocab["sd"])
    best, bd = "neutral", None
    for sid, s in vocab["styles"].items():
        d = float(np.linalg.norm(z - np.array(s["centroid"])))
        if bd is None or d < bd:
            best, bd = sid, d
    return best


if __name__ == "__main__":
    fit_vocab()
