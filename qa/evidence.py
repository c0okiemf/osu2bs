"""Independent-QA Task 7 (bundles): blind multimodal evidence (spec §7).

A bundle is an ALLOWLISTED record for one opaque candidate id: chart/
audio/Info hashes, scope+coverage, scene JSON, complete per-window
metrics, full-song rasters, six deterministic 8s witness windows with
pinned-camera front/side/top contact sheets, model quantiles, neighbour
exemplars (OBSERVED paths) vs model illustrations (ILLUSTRATIVE, never
player motion). No generator metadata can enter: the builder copies only
allowlisted fields and refuses unknown keys. Chart strings are inert data.
Render caches key on source + settings hashes; a mean between opposing
modes is never drawn as a real movement — exemplars render individually.
"""
import hashlib
import json
import math
from pathlib import Path

ALLOWED_TOP = {"candidate_id", "chart_sha256", "audio_sha256",
               "info_sha256", "scope", "coverage", "scene_summary",
               "physics", "support", "windows", "raster_png",
               "contact_sheets", "model_quantiles", "neighbours",
               "audio_evidence_ref", "mandatory_pass", "missing_evidence",
               "bundle_sha256", "modality_note"}
FORBIDDEN_SUBSTRINGS = ("critic", "logit", "seed", "checkpoint",
                        "generator", "arm", "winner", "repair")
CAMERAS = (("front", 0, 1), ("side", 2, 1), ("top", 0, 2))


def _sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str)
                          .encode()).hexdigest()


def opaque_id(source_key, salt):
    """Custodian-side opaque candidate id; the origin map stays elsewhere."""
    return "c-" + hashlib.sha256(f"{salt}:{source_key}".encode()) \
        .hexdigest()[:12]


def render_contact_sheet(scene, window_s, out_path, exemplars=None,
                         title=""):
    """Pinned-camera orthographic front/side/top sheet of one 8 s window:
    note cells with direction arrows + optional OBSERVED exemplar paths
    (each drawn individually, labelled; no averaged trajectory)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qa.physics import note_center
    a, b = window_s
    notes = [(t, li, ll, c, d) for t, li, ll, c, d in scene["notes"]
             if a <= t < b]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, ix, iy) in zip(axes, CAMERAS):
        for t, li, ll, c, d in notes:
            pos = note_center(li, ll) + ((t - a) / max(b - a, 1e-9),)
            xy = (pos[ix] if ix < 3 else pos[3],
                  pos[iy] if iy < 3 else pos[3])
            if name in ("front",):
                x, y = pos[0], pos[1]
            elif name == "side":
                x, y = (t - a) * 0.5, pos[1]     # time as depth proxy
            else:
                x, y = pos[0], (t - a) * 0.5
            ax.scatter([x], [y], s=90,
                       c=("tab:red" if c == 0 else "tab:blue"),
                       marker="s", alpha=0.7)
            ax.annotate(f"{t - a:.1f}", (x, y), fontsize=6, alpha=0.6)
        for ex in (exemplars or [])[:5]:
            pts = ex.get("rel_path_24") or []
            if name == "front" and pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        alpha=0.5, linewidth=1)
        ax.set_title(f"{name} — {title}", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
    if exemplars:
        axes[0].text(0.02, 0.98, "paths: OBSERVED human exemplars",
                     transform=axes[0].transAxes, fontsize=7, va="top")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def render_raster(scene, out_path):
    """Full-song timing/direction raster (time x lane, arrows by color)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 3))
    for t, li, ll, c, d in scene["notes"]:
        ax.scatter([t], [li + ll * 0.22],
                   c=("tab:red" if c == 0 else "tab:blue"), s=4, alpha=0.5)
    ax.set_xlabel("seconds")
    ax.set_ylabel("lane (+layer)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


def _scrub(obj):
    """Refuse generator-identifying keys anywhere in the payload."""
    if isinstance(obj, dict):
        for k in obj:
            lk = str(k).lower()
            if any(s in lk for s in FORBIDDEN_SUBSTRINGS):
                raise ValueError(f"forbidden key in evidence: {k!r}")
            _scrub(obj[k])
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scrub(v)


# ---------------- comparator v2 (spec 2f687f5 §5) ----------------
# v1 stays untouched above for historical reproducibility; v2 renders
# EVERY cut arrow/dot with hands and simultaneous groups, labels
# time-vs-lane views as TIME PROJECTIONS (never cameras), shows observed
# raw exemplars in their OWN coordinates with missing components visible,
# and scrubs nested values and path-like strings — an old v1
# mandatory_pass can never bypass this contract.

V2_ALLOWED_TOP = {"schema_version", "candidate_id", "contract_sha256",
                  "chart_sha256", "audio_sha256", "info_sha256", "scope",
                  "scene_summary", "contradictions", "support",
                  "witnesses", "renders", "audio_evidence",
                  "modalities", "missing_evidence", "bundle_sha256"}
V2_FORBIDDEN_KEYS = FORBIDDEN_SUBSTRINGS + (
    "mutation", "recipe", "expected", "player_token", "custodian",
    "label", "origin", "mapper", "title", "model_quantiles", "mixture",
    "latent", "quantile")
_DIR_DEG = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}
SLICE_HEADS = 8


def _looks_like_path(s):
    return (s.startswith("/") or s.startswith("~") or ".." in s
            or "experiments/" in s or s.endswith(".pt")
            or s.endswith(".bsor") or s.endswith(".json"))


def _scrub_v2(obj, where="bundle"):
    """Refuse forbidden keys anywhere AND path-like/identity string
    values (nested), except declared render asset names (opaque,
    bundle-relative)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if any(s in lk for s in V2_FORBIDDEN_KEYS):
                raise ValueError(f"forbidden key in {where}: {k!r}")
            _scrub_v2(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scrub_v2(v, where)
    elif isinstance(obj, str):
        if _looks_like_path(obj) and not where.endswith(".asset"):
            raise ValueError(f"path-like value in {where}: {obj!r}")


def _head_groups(scene, interval_s):
    a, b = interval_s
    groups = {}
    for t, li, ll, c, d in scene["notes"]:
        if a <= t < b:
            groups.setdefault(round(float(t), 6), []).append(
                (li, ll, c, d))
    return sorted(groups.items())


def sequence_primitives(scene, interval_s):
    """Deterministic render primitives: every note in the interval as an
    arrow (angle from cut direction) or a dot, with hand, simultaneous
    grouping and absolute-seconds timestamps, in chronological order."""
    prims = []
    for t, notes in _head_groups(scene, interval_s):
        simul = len({c for _li, _ll, c, _d in notes}) > 1
        for li, ll, c, d in sorted(notes):
            prims.append({"type": "dot" if d == 8 else "arrow",
                          "t": t, "col": li, "row": ll,
                          "hand": "right" if c == 1 else "left",
                          "angle_deg": _DIR_DEG.get(d),
                          "simultaneous": simul})
    return prims


def render_sequence(scene, interval_s, output_dir, settings=None):
    """Ordered note slices + a labeled time projection for one interval.
    Returns the primitive manifest and image hashes (cached by content)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = settings or {}
    prims = sequence_primitives(scene, interval_s)
    key = _sha_obj({"prims": prims, "interval": list(interval_s),
                    "settings": settings})[:16]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_p = out / f"seq-{key}.png"
    if not img_p.exists():
        groups = _head_groups(scene, interval_s)
        slices = [groups[i:i + SLICE_HEADS]
                  for i in range(0, max(len(groups), 1), SLICE_HEADS)]
        n_rows = len(slices) + 1
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(12, 2.6 * n_rows))
        axes = [axes] if n_rows == 1 else list(axes)
        for si, sl in enumerate(slices):
            ax = axes[si]
            for gi, (t, notes) in enumerate(sl):
                simul = len({c for _l, _y, c, _d in notes}) > 1
                for li, ll, c, d in sorted(notes):
                    x = gi * 1.6 + li * 0.3
                    y = ll * 0.45
                    color = "tab:blue" if c == 1 else "tab:red"
                    if d == 8:
                        ax.plot([x], [y], "o", color=color, ms=10)
                    else:
                        ang = math.radians(_DIR_DEG[d])
                        ax.annotate(
                            "", xy=(x + 0.14 * math.cos(ang),
                                    y + 0.14 * math.sin(ang)),
                            xytext=(x, y),
                            arrowprops={"color": color, "width": 2,
                                        "headwidth": 7})
                    if simul:
                        ax.plot([x], [y], "s", mfc="none", mec="k",
                                ms=14)
                ax.text(gi * 1.6, -0.45, f"{t:.3f}s", fontsize=7)
            ax.set_xlim(-0.5, SLICE_HEADS * 1.6)
            ax.set_ylim(-0.7, 1.5)
            ax.set_title(f"ordered slice {si + 1}/{len(slices)} — "
                         "grid columns per head, arrows = cut "
                         "direction, square = simultaneous group",
                         fontsize=8)
            ax.set_xticks([])
        axp = axes[-1]
        for p in prims:
            axp.plot([p["t"]], [p["col"] + p["row"] * 0.22],
                     "o" if p["type"] == "dot" else "^",
                     color="tab:blue" if p["hand"] == "right"
                     else "tab:red", ms=4)
        axp.set_xlabel("seconds")
        axp.set_ylabel("lane (+layer)")
        axp.set_title("TIME PROJECTION (time vs lane/layer; no physical "
                      "depth)", fontsize=8)
        fig.tight_layout()
        fig.savefig(img_p, dpi=110)
        plt.close(fig)
    return {"image": img_p.name,
            "image_sha256": hashlib.sha256(img_p.read_bytes())
            .hexdigest(),
            "interval_s": [float(interval_s[0]), float(interval_s[1])],
            "primitives": prims,
            "primitives_sha256": _sha_obj(prims),
            "time_projections": [{"time_axis_label": "seconds",
                                  "label": "time projection",
                                  "physical_depth": False}]}


def render_observed_reference(reference, source_scene, output_dir,
                              settings=None):
    """Front/side/top views of ONE observed raw exemplar in its own
    coordinates. Missing components (head/other hand/orientation) are
    explicitly visible, never silently absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = reference.get("rel_path_24") or []
    missing = [k for k in ("head_path", "other_hand_path", "orientation")
               if not reference.get(k)]
    key = _sha_obj({"pts": pts, "missing": missing})[:16]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_p = out / f"ref-{key}.png"
    views = (("front", 0, 1), ("side", 2, 1), ("top", 0, 2))
    if not img_p.exists():
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
        for ax, (name, ix, iy) in zip(axes, views):
            if pts:
                ax.plot([p[ix] for p in pts], [p[iy] for p in pts],
                        "-o", ms=2, color="tab:green")
            ax.set_title(f"{name} — OBSERVED exemplar (own coordinates, "
                         f"anchor-relative)", fontsize=7)
            ax.set_aspect("equal", adjustable="datalim")
        if missing:
            axes[0].text(0.02, 0.02, "MISSING: " + ", ".join(missing),
                         transform=axes[0].transAxes, fontsize=7,
                         color="crimson")
        fig.tight_layout()
        fig.savefig(img_p, dpi=110)
        plt.close(fig)
    return {"image": img_p.name,
            "image_sha256": hashlib.sha256(img_p.read_bytes())
            .hexdigest(),
            "views": [v[0] for v in views],
            "observed": True, "missing_components": missing,
            "source_time_s": reference.get("event_time")}


def render_signal_panel(audio, interval_s, output_dir, settings=None):
    """AUDIO SIGNAL panel for one interval: per-second intensity/
    harmonic/percussive energy, beat ticks and section bounds, in the
    panel's OWN time origin (synchronized provenance). Explicitly not
    audible; missing audio returns typed missing evidence."""
    if not audio or not audio.get("supported"):
        return {"missing": True, "type": "audio_evidence_unavailable"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = interval_s
    ps = audio.get("per_second") or {}
    key = _sha_obj({"iv": [a, b],
                    "rms": ps.get("intensity_rms"),
                    "h": ps.get("harmonic_rms"),
                    "p": ps.get("percussive_rms"),
                    "beats": audio.get("beat_times"),
                    "sections": audio.get("section_bounds")})[:16]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_p = out / f"sig-{key}.png"
    if not img_p.exists():
        lo, hi = int(max(0, a - 1)), int(b + 2)
        xs = list(range(lo, hi))
        fig, ax = plt.subplots(figsize=(10, 2.4))
        for name, kk in (("intensity", "intensity_rms"),
                         ("harmonic", "harmonic_rms"),
                         ("percussive", "percussive_rms")):
            arr = ps.get(kk) or []
            ax.plot(xs, [arr[i] if 0 <= i < len(arr) else 0.0
                         for i in xs], label=name, linewidth=1)
        for t in audio.get("beat_times") or []:
            if a <= t <= b:
                ax.axvline(t, color="k", alpha=0.12, linewidth=0.6)
        for t in audio.get("section_bounds") or []:
            if a - 1 <= t <= b + 1:
                ax.axvline(t, color="tab:purple", alpha=0.6,
                           linestyle="--", linewidth=1)
        ax.axvspan(a, b, color="tab:orange", alpha=0.07)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("seconds (this side's own time origin)")
        ax.legend(fontsize=6, loc="upper right")
        ax.set_title("AUDIO SIGNALS — per-second energy, beat ticks, "
                     "dashed section bounds (evidence, not audible)",
                     fontsize=8)
        fig.tight_layout()
        fig.savefig(img_p, dpi=110)
        plt.close(fig)
    return {"image": img_p.name,
            "image_sha256": hashlib.sha256(img_p.read_bytes())
            .hexdigest(),
            "interval_s": [float(a), float(b)], "missing": False}


def bundle_v2(candidate_id, scene, contradiction_report, support_report,
              audio_evidence, witnesses, artifacts, contract):
    """Allowlisted blind v2 bundle. artifacts: {witness_name: render
    manifest}; contract: {contract_sha256, modalities}."""
    sup = {"heads_total": support_report.get("heads_total"),
           "heads_supported": support_report.get("heads_supported"),
           "share_supported": support_report.get("share_supported"),
           "unsupported_runs": support_report.get("unsupported_runs"),
           "bins_2s": support_report.get("bins_2s"),
           "machine_support_pass":
               support_report.get("machine_support_pass")}
    contr = {"status": contradiction_report.get("status"),
             "structural": contradiction_report.get("structural"),
             "model": [{"type": m.get("type"),
                        "interval_s": m.get("interval_s")}
                       for m in contradiction_report.get("model") or []],
             "warnings": contradiction_report.get("warnings"),
             "unknowns": contradiction_report.get("unknowns")}
    missing = []
    aud = None
    if audio_evidence and (audio_evidence.get("audio") or {}) \
            .get("supported"):
        a = audio_evidence["audio"]
        aud = {"per_second": a.get("per_second"),
               "beat_times": a.get("beat_times"),
               "section_bounds": a.get("section_bounds"),
               "duration_s": a.get("duration_s")}
    else:
        missing.append({"type": "audio_evidence_unavailable"})
    out = {"schema_version": 2, "candidate_id": candidate_id,
           "contract_sha256": contract.get("contract_sha256"),
           "chart_sha256": scene.get("chart_sha256"),
           "audio_sha256": scene.get("audio_sha256"),
           "info_sha256": scene.get("info_sha256"),
           "scope": scene.get("scope"),
           "scene_summary": {"n_notes": len(scene.get("notes") or []),
                             "n_walls": len(scene.get("walls") or []),
                             "settings": scene.get("settings")},
           "contradictions": contr, "support": sup,
           "witnesses": witnesses, "renders": artifacts,
           "audio_evidence": aud,
           "modalities": contract.get("modalities")
           or {"received": ["image", "signals"],
               "used": []},
           "missing_evidence": missing
           + list(contract.get("missing_evidence") or ())}
    validate_bundle_v2(out)
    out["bundle_sha256"] = _sha_obj(out)
    return out


def validate_bundle_v2(bundle):
    unknown = set(bundle) - V2_ALLOWED_TOP - {"bundle_sha256"}
    if unknown:
        raise ValueError(f"non-allowlisted v2 bundle keys: {unknown}")
    # chart text under scene_summary.settings is inert data; everything
    # else is scrubbed for forbidden keys and path-like values
    probe = dict(bundle)
    ss = (probe.get("scene_summary") or {}).copy()
    ss.pop("settings", None)
    probe["scene_summary"] = ss
    _scrub_v2(probe)
    for name, art in (bundle.get("renders") or {}).items():
        for k in ("image",):
            v = (art or {}).get(k)
            if v and ("/" in v or "\\" in v or ".." in v):
                raise ValueError(f"render asset {name} must be an opaque "
                                 f"bundle-relative name, got {v!r}")


def select_witnesses(scene, support_report, contradiction_report,
                     candidate_id, salt, span_s=8.0):
    """Six deterministic 8 s witnesses (spec §5), deduplicated with
    next-ranked fill."""
    heads = sorted({t for t, *_ in scene["notes"]})
    if not heads:
        return []

    def window(center):
        a = max(0.0, center - span_s / 2)
        return [round(a, 3), round(a + span_s, 3)]
    ranked = {}
    ranked["first_active"] = [window(heads[0] + span_s / 2)]
    cad = []
    for t in heads:
        n = sum(1 for u in heads if t <= u < t + 2.0)
        cad.append((n, t))
    ranked["peak_cadence"] = [window(t + 1.0) for n, t in
                              sorted(cad, key=lambda x: (-x[0], x[1]))]
    warns = sorted((contradiction_report.get("warnings") or []),
                   key=lambda w: -(w.get("lower_bound") or 0))
    ranked["greatest_geometric_warning"] = \
        [window(scene["notes"][w["index"]][0]) for w in warns
         if w.get("index") is not None] or list(ranked["peak_cadence"])
    ph = [(r.get("distance") or 0, r["t"])
          for r in support_report.get("per_head") or []]
    ranked["greatest_support_distance"] = \
        [window(t) for _d, t in sorted(ph, key=lambda x: (-x[0], x[1]))] \
        or list(ranked["peak_cadence"])
    toks = [(t, li, ll, c, d) for t, li, ll, c, d in scene["notes"]]
    grams = {}
    for i in range(len(toks) - 3):
        g = tuple((li, ll, c, d) for _t, li, ll, c, d in toks[i:i + 4])
        grams.setdefault(g, []).append(toks[i][0])
    motif = sorted(grams.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
    ranked["longest_repeated_motif"] = \
        [window(ts[0] + 1.0) for _g, ts in motif] \
        or list(ranked["peak_cadence"])
    hs = sorted(heads, key=lambda t: hashlib.sha256(
        f"{salt}:{candidate_id}:{t}".encode()).hexdigest())
    ranked["hash_selected"] = [window(t) for t in hs]
    chosen, used = [], []
    for name in ("first_active", "peak_cadence",
                 "greatest_geometric_warning",
                 "greatest_support_distance", "longest_repeated_motif",
                 "hash_selected"):
        pick = None
        for iv in ranked[name]:
            if all(min(iv[1], u[1]) - max(iv[0], u[0]) < span_s / 2
                   for u in used):
                pick = iv
                break
        if pick is None and ranked[name]:
            pick = ranked[name][0]
        if pick is not None:
            chosen.append({"name": name, "interval_s": pick,
                           "context_s": [max(0.0, pick[0] - 2.0),
                                         pick[1] + 2.0]})
            used.append(pick)
    return chosen


def bundle(candidate_id, scene, physics_report, support, windows,
           renders, model_quantiles, neighbours, audio_ref,
           mandatory_pass, missing_evidence=()):
    """Assemble + validate one blind bundle."""
    out = {"candidate_id": candidate_id,
           "chart_sha256": scene.get("chart_sha256"),
           "audio_sha256": scene.get("audio_sha256"),
           "info_sha256": scene.get("info_sha256"),
           "scope": scene.get("scope"),
           "coverage": support.get("share_supported")
           if isinstance(support, dict) else None,
           "scene_summary": {"n_notes": len(scene.get("notes") or []),
                             "n_walls": len(scene.get("walls") or []),
                             "settings": scene.get("settings")},
           "physics": physics_report, "support": support,
           "windows": windows, "raster_png": renders.get("raster"),
           "contact_sheets": renders.get("sheets"),
           "model_quantiles": model_quantiles,
           "neighbours": neighbours,
           "audio_evidence_ref": audio_ref,
           "mandatory_pass": bool(mandatory_pass),
           "missing_evidence": list(missing_evidence),
           "modality_note": "images+signals only; no audio playback claim"}
    unknown = set(out) - ALLOWED_TOP
    if unknown:
        raise ValueError(f"non-allowlisted bundle keys: {unknown}")
    _scrub({k: v for k, v in out.items()
            if k not in ("physics",)})        # physics keys are our own
    out["bundle_sha256"] = _sha_obj(out)
    return out
