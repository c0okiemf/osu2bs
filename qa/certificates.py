"""Comparator Task 2 (spec §3): typed contradictions and independently
verified certificates.

contradictions() returns TYPED results, never a universal safety bool:
- structural: malformed/nonfinite required values, invalid schema values
  (bool-as-int included), negative durations, hash-bound asset mismatch.
  Independently reproducible from source; routes HARD_FAIL.
- model: an independently VERIFIED certificate contradicts a declared
  geometric scenario (wall corridor under the frozen stance model).
  Routes REGENERATE/MODEL_CONDITIONAL_CONTRADICTION — never a universal
  biological impossibility claim.
- warnings: the exact SERIALIZED QA-train p99.5 speed warning (an
  empirical warning, not a floor or maximum). Missing artifact means the
  evidence is UNKNOWN — never the historical 12.0 default.
- unknowns: unsupported mechanics/scope, ambiguous duplicate occupancy
  (MUST_ABSTAIN under the v2 contract), missing settings, invalid
  certificates. Nothing is silently dropped to make scope look known.
- status NO_CONTRADICTION_FOUND means these checks found nothing — it is
  not a constructive execution proof.

verify_certificate() is deliberately INDEPENDENT of the detector: it
never imports or calls qa.physics._corridor or its helpers; it
re-partitions time at obstacle endpoints with its own half-open
membership and re-derives reach inequalities with its own (documented
duplicate) grid arithmetic. An invalid certificate is
UNKNOWN/CERTIFICATE_INVALID, never a successful detection.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEED_P = ROOT / "experiments" / "qa-v1" / "speed_threshold.json"
CHECKER_VERSION = 1
STANCE_COLUMNS = (0, 1, 2, 3)
STRUCTURAL_SCOPES = ("invalid_bpm", "invalid_note_time")
# deliberate independent duplicates of the pinned grid/relaxation
# constants (spec: the checker must not call the detector's helpers)
_SPACING = 0.6
_NOTE_R = 0.25 * math.sqrt(3)
_SABER = 1.0
REACH_TOL = 1e-9


def _sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str)
                          .encode()).hexdigest()


def load_speed_warning(path=SPEED_P):
    """The exact frozen artifact or None (evidence unknown). NEVER the
    historical 12.0 default."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"value": d["pooled_p995"], "n": d.get("n"),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "kind": "empirical_warning_not_biological_max"}


def make_wall_certificate(scene, interval_s):
    return {"schema_version": 1, "kind": "wall_corridor",
            "checker_version": CHECKER_VERSION,
            "source": {"chart_sha256": scene.get("chart_sha256"),
                       "info_sha256": scene.get("info_sha256")},
            "interval_s": [float(interval_s[0]), float(interval_s[1])],
            "assumptions": {"stance_columns": list(STANCE_COLUMNS),
                            "full_height_only": True,
                            "membership": "half-open [start, start+dur)",
                            "heights_m": [1.4, 1.7, 2.0],
                            "domain": "bounded standing stances at grid "
                                      "columns; crouch/out-of-domain "
                                      "invalidates universal conclusions"},
            "witness": {"walls": [list(w) for w in
                                  (scene.get("walls") or [])]}}


def make_reach_certificate(scene, i_from, i_to, dt, min_speed):
    return {"schema_version": 1, "kind": "reach",
            "checker_version": CHECKER_VERSION,
            "source": {"chart_sha256": scene.get("chart_sha256"),
                       "info_sha256": scene.get("info_sha256")},
            "from_cell": list(i_from), "to_cell": list(i_to),
            "dt_s": float(dt), "claimed_min_speed": float(min_speed),
            "assumptions": {"relaxation": "containing balls; grip radius "
                                          "= note_r + saber",
                            "tolerance": REACH_TOL},
            "interval_s": None}


def _cell_center(li, ll):
    # independent duplicate of the pinned grid (documented; do not
    # "deduplicate" into qa.physics — independence is the point)
    return ((li - 1.5) * _SPACING, _SPACING + ll * _SPACING, 0.0)


def verify_certificate(scene, certificate):
    """Independent verification. Returns {valid, kind, interval_s,
    assumptions, reason}."""
    out = {"valid": False, "kind": certificate.get("kind"),
           "interval_s": certificate.get("interval_s"),
           "assumptions": certificate.get("assumptions"), "reason": None}

    def fail(reason):
        out["reason"] = reason
        return out
    if certificate.get("checker_version") != CHECKER_VERSION:
        return fail("checker version mismatch")
    src = certificate.get("source") or {}
    if src.get("chart_sha256") != scene.get("chart_sha256"):
        return fail("source hash mismatch")
    if certificate.get("kind") == "wall_corridor":
        a = certificate.get("assumptions") or {}
        if list(a.get("stance_columns") or ()) != list(STANCE_COLUMNS) \
                or not a.get("full_height_only"):
            return fail("assumptions outside the declared stance domain")
        iv = certificate.get("interval_s")
        if not iv or len(iv) != 2 or not all(math.isfinite(v)
                                             for v in iv) \
                or iv[1] <= iv[0]:
            return fail("interval must be finite with positive duration")
        walls = scene.get("walls") or []
        spans = []
        for w in walls:
            t, li, typ, dur, width = w
            if not all(isinstance(v, (int, float))
                       and not isinstance(v, bool)
                       and math.isfinite(float(v))
                       for v in (t, dur)) or dur < 0:
                return fail("malformed wall in scene")
            if typ != 0:
                # crouch/custom overlapping the claim leaves the domain
                if t < iv[1] and t + dur > iv[0]:
                    return fail("non-full-height wall inside the claimed "
                                "interval: outside certified domain")
                continue
            try:
                cols = set(range(int(li), int(li) + int(width)))
            except (TypeError, ValueError):
                return fail("malformed wall columns")
            if dur > 0:
                spans.append((float(t), float(t) + float(dur), cols))
        # independent partition at obstacle endpoints inside the claim
        cuts = sorted({iv[0], iv[1]}
                      | {v for s, e, _c in spans for v in (s, e)
                         if iv[0] < v < iv[1]})
        for lo, hi in zip(cuts, cuts[1:]):
            mid = (lo + hi) / 2.0
            covered = set()
            for s, e, cols in spans:
                if s <= mid < e:                    # half-open membership
                    covered |= cols
            if not covered.issuperset(STANCE_COLUMNS):
                return fail(f"interval not covered at t={mid:.4f} under "
                            "the declared model")
        out["valid"] = True
        return out
    if certificate.get("kind") == "reach":
        dt = certificate.get("dt_s")
        if not isinstance(dt, (int, float)) or dt <= 0 \
                or not math.isfinite(dt):
            return fail("invalid dt")
        c1 = _cell_center(*certificate["from_cell"])
        c2 = _cell_center(*certificate["to_cell"])
        d = max(0.0, math.dist(c1, c2) - 2 * (_NOTE_R + _SABER))
        speed = d / dt
        if speed + REACH_TOL < certificate.get("claimed_min_speed", 0.0):
            return fail(f"claimed minimum speed not reproduced "
                        f"(independent bound {speed:.6f})")
        out["valid"] = True
        return out
    return fail(f"unknown certificate kind {certificate.get('kind')!r}")


def _structural_scene_issues(scene):
    issues = []
    for i, (t, li, ll, c, d) in enumerate(scene.get("notes") or []):
        for name, v in (("time", t), ("line_index", li),
                        ("line_layer", ll), ("cut_direction", d)):
            if isinstance(v, bool):
                issues.append({"kind": "bool_as_int", "field": name,
                               "index": i})
            elif isinstance(v, (int, float)) and not math.isfinite(
                    float(v)):
                issues.append({"kind": "nonfinite", "field": name,
                               "index": i})
            elif v is None:
                issues.append({"kind": "missing_required", "field": name,
                               "index": i})
    for i, w in enumerate(scene.get("walls") or []):
        t, li, typ, dur, width = w
        if isinstance(dur, (int, float)) and not isinstance(dur, bool) \
                and (not math.isfinite(float(dur)) or dur < 0):
            issues.append({"kind": "malformed_duration", "field": "wall",
                           "index": i})
        if isinstance(typ, bool):
            issues.append({"kind": "bool_as_int", "field": "wall_type",
                           "index": i})
    declared = scene.get("declared_chart_sha256")
    if declared is not None and declared != scene.get("chart_sha256"):
        issues.append({"kind": "asset_mismatch", "field": "chart_sha256"})
    return issues


def _duplicate_occupancy(scene):
    seen, dups = set(), []
    for i, (t, li, ll, c, d) in enumerate(scene.get("notes") or []):
        key = (round(float(t), 6) if isinstance(t, (int, float))
               and math.isfinite(float(t)) else t, li, ll, c)
        if key in seen:
            dups.append({"index": i, "occupancy": list(key)})
        seen.add(key)
    return dups


def contradictions(scene, constants=None):
    """Typed contradiction report for one scene."""
    report = {"schema_version": 1, "checker_version": CHECKER_VERSION,
              "source": {"chart_sha256": scene.get("chart_sha256"),
                         "info_sha256": scene.get("info_sha256")},
              "structural": [], "model": [], "warnings": [],
              "unknowns": []}
    scope = scene.get("scope")
    if scope is not None:
        entry = {"kind": "scope", "reason": scope}
        if any(scope.startswith(s) for s in STRUCTURAL_SCOPES) \
                or scope.startswith("unsupported_note_type") \
                or scope.startswith("invalid_"):
            report["structural"].append(entry)
        else:
            report["unknowns"].append({**entry,
                                       "type": "unsupported_mechanics"})
        report["status"] = "STRUCTURAL_CONTRADICTION" \
            if report["structural"] else "UNKNOWN"
        return report
    report["structural"] += _structural_scene_issues(scene)
    for dup in _duplicate_occupancy(scene):
        report["unknowns"].append({"type": "duplicate_occupancy",
                                   **dup,
                                   "note": "ambiguous game semantics; "
                                           "MUST_ABSTAIN (v2 contract)"})
    # empirical speed warning from the exact frozen artifact
    warn = load_speed_warning() if constants is None \
        else constants.get("speed_warning")
    if warn is None:
        report["unknowns"].append({"type": "speed_evidence_unknown",
                                   "note": "frozen threshold artifact "
                                           "unavailable; no default is "
                                           "substituted"})
    elif not report["structural"]:
        from qa.physics import physics
        ph = physics(scene, speed_warn=warn["value"])
        if ph.get("status") == "ok":
            for v in ph["grip_speed"]["violations"]:
                report["warnings"].append({"type": "speed_warning", **v,
                                           "threshold_sha256":
                                               warn.get("sha256")})
    # wall-corridor DETECTION (detector) + INDEPENDENT verification
    if not report["structural"]:
        from qa.physics import _corridor
        cor = _corridor(scene.get("walls") or [], 0.0, 0.0)
        for iv in cor["empty_intervals"]:
            cert = make_wall_certificate(scene, iv)
            v = verify_certificate(scene, cert)
            if v["valid"]:
                report["model"].append(
                    {"type": "MODEL_CONDITIONAL_CONTRADICTION",
                     "certificate": cert, "verified": True,
                     "interval_s": iv})
            else:
                report["unknowns"].append({"type": "CERTIFICATE_INVALID",
                                           "interval_s": iv,
                                           "reason": v["reason"]})
        for iv in cor["unknown_intervals"]:
            report["unknowns"].append({"type": "wall_unknown",
                                       "interval_s": iv})
    report["status"] = ("STRUCTURAL_CONTRADICTION" if report["structural"]
                        else "MODEL_CONTRADICTION" if report["model"]
                        else "NO_CONTRADICTION_FOUND")
    report["note"] = ("NO_CONTRADICTION_FOUND is not a constructive "
                      "execution proof; warnings route to review and an "
                      "unresolved material warning prevents PASS")
    return report
