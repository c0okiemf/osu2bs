"""Comparator Task 5 (spec §6): frozen validation specimens with
INDEPENDENT expected outcomes.

Per family: ten structural negatives (5 nonfinite required-time + 5
invalid required-type at hash-ranked distinct note indices), ten
separately certified wall contradictions (durations {0.25,0.5,1,2,4}s x
one-spanning-wall / two-jointly-spanning layouts at hash-ranked occupied
timestamps, each independently verified BEFORE use), machine invariance
transforms (exact serialization, seconds-equivalent BPM encoding,
synchronized time-origin shifts that actually move chart AND audio,
mirror of chart+reference poses together, non-clipping gain), five
expression-destruction pairs and five benign pairs with nonoverlapping
8s cores from at most 40 hash-ranked eligible windows.

Expression collapse preserves audio, every timestamp, hand assignment
and per-hand cut directions; left maps to column0/layer0, right to
column3/layer0 across the core. Contraction is required in the SUM OF
PER-HAND spans separately for horizontal and vertical axes (the
combined two-hand span deliberately cannot contract). It is a RELATIVE
expressive-information destruction probe, never a proof such charts are
unpleasant. Benign expectations are comparative preservation/TIE, not
"unmodified charts are good". Labels live in the custodian manifest and
never enter evidence bundles.

score_case: certified-negative credit requires the expected reason AND
interval overlap (unrelated missing evidence is NOT detection);
expression credit requires preferring the source WITH a localized
axis/evidence rationale (generic "more movement is better" earns zero);
benign credit requires TIE with no invented change-specific defect.
"""
import copy
import hashlib
import json

from qa.certificates import (contradictions, make_wall_certificate,
                             verify_certificate)

SPAN_S = 8.0
CTX_S = 2.0
MAX_ELIGIBLE = 40
WALL_DURATIONS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_STRUCTURAL = 10
N_WALL = 10
N_PAIRS = 5
EXPRESSION_AXES = ("vocabulary", "coordination", "audio_correspondence",
                   "continuity")


def _h(salt, s):
    return hashlib.sha256(f"{salt}:{s}".encode()).hexdigest()


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


# ---------------- eligibility and collapse ----------------

def _heads_in(scene, core):
    a, b = core
    groups = {}
    for i, (t, li, ll, c, d) in enumerate(scene["notes"]):
        if a <= t < b:
            groups.setdefault((round(float(t), 6), c),
                              []).append((li, ll, d, i))
    return groups


def eligible_windows(scene, salt, span=SPAN_S):
    """Hash-ranked nonoverlapping 8s cores meeting the expression
    eligibility floor; at most MAX_ELIGIBLE."""
    times = sorted({t for t, *_ in scene["notes"]})
    cands = []
    for t0 in times:
        core = (t0, t0 + span)
        heads = _heads_in(scene, core)
        if len(heads) < 16:
            continue
        hands = {h for _t, h in heads}
        cols = {li for g in heads.values() for li, _ll, _d, _i in g}
        layers = {ll for g in heads.values() for _li, ll, _d, _i in g}
        toks = {(h, li, ll, d) for (_t, h), g in heads.items()
                for li, ll, d, _i in g}
        if len(hands) < 2 or len(cols) < 3 or len(layers) < 2 \
                or len(toks) < 4:
            continue
        cands.append(core)
    ranked = sorted(cands, key=lambda c: _h(salt, f"{c[0]:.6f}"))
    picked = []
    for c in ranked:
        if len(picked) >= MAX_ELIGIBLE:
            break
        if all(_overlap(c, p) == 0 for p in picked):
            picked.append(c)
    return picked


def per_hand_spans(scene, core):
    """Sum of per-hand coordinate spans, horizontal and vertical
    separately (the required contraction measure)."""
    h_sum = v_sum = 0.0
    for hand in (0, 1):
        cols = [li for t, li, ll, c, d in scene["notes"]
                if core[0] <= t < core[1] and c == hand]
        rows = [ll for t, li, ll, c, d in scene["notes"]
                if core[0] <= t < core[1] and c == hand]
        if cols:
            h_sum += max(cols) - min(cols)
            v_sum += max(rows) - min(rows)
    return {"h_sum": float(h_sum), "v_sum": float(v_sum)}


def collapse_window(scene, core):
    """Expression-destruction mutant: left -> col0/layer0, right ->
    col3/layer0 for the whole core; audio/timestamps/hands/directions
    preserved. Returns (mutant, check) — check.valid False means the
    construction is rejected and the next window must be used."""
    mut = copy.deepcopy(scene)
    notes = []
    occupied = set()
    for t, li, ll, c, d in mut["notes"]:
        if core[0] <= t < core[1]:
            li2 = 0 if c == 0 else 3
            ll2 = 0
            key = (round(float(t), 6), li2, ll2, c)
            if key in occupied:
                return None, {"valid": False,
                              "reason": "duplicate_same_hand_occupancy"}
            occupied.add(key)
            notes.append((t, li2, ll2, c, d))
        else:
            notes.append((t, li, ll, c, d))
    mut["notes"] = notes
    src = per_hand_spans(scene, core)
    got = per_hand_spans(mut, core)
    if not (src["h_sum"] > 0 and src["v_sum"] > 0
            and got["h_sum"] == 0 and got["v_sum"] == 0):
        return None, {"valid": False, "reason": "no_span_contraction",
                      "src": src, "mut": got}
    rep = contradictions(mut, {"speed_warning": None})
    if rep["structural"] or rep["model"]:
        return None, {"valid": False,
                      "reason": "new_certified_contradiction"}
    if mut.get("scope") is not None:
        return None, {"valid": False, "reason": "unsupported_scope"}
    return mut, {"valid": True, "src_spans": src, "mut_spans": got}


# ---------------- negatives ----------------

def structural_cases(scene, salt):
    n = len(scene["notes"])
    order = sorted(range(n), key=lambda i: _h(salt, f"structural:{i}"))
    cases = []
    for j, i in enumerate(order[:N_STRUCTURAL]):
        mut = copy.deepcopy(scene)
        t, li, ll, c, d = mut["notes"][i]
        if j < N_STRUCTURAL // 2:
            mut["notes"][i] = (float("inf"), li, ll, c, d)
            kind, field = "nonfinite", "time"
        else:
            mut["notes"][i] = (t, li, ll, c, True)
            kind, field = "bool_as_int", "cut_direction"
        cases.append({"kind": "structural_negative", "scene": mut,
                      "expected": {"status": "STRUCTURAL_CONTRADICTION",
                                   "field": field, "issue": kind,
                                   "index": i}})
    return cases


def wall_cases(scene, salt):
    times = sorted({t for t, *_ in scene["notes"]})
    order = sorted(times, key=lambda t: _h(salt, f"wall:{t:.6f}"))
    cases = []
    for j, t0 in enumerate(order):
        if len(cases) >= N_WALL:
            break
        dur = WALL_DURATIONS[j % len(WALL_DURATIONS)]
        mut = copy.deepcopy(scene)
        if j % 2 == 0:
            walls = [(t0, 0, 0, dur, 4)]                # one spanning
        else:
            walls = [(t0, 0, 0, dur, 2), (t0, 2, 0, dur, 2)]
        mut["walls"] = list(mut.get("walls") or []) + walls
        cert = make_wall_certificate(mut, (t0, t0 + dur))
        v = verify_certificate(mut, cert)
        if not v["valid"]:
            continue                                    # never uncertified
        cases.append({"kind": "wall_negative", "scene": mut,
                      "certificate": cert,
                      "expected": {"status": "MODEL_CONTRADICTION",
                                   "reason":
                                       "MODEL_CONDITIONAL_CONTRADICTION",
                                   "interval_s": [t0, t0 + dur]}})
    return cases


# ---------------- invariance transforms ----------------

def roundtrip_scene(scene):
    d = json.loads(json.dumps(scene))
    d["notes"] = [tuple(n) for n in d["notes"]]
    d["walls"] = [tuple(w) for w in d.get("walls") or []]
    d["bombs"] = [tuple(b) for b in d.get("bombs") or []]
    return d


def time_origin_shift(scene, audio, delta_s):
    """SYNCHRONIZED origin shift: chart times AND audio evidence move
    together (silence prepended to per-second arrays)."""
    sc = copy.deepcopy(scene)
    sc["notes"] = [(t + delta_s, li, ll, c, d)
                   for t, li, ll, c, d in sc["notes"]]
    sc["walls"] = [(t + delta_s, li, ty, du, w)
                   for t, li, ty, du, w in sc.get("walls") or []]
    au = copy.deepcopy(audio) if audio else None
    if au and au.get("supported"):
        pad = int(round(delta_s))
        for k, arr in (au.get("per_second") or {}).items():
            au["per_second"][k] = [0.0] * pad + arr
        au["duration_s"] = au["duration_s"] + delta_s
        if au.get("frames"):
            au["frames"]["times"] = [t + delta_s
                                     for t in au["frames"]["times"]]
        for k in ("beat_times", "onset_times", "section_bounds"):
            if au.get(k):
                au[k] = [t + delta_s for t in au[k]]
    return sc, au


def gain_scene_audio(audio, db):
    au = copy.deepcopy(audio)
    f = 10 ** (db / 20.0)
    for k, arr in (au.get("per_second") or {}).items():
        au["per_second"][k] = [v * f for v in arr]
    return au


def mirror_pair(scene, reference):
    """Mirror chart AND reference poses TOGETHER (hand semantics stay
    consistent); measured separately from exact serialization."""
    from qa.telemetry_v2 import mirror_pose
    from qa.train import mirror_scene
    ref = copy.deepcopy(reference)
    if ref.get("rel_path_24"):
        ref["rel_path_24"] = [list(mirror_pose(tuple(p), (0, 0, 0, 1))[0])
                              for p in ref["rel_path_24"]]
    return mirror_scene(scene), ref


# ---------------- case assembly and scoring ----------------

def make_cases(family, scene, audio, contract):
    """Custodian manifest for one family; opaque case ids; labels stay
    HERE and never in bundles."""
    salt = f"{contract.get('label', 'comparator-v1')}:{family}"
    cores = eligible_windows(scene, salt)
    expr, benign, skipped = [], [], []
    for core in cores:
        if len(expr) >= N_PAIRS:
            break
        mut, chk = collapse_window(scene, core)
        if mut is None:
            skipped.append({"core": list(core), **chk})
            continue
        expr.append({"kind": "expression_pair", "core_s": list(core),
                     "context_s": [core[0] - CTX_S, core[1] + CTX_S],
                     "mutant": mut, "check": chk,
                     "expected": {"prefer": "source",
                                  "axis": "vocabulary",
                                  "localized_to": list(core)}})
    # benign pairs: 2 serialization/BPM, 1 time-origin, 1 mirror, 1 gain;
    # >=2 sourced from the highest repeated-token-share eligible windows
    rep_rank = sorted(cores, key=lambda c: -_gram_share(scene, c))
    benign_defs = (("serialization", None), ("serialization", None),
                   ("time_origin", 2.0), ("mirror", None),
                   ("gain", -6.0))
    for i, (kindb, arg) in enumerate(benign_defs):
        core = (rep_rank[i % max(1, len(rep_rank))]
                if i < 2 and rep_rank else
                cores[i % max(1, len(cores))] if cores else None)
        if core is None:
            break
        other_audio = None
        shift_s = 0.0
        if kindb == "serialization":
            other = roundtrip_scene(scene)
            note = "exact serialization roundtrip"
        elif kindb == "time_origin":
            other, other_audio = time_origin_shift(scene, audio, arg)
            shift_s = arg
            note = "synchronized time-origin shift"
        elif kindb == "mirror":
            other, _ref = mirror_pair(scene, {"rel_path_24": []})
            note = "mirrored scene + reference poses"
        else:
            other = scene
            other_audio = gain_scene_audio(audio, arg) if audio else None
            note = f"non-clipping gain {arg}dB (audio-side)"
        benign.append({"kind": "benign_pair", "transform": kindb,
                       "core_s": list(core), "other": other,
                       "other_audio": other_audio,
                       "other_shift_s": shift_s,
                       "note": note, "expected": {"pair": "TIE"}})
    status = "OK" if len(expr) == N_PAIRS and len(benign) == N_PAIRS \
        else "INSUFFICIENT"
    return {"family": family, "status": status,
            "structural": structural_cases(scene, salt),
            "wall": wall_cases(scene, salt),
            "expression": expr, "benign": benign,
            "skipped_constructions": skipped,
            "counts": {"structural": N_STRUCTURAL, "wall": N_WALL,
                       "expression": len(expr), "benign": len(benign),
                       "eligible_windows": len(cores)}}


def _gram_share(scene, core):
    toks = [(li, ll, c, d) for t, li, ll, c, d in scene["notes"]
            if core[0] <= t < core[1]]
    if len(toks) < 4:
        return 0.0
    grams = {}
    for i in range(len(toks) - 3):
        g = tuple(toks[i:i + 4])
        grams[g] = grams.get(g, 0) + 1
    return max(grams.values()) / (len(toks) - 3)


def score_case(expected, machine, judgments):
    kind = expected.get("kind") or expected.get("status")
    if expected.get("status") == "MODEL_CONTRADICTION" \
            or kind == "wall_negative":
        det = False
        if machine and machine.get("status") == "MODEL_CONTRADICTION":
            det = any(_overlap(m.get("interval_s") or (0, 0),
                               expected["interval_s"]) > 0
                      for m in machine.get("model") or [])
        return {"detected": det,
                "note": "unrelated missing evidence is not detection; "
                        "relative outcomes never imply structural "
                        "impossibility"}
    if expected.get("status") == "STRUCTURAL_CONTRADICTION" \
            or kind == "structural_negative":
        det = bool(machine
                   and machine.get("status") == "STRUCTURAL_CONTRADICTION"
                   and any(i.get("field") == expected.get("field")
                           or i.get("kind") == expected.get("issue")
                           for i in machine.get("structural") or []))
        return {"detected": det}
    if expected.get("prefer") == "source" or kind == "expression_pair":
        credit = 0
        for j in judgments:
            if j.get("pair_verdict") not in ("LEFT_BETTER",
                                             "RIGHT_BETTER"):
                continue
            if j.get("preferred") != "source":
                continue
            localized = any(
                c.get("interval_s")
                and _overlap(c["interval_s"],
                             expected["localized_to"]) > 0
                and c.get("axis") in EXPRESSION_AXES
                for c in j.get("claims") or [])
            if localized:
                credit = 1
                break
        return {"credit": credit,
                "note": "generic more-movement rationale earns zero; a "
                        "valid musical counterargument counts against "
                        "sensitivity, never relabels the case"}
    if expected.get("pair") == "TIE" or kind == "benign_pair":
        ok = bool(judgments) and all(
            j.get("pair_verdict") == "TIE"
            and not any(c.get("change_specific_defect")
                        for c in j.get("claims") or [])
            for j in judgments)
        return {"preserved": ok,
                "note": "a TIE does not declare the source "
                        "aesthetically good"}
    raise ValueError(f"unknown expected case {expected!r}")
