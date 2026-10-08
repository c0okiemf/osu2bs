"""Independent-QA Task 6: red-team fixtures and held-out controls (spec §8).

Every mutant carries source hashes, the changed fields, its CLASS, an
expected outcome with rationale, and — for MUST_NOT_PASS only — an ANALYTIC
CERTIFICATE (a mutation without a certificate is a CHALLENGE example,
never a guaranteed negative). Cosmetic/diagonal changes are challenges,
not automatic defects. MUST_PRESERVE invariances must hold exactly for
serialization/time/BPM. Selection windows are deterministic by hash and
never chosen by detector outcome.

Classes:
  MUST_HARD_FAIL — independently reproducible structural defects.
  MUST_NOT_PASS  — certified model-conditional violations (constructed
                   wall corridors whose emptiness qa.physics proves).
  CHALLENGE      — expression probes (mechanical collapse, diagonal
                   substitution, cosmetic shifts, gap compression, sparse
                   low cuts) judged blind, plus intentional-repetition
                   foils that must NOT be condemned.
  MUST_PRESERVE  — byte/time/BPM/gain/mirror invariances.
"""
import copy
import hashlib
import json

# v2 (comparator contract, spec §3): duplicate occupancy is AMBIGUOUS
# game semantics, not an independently reproducible structural defect —
# it must be abstained on, never counted as a hard negative. v1 RECIPES
# stay untouched for historical result reproducibility.
RECIPES_VERSION_V2 = 2


def recipes_v2():
    out = dict(RECIPES)
    out["duplicate_occupancy"] = {
        "class": "MUST_ABSTAIN",
        "rationale": "exact duplicate occupancy is ambiguous game "
                     "semantics; abstain unless a verified game-semantic "
                     "contradiction is supplied"}
    return out


RECIPES = {
    "nonfinite_time": {"class": "MUST_HARD_FAIL",
                       "rationale": "nonfinite required timing field"},
    "invalid_type": {"class": "MUST_HARD_FAIL",
                     "rationale": "invalid required note type"},
    "duplicate_occupancy": {"class": "MUST_HARD_FAIL",
                            "rationale": "exact duplicate note occupancy"},
    "empty_corridor": {"class": "MUST_NOT_PASS",
                       "rationale": "full-height walls close every stance "
                                    "column simultaneously"},
    "compress_recovery": {"class": "CHALLENGE",
                          "rationale": "halved gaps; no analytic certificate"
                                       " on the standard grid relaxation"},
    "mechanical_collapse": {"class": "CHALLENGE",
                            "rationale": "varied passage replaced by one "
                                         "repeated motif >=8x, audio fixed"},
    "diagonal_substitution": {"class": "CHALLENGE",
                              "rationale": "cardinal cuts remapped to "
                                           "diagonals; geometry unchanged"},
    "cosmetic_shift": {"class": "CHALLENGE",
                       "rationale": "single-column lateral shift of a "
                                    "window; not automatically a defect"},
    "sparse_low_cuts": {"class": "CHALLENGE",
                        "rationale": "window thinned to alternating low "
                                     "verticals against unchanged audio"},
    "intentional_repetition_foil": {"class": "CHALLENGE",
                                    "rationale": "steady-beat repetition "
                                                 "kept: must NOT be "
                                                 "condemned"},
    "time_translate": {"class": "MUST_PRESERVE",
                       "rationale": "translation with matching audio "
                                    "origin"},
    "bpm_reencode": {"class": "MUST_PRESERVE",
                     "rationale": "seconds-equivalent BPM re-encoding"},
    "serialize_roundtrip": {"class": "MUST_PRESERVE",
                            "rationale": "byte-equivalent serialization"},
    "mirror_chart_and_replay": {"class": "MUST_PRESERVE",
                                "rationale": "chart+replay mirrored "
                                             "together"},
}

_DIAG = {0: 4, 1: 6, 2: 6, 3: 5}       # cardinal -> a diagonal neighbour


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()) \
        .hexdigest()


def select_windows(chart, seed, n=6, span_beats=16.0):
    """Deterministic nonoverlapping windows by hash order over candidate
    starts; never conditioned on detector outcomes."""
    notes = sorted(chart.get("_notes", []), key=lambda x: x["_time"])
    if not notes:
        return []
    t0, t1 = notes[0]["_time"], notes[-1]["_time"]
    starts = []
    s = t0
    while s + span_beats <= t1:
        cnt = sum(1 for x in notes if s <= x["_time"] < s + span_beats)
        if cnt >= 8:
            starts.append(round(s, 3))
        s += span_beats / 2
    ranked = sorted(starts, key=lambda x: hashlib.sha256(
        f"{seed}:{x}".encode()).hexdigest())
    out = []
    for s in ranked:
        if all(s + span_beats <= a or s >= b for a, b in out):
            out.append((s, s + span_beats))
        if len(out) >= n:
            break
    return sorted(out)


def mutate(chart, audio, recipe, seed):
    """One mutant chart per recipe. Locality: only the declared fields/
    window change; original audio is preserved except explicitly benign
    synchronized transforms (declared in the record)."""
    rec = RECIPES[recipe]
    src_hash = _sha(chart)
    c = copy.deepcopy(chart)
    notes = sorted(c.get("_notes", []), key=lambda x: x["_time"])
    c["_notes"] = notes
    out = {"recipe": recipe, "class": rec["class"],
           "expected": rec["class"], "rationale": rec["rationale"],
           "source_sha256": src_hash, "certificate": None,
           "audio_transform": None, "interval_beats": None}
    wins = select_windows(chart, seed=f"{recipe}:{seed}")
    win = wins[seed % len(wins)] if wins else None

    if recipe == "nonfinite_time":
        notes[len(notes) // 2]["_time"] = float("nan")
    elif recipe == "invalid_type":
        notes[len(notes) // 2]["_type"] = 7
    elif recipe == "duplicate_occupancy":
        n0 = dict(notes[len(notes) // 2])
        notes.append(dict(n0))
        c["_notes"] = sorted(notes, key=lambda x: (x["_time"] if
                                                   x["_time"] == x["_time"]
                                                   else 0))
    elif recipe == "empty_corridor":
        t = notes[len(notes) // 2]["_time"]
        c.setdefault("_obstacles", [])
        c["_obstacles"] += [
            {"_time": t - 1.0, "_lineIndex": 0, "_type": 0,
             "_duration": 2.0, "_width": 2},
            {"_time": t - 1.0, "_lineIndex": 2, "_type": 0,
             "_duration": 2.0, "_width": 2}]
        from qa.physics import _corridor
        cert = _corridor([(w["_time"], w["_lineIndex"], w["_type"],
                           w["_duration"], w["_width"])
                          for w in c["_obstacles"]], 0, 1e9)
        out["certificate"] = cert if cert["empty_intervals"] else None
        if out["certificate"] is None:
            out["expected"] = "CHALLENGE"      # no certificate, no claim
    elif recipe == "compress_recovery":
        if win:
            w0, w1 = win
            out["interval_beats"] = list(win)
            for x in notes:
                if w0 <= x["_time"] < w1:
                    x["_time"] = w0 + (x["_time"] - w0) * 0.5
            c["_notes"] = sorted(notes, key=lambda x: x["_time"])
        # no analytic certificate on the standard grid: stays CHALLENGE
    elif recipe == "mechanical_collapse":
        if not win:
            return {**out, "expected": "SKIP",
                    "rationale": "no supported window"}
        w0, w1 = win
        inside = [x for x in notes if w0 <= x["_time"] < w1]
        motifs = {(x["_lineIndex"], x["_lineLayer"], x["_cutDirection"])
                  for x in inside}
        if len(motifs) < 4 or len(inside) < 8:
            return {**out, "expected": "SKIP",
                    "rationale": "original lacks >=4 distinct motifs / "
                                 ">=8 events in the window"}
        out["interval_beats"] = list(win)
        proto = inside[0]
        for x in inside:
            x["_lineIndex"] = proto["_lineIndex"]
            x["_lineLayer"] = proto["_lineLayer"]
            x["_cutDirection"] = proto["_cutDirection"]
    elif recipe == "diagonal_substitution":
        for x in notes:
            if x["_cutDirection"] in _DIAG:
                x["_cutDirection"] = _DIAG[x["_cutDirection"]]
    elif recipe == "cosmetic_shift":
        if win:
            w0, w1 = win
            out["interval_beats"] = list(win)
            for x in notes:
                if w0 <= x["_time"] < w1 and 0 <= x["_lineIndex"] <= 2:
                    x["_lineIndex"] += 1
    elif recipe == "sparse_low_cuts":
        if not win:
            return {**out, "expected": "SKIP",
                    "rationale": "no supported window"}
        w0, w1 = win
        out["interval_beats"] = list(win)
        kept, i = [], 0
        for x in notes:
            if w0 <= x["_time"] < w1:
                if i % 2 == 0:
                    x["_lineLayer"] = 0
                    x["_cutDirection"] = 0 if i % 4 == 0 else 1
                    kept.append(x)
                i += 1
            else:
                kept.append(x)
        c["_notes"] = kept
    elif recipe == "intentional_repetition_foil":
        pass                                    # unmodified: the foil IS
        out["rationale"] += " (chart unmodified)"
    elif recipe == "time_translate":
        for x in notes:
            x["_time"] += 16.0
        for w in c.get("_obstacles", []):
            w["_time"] += 16.0
        out["audio_transform"] = "prepend 16 beats of silence (declared)"
    elif recipe == "bpm_reencode":
        for x in notes:
            x["_time"] *= 2.0
        for w in c.get("_obstacles", []):
            w["_time"] *= 2.0
            w["_duration"] *= 2.0
        out["audio_transform"] = "info bpm doubled (seconds identical)"
        out["bpm_factor"] = 2.0
    elif recipe == "serialize_roundtrip":
        c = json.loads(json.dumps(chart, sort_keys=True))
    elif recipe == "mirror_chart_and_replay":
        from qa.scene import MIRROR_DIR
        for x in notes:
            x["_lineIndex"] = 3 - x["_lineIndex"]
            if x["_type"] in (0, 1):
                x["_type"] = 1 - x["_type"]
            x["_cutDirection"] = MIRROR_DIR.get(x["_cutDirection"],
                                                x["_cutDirection"])
        out["replay_transform"] = "mirror replays with qa.telemetry_v2." \
                                  "mirror_pose (declared, applied together)"
    else:
        raise KeyError(recipe)
    out["chart"] = c
    out["mutant_sha256"] = _sha(c)
    return out
