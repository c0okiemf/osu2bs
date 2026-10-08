"""Post-failure diagnostic adjudication.

Development already FAILED (retention); this stage runs the ALREADY
BUDGETED development adjudicator tests as diagnostic evidence only —
no threshold/selector/denominator changes, no replacement cases, and it
can never become a development PASS.

- build_pairs stages every expression-destruction and benign pair of
  the 8 development families as a BLIND on-disk case: left/right
  rendered sequence sheets of the changed window (+2 s context), an
  opaque case id, and NO labels/reasons. Which side is the source is a
  deterministic hash bit, recorded only in the custodian manifest.
- batch_prompt builds one isolated-judge request (<=4 pairs) carrying
  the frozen rubric and image paths; primary and reversed orientations
  are SEPARATE calls in FRESH contexts (never the design thread).
- ingest_response validates/normalizes replies (typed pair verdicts,
  claims with intervals/axes, change-specific defect flags) and caches
  by full identity; score_pairs applies the frozen scoring
  (qa.comparator_controls.score_case) plus reversed-order consistency.
"""
import hashlib
import json
from pathlib import Path

from qa.adjudication import PAIR_VERDICTS, normalize_pair, \
    rubric_v2_sha256
from qa.comparator_contract import LABEL, OUT_ROOT, atomic_json
from qa.comparator_controls import EXPRESSION_AXES, score_case

# v2 : corrected bundles carry the complete audio-signal
# panel per side with synchronized time-origin provenance; the first 8
# judgments under pairs/ are retained as SUPERSEDED diagnostics.
PAIRS_DIR = OUT_ROOT / "pairs2"
CUSTODIAN_P = OUT_ROOT / "custodian_pairs2.json"
JUDGE_DIR = OUT_ROOT / "judgments2"
BATCH = 4
CTX_S = 2.0


def _case_id(kind, fam, i):
    return "p-" + hashlib.sha256(
        f"{LABEL}:{kind}:{fam}:{i}".encode()).hexdigest()[:12]


def _source_left(case_id):
    return int(hashlib.sha256(f"side:{case_id}".encode())
               .hexdigest(), 16) % 2 == 0


def build_pairs():
    """Stage all pair cases; idempotent (renders cached by content)."""
    from qa.certificates import load_speed_warning
    from qa.comparator_controls import make_cases
    from qa.evidence import render_sequence
    from qa.features import family_evidence
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    from qa.expand import EVAL_FREEZE_P
    if CUSTODIAN_P.exists():
        return json.loads(CUSTODIAN_P.read_text())
    eman = json.loads(EVAL_FREEZE_P.read_text())
    state = json.loads((Path(__file__).resolve().parent.parent
                        / "experiments/qa-v1/collect2_state.json")
                       .read_text())
    manifest = json.loads((Path(__file__).resolve().parent.parent
                           / "eval/corpus_manifest.json").read_text())
    _ = load_speed_warning()
    custodian = {"cases": {}, "completed": False}
    for fam in sorted(eman["samples"]):
        dat_p, info_p = _family_chart(
            manifest, fam, state["charts"][fam]["difficulty"])
        scene = read_scene(dat_p, info_p)
        ev = family_evidence(fam)
        aud = ev.get("audio") or {}
        cases = make_cases(fam, scene, aud if aud.get("supported")
                           else None, {"label": LABEL})
        for kind, items in (("expr", cases["expression"]),
                            ("benign", cases["benign"])):
            for i, c in enumerate(items):
                cid = _case_id(kind, fam, i)
                d = PAIRS_DIR / cid
                d.mkdir(parents=True, exist_ok=True)
                core = c["core_s"]
                iv = (max(0.0, core[0] - CTX_S), core[1] + CTX_S)
                if kind == "expr":
                    a_scene, b_scene = scene, c["mutant"]
                else:
                    a_scene, b_scene = scene, c["other"]
                a_audio = aud if aud.get("supported") else None
                if kind == "benign":
                    b_audio = c.get("other_audio") or a_audio
                    b_shift = c.get("other_shift_s", 0.0)
                else:
                    b_audio, b_shift = a_audio, 0.0
                src_left = _source_left(cid)
                left_sc = a_scene if src_left else b_scene
                right_sc = b_scene if src_left else a_scene
                left = render_sequence(left_sc, iv, d / "left")
                right = render_sequence(right_sc, iv, d / "right")
                from qa.evidence import render_signal_panel
                b_iv = (iv[0] + b_shift, iv[1] + b_shift)
                a_sig = render_signal_panel(a_audio, iv, d / "left"
                                            if src_left else d / "right")
                b_sig = render_signal_panel(b_audio, b_iv, d / "right"
                                            if src_left else d / "left")
                l_sig = a_sig if src_left else b_sig
                r_sig = b_sig if src_left else a_sig
                atomic_json(d / "meta.json",
                            {"case_id": cid, "interval_s": list(iv),
                             "changed_window_s": list(core),
                             "note": "the changed window is shown; the "
                                     "reason for selection is not "
                                     "disclosed",
                             "left_image": f"left/{left['image']}",
                             "right_image": f"right/{right['image']}",
                             "left_audio": ("left/" + l_sig["image"])
                             if not l_sig.get("missing") else None,
                             "right_audio": ("right/" + r_sig["image"])
                             if not r_sig.get("missing") else None,
                             "completed": True})
                custodian["cases"][cid] = {
                    "family": fam, "kind": kind, "index": i,
                    "core_s": list(core),
                    "source_side_primary": "left" if src_left
                    else "right",
                    "expected": c["expected"],
                    "left_sha256": left["image_sha256"],
                    "right_sha256": right["image_sha256"]}
        print(f"  [pairs] {fam}: staged", flush=True)
    custodian["completed"] = True
    atomic_json(CUSTODIAN_P, custodian)
    print(f"staged {len(custodian['cases'])} pair cases")
    return custodian


def batch_prompt(case_ids, orientation="primary"):
    """One isolated-judge request for <=4 pairs. The judge sees images +
    rubric only; reversed orientation swaps the presented sides."""
    rubric = (Path(__file__).resolve().parent / "prompts"
              / "comparator-v1.md").read_text()
    lines = [rubric, "\n---\n",
             "You are judging PAIRS of chart excerpts (LEFT vs RIGHT), "
             "both rendered over the same seconds. For each case give "
             "ONLY a JSON object per the schema below. The changed "
             "window is indicated; why it was selected is not "
             "disclosed. Respond with a single JSON array, one object "
             "per case, no prose outside it.",
             'Schema per case: {"case_id": "...", "pair_verdict": '
             '"LEFT_BETTER"|"TIE"|"RIGHT_BETTER"|"INSUFFICIENT", '
             '"claims": [{"interval_s": [a,b], "axis": '
             '"continuity"|"coordination"|"vocabulary"|'
             '"audio_correspondence", "text": "...", "evidence_ref": '
             '"left"|"right", "change_specific_defect": false}], '
             '"limitations": ["..."]}',
             "Claims need absolute-second intervals inside the shown "
             "window and a concrete comparison; 'more movement is "
             "better' alone is worthless. If the two sides are "
             "equivalent in mapping quality, say TIE. Do not invent "
             "defects you cannot point to.\n"]
    lines.append(
        "Each side also has an AUDIO SIGNALS panel (per-second energy, "
        "beat ticks, dashed section bounds) rendered in THAT SIDE'S own "
        "time origin — the highlighted span is the shown window. Use it "
        "for audio-correspondence claims; a time-shifted side has a "
        "correspondingly shifted panel (synchronized provenance).\n")
    for cid in case_ids:
        meta = json.loads((PAIRS_DIR / cid / "meta.json").read_text())
        li, ri = meta["left_image"], meta["right_image"]
        la, ra = meta.get("left_audio"), meta.get("right_audio")
        if orientation == "reversed":
            li, ri = ri, li
            la, ra = ra, la
        line = (f"CASE {cid}: window "
                f"{meta['changed_window_s'][0]:.2f}-"
                f"{meta['changed_window_s'][1]:.2f}s; LEFT image: "
                f"{PAIRS_DIR / cid / li}; RIGHT image: "
                f"{PAIRS_DIR / cid / ri}")
        line += (f"; LEFT audio panel: {PAIRS_DIR / cid / la}"
                 if la else "; LEFT audio panel: MISSING")
        line += (f"; RIGHT audio panel: {PAIRS_DIR / cid / ra}"
                 if ra else "; RIGHT audio panel: MISSING")
        lines.append(line)
    return "\n".join(lines)


def make_batches():
    custodian = json.loads(CUSTODIAN_P.read_text())
    ids = sorted(custodian["cases"])
    return [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]


def _identity(orientation):
    recipe = json.loads((OUT_ROOT / "recipe.json").read_text())
    return {"contract_sha256": recipe["recipe_sha256"],
            "model": recipe["adjudicator"]["model"],
            "settings": json.dumps(recipe["adjudicator"]["settings"],
                                   sort_keys=True),
            "prompt_sha256": rubric_v2_sha256(),
            "orientation": orientation}


def ingest_response(raw_text, case_ids, orientation):
    """Validate + cache one batch reply; first valid per case is final."""
    import re
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    m = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not m:
        raise ValueError("no JSON array in judge response")
    arr = json.loads(m.group(0))
    got = {}
    for obj in arr:
        cid = obj.get("case_id")
        if cid not in case_ids:
            continue
        pv = obj.get("pair_verdict")
        if pv not in PAIR_VERDICTS:
            raise ValueError(f"{cid}: bad pair verdict {pv!r}")
        for c in obj.get("claims") or []:
            iv = c.get("interval_s")
            if not (isinstance(iv, (list, tuple)) and len(iv) == 2
                    and iv[0] < iv[1]):
                raise ValueError(f"{cid}: claim needs an ordered "
                                 "interval")
        key = hashlib.sha256(
            f"{json.dumps(_identity(orientation), sort_keys=True)}:"
            f"{cid}".encode()).hexdigest()[:24]
        p = JUDGE_DIR / f"{cid}.{orientation}.json"
        if not p.exists():
            atomic_json(p, {"case_id": cid, "orientation": orientation,
                            "verdict": obj, "cache_key": key,
                            "completed": True})
        got[cid] = obj
    missing = [c for c in case_ids if c not in got]
    return {"ingested": sorted(got), "missing": missing}


def score_pairs():
    """Frozen scoring over all ingested judgments; per-family expression
    credits, benign preservation and reversed-order consistency."""
    custodian = json.loads(CUSTODIAN_P.read_text())
    by_family = {}
    for cid, lab in sorted(custodian["cases"].items()):
        fam = lab["family"]
        rec = by_family.setdefault(
            fam, {"expr_credits": [], "benign_preserved": [],
                  "consistency": [], "missing": []})
        entry = {}
        for orient in ("primary", "reversed"):
            p = JUDGE_DIR / f"{cid}.{orient}.json"
            if p.exists():
                entry[orient] = json.loads(p.read_text())["verdict"]
        if "primary" not in entry:
            rec["missing"].append(cid)
            continue
        prim = entry["primary"]
        src_side = lab["source_side_primary"]
        preferred = None
        if prim["pair_verdict"] in ("LEFT_BETTER", "RIGHT_BETTER"):
            win = "left" if prim["pair_verdict"] == "LEFT_BETTER" \
                else "right"
            preferred = "source" if win == src_side else "mutant"
        judgment = {"pair_verdict": prim["pair_verdict"],
                    "preferred": preferred,
                    "claims": prim.get("claims") or []}
        if lab["kind"] == "expr":
            exp = {"kind": "expression_pair", "prefer": "source",
                   "localized_to": lab["core_s"]}
            rec["expr_credits"].append(
                score_case(exp, {}, [judgment])["credit"])
        else:
            exp = {"kind": "benign_pair", "pair": "TIE"}
            rec["benign_preserved"].append(
                int(score_case(exp, {}, [judgment])["preserved"]))
        if "reversed" in entry:
            a = prim["pair_verdict"]
            b = normalize_pair(entry["reversed"]["pair_verdict"],
                               reversed_order=True)
            rec["consistency"].append(int(a == b))
    report = {"per_family": by_family,
              "totals": {
                  "expr": [sum(r["expr_credits"]) for r in
                           by_family.values()],
                  "benign": [sum(r["benign_preserved"]) for r in
                             by_family.values()]},
              "note": "POST-FAILURE DIAGNOSTIC : "
                      "development remains FAILED/EVALUATOR_NOT_READY; "
                      "this can never become a development PASS",
              "completed": False}
    atomic_json(OUT_ROOT / "diagnostic_adjudication.json", report)
    return report


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build_pairs()
    elif cmd == "score":
        print(json.dumps(score_pairs()["totals"], indent=1))
    elif cmd == "ingest":                    # ingest NN ORIENT
        import shutil
        b, orient = sys.argv[2], sys.argv[3]
        src = Path(f"/tmp/batch{b}.{orient}.json")
        ids = json.loads((OUT_ROOT / f"batch_prompts2/batch{b}.ids.json")
                         .read_text())
        print(ingest_response(src.read_text(), ids, orient))
        (OUT_ROOT / "judge_out").mkdir(exist_ok=True)
        shutil.copy(src, OUT_ROOT / f"judge_out/batch{b}.{orient}.json")
    else:
        raise SystemExit(f"unknown command {cmd!r}")
