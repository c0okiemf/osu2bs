"""Development-only rubric-calibration packet.

Separates Q1 observed expressive loss from Q2 overall mapping preference
on the SAME 78 v2 development bundles (pairs2), frozen before any call.
Cannot pass or rename any earlier gate; development stays FAILED.

- Short-circuit ONLY verified semantic equivalence: benign serialization
  pairs whose re-serialized scene equals the source (notes+walls) with the
  same audio. They get no judge call and leave the judge denominators.
- Batching comparison before the main run: 12 hash-ranked judged cases,
  primary orientation, as 3 calls x 4 and 1 call x 12. Adopt batch 12 iff
  both dimensions agree on >= 11/12 cases; otherwise keep batch 4.
- Frozen scoring: recognition (Q1 names the mutant side with a localized
  vocabulary/coordination claim) >= 80% pooled and >= 3/5 per family;
  benign no-false-loss >= 90%; Q1 order consistency >= 90%. Q2 is scored
  with the old source-preference/consistency gates and reported
  separately. fam:23924's construction shortfall is preserved.
"""
import hashlib
import json
from pathlib import Path

from qa.adjudication import normalize_pair
from qa.comparator_contract import OUT_ROOT, atomic_json
from qa.comparator_pairs import CUSTODIAN_P, PAIRS_DIR, _case_id

CAL = OUT_ROOT / "calib"
PLAN_P = CAL / "plan.json"
RUBRIC_P = Path(__file__).resolve().parent / "prompts" / \
    "comparator-calib-pairs.md"
LOSS_AXES = ("vocabulary", "coordination")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plan():
    """Freeze case list, short-circuits, batching subset. Idempotent."""
    if PLAN_P.exists():
        return json.loads(PLAN_P.read_text())
    from qa.comparator_controls import make_cases
    from qa.comparator_contract import LABEL
    from qa.features import family_evidence
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    root = Path(__file__).resolve().parent.parent
    state = json.loads((root / "experiments/qa-v1/collect2_state.json")
                       .read_text())
    manifest = json.loads((root / "eval/corpus_manifest.json").read_text())
    cust = json.loads(CUSTODIAN_P.read_text())["cases"]
    short = []
    for fam in sorted({c["family"] for c in cust.values()}):
        dat_p, info_p = _family_chart(manifest, fam,
                                      state["charts"][fam]["difficulty"])
        scene = read_scene(dat_p, info_p)
        aud = family_evidence(fam).get("audio") or {}
        cases = make_cases(fam, scene, aud if aud.get("supported")
                           else None, {"label": LABEL})
        for i, c in enumerate(cases["benign"]):
            o = c["other"]
            if (c["transform"] == "serialization"
                    and c.get("other_audio") is None
                    and o["notes"] == [tuple(n) for n in scene["notes"]]
                    and o["walls"] == [tuple(w) for w in
                                       scene.get("walls") or []]):
                short.append(_case_id("benign", fam, i))
    judged = sorted(set(cust) - set(short))
    subset = sorted(judged, key=lambda c: hashlib.sha256(
        f"calib-batch:{c}".encode()).hexdigest())[:12]
    rec = {"ruling": "review", "rubric_sha256": _sha(RUBRIC_P),
           "short_circuit": short, "judged": judged,
           "batch_subset": subset,
           "rule": "adopt batch 12 iff Q1 side AND Q2 verdict agree on "
                   ">=11/12 subset cases between 3x4 and 1x12 calls",
           "completed": True}
    CAL.mkdir(parents=True, exist_ok=True)
    atomic_json(PLAN_P, rec)
    return rec


def prompt(case_ids, orientation):
    lines = [RUBRIC_P.read_text(), "\n## Cases\n"]
    for cid in case_ids:
        m = json.loads((PAIRS_DIR / cid / "meta.json").read_text())
        li, ri = m["left_image"], m["right_image"]
        la, ra = m.get("left_audio"), m.get("right_audio")
        if orientation == "reversed":
            li, ri, la, ra = ri, li, ra, la
        d = PAIRS_DIR / cid
        lines.append(
            f"CASE {cid}: window {m['changed_window_s'][0]:.2f}-"
            f"{m['changed_window_s'][1]:.2f}s; LEFT sequence {d / li}; "
            f"LEFT audio {d / la if la else 'MISSING'}; RIGHT sequence "
            f"{d / ri}; RIGHT audio {d / ra if ra else 'MISSING'}")
    return "\n".join(lines)


def write_call(name, case_ids, orientation):
    CAL.mkdir(parents=True, exist_ok=True)
    (CAL / f"{name}.txt").write_text(prompt(case_ids, orientation))
    (CAL / f"{name}.ids.json").write_text(json.dumps(
        {"ids": case_ids, "orientation": orientation}))
    return CAL / f"{name}.txt"


def ingest(name, arm="main"):
    """/tmp/<name>.json -> calib/judgments/<arm>/<cid>.<orient>.json;
    first valid per case is final."""
    spec = json.loads((CAL / f"{name}.ids.json").read_text())
    import re
    raw = Path(f"/tmp/{name}.json").read_text()
    arr = json.loads(re.search(r"\[.*\]", raw, re.DOTALL).group(0))
    out = CAL / "judgments" / arm
    out.mkdir(parents=True, exist_ok=True)
    got = []
    for o in arr:
        cid = o.get("case_id")
        if cid not in spec["ids"]:
            continue
        side = (o.get("observed_loss") or {}).get("side")
        pref = o.get("overall_preference")
        if side not in ("LEFT", "RIGHT", "NONE", "UNKNOWN") or pref not in (
                "LEFT_BETTER", "TIE", "RIGHT_BETTER", "INSUFFICIENT"):
            raise ValueError(f"{cid}: invalid Q1/Q2 answer")
        p = out / f"{cid}.{spec['orientation']}.json"
        if not p.exists():
            atomic_json(p, {"case_id": cid, "verdict": o,
                            "call": name, "completed": True})
        got.append(cid)
    missing = sorted(set(spec["ids"]) - set(got))
    print({"ingested": len(got), "missing": missing})
    return missing


def _norm_side(side, reversed_order):
    if not reversed_order or side not in ("LEFT", "RIGHT"):
        return side
    return "RIGHT" if side == "LEFT" else "LEFT"


def batch_agreement():
    pl = json.loads(PLAN_P.read_text())
    agree = 0
    for cid in pl["batch_subset"]:
        a = json.loads((CAL / "judgments/b4" / f"{cid}.primary.json")
                       .read_text())["verdict"]
        b = json.loads((CAL / "judgments/main" / f"{cid}.primary.json")
                       .read_text())["verdict"]
        agree += int(a["observed_loss"]["side"]
                     == b["observed_loss"]["side"]
                     and a["overall_preference"]
                     == b["overall_preference"])
    return agree


def score():
    pl = json.loads(PLAN_P.read_text())
    cust = json.loads(CUSTODIAN_P.read_text())["cases"]
    J = CAL / "judgments/main"
    fams = {}
    for cid in pl["judged"]:
        lab = cust[cid]
        f = fams.setdefault(lab["family"], {"recog": [], "no_false_loss": [],
                                            "pref": [], "q1_consist": [],
                                            "q2_consist": []})
        pp = J / f"{cid}.primary.json"
        if not pp.exists():
            continue
        v = json.loads(pp.read_text())["verdict"]
        src = lab["source_side_primary"].upper()
        mut = "RIGHT" if src == "LEFT" else "LEFT"
        side = v["observed_loss"]["side"]
        if lab["kind"] == "expr":
            loc = any(c.get("axis") in LOSS_AXES and c.get("interval_s")
                      and min(c["interval_s"][1], lab["core_s"][1])
                      - max(c["interval_s"][0], lab["core_s"][0]) > 0
                      for c in v["observed_loss"].get("claims") or [])
            f["recog"].append(int(side == mut and loc))
            want = f"{src}_BETTER"
            f["pref"].append(int(v["overall_preference"] == want))
        else:
            f["no_false_loss"].append(int(side in ("NONE", "UNKNOWN")))
        rp = J / f"{cid}.reversed.json"
        if rp.exists():
            r = json.loads(rp.read_text())["verdict"]
            f["q1_consist"].append(int(
                _norm_side(r["observed_loss"]["side"], True) == side))
            f["q2_consist"].append(int(
                normalize_pair(r["overall_preference"], True)
                == v["overall_preference"]))

    def pooled(key):
        h = sum(sum(v[key]) for v in fams.values())
        n = sum(len(v[key]) for v in fams.values())
        return h, n, (h / n if n else None)
    out = {"per_family": {k: {kk: [sum(vv), len(vv)] for kk, vv in v.items()}
                          for k, v in sorted(fams.items())}}
    for key in ("recog", "no_false_loss", "pref", "q1_consist",
                "q2_consist"):
        out[key] = pooled(key)
    r_ok = (out["recog"][2] or 0) >= 0.80 and all(
        v["recog"] and sum(v["recog"]) >= 3 and len(v["recog"]) >= 5
        for v in fams.values())
    out["gates"] = {
        "recognition": r_ok,
        "benign_no_false_loss": (out["no_false_loss"][2] or 0) >= 0.90,
        "q1_consistency": (out["q1_consist"][2] or 0) >= 0.90,
        "q2_preference_old_gate": (out["pref"][2] or 0) >= 0.80,
        "q2_consistency_old_gate": (out["q2_consist"][2] or 0) >= 0.90}
    out["short_circuit_n"] = len(pl["short_circuit"])
    out["note"] = ("development-only calibration diagnostic; cannot pass "
                   "or rename earlier gates; fam:23924 construction "
                   "shortfall preserved")
    out["completed"] = True
    atomic_json(CAL / "calib_report.json", out)
    return out


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1]
    if cmd == "plan":
        p = plan()
        print(len(p["short_circuit"]), "short-circuit;", len(p["judged"]),
              "judged")
    elif cmd == "ingest":
        ingest(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "main")
    elif cmd == "agree":
        print(batch_agreement(), "/ 12")
    elif cmd == "score":
        print(json.dumps({k: v for k, v in score().items()
                          if k != "per_family"}, indent=1))
