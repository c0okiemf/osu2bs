"""Independent-QA Task 7: verdict schema, validation and routing (spec §7).

validate_verdict enforces: machine-readable schema; PASS can never
override a failed mandatory gate or missing required evidence; claims
need intervals; pairwise verdicts carry their own field. Bundles carry an
opaque candidate id; the custodian's origin map never enters this module.
The first completed VALID response is final — schema-invalid transport
retries are logged and never selected by favorability.
"""
import json
import re
from pathlib import Path

RUBRIC_PATH = Path(__file__).resolve().parent / "prompts" / \
    "independent-v1.md"
VERDICTS = ("PASS", "REGENERATE", "HARD_FAIL")
PAIR_VERDICTS = ("LEFT_BETTER", "TIE", "RIGHT_BETTER", "INSUFFICIENT")


def rubric_sha256():
    import hashlib
    return hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()


def validate_verdict(bundle, verdict):
    """Decision from one adjudication response against its bundle."""
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be a dict")
    v = verdict.get("verdict")
    if v not in VERDICTS:
        raise ValueError(f"unknown verdict {v!r}")
    if v == "PASS":
        if not bundle.get("mandatory_pass", False):
            raise ValueError(
                "PASS cannot override failed mandatory machine gates")
        if bundle.get("missing_evidence"):
            raise ValueError("PASS with missing required evidence")
    for c in verdict.get("claims") or []:
        iv = c.get("interval_s")
        if not (isinstance(iv, (list, tuple)) and len(iv) == 2
                and all(isinstance(x, (int, float)) for x in iv)):
            raise ValueError("every claim needs a numeric interval_s")
    pv = verdict.get("pair_verdict")
    if pv is not None and pv not in PAIR_VERDICTS:
        raise ValueError(f"unknown pair verdict {pv!r}")
    return {"candidate_id": bundle["candidate_id"], "verdict": v,
            "pair_verdict": pv,
            "reason_codes": verdict.get("reason_codes") or [],
            "unknowns": verdict.get("unknowns") or [],
            "modality_used": verdict.get("modality_used") or [],
            "rubric_sha256": rubric_sha256()}


# ---------------- comparator v2 (spec 2f687f5 §5) ----------------

RUBRIC_V2_PATH = Path(__file__).resolve().parent / "prompts" / \
    "comparator-v1.md"
AXES = ("continuity", "coordination", "vocabulary",
        "audio_correspondence")
AXIS_STATES = ("SUPPORTED", "DEFECT", "UNKNOWN")
REGEN_REASONS = ("MAP_DEFECT", "SUPPORT_UNKNOWN", "SCOPE_UNKNOWN",
                 "EVIDENCE_UNAVAILABLE", "JUDGMENT_UNRESOLVED")
_MODEL_RATIONALE = ("mixture", "nll", "quantile", "critic score",
                    "model score", "latent")


def rubric_v2_sha256():
    import hashlib
    return hashlib.sha256(RUBRIC_V2_PATH.read_bytes()).hexdigest()


def cache_key(identity, bundle):
    """contract+model+settings+prompt+bundle+orientation — a candidate id
    alone can NEVER key a verdict."""
    import hashlib
    for k in ("contract_sha256", "model", "settings", "prompt_sha256",
              "orientation"):
        if identity.get(k) in (None, ""):
            raise ValueError(f"cache identity missing {k!r}")
    payload = {k: identity[k] for k in ("contract_sha256", "model",
                                        "settings", "prompt_sha256",
                                        "orientation")}
    payload["bundle_sha256"] = bundle["bundle_sha256"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     default=str).encode()).hexdigest()


def normalize_pair(verdict, reversed_order):
    if verdict not in PAIR_VERDICTS:
        raise ValueError(f"unknown pair verdict {verdict!r}")
    if not reversed_order:
        return verdict
    return {"LEFT_BETTER": "RIGHT_BETTER",
            "RIGHT_BETTER": "LEFT_BETTER"}.get(verdict, verdict)


def _bundle_asset_names(bundle):
    names = {w.get("name") for w in bundle.get("witnesses") or []}
    names |= set((bundle.get("renders") or {}).keys())
    return {n for n in names if n}


def validate_verdict_v2(bundle, verdict):
    """Schema + honesty validation for one v2 judgment."""
    import math
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be a dict")
    v = verdict.get("verdict")
    if v not in VERDICTS:
        raise ValueError(f"unknown verdict {v!r}")
    axes = verdict.get("axes") or {}
    if set(axes) != set(AXES) or any(axes[a] not in AXIS_STATES
                                     for a in AXES):
        raise ValueError("all four rubric axes must be judged "
                         "SUPPORTED/DEFECT/UNKNOWN")
    received = set((bundle.get("modalities") or {}).get("received") or ())
    used = verdict.get("modalities_used") or []
    for m in used:
        if m not in received:
            raise ValueError(f"claimed modality {m!r} was not received "
                             "(no false listening claims)")
    assets = _bundle_asset_names(bundle)
    for c in verdict.get("claims") or []:
        iv = c.get("interval_s")
        if not (isinstance(iv, (list, tuple)) and len(iv) == 2
                and all(isinstance(x, (int, float))
                        and math.isfinite(float(x)) for x in iv)
                and 0 <= iv[0] < iv[1]):
            raise ValueError("claim intervals must be finite, ordered, "
                             "in-range absolute seconds")
        if c.get("axis") not in AXES:
            raise ValueError(f"claim axis {c.get('axis')!r} unknown")
        ref = c.get("evidence_ref")
        if not ref or ref not in assets:
            raise ValueError(f"claim cites nonexistent evidence {ref!r}")
        text = (c.get("text") or "").lower()
        if any(m in text for m in _MODEL_RATIONALE):
            raise ValueError("diagnostic model output cannot be a "
                             "rationale")
    if v == "HARD_FAIL":
        if (bundle.get("contradictions") or {}).get("status") \
                != "STRUCTURAL_CONTRADICTION":
            raise ValueError("HARD_FAIL requires a structural "
                             "certificate in the bundle")
    if v == "PASS":
        sup = bundle.get("support") or {}
        if not sup.get("machine_support_pass"):
            raise ValueError("PASS cannot override machine eligibility")
        contr = bundle.get("contradictions") or {}
        if contr.get("status") != "NO_CONTRADICTION_FOUND":
            raise ValueError("PASS with unresolved contradictions")
        if bundle.get("missing_evidence"):
            raise ValueError("PASS with missing required evidence")
        if any(axes[a] == "DEFECT" for a in AXES):
            raise ValueError("PASS with a DEFECT axis")
        if any(axes[a] == "UNKNOWN" for a in AXES):
            raise ValueError("PASS with a material UNKNOWN axis")
        claims = verdict.get("claims") or []
        if not any(c.get("axis") in ("continuity", "coordination")
                   for c in claims):
            raise ValueError("PASS needs a coordination/recovery witness "
                             "claim")
        if not any(c.get("axis") in ("vocabulary",
                                     "audio_correspondence")
                   for c in claims):
            raise ValueError("PASS needs an expression/audio witness "
                             "claim")
        warns = contr.get("warnings") or []
        addressed = set(verdict.get("warnings_addressed") or ())
        if warns and addressed != set(range(len(warns))):
            raise ValueError("PASS with unresolved material warnings")
    if v == "REGENERATE" and verdict.get("reason") not in REGEN_REASONS:
        raise ValueError("REGENERATE needs one typed reason")
    pv = verdict.get("pair_verdict")
    if pv is not None and pv not in PAIR_VERDICTS:
        raise ValueError(f"unknown pair verdict {pv!r}")
    return {"verdict": v, "reason": verdict.get("reason"),
            "axes": dict(axes),
            "claims": [dict(c) for c in verdict.get("claims") or []],
            "warnings_addressed":
                list(verdict.get("warnings_addressed") or ()),
            "strongest_contrary": verdict.get("strongest_contrary"),
            "limitations": list(verdict.get("limitations") or ()),
            "modalities_used": list(used),
            "pair_verdict": pv,
            "rubric_sha256": rubric_v2_sha256(),
            "candidate_id": bundle.get("candidate_id")}


def record_judgment_v2(store_dir, bundle, verdict, raw_text, identity):
    """Append-only, keyed by the FULL cache identity; the first
    schema-valid response is final."""
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    key = cache_key(identity, bundle)
    final_p = store / f"{key}.json"
    if final_p.exists():
        return json.loads(final_p.read_text())
    decision = validate_verdict_v2(bundle, verdict)
    rec = {"decision": decision, "raw": raw_text,
           "identity": {k: identity[k] for k in
                        ("contract_sha256", "model", "settings",
                         "prompt_sha256", "orientation")},
           "bundle_sha256": bundle.get("bundle_sha256"),
           "cache_key": key, "completed": True}
    tmp = final_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True,
                              default=str))
    tmp.replace(final_p)
    return rec


def stage_request(store_dir, bundle, identity, orientation="primary"):
    """When no isolated adjudication tooling is available, emit a PENDING
    bundle manifest for review — never a fabricated automated reply."""
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    key = cache_key({**identity, "orientation": orientation}, bundle)
    p = store / f"pending-{key}.json"
    if not p.exists():
        rec = {"status": "PENDING_ADJUDICATION", "cache_key": key,
               "orientation": orientation,
               "bundle_sha256": bundle.get("bundle_sha256"),
               "rubric_sha256": rubric_v2_sha256(),
               "note": "isolated QA context required; this manifest is "
                       "not a judgment"}
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
        tmp.replace(p)
    return json.loads(p.read_text())


def parse_response(text):
    """Extract the first JSON object from an adjudicator response; a
    non-JSON response is a transport failure (retryable with the SAME
    content), never silently coerced."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in adjudication response")
    return json.loads(m.group(0))


def record_judgment(store_dir, candidate_id, bundle, verdict, raw_text,
                    attempt=1):
    """Append-only judgment cache: the first VALID response is final."""
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    final_p = store / f"{candidate_id}.json"
    if final_p.exists():
        return json.loads(final_p.read_text())     # final stays final
    decision = validate_verdict(bundle, verdict)
    rec = {"decision": decision, "raw": raw_text, "attempt": attempt,
           "bundle_sha256": bundle.get("bundle_sha256")}
    tmp = final_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True))
    tmp.replace(final_p)
    return rec
