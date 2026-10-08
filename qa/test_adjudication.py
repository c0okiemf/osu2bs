"""Task 7 tests: verdict schema and routing.
  .venv/bin/python -m unittest qa.test_adjudication -q
"""
import tempfile
import unittest

from qa.adjudication import (cache_key, normalize_pair, parse_response,
                             record_judgment, record_judgment_v2,
                             stage_request, validate_verdict,
                             validate_verdict_v2)


def fixture_bundle(mandatory_pass=True, missing=False, cid="c-01"):
    return {"candidate_id": cid, "mandatory_pass": mandatory_pass,
            "missing_evidence": ["render"] if missing else [],
            "bundle_sha256": "ab" * 32}


def verdict_fixture(v="PASS", **kw):
    return {"verdict": v, "reason_codes": [], "claims": [],
            "unknowns": [], "modality_used": ["image"], **kw}


class TestValidation(unittest.TestCase):
    def test_judge_cannot_override_missing_support(self):
        with self.assertRaises(ValueError):
            validate_verdict(fixture_bundle(mandatory_pass=False),
                             verdict_fixture("PASS"))

    def test_pass_with_missing_evidence_refused(self):
        with self.assertRaises(ValueError):
            validate_verdict(fixture_bundle(missing=True),
                             verdict_fixture("PASS"))

    def test_regenerate_allowed_despite_gates(self):
        d = validate_verdict(fixture_bundle(mandatory_pass=False),
                             verdict_fixture("REGENERATE",
                                             reason_codes=["OUT_OF_SUPPORT"]))
        self.assertEqual(d["verdict"], "REGENERATE")

    def test_claims_need_intervals(self):
        v = verdict_fixture("REGENERATE",
                            claims=[{"claim": "boring", "evidence": "x"}])
        with self.assertRaisesRegex(ValueError, "interval"):
            validate_verdict(fixture_bundle(), v)

    def test_pair_verdict_schema(self):
        d = validate_verdict(fixture_bundle(),
                             verdict_fixture("PASS",
                                             pair_verdict="LEFT_BETTER"))
        self.assertEqual(d["pair_verdict"], "LEFT_BETTER")
        with self.assertRaises(ValueError):
            validate_verdict(fixture_bundle(),
                             verdict_fixture("PASS", pair_verdict="MAYBE"))


class TestRouting(unittest.TestCase):
    def test_first_valid_response_is_final(self):
        with tempfile.TemporaryDirectory() as td:
            a = record_judgment(td, "c-01", fixture_bundle(),
                                verdict_fixture("REGENERATE"), "raw1")
            b = record_judgment(td, "c-01", fixture_bundle(),
                                verdict_fixture("PASS"), "raw2")
            self.assertEqual(b["decision"]["verdict"], "REGENERATE")
            self.assertEqual(a["raw"], b["raw"])

    def test_parse_response_extracts_json(self):
        v = parse_response('noise {"verdict": "PASS"} trailing')
        self.assertEqual(v["verdict"], "PASS")
        with self.assertRaises(ValueError):
            parse_response("no json here")


def v2_bundle(status="NO_CONTRADICTION_FOUND", support_pass=True,
              missing=(), warnings=(), sha="ab" * 32):
    return {"candidate_id": "c-01", "bundle_sha256": sha,
            "contradictions": {"status": status, "structural": [],
                               "model": [], "warnings": list(warnings),
                               "unknowns": []},
            "support": {"machine_support_pass": support_pass},
            "witnesses": [{"name": "first_active"},
                          {"name": "peak_cadence"}],
            "renders": {"first_active": {"image": "a.png"}},
            "modalities": {"received": ["image", "signals"],
                           "used": []},
            "missing_evidence": list(missing)}


def v2_verdict(v="REGENERATE", **kw):
    d = {"verdict": v,
         "reason": "MAP_DEFECT" if v == "REGENERATE" else None,
         "axes": {"continuity": "SUPPORTED",
                  "coordination": "SUPPORTED",
                  "vocabulary": "SUPPORTED",
                  "audio_correspondence": "SUPPORTED"},
         "claims": [{"interval_s": [1.0, 3.0], "axis": "coordination",
                     "text": "clean alternation into the doubles",
                     "evidence_ref": "first_active"},
                    {"interval_s": [4.0, 6.0], "axis": "vocabulary",
                     "text": "arc development answers the build",
                     "evidence_ref": "peak_cadence"}],
         "modalities_used": ["image"], "limitations": []}
    d.update(kw)
    return d


IDENTITY = {"contract_sha256": "cc" * 32, "model": "gpt-6-astra",
            "settings": "high", "prompt_sha256": "dd" * 32,
            "orientation": "primary"}


class TestValidateV2(unittest.TestCase):
    def test_fake_listening_rejected(self):
        with self.assertRaisesRegex(ValueError, "listening|received"):
            validate_verdict_v2(v2_bundle(),
                                v2_verdict(modalities_used=["audio"]))

    def test_nonexistent_evidence_rejected(self):
        v = v2_verdict()
        v["claims"][0]["evidence_ref"] = "no_such_witness"
        with self.assertRaisesRegex(ValueError, "nonexistent"):
            validate_verdict_v2(v2_bundle(), v)

    def test_bad_intervals_rejected(self):
        for iv in ([3.0, 1.0], [float("nan"), 2.0], [-1.0, 2.0]):
            v = v2_verdict()
            v["claims"][0]["interval_s"] = iv
            with self.assertRaises(ValueError):
                validate_verdict_v2(v2_bundle(), v)

    def test_pass_with_unknown_axis_rejected(self):
        v = v2_verdict("PASS")
        v["axes"]["vocabulary"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            validate_verdict_v2(v2_bundle(), v)

    def test_model_score_rationale_rejected(self):
        v = v2_verdict()
        v["claims"][0]["text"] = "the mixture NLL prefers this"
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            validate_verdict_v2(v2_bundle(), v)

    def test_hard_fail_needs_structural_certificate(self):
        with self.assertRaisesRegex(ValueError, "structural"):
            validate_verdict_v2(v2_bundle(),
                                v2_verdict("HARD_FAIL", reason=None))
        ok = validate_verdict_v2(
            v2_bundle(status="STRUCTURAL_CONTRADICTION"),
            v2_verdict("HARD_FAIL", reason=None))
        self.assertEqual(ok["verdict"], "HARD_FAIL")

    def test_pass_needs_both_witness_claim_kinds(self):
        v = v2_verdict("PASS", reason=None)
        v["claims"] = [v["claims"][0]]          # coordination only
        with self.assertRaisesRegex(ValueError, "expression/audio"):
            validate_verdict_v2(v2_bundle(), v)

    def test_pass_with_unresolved_warning_rejected(self):
        b = v2_bundle(warnings=[{"type": "speed_warning"}])
        with self.assertRaisesRegex(ValueError, "warning"):
            validate_verdict_v2(b, v2_verdict("PASS", reason=None))
        ok = validate_verdict_v2(
            b, v2_verdict("PASS", reason=None, warnings_addressed=[0]))
        self.assertEqual(ok["verdict"], "PASS")

    def test_claims_preserved_in_normalized_decision(self):
        d = validate_verdict_v2(v2_bundle(), v2_verdict())
        self.assertEqual(len(d["claims"]), 2)
        self.assertEqual(d["axes"]["coordination"], "SUPPORTED")


class TestCacheAndRouting(unittest.TestCase):
    def test_candidate_id_alone_cannot_key_cache(self):
        a = cache_key(IDENTITY, v2_bundle(sha="ab" * 32))
        b = cache_key(IDENTITY, v2_bundle(sha="ba" * 32))
        self.assertNotEqual(a, b)               # same candidate_id!
        with self.assertRaises(ValueError):
            cache_key({**IDENTITY, "prompt_sha256": None}, v2_bundle())

    def test_normalize_pair_reversal(self):
        self.assertEqual(normalize_pair("LEFT_BETTER", True),
                         "RIGHT_BETTER")
        self.assertEqual(normalize_pair("TIE", True), "TIE")
        with self.assertRaises(ValueError):
            normalize_pair("MAYBE", False)

    def test_first_valid_is_final_per_identity(self):
        with tempfile.TemporaryDirectory() as td:
            a = record_judgment_v2(td, v2_bundle(),
                                   v2_verdict("REGENERATE"), "raw1",
                                   IDENTITY)
            b = record_judgment_v2(td, v2_bundle(),
                                   v2_verdict("REGENERATE",
                                              reason="SUPPORT_UNKNOWN"),
                                   "raw2", IDENTITY)
            self.assertEqual(b["decision"]["reason"], "MAP_DEFECT")
            rev = record_judgment_v2(td, v2_bundle(),
                                     v2_verdict("REGENERATE"), "raw3",
                                     {**IDENTITY,
                                      "orientation": "reversed"})
            self.assertNotEqual(rev["cache_key"], a["cache_key"])

    def test_stage_request_is_pending_not_a_judgment(self):
        with tempfile.TemporaryDirectory() as td:
            r = stage_request(td, v2_bundle(), IDENTITY)
            self.assertEqual(r["status"], "PENDING_ADJUDICATION")
            self.assertNotIn("decision", r)


if __name__ == "__main__":
    unittest.main()
