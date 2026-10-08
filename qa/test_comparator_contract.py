"""Comparator Task 1 tests: contract, lifecycle, freeze machinery.
  .venv/bin/python -m unittest qa.test_comparator_contract -q
"""
import json
import tempfile
import unittest
from pathlib import Path

from qa import comparator_contract as cc


def inputs_fixture():
    return {"adjudicator": {"model": "gpt-6-astra", "version": "2026-09",
                            "settings": {"reasoning": "high"}},
            "modality_profile": "images+signals"}


class TestRecipe(unittest.TestCase):
    def test_model_role_is_diagnostic_only(self):
        r = cc.build_recipe(inputs_fixture())
        self.assertEqual(r["model_decision_role"], "diagnostic_only")

    def test_mandatory_model_role_is_rejected(self):
        r = cc.build_recipe(inputs_fixture())
        with self.assertRaises(ValueError):
            cc.validate_recipe({**r, "model_decision_role": "mandatory"})

    def test_missing_code_files_block_freeze(self):
        with unittest.mock.patch.object(
                cc, "CODE_FILES", cc.CODE_FILES + ("qa/nonexistent.py",)):
            r = cc.build_recipe(inputs_fixture())
        self.assertEqual(r["missing_code_files"], ["qa/nonexistent.py"])
        with self.assertRaisesRegex(ValueError, "missing code"):
            cc.validate_recipe(r)

    def test_unresolved_adjudicator_blocks_freeze(self):
        r = cc.build_recipe({"modality_profile": "images+signals"})
        with self.assertRaisesRegex(ValueError, "adjudicator"):
            cc.validate_recipe(r)

    def test_tampered_hash_rejected(self):
        r = cc.build_recipe(inputs_fixture())
        r["artifact_sha256"]["speed_threshold"] = "aa" * 32
        r["recipe_sha256"] = cc._sha_obj(
            {k: v for k, v in r.items() if k != "recipe_sha256"})
        cc.validate_recipe(r)
        r["budgets"] = {**r["budgets"], "fresh_replays_max": 999}
        with self.assertRaisesRegex(ValueError, "hash"):
            cc.validate_recipe(r)


class TestLifecycle(unittest.TestCase):
    def test_skip_transition_refused(self):
        with self.assertRaises(PermissionError):
            cc.advance({"stage": "RECIPE_FROZEN"}, "FRESH_PASS", {})

    def test_ordered_transitions_require_evidence(self):
        s = {"stage": "BUILDING"}
        with self.assertRaises(ValueError):
            cc.advance(s, "RECIPE_FROZEN", {})
        s = cc.advance(s, "RECIPE_FROZEN", {"recipe_sha256": "aa"})
        self.assertEqual(s["stage"], "RECIPE_FROZEN")
        with self.assertRaises(PermissionError):
            cc.advance(s, "EVALUATOR_FROZEN",
                       {"evaluator_freeze_sha256": "bb"})
        s = cc.advance(s, "DEVELOPMENT_PASS",
                       {"development_report_sha256": "cc"})
        self.assertEqual(len(s["history"]), 2)

    def test_failed_records_stage_and_dimension_and_locks(self):
        s = cc.advance({"stage": "RECIPE_FROZEN"}, "FAILED",
                       {"stage": "development",
                        "dimension": "quality-probe-sensitivity"})
        self.assertEqual(s["stage"], "FAILED")
        with self.assertRaises(PermissionError):
            cc.advance(s, "DEVELOPMENT_PASS",
                       {"development_report_sha256": "cc"})

    def test_unauthorized_fresh_and_seal_access(self):
        with self.assertRaises(PermissionError):
            cc.require_stage({"stage": "DEVELOPMENT_PASS"},
                             "EVALUATOR_FROZEN")
        with self.assertRaises(PermissionError):
            cc.require_stage({"stage": "EVALUATOR_FROZEN"}, "FRESH_PASS")
        cc.require_stage({"stage": "FRESH_PASS"}, "EVALUATOR_FROZEN")

    def test_role_matrix(self):
        from qa.features import records_for_role
        for purpose in ("fit", "reference", "calibrate", "confirm"):
            with self.assertRaises(PermissionError):
                next(records_for_role("qa_validate_comparator", purpose))


class TestAtomicJson(unittest.TestCase):
    def test_completed_collision_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            cc.atomic_json(p, {"completed": True, "v": 1})
            cc.atomic_json(p, {"completed": True, "v": 1})   # same = fine
            with self.assertRaisesRegex(ValueError, "collision"):
                cc.atomic_json(p, {"completed": True, "v": 2})
            self.assertEqual(json.loads(p.read_text())["v"], 1)

    def test_incomplete_records_may_progress(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            cc.atomic_json(p, {"completed": False, "v": 1})
            cc.atomic_json(p, {"completed": True, "v": 2})
            self.assertEqual(json.loads(p.read_text())["v"], 2)

    def test_no_partial_file_visible(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.json"
            cc.atomic_json(p, {"v": 1})
            # tmp file never remains
            self.assertEqual([f.name for f in Path(td).iterdir()],
                             ["r.json"])


if __name__ == "__main__":
    unittest.main()
