"""Seal runner guards.  .venv/bin/python -m unittest qa.test_comparator_seal -q"""
import unittest
import unittest.mock

from qa import comparator_seal as cs


class TestSealGuards(unittest.TestCase):
    def test_refuses_before_fresh_pass(self):
        for stage in ("RECIPE_FROZEN", "EVALUATOR_FROZEN", "FAILED"):
            with unittest.mock.patch.object(cs.v2, "load_state",
                                            lambda s=stage: {"stage": s}):
                with self.assertRaises(PermissionError):
                    cs.confirm()

    def test_serve_only_named_families(self):
        with cs.serve({"f:a": [{"x": 1}], "f:b": [{"x": 2}]}):
            from qa.features import records_for_role
            self.assertEqual(list(records_for_role("r", "p", {"f:a"})),
                             [{"x": 1}])


if __name__ == "__main__":
    unittest.main()
