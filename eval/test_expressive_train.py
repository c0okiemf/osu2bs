"""Task 3 tests: shipped-checkpoint adaptation arms.
  .venv/bin/python -m unittest eval.test_expressive_train -q
"""
import unittest

import torch

from eval.expressive_train import (SampleStream, initialized_models,
                                   select_snapshot, teacher_kl)


class TestObjective(unittest.TestCase):
    def test_teacher_match_has_zero_kl(self):
        logits = torch.tensor([[0.2, -0.3, 0.9]])
        self.assertLess(teacher_kl(logits, logits).abs().item(), 1e-6)

    def test_kl_orientation_teacher_first(self):
        """KL(teacher || student): mass where the TEACHER puts it counts."""
        t = torch.tensor([[10.0, 0.0, 0.0]])       # teacher certain of 0
        s_bad = torch.tensor([[0.0, 10.0, 0.0]])   # student certain of 1
        s_med = torch.tensor([[0.0, 0.0, 0.0]])    # student uniform
        self.assertGreater(teacher_kl(s_bad, t).item(),
                           teacher_kl(s_med, t).item())

    def test_no_gradient_through_teacher(self):
        s = torch.zeros(1, 3, requires_grad=True)
        t = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        teacher_kl(s, t).backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)


class TestInit(unittest.TestCase):
    def _ckpt(self, tmp):
        from groom import Flow
        torch.manual_seed(7)
        m = Flow()
        p = tmp / "flow_fixture.pt"
        torch.save(m.state_dict(), p)
        return p

    def test_init_is_shipped_flow(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            p = self._ckpt(Path(td))
            student, teacher = initialized_models(p)
            self.assertTrue(all(
                torch.equal(v, teacher.state_dict()[k])
                for k, v in student.state_dict().items()))
            self.assertFalse(any(q.requires_grad
                                 for q in teacher.parameters()))
            self.assertTrue(any(q.requires_grad
                                for q in student.parameters()))


def fixture_strata():
    """approved fam a1 has ONE chart; general fam g1 has FIVE charts —
    family sampling must not favor g1's chart count."""
    return {"approved": {"a1": ["a1c0"], "a2": ["a2c0"]},
            "general": {"g1": [f"g1c{i}" for i in range(5)],
                        "g2": ["g2c0"]}}


class TestSampler(unittest.TestCase):
    def test_stratum_probability_and_family_uniformity(self):
        st = SampleStream(fixture_strata(), approved_probability=0.75,
                          seed=1)
        ids = [st.next() for _ in range(4000)]
        appr = sum(1 for s in ids if s["stratum"] == "approved") / len(ids)
        self.assertAlmostEqual(appr, 0.75, delta=0.03)
        g = [s for s in ids if s["stratum"] == "general"]
        g1 = sum(1 for s in g if s["family"] == "g1") / len(g)
        self.assertAlmostEqual(g1, 0.5, delta=0.05)   # family, not chart count

    def test_mirror_probability_half(self):
        st = SampleStream(fixture_strata(), 0.5, seed=2)
        ids = [st.next() for _ in range(2000)]
        m = sum(1 for s in ids if s["mirror"]) / len(ids)
        self.assertAlmostEqual(m, 0.5, delta=0.04)

    def test_missing_stratum_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "stratum"):
            SampleStream({"approved": {}, "general": {"g": ["c"]}}, 0.5,
                         seed=0)

    def test_resume_reproduces_next_samples(self):
        a = SampleStream(fixture_strata(), 0.5, seed=3)
        for _ in range(137):
            a.next()
        state = a.state_dict()
        want = [a.next() for _ in range(20)]
        b = SampleStream(fixture_strata(), 0.5, seed=3)
        b.load_state_dict(state)
        got = [b.next() for _ in range(20)]
        self.assertEqual(want, got)


class TestSelection(unittest.TestCase):
    def test_select_lowest_stratified_macro_ce(self):
        history = [
            {"update": 0, "val": {"approved_macro": 1.0,
                                  "general_macro": 1.0}},
            {"update": 600, "val": {"approved_macro": 0.8,
                                    "general_macro": 0.9}},
            {"update": 1800, "val": {"approved_macro": 0.85,
                                     "general_macro": 0.7}},
        ]
        sel = select_snapshot(history, min_approved_families=5,
                              approved_families_n=11)
        self.assertEqual(sel["update"], 1800)      # (0.85+0.7)/2 lowest

    def test_tie_chooses_earlier_update(self):
        history = [
            {"update": 600, "val": {"approved_macro": 0.8,
                                    "general_macro": 0.8}},
            {"update": 1800, "val": {"approved_macro": 0.8,
                                     "general_macro": 0.8}},
        ]
        sel = select_snapshot(history, 5, 11)
        self.assertEqual(sel["update"], 600)

    def test_step_zero_winner_is_no_new_checkpoint(self):
        history = [
            {"update": 0, "val": {"approved_macro": 0.5,
                                  "general_macro": 0.5}},
            {"update": 600, "val": {"approved_macro": 0.9,
                                    "general_macro": 0.9}},
        ]
        sel = select_snapshot(history, 5, 11)
        self.assertEqual(sel["update"], 0)
        self.assertTrue(sel["step_zero"])

    def test_fallback_metric_when_approved_val_thin(self):
        history = [{"update": 600, "val": {"approved_macro": None,
                                           "general_macro": 0.9,
                                           "overall_macro": 0.7}},
                   {"update": 1800, "val": {"approved_macro": None,
                                            "general_macro": 0.5,
                                            "overall_macro": 0.8}}]
        sel = select_snapshot(history, min_approved_families=5,
                              approved_families_n=3)
        self.assertEqual(sel["update"], 600)       # overall fallback, frozen
        self.assertEqual(sel["metric"], "overall_macro")


if __name__ == "__main__":
    unittest.main()
