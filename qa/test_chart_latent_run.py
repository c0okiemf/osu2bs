"""Chart-latent Task 5 tests: development evaluation + candidate freeze.
  .venv/bin/python -m unittest qa.test_chart_latent_run -q
"""
import json
import math
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from qa import chart_latent_run as clr


class TestNLL(unittest.TestCase):
    def test_single_gaussian_matches_analytic(self):
        lw = torch.zeros(1, 1)
        mu = torch.zeros(1, 1, 10)
        ls = torch.zeros(1, 1, 10)
        y = torch.full((1, 10), 0.5)
        mask = torch.zeros(1, 10, dtype=torch.bool)
        mask[0, :2] = True
        rows, keep = clr.marginal_nll_rows(lw, mu, ls, y, mask)
        want = -(2 * (-0.5 * 0.25 - 0.5 * math.log(2 * math.pi))) / 2
        self.assertLess(abs(float(rows[0]) - want), 1e-6)
        self.assertTrue(bool(keep[0]))

    def test_equal_player_mean_weights_players_equally(self):
        rows = torch.tensor([1.0] * 9 + [11.0])
        keep = torch.ones(10, dtype=torch.bool)
        players = ["a"] * 9 + ["b"]
        self.assertAlmostEqual(
            clr.equal_player_mean(rows, keep, players), 6.0, places=6)

    def test_fully_masked_rows_are_excluded(self):
        rows = torch.tensor([1.0, 99.0])
        keep = torch.tensor([True, False])
        self.assertAlmostEqual(
            clr.equal_player_mean(rows, keep, ["a", "b"]), 1.0, places=6)


class TestSelection(unittest.TestCase):
    def test_step_zero_winner_is_not_learned(self):
        best, status = clr.select_snapshot({0: 1.0, 500: 1.2, 1000: 1.3,
                                            2000: 1.4})
        self.assertEqual(best, 0)
        self.assertEqual(status, "DEVELOPMENT_FAILED")

    def test_tie_within_1e6_prefers_earlier_update(self):
        best, status = clr.select_snapshot({0: 2.0, 500: 1.0000004,
                                            1000: 1.0, 2000: 1.5})
        self.assertEqual(best, 500)
        self.assertEqual(status, "OK")

    def test_clear_minimum_wins(self):
        best, _ = clr.select_snapshot({0: 2.0, 500: 1.5, 1000: 1.2,
                                       2000: 1.3})
        self.assertEqual(best, 1000)


class TestFreezeGuards(unittest.TestCase):
    def _freeze(self, td, dev_status, adjudication=None):
        dp = Path(td) / "dev.json"
        dp.write_text(json.dumps({"status": dev_status,
                                  "selected_snapshot": 1000,
                                  "selected_factor": 1.5,
                                  "thresholds": {}}))
        ap = Path(td) / "adj.json"
        if adjudication is not None:
            ap.write_text(json.dumps(adjudication))
        with unittest.mock.patch.object(clr, "DEV_REPORT_P", dp), \
             unittest.mock.patch.object(clr, "ADJUDICATION_P", ap), \
             unittest.mock.patch.object(clr, "CANDIDATE_P",
                                        Path(td) / "cand.json"):
            return clr.freeze_candidate()

    def test_failed_development_blocks_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,
                                        "DEVELOPMENT_FAILED"):
                self._freeze(td, "DEVELOPMENT_FAILED")

    def test_missing_adjudication_blocks_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "adjudication"):
                self._freeze(td, "DEVELOPMENT_GATES_1_3_PASSED")

    def test_failed_adjudication_blocks_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, "adjudication"):
                self._freeze(td, "DEVELOPMENT_GATES_1_3_PASSED",
                             adjudication={"status": "FAILED"})


if __name__ == "__main__":
    unittest.main()
