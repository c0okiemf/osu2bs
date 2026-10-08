"""Task 2 tests (feature-parity repair): order-invariant equal-player
sampling and complete coverage denominators.
  .venv/bin/python -m unittest qa.test_coverage_sample -q
"""
import unittest

from qa.coverage_sample import coverage_table, select_windows

N_T = 10


def players_fixture(counts):
    rows = []
    for pi, n in enumerate(counts):
        for wi in range(n):
            rows.append({"player_token": f"player{pi}",
                         "window_id": f"fam:x:{pi}:{wi}"})
    return rows


def obs(player, wid, inside, observed, nn, fam="fam:a", t=None):
    o = {"family": fam, "player": player, "window_id": wid,
         "inside": inside, "observed": observed, "nn_supported": nn}
    if t is not None:
        o["event_time"] = t
    return o


def outcomes_fixture(nn_missing=False):
    return [obs("p1", f"w{i}", [True] * N_T, [True] * N_T,
                not nn_missing) for i in range(10)]


class TestSelect(unittest.TestCase):
    def test_replay_order_does_not_select_a_different_population(self):
        rows = players_fixture(counts=[5000, 700, 90])
        self.assertEqual(select_windows(rows),
                         select_windows(list(reversed(rows))))

    def test_equal_player_water_filling(self):
        rows = players_fixture(counts=[5000, 700, 90])
        sel = select_windows(rows, cap=2000)
        self.assertEqual(len(sel), 2000)
        per = {}
        for w in sel:
            per[w.split(":")[2]] = per.get(w.split(":")[2], 0) + 1
        # water level: 90 + 700 + 1210 = 2000
        self.assertEqual(sorted(per.values()), [90, 700, 1210])

    def test_short_players_keep_everything(self):
        rows = players_fixture(counts=[5, 3])
        self.assertEqual(len(select_windows(rows, cap=2000)), 8)

    def test_selection_spans_the_full_replay_not_a_prefix(self):
        rows = players_fixture(counts=[1000])
        sel = select_windows(rows, cap=100)
        idxs = sorted(int(w.split(":")[3]) for w in sel)
        self.assertGreater(max(idxs), 500)     # hash order, not prefix

    def test_cap_respected(self):
        rows = players_fixture(counts=[10, 10])
        self.assertEqual(len(select_windows(rows, cap=7)), 7)


class TestCoverage(unittest.TestCase):
    def test_missing_neighbour_does_not_remove_prediction_outcome(self):
        tab = coverage_table(outcomes_fixture(nn_missing=True))
        self.assertEqual(tab["predictive_observation_count"], 10)
        self.assertEqual(tab["nn_supported_count"], 0)
        self.assertEqual(tab["families"]["fam:a"]["pooled_coverage"], 1.0)

    def test_absent_masks_do_not_shrink_into_fake_coverage(self):
        rows = [obs("p1", "w0", [False] * N_T, [False] * N_T, True)]
        tab = coverage_table(rows)
        f = tab["families"]["fam:a"]
        self.assertIsNone(f["pooled_coverage"])
        self.assertEqual(f["status"], "INSUFFICIENT")

    def test_equal_player_average(self):
        rows = ([obs("p1", f"a{i}", [True] * N_T, [True] * N_T, True)
                 for i in range(9)]
                + [obs("p2", "b0", [False] * N_T, [True] * N_T, True)])
        f = coverage_table(rows)["families"]["fam:a"]
        self.assertAlmostEqual(f["pooled_coverage"], 0.9, places=6)
        self.assertAlmostEqual(f["equal_player_coverage"], 0.5, places=6)

    def test_expected_family_with_no_observations_is_insufficient(self):
        tab = coverage_table(outcomes_fixture(),
                             expected_families=("fam:a", "fam:b"))
        self.assertEqual(tab["families"]["fam:b"]["status"], "INSUFFICIENT")

    def test_overlap_report(self):
        rows = [obs("p1", "w0", [True] * N_T, [True] * N_T, True, t=1.0),
                obs("p2", "w1", [True] * N_T, [True] * N_T, True, t=1.3),
                obs("p1", "w2", [True] * N_T, [True] * N_T, True, t=9.0)]
        f = coverage_table(rows)["families"]["fam:a"]
        self.assertEqual(f["overlapping_window_count"], 2)


if __name__ == "__main__":
    unittest.main()
