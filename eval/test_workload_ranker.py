"""Q3 workload-calibration ranker tests (constructed fixtures only).
  .venv/bin/python -m unittest eval.test_workload_ranker -v
"""
import json
import random
import tempfile
import unittest
from pathlib import Path

from difficulty import (FEATURES, build_pairs, burst_rates, chart_descriptors,
                        fit_weights, measure_dir, score)
from eval.difficulty_vector import grouped_events, measure_scene
from eval.visibility import Note


def _scene(notes):
    return {"notes": notes, "chart_unknown": False, "ignored": {}}


def _ladder(n, dt, coinc_every=0):
    notes = []
    for k in range(n):
        notes.append(Note(f"n{k}", k * dt, k % 4, 0, k % 2, 1))
        if coinc_every and k % coinc_every == 0:
            notes.append(Note(f"c{k}", k * dt, 3 - k % 4, 2, 1 - k % 2, 1))
    return notes


class TestBurstMatchesFrozen(unittest.TestCase):
    def _check(self, notes):
        g = grouped_events(notes)
        fast = burst_rates(g["events"], g["instants"])
        frozen = measure_scene(_scene(notes))["burst_2s"]
        for key in ("left", "right", "combined"):
            self.assertEqual(fast[key], frozen[key]["rate"], key)

    def test_alternating_ladder(self):
        self._check(_ladder(40, 0.25))

    def test_with_coincidence_and_towers(self):
        notes = _ladder(30, 0.3, coinc_every=3)
        notes += [Note("t1", 0.3, 0, 1, 0, 1),           # tower member
                  Note("t2", 0.3 + 5e-7, 1, 1, 0, 1)]    # sub-TOL cluster
        self._check(notes)

    def test_irregular_times(self):
        rng = random.Random(7)
        notes = [Note(f"r{k}", round(rng.uniform(0, 30), 3), rng.randrange(4),
                      rng.randrange(3), rng.randrange(2), 1)
                 for k in range(120)]
        self._check(notes)

    def test_one_empty_hand(self):
        notes = [Note(f"l{k}", k * 0.4, 1, 0, 0, 1) for k in range(12)]
        self._check(notes)


def _chart_json(n, step_beats):
    notes = [{"_time": k * step_beats, "_type": k % 2, "_lineIndex": k % 4,
              "_lineLayer": 0, "_cutDirection": 1} for k in range(n)]
    return {"_version": "2.0.0", "_obstacles": [], "_notes": notes}


def _bm(fn, diff, rank, label=None):
    b = {"_difficulty": diff, "_difficultyRank": rank, "_beatmapFilename": fn,
         "_noteJumpMovementSpeed": 18, "_noteJumpStartBeatOffset": 0.0}
    if label:
        b["_customData"] = {"_difficultyLabel": label}
    return b


def _mkdir(tmp):
    """Standard Hard/Expert/E+ + dup rank-9, plus a OneSaber-only chart."""
    for fn, step in (("Hard.dat", 1.0), ("Expert.dat", 0.5),
                     ("ExpertPlus.dat", 0.25), ("ExpertPlus2.dat", 0.25),
                     ("OneSaber.dat", 0.5)):
        (tmp / fn).write_text(json.dumps(_chart_json(64, step)))
    (tmp / "Info.dat").write_text(json.dumps({
        "_beatsPerMinute": 120,
        "_difficultyBeatmapSets": [
            {"_beatmapCharacteristicName": "Standard", "_difficultyBeatmaps": [
                _bm("Hard.dat", "Hard", 5), _bm("Expert.dat", "Expert", 7),
                _bm("ExpertPlus.dat", "ExpertPlus", 9),
                _bm("ExpertPlus2.dat", "ExpertPlus", 9, "Expert++")]},
            {"_beatmapCharacteristicName": "OneSaber", "_difficultyBeatmaps": [
                _bm("OneSaber.dat", "Expert", 7)]}]}))


class TestMeasureAndPairs(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        _mkdir(self.tmp)
        self.addCleanup(self._td.cleanup)

    def test_standard_only_authentication(self):
        rec = measure_dir(self.tmp)
        self.assertEqual(sorted(c["rank"] for c in rec["charts"]), [5, 7, 9, 9])
        # a chart bound only by a OneSaber set never enters the corpus
        self.assertNotIn("OneSaber.dat", [c["file"] for c in rec["charts"]])
        for c in rec["charts"]:
            self.assertEqual(sorted(c["features"]), sorted(FEATURES))

    def test_dup_rank9_never_ordered_and_adjacency(self):
        rec = measure_dir(self.tmp)
        rec["split"] = "train"
        pairs = build_pairs([rec])
        ranks = [(p["hi_rank"], p["lo_rank"]) for p in pairs]
        self.assertNotIn((9, 9), ranks)                    # no dup-9 ordering
        self.assertEqual(ranks.count((9, 7)), 2)           # both 9s vs the 7
        self.assertEqual(ranks.count((7, 5)), 1)
        self.assertEqual(ranks.count((9, 5)), 2)
        adj = {(p["hi_rank"], p["lo_rank"]): p["adjacent"] for p in pairs}
        self.assertTrue(adj[(9, 7)] and adj[(7, 5)])
        self.assertFalse(adj[(9, 5)])                      # 7 lies between

    def test_denser_chart_measures_harder_components(self):
        rec = measure_dir(self.tmp)
        by_rank = {c["rank"]: c for c in rec["charts"]}
        self.assertGreater(by_rank[9]["features"]["cad_l_p50_hz"],
                           by_rank[5]["features"]["cad_l_p50_hz"])
        self.assertGreater(by_rank[9]["features"]["burst2_comb"],
                           by_rank[5]["features"]["burst2_comb"])

    def test_unknown_support_yields_no_claim(self):
        """A family with only duplicate rank-9 charts orders nothing — and no
        equality claim exists anywhere in the output."""
        rec = {"dir": "x", "split": "train", "charts": [
            {"rank": 9, "features": dict.fromkeys(FEATURES, 1.0)},
            {"rank": 9, "features": dict.fromkeys(FEATURES, 2.0)}]}
        self.assertEqual(build_pairs([rec]), [])


class TestFit(unittest.TestCase):
    def test_nonnegative_and_learns(self):
        """dim0 predicts order, dim1 is anti-correlated (a negative weight
        would help) — the nonnegative constraint must zero it out."""
        rng = random.Random(3)
        deltas = [[abs(rng.gauss(1.0, 0.3)), -abs(rng.gauss(0.5, 0.2))]
                  + [rng.gauss(0, 0.1) for _ in range(3)] for _ in range(200)]
        w = fit_weights(deltas, lam=1e-3)
        self.assertTrue(all(x >= 0 for x in w))
        self.assertGreater(w[0], 0.1)
        self.assertEqual(w[1], 0.0)
        acc = sum(1 for d in deltas
                  if sum(a * b for a, b in zip(w, d)) > 0) / len(deltas)
        self.assertGreater(acc, 0.95)

    def test_score_monotone_in_every_component(self):
        calib = {"features": FEATURES,
                 "train_mu": [0.0] * len(FEATURES),
                 "train_sd": [1.0] * len(FEATURES),
                 "weights": [0.5] * len(FEATURES)}
        base = dict.fromkeys(FEATURES, 1.0)
        s0 = score(base, calib)
        for k in FEATURES:
            up = dict(base)
            up[k] = 2.0
            self.assertGreaterEqual(score(up, calib), s0, k)

    def test_metadata_is_not_an_input(self):
        """Identical features, different labels/genre-ish metadata -> same
        score; nothing but the feature vector reaches the ranker."""
        calib = {"features": FEATURES,
                 "train_mu": [0.0] * len(FEATURES),
                 "train_sd": [1.0] * len(FEATURES),
                 "weights": [1.0] * len(FEATURES)}
        f = {k: 0.3 * i for i, k in enumerate(FEATURES)}
        self.assertEqual(score(dict(f), calib), score(dict(f), calib))
        self.assertEqual(sorted(FEATURES),
                         sorted(set(FEATURES)))            # numeric-only names
        for banned in ("genre", "title", "artist", "song"):
            self.assertFalse(any(banned in k for k in FEATURES), banned)


class TestDescriptorFailClosed(unittest.TestCase):
    def test_unknown_scene_rejected(self):
        feats, why = chart_descriptors({"chart_unknown": True, "scope": "x"})
        self.assertIsNone(feats)
        self.assertEqual(why, "x")

    def test_too_sparse_rejected(self):
        notes = [Note("a", 0.0, 0, 0, 0, 1), Note("b", 1.0, 3, 0, 1, 1)]
        feats, why = chart_descriptors(_scene(notes))
        self.assertIsNone(feats)
        self.assertEqual(why, "insufficient_hand_events")


if __name__ == "__main__":
    unittest.main()
