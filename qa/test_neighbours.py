"""Task 5 tests: raw reference retrieval.
  .venv/bin/python -m unittest qa.test_neighbours -q
"""
import unittest

import torch

from qa.neighbours import Bank, build_bank, descriptor, retrieve


def rows(n=40, fams=4, players=6, base=0.0):
    out = []
    for i in range(n):
        out.append({"role": "qa_train",
                    "family": f"fam:{i % fams}",
                    "player": f"p{i % players}",
                    "desc": torch.tensor([base + i * 0.1, 0.0, 1.0]),
                    "targets": torch.tensor([0.5] * 3),
                    "traj_ref": f"w{i}",
                    "content_hash": f"{i:04x}"})
    return out


class TestBank(unittest.TestCase):
    def test_calibration_never_enters_bank(self):
        bad = rows()
        for r in bad:
            r["role"] = "qa_calib"
        with self.assertRaises(PermissionError):
            build_bank(bad, "id1")

    def test_bank_standardizes_on_train(self):
        b = build_bank(rows(), "id1")
        self.assertEqual(b.identity, "id1")
        self.assertEqual(b.desc.shape[0], 40)


class TestRetrieve(unittest.TestCase):
    def test_self_neighbour_cannot_certify_control(self):
        b = build_bank(rows(), "id1")
        ns = retrieve(b, torch.tensor([1.2, 0.0, 1.0]),
                      exclude={"families": {"fam:0"}, "players": {"p0"}})
        self.assertEqual(len(ns["neighbours"]), 5)
        for n in ns["neighbours"]:
            self.assertNotEqual(n["family"], "fam:0")
            self.assertNotEqual(n["player"], "p0")

    def test_diversity_constraints(self):
        b = build_bank(rows(), "id1")
        ns = retrieve(b, torch.tensor([0.0, 0.0, 1.0]), exclude=None)
        fams = {n["family"] for n in ns["neighbours"]}
        players = {n["player"] for n in ns["neighbours"]}
        self.assertGreaterEqual(len(fams), 3)
        self.assertGreaterEqual(len(players), 3)
        self.assertIsNotNone(ns["support_distance"])

    def test_tie_break_by_content_hash(self):
        rs = rows(10, fams=5, players=5)
        for r in rs:
            r["desc"] = torch.tensor([1.0, 0.0, 1.0])   # all equidistant
        b = build_bank(rs, "id1")
        a = retrieve(b, torch.tensor([1.0, 0.0, 1.0]), exclude=None)
        c = retrieve(b, torch.tensor([1.0, 0.0, 1.0]), exclude=None)
        self.assertEqual([n["traj_ref"] for n in a["neighbours"]],
                         [n["traj_ref"] for n in c["neighbours"]])

    def test_insufficient_support_is_unknown(self):
        rs = rows(6, fams=2, players=2)
        b = build_bank(rs, "id1")
        ns = retrieve(b, torch.tensor([0.0, 0.0, 1.0]),
                      exclude={"families": {"fam:0"}, "players": set()})
        self.assertEqual(ns["status"], "insufficient_support")


class TestDescriptor(unittest.TestCase):
    def test_descriptor_deterministic(self):
        scene = {"scope": None, "notes": [(0.5 * i, i % 4, i % 3, i % 2, 1)
                                         for i in range(10)],
                 "bombs": [], "walls": [],
                 "settings": {"njs": 18, "offset_beats": 0}}
        a = descriptor(scene, 4)
        b = descriptor(scene, 4)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.isfinite(a).all())


if __name__ == "__main__":
    unittest.main()


class TestPrefilterExactness(unittest.TestCase):
    def test_prefilter_matches_full_sort(self):
        import random
        import unittest.mock
        from qa import neighbours as nb
        rng = random.Random(7)
        for trial in range(200):
            n = rng.randint(20, 120)
            recs = [{"role": "qa_train", "family": f"f{rng.randint(0, 4)}",
                     "player": f"p{rng.randint(0, 6)}",
                     "desc": torch.tensor([float(rng.randint(0, 3)),
                                           float(rng.randint(0, 3))]),
                     "targets": torch.zeros(2), "traj_ref": f"r{i}",
                     "content_hash": f"{rng.getrandbits(32):08x}"}
                    for i in range(n)]
            bank = nb.build_bank(recs, "t")
            q = torch.tensor([float(rng.randint(0, 3)),
                              float(rng.randint(0, 3))])
            excl = {"families": {f"f{rng.randint(0, 4)}"},
                    "players": {f"p{rng.randint(0, 6)}"}} \
                if trial % 2 else None
            with unittest.mock.patch.object(nb, "PREFILTER_K", 10 ** 9):
                full = nb.retrieve(bank, q, exclude=excl)
            bank.hash_arr = None
            with unittest.mock.patch.object(nb, "PREFILTER_K",
                                            rng.randint(1, 12)):
                fast = nb.retrieve(bank, q, exclude=excl)
            self.assertEqual(full, fast, f"trial {trial}")
