"""QA packet 1 tests: BSOR parser + split assignment + note alignment.
  .venv/bin/python -m unittest eval.test_qa_replays -q
"""
import struct
import unittest

from qa.bsor import BsorError, decode_note_id, parse_bsor
from qa.replays import align_notes, assign_side, pseudonymize


def _s(x):
    b = x.encode()
    return struct.pack("<i", len(b)) + b


def synthetic_bsor(modifiers="", fail_time=0.0, speed=1.0, start=0.0,
                   n_frames=3, note_ids=(18010,), left_handed=False):
    out = struct.pack("<i", 0x442D3D69) + bytes([1, 0])
    for v in ("mod1", "1.29", "170000", "PLAYER123", "SomeName", "steam",
              "OpenVR", "Quest2", "Touch", "AB" * 20, "Song", "Mapper",
              "ExpertPlus"):
        out += _s(v)
    out += struct.pack("<i", 12345)                       # score
    out += _s("Standard") + _s("Env") + _s(modifiers)
    out += struct.pack("<f", 18.0)                        # jumpDistance
    out += bytes([1 if left_handed else 0])
    out += struct.pack("<f", 1.7)                         # height
    out += struct.pack("<fff", start, fail_time, speed)
    out += bytes([1]) + struct.pack("<i", n_frames)
    for i in range(n_frames):
        out += struct.pack("<f", i * 0.5) + struct.pack("<i", 90)
        for _ in range(3):
            out += struct.pack("<7f", 0.1 * i, 1.5, 0.0, 0, 0, 0, 1)
    out += bytes([2]) + struct.pack("<i", len(note_ids))
    for nid in note_ids:
        out += struct.pack("<i", nid) + struct.pack("<ff", 1.0, 0.6)
        out += struct.pack("<i", 2)                       # miss: no cut info
    out += bytes([3]) + struct.pack("<i", 0)
    out += bytes([4]) + struct.pack("<i", 0)
    out += bytes([5]) + struct.pack("<i", 0)
    return out


class TestParser(unittest.TestCase):
    def test_roundtrip_and_identity_separation(self):
        r = parse_bsor(synthetic_bsor())
        self.assertEqual(r["info"]["score"], 12345)
        self.assertEqual(r["identity"]["playerID"], "PLAYER123")
        self.assertNotIn("playerID", r["info"])
        self.assertNotIn("playerName", r["info"])
        self.assertEqual(len(r["frames"]), 3)
        self.assertEqual(r["notes"][0]["event_type"], 2)

    def test_note_id_decoding(self):
        d = decode_note_id(3 * 10000 + 2 * 1000 + 1 * 100 + 1 * 10 + 4)
        self.assertEqual(d, {"scoring_type": 1, "line_index": 2,
                             "line_layer": 1, "color": 1,
                             "cut_direction": 4})

    def test_bad_magic_rejected(self):
        with self.assertRaises(BsorError):
            parse_bsor(b"\x00\x00\x00\x00\x01")

    def test_truncated_rejected(self):
        with self.assertRaises(BsorError):
            parse_bsor(synthetic_bsor()[:-8])


class TestSplit(unittest.TestCase):
    def test_family_side_deterministic_and_balanced(self):
        fams = [f"fam:{i}" for i in range(40)]
        sides = {f: assign_side(f, seed="qa-v1-seal") for f in fams}
        again = {f: assign_side(f, seed="qa-v1-seal") for f in fams}
        self.assertEqual(sides, again)
        n_gen = sum(1 for s in sides.values() if s == "gen")
        self.assertGreater(n_gen, 10)
        self.assertGreater(40 - n_gen, 10)

    def test_pseudonym_stable_and_keyed(self):
        a = pseudonymize("76561198000000", key="k1")
        self.assertEqual(a, pseudonymize("76561198000000", key="k1"))
        self.assertNotEqual(a, pseudonymize("76561198000000", key="k2"))
        self.assertNotIn("7656", a)


class TestAlignment(unittest.TestCase):
    def _chart(self):
        # (time_s, line_index, line_layer, color, cut_direction)
        return [(1.0, 2, 0, 0, 1), (1.5, 1, 0, 1, 0), (2.0, 2, 0, 0, 1)]

    def _ev(self, t, li, ll, col, cd):
        return {"event_time": t, "line_index": li, "line_layer": ll,
                "color": col, "cut_direction": cd, "event_type": 0}

    def test_attribute_and_order_alignment(self):
        evs = [self._ev(1.02, 2, 0, 0, 1), self._ev(1.51, 1, 0, 1, 0),
               self._ev(2.03, 2, 0, 0, 1)]
        m = align_notes(self._chart(), evs, tol_s=0.2)
        self.assertEqual(m["aligned"], [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(m["ambiguous"], 0)

    def test_ambiguous_rejected_not_guessed(self):
        # two identical-attribute chart notes inside one tolerance window
        chart = [(1.00, 2, 0, 0, 1), (1.05, 2, 0, 0, 1)]
        evs = [self._ev(1.02, 2, 0, 0, 1)]
        m = align_notes(chart, evs, tol_s=0.2)
        self.assertEqual(m["aligned"], [])
        self.assertEqual(m["ambiguous"], 1)

    def test_unmatched_reported(self):
        m = align_notes(self._chart(), [self._ev(9.0, 3, 2, 1, 8)],
                        tol_s=0.2)
        self.assertEqual(m["aligned"], [])
        self.assertEqual(m["unmatched_events"], 1)


if __name__ == "__main__":
    unittest.main()
