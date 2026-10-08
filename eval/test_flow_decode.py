"""Q2 fast fixtures: EventSchedule semantics (no real decodes here; the
baseline-equivalence and rollout integration tests live in the eval driver
and the seam smoke, which decode real songs)."""
import unittest

from timing import TimeGrid

from flow_decode import EventSchedule

GRID = TimeGrid.uniform(125.0, 0.0, 400)


class TestEventSchedule(unittest.TestCase):
    RAW = [(0, 0, 1, 0, 0), (4, 0, 2, 0, 1), (4, 0, 2, 1, 8),
           (4, 1, 3, 2, 3), (8, 1, 0, 0, 2)]
    WALLS = [(10, 4, 0)]

    def test_from_raw_groups_hands_and_followers(self):
        es = EventSchedule.from_raw(self.RAW, self.WALLS, GRID, 400, 125.0)
        self.assertEqual(es.entries[4]["hands"], (0, 1))
        self.assertEqual(es.entries[4]["k"], {0: 1, 1: 0})
        self.assertEqual(es.entries[0], {"hands": (0,), "k": {0: 0}})
        self.assertEqual(es.walls, [(10, 4, 0)])

    def test_json_roundtrip_exact(self):
        es = EventSchedule.from_raw(self.RAW, self.WALLS, GRID, 400, 125.0,
                                    meta={"song": "x"})
        es2 = EventSchedule.from_json(es.to_json())
        self.assertEqual(es2.entries, es.entries)
        self.assertEqual(es2.walls, es.walls)
        self.assertEqual(es2.source_signature, es.source_signature)
        self.assertEqual(es2.to_json(), es.to_json())

    def test_equal_density_plans_are_not_interchangeable(self):
        # same note count and density, different event placement: the
        # schedules and their source signatures must differ
        raw_b = [(0, 0, 1, 0, 0), (6, 0, 2, 0, 1), (6, 0, 2, 1, 8),
                 (6, 1, 3, 2, 3), (8, 1, 0, 0, 2)]     # step 4 -> 6
        a = EventSchedule.from_raw(self.RAW, self.WALLS, GRID, 400, 125.0)
        b = EventSchedule.from_raw(raw_b, self.WALLS, GRID, 400, 125.0)
        self.assertEqual(len(self.RAW), len(raw_b))
        self.assertNotEqual(a.entries, b.entries)
        self.assertNotEqual(a.source_signature, b.source_signature)

    def test_signature_of_verifies_honoring(self):
        es = EventSchedule.from_raw(self.RAW, self.WALLS, GRID, 400, 125.0)
        self.assertTrue(es.signature_of(self.RAW, self.WALLS, GRID))
        moved = [(0, 0, 3, 2, 1)] + self.RAW[1:]       # geometry may change
        self.assertTrue(es.signature_of(moved, self.WALLS, GRID))
        dropped = self.RAW[:-1]                        # events may NOT
        self.assertFalse(es.signature_of(dropped, self.WALLS, GRID))
        rehanded = [(0, 1, 1, 0, 0)] + self.RAW[1:]
        self.assertFalse(es.signature_of(rehanded, self.WALLS, GRID))


if __name__ == "__main__":
    unittest.main()
