"""Literal decoding, full-scene export and scoped QA regression checks."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from eval.joint_decode import JointState, sample_event
from eval.joint_export import assert_same, export_chart, qa_scene, read_chart
from eval.joint_phrase import UnsupportedSource


def source():
    return {"bpm": 111.984123, "duration_beats": 25,
            "notes": [(0, 0, 0, 0, 1), (0, 0, 1, 0, 1), (0, 0, 2, 0, 8),
                      (1 / 3, 1, 3, 2, 8), (8, 0, 0, 2, 0)],
            "bombs": [(1 / 3, 1, 1)],
            "walls": [(7, 3, 0, 2, 2, 3), (0, 1, 3, 1, 0, 5),
                      (10, 1, 0, 1, 0, 2)]}


class JointDecodeTests(unittest.TestCase):
    def test_state_preserves_hands_across_rests_and_copies_input(self):
        state = JointState(25)
        event = {"beat": 0, "hands": [[(0, 0, 1), (1, 0, 1)], []]}
        state.append(event)
        event["hands"][0].clear()
        state.rest(8)
        state.append({"beat": 8, "hands": [[], [(3, 2, 8)]]})
        state.rest(24)
        self.assertEqual(state.last_by_hand[0]["beat"], 0)
        self.assertEqual(state.last_by_hand[1]["beat"], 8)
        self.assertEqual(len(state.notes()), 3)
        for b in (8, 23, 25, float("nan")):
            before = state.notes()
            with self.assertRaises(UnsupportedSource):
                state.append({"beat": b, "hands": [[(0, 0, 1)], []]})
            self.assertEqual(before, state.notes())
        with self.assertRaises(UnsupportedSource):
            state.rest(23)

    def test_sample_is_deterministic_joint_and_collision_free(self):
        counts = torch.full((16,), -1000.)
        counts[15] = 1000.  # three notes on each hand
        seen = []

        def slots(hand, slot, prefix):
            seen.append((hand, slot, prefix))
            logits = torch.zeros(108)
            logits[9 * 4 + 8] = 1000  # shared desired cell; must not collide
            return logits

        a = sample_event(1 / 3, counts, slots, torch.Generator().manual_seed(3))
        b = sample_event(1 / 3, counts, slots, torch.Generator().manual_seed(3))
        self.assertEqual(a, b)
        self.assertEqual([len(ns) for ns in a["hands"]], [3, 3])
        cells = [(c, l) for ns in a["hands"] for c, l, d in ns]
        self.assertEqual(len(set(cells)), 6)
        for notes in a["hands"]:
            self.assertEqual(notes, sorted(notes))
        self.assertEqual(len(seen[3][2][0]), 3)  # other hand context visible

    def test_bad_logits_and_temperatures_fail_without_emission(self):
        def slots(*args):
            return torch.zeros(108)
        for logits in (torch.zeros(15), torch.full((16,), float("nan"))):
            with self.assertRaises(UnsupportedSource):
                sample_event(0, logits, slots, torch.Generator())
        for temp in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                sample_event(0, torch.zeros(16), slots, torch.Generator(), temp)
        with self.assertRaises(UnsupportedSource):
            sample_event(0, torch.zeros(16), lambda *a: torch.zeros(2), torch.Generator())

    def test_v3_roundtrip_and_qa_retains_every_object(self):
        src = source()
        with tempfile.TemporaryDirectory() as tmp:
            read = export_chart(src, tmp, njs=19.25, offset=-0.125)
            assert_same(src, read, 19.25, -0.125)
            scene = qa_scene(read)
            self.assertEqual(len(scene["notes"]), len(src["notes"]))
            self.assertEqual(len(scene["bombs"]), 1)
            self.assertEqual(len(scene["walls"]), 3)
            self.assertEqual({w[2] for w in scene["walls"]}, {0, 1, 2})
            from qa.certificates import contradictions
            from qa.comparator_v2 import chart_verdict
            result = contradictions(scene, {"speed_warning": {"value": 100, "sha256": "fixture"}})
            self.assertTrue(any(x["type"] == "wall_unknown" for x in result["unknowns"]))
            self.assertEqual(chart_verdict(result, {"share_supported": 1, "run_gate": True}),
                             "SCOPE_OR_EVIDENCE_UNKNOWN")

    def test_all_observed_wall_heights_survive(self):
        src = source()
        src["walls"] = [(0, 1, 0, 1, y, h) for y, h in
                        [(0, 5), (2, 3), (0, 2), (0, 1), (0, 3), (0, 4), (2, 1), (2, 2)]]
        with tempfile.TemporaryDirectory() as tmp:
            assert_same(src, export_chart(src, tmp), 18, 0)

    def test_readback_rejects_invalid_info_and_mutated_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = source()
            export_chart(src, root)
            cp, ip = root / "ExpertPlus.dat", root / "Info.dat"
            dat = json.loads(cp.read_text())
            dat["colorNotes"][0]["d"] = 7
            cp.write_text(json.dumps(dat))
            with self.assertRaisesRegex(ValueError, "roundtrip"):
                assert_same(src, read_chart(cp, ip), 18, 0)
            info = json.loads(ip.read_text())
            info["_difficultyBeatmapSets"][0]["_difficultyBeatmaps"][0]["_beatmapFilename"] = "else.dat"
            ip.write_text(json.dumps(info))
            with self.assertRaisesRegex(UnsupportedSource, "binding"):
                read_chart(cp, ip)


if __name__ == "__main__":
    unittest.main()
