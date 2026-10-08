"""Acceptance for the no-op decode-plan adapter (eval/phrase_plan).

The plan captured from a baseline decode, after a JSON round-trip, must drive a
BIT-IDENTICAL decode (raw notes + walls) for every panel song under every
production seed. This proves the phrase-plan -> flow-decode data path changes
nothing about generation.

Run directly for the full 8-song x 6-seed evidence; the unittest checks a fast
subset (1 song x 2 seeds).
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "eval" / "clean_rhythm_panel.json"
SEEDS = [(s, (0.85, 1.0, 1.15)[s % 3]) for s in range(6)]  # production schedule


def _entries():
    import json
    return json.loads(PANEL.read_text())["entries"]


def check_song(entry, seeds):
    """Return list of (seed, ok) for one panel song."""
    from convert import parse_osu
    from eval.phrase_plan import roundtrip_identical
    _meta, objects, bpm, offset = parse_osu(ROOT / entry["osu"])
    audio = str(ROOT / entry["audio"])
    out = []
    for seed, temp in seeds:
        ok, _plan = roundtrip_identical(objects, bpm, offset, audio,
                                        seed=seed, temp=temp)
        out.append((seed, ok))
    return out


def convert_check_song(entry):
    """Full-production-path capture->replay equivalence for one panel song."""
    from convert import parse_osu
    from eval.phrase_plan import convert_roundtrip_identical
    _meta, objects, bpm, offset = parse_osu(ROOT / entry["osu"])
    return convert_roundtrip_identical(
        objects, bpm, offset, str(ROOT / entry["audio"]),
        diff="ExpertPlus", replay_mode="off", thin=True, calibrate=True)


def retry_fixture():
    """Force the off-band rate-retry (pass 1) on the densest panel song by
    narrowing the band far below its natural rate — the calibration offset cap
    (+-0.35) cannot reach it, so pass 1 MUST fire. Capture and replay see the
    same narrowed band, so the real retry path is exercised end to end."""
    from unittest import mock
    import convert
    from eval.phrase_plan import convert_roundtrip_identical
    entry = next(e for e in _entries() if e["song"] == "still_waiting")
    _meta, objects, bpm, offset = convert.parse_osu(ROOT / entry["osu"])
    real = convert.diff_spec

    def narrow(diff):
        spec = dict(real(diff))
        spec["band"] = (2.0, 3.0)                 # unreachable by calibration
        return spec

    with mock.patch.object(convert, "diff_spec", narrow):
        ok, detail = convert_roundtrip_identical(
            objects, bpm, offset, str(ROOT / entry["audio"]),
            diff="ExpertPlus", replay_mode="off", thin=True, calibrate=True)
    return ok, detail


class TestPhrasePlanRoundtrip(unittest.TestCase):
    def test_one_song_two_seeds_bit_identical(self):
        import torch
        torch.set_num_threads(4)                     # tiny-GEMM decode gotcha
        entry = _entries()[0]
        for seed, ok in check_song(entry, SEEDS[:2]):
            self.assertTrue(ok, f"{entry['song']} seed {seed} not bit-identical")

    def test_extend_draft_leaves_phrase_fields_untouched(self):
        from eval.phrase_plan import extend_draft
        draft = {"song": "x", "phrases": [{"section_id": "S0",
                 "musical_focus": "unknown", "motif_id": None}]}
        ext = extend_draft(draft, {"version": 1})
        self.assertEqual(ext["phrases"], draft["phrases"])   # diagnostic slots kept
        self.assertEqual(ext["decode_plan"]["version"], 1)
        self.assertNotIn("decode_plan", draft)               # input not mutated


def main():
    import sys
    import torch
    torch.set_num_threads(4)
    if "--convert-only" not in sys.argv:
        total, bad = 0, 0
        for e in _entries():
            results = check_song(e, SEEDS)
            n_ok = sum(ok for _, ok in results)
            total += len(results)
            bad += len(results) - n_ok
            flag = "OK" if n_ok == len(results) else "FAIL"
            print(f"  {e['song']:<18} {n_ok}/{len(results)} seeds bit-identical  [{flag}]")
        print(f"PANEL(groom seam): {total - bad}/{total} decodes bit-identical, "
              f"{bad} mismatches")
        assert bad == 0, "decode plan is not a no-op at the groom seam"
    # full production path: per-attempt capture -> replay through convert_groomed
    bad = 0
    for e in _entries():
        ok, d = convert_check_song(e)
        bad += not ok
        print(f"  {e['song']:<18} attempts {d['attempts_a']}=={d['attempts_b']} "
              f"retry={d['retry']} attempts_equal={d['attempts_equal']} "
              f"output_equal={d['output_equal']}  [{'OK' if ok else 'FAIL'}]")
    print(f"PANEL(convert): {8 - bad}/8 songs equivalent")
    assert bad == 0, "per-attempt plan replay is not a no-op through convert"
    ok, d = retry_fixture()
    print(f"  retry fixture: retry={d['retry']} attempts "
          f"{d['attempts_a']}=={d['attempts_b']} "
          f"attempts_equal={d['attempts_equal']} output_equal={d['output_equal']}"
          f"  [{'OK' if ok else 'FAIL'}]")
    assert d["retry"], "fixture failed to trigger the rate-retry pass"
    assert ok, "retry-pass replay is not equivalent"
    print("PASS no-op decode-plan adapter (groom seam + convert per-attempt + retry)")


if __name__ == "__main__":
    main()
