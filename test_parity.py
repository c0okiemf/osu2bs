"""Self-checks for the parity machine and converter invariants. Run: python test_parity.py"""
import random

from parity import HandParity, violations, family, FOREHAND, BACKHAND
import convert


def test_machine_alternates():
    m = HandParity()
    t, last = 0, None
    for _ in range(200):
        t += random.choice([120, 250, 400])
        d = m.next_direction(random.randint(0, 8), t)
        fam = family(d)
        if fam == "lateral":  # laterals flip parity, same as the machine's model
            last = {"fore": "back", "back": "fore", None: "fore"}[last]
        else:
            assert fam != last, "same family twice in a row"
            last = fam
    assert violations([(i * 200, 0, d) for i, d in
                       enumerate([1, 0, 1, 5, 6, 0])]) == 0


def test_violations_detects():
    assert violations([(0, 0, 1), (200, 0, 1)]) == 1        # double-down
    assert violations([(0, 0, 1), (2000, 0, 1)]) == 0       # reset after gap
    assert violations([(0, 0, 1), (200, 1, 1)]) == 0        # different hands


def test_reset_after_gap():
    m = HandParity()
    assert m.next_direction(1, 0) == 1
    assert m.next_direction(1, 5000) == 1  # gap > RESET_MS: down allowed again


def test_correction_stays_close():
    m = HandParity()
    m.next_direction(1, 0)                  # forehand down
    d = m.next_direction(7, 200)            # wants downRight; must go backhand
    assert d in BACKHAND


def test_converter_invariants():
    random.seed(1)
    t, objs = 1000, []
    for i in range(400):
        t += random.choice([100, 150, 300, 60, 60])
        objs.append({"t": t, "x": random.randint(0, 512), "y": random.randint(0, 384),
                     "kind": "circle", "new_combo": i % 8 == 0,
                     "end_t": t, "end_x": 0, "end_y": 0})
    notes = convert.convert(objs, 128.0)
    problems = convert.check(notes, 128.0)
    assert not problems, problems
    kept = {n["t"] for n in notes}
    assert kept <= {o["t"] for o in objs}, "converter invented timestamps"


def test_lean_walls():
    beat = 60000 / 120
    # col 0 free between two uses -> a wall there; all walls lean-only, no collisions
    notes = [{"t": i * beat, "col": 0 if i in (0, 30) else 1 + i % 3,
              "layer": 0, "hand": i % 2, "dir": 8} for i in range(31)]
    walls = convert.lean_walls(notes, 120)
    assert walls, "expected a lean wall in the col-0 gap"
    for w in walls:
        assert w["col"] in (0, 3) and w["dur"] > 0
        assert not any(n["col"] == w["col"] and w["t"] <= n["t"] <= w["t"] + w["dur"]
                       for n in notes)
    # every column busy everywhere -> no walls
    busy = [{"t": i * beat / 2, "col": i % 4, "layer": 0} for i in range(80)]
    assert convert.lean_walls(busy, 120) == []
    assert not convert.check(notes, 120, walls)


def test_pick_best_valid_beats_higher_scoring_invalid():
    # a higher-scoring INVALID candidate must never win over a valid one
    cands = [((False, 5.0), "bad-but-high"), ((True, -8.0), "good-but-low")]
    assert convert.pick_best(cands)[1] == "good-but-low"
    # among valid, higher score wins; among all-invalid, the winner is still
    # flagged not-ok so convert_groomed raises instead of exporting it
    valids = [((True, -8.0), "lo"), ((True, -2.0), "hi")]
    assert convert.pick_best(valids)[1] == "hi"
    allbad = [((False, 1.0), "a"), ((False, 9.0), "b")]
    assert convert.pick_best(allbad)[0][0] is False


def test_all_invalid_candidates_raise(tmp_dir="/tmp/osu2bs_test_gen"):
    # main() must raise (not silently export) when every difficulty's decode
    # fails hard validation. Stub the heavy deps; groom.pt/flow.pt exist in
    # the repo so the groomed path is taken naturally.
    import convert as c
    import os
    saved = (c.parse_osu, c.convert_groomed, c.write_map)
    c.parse_osu = lambda p: ({"Title": "T", "Artist": "A"}, [], 120.0, 0.0)
    c.convert_groomed = lambda *a, **k: (_ for _ in ()).throw(
        c.GenerationError("all gated"))
    written = []
    c.write_map = lambda *a, **k: written.append(a)
    try:
        raised = False
        try:
            c.main("x.osu", "x.egg", tmp_dir, diffs="ExpertPlus")
        except c.GenerationError:
            raised = True
        assert raised, "main must raise GenerationError when all diffs fail"
        assert not written, "must not write a map when nothing is valid"
    finally:
        c.parse_osu, c.convert_groomed, c.write_map = saved


def test_grid_piecewise_only_when_it_helps():
    # a CLEAN 2x tempo change with hits exactly on the piecewise grid: the
    # uniform grid misplaces the fast section, so piecewise is adopted and
    # snaps every hit exactly.
    from timing import TimeGrid
    timing = [(0.0, 500.0), (4000.0, 400.0)]  # 120bpm then 150bpm at 4s
    grid = TimeGrid.covering(timing, 0.0, 9000.0, 125.0)
    objs = [{"kind": "circle", "new_combo": False, "t": grid.time(s),
             "x": 200, "y": 200, "end_t": grid.time(s)}
            for s in range(grid.n)]  # every step; both segments' step > 90ms
    steps, T, step_ms, offset, g = convert.grid_steps(objs, 120.0, 0.0, timing)
    assert len(g.segments) > 1, "clean tempo change must adopt piecewise grid"
    # noisy near-equal timing on the SAME hits stays uniform (byte-identical
    # step set to no-timing)
    noisy = [(0.0, 500.0), (2000.0, 495.0), (5000.0, 510.0)]
    s_noisy, *_ , gn = convert.grid_steps(objs, 120.0, 0.0, noisy)
    s_uni, *_, gu = convert.grid_steps(objs, 120.0, 0.0, None)
    assert s_noisy == s_uni and gn.decision["grid"] == "uniform"
    # a same-BPM OFFSET reset (review follow-up 1): notes on 125ms lines then
    # a +50ms re-anchor at 4050. Uniform leaves 50ms error; piecewise adopts.
    ro = [{"kind": "circle", "new_combo": False, "t": t, "x": 200, "y": 200,
           "end_t": t} for t in [i * 125.0 for i in range(32)]
          + [4050.0 + i * 125.0 for i in range(30)]]
    _, _, _, _, g2 = convert.grid_steps(ro, 120.0, 0.0, [(0.0, 500.0), (4050.0, 500.0)])
    assert g2.decision["grid"] == "piecewise", g2.decision
    # a 12-segment noise map (rather_be-style) is capped out even though it
    # would lower snap error by following MI's own wobbly lines
    noisy_many = [(0.0, 500.0)] + [(i * 1000.0, 500.0 + (i % 3) * 120.0)
                                   for i in range(1, 13)]
    _, _, _, _, g3 = convert.grid_steps(objs, 120.0, 0.0, noisy_many)
    assert g3.decision["grid"] == "uniform", g3.decision


def test_export_bpm_roundtrip():
    # fractional BPM + long duration + lead shift: the Info BPM the player
    # reads MUST match the BPM notes were converted with, or every note
    # drifts (review follow-up 4). Round-trip a late note through export_charts.
    import tempfile, json as _json
    bpm = 174.1372  # >3 decimals: Info rounds to 174.137, so notes must
                    # convert with THAT, not the raw value, or playback drifts
    spec = convert.diff_spec("ExpertPlus")
    t_ms = 200000.0  # a note >3min in, where rounding drift would show
    with tempfile.TemporaryDirectory() as d:
        convert.export_charts({"ExpertPlus": ([{"t": t_ms, "hand": 0, "col": 1,
                            "layer": 0, "dir": 1}], [], spec)}, bpm,
                          {"Title": "T", "Artist": "A"}, d)
        info = _json.loads((__import__("pathlib").Path(d) / "Info.dat").read_text())
        dat = _json.loads((__import__("pathlib").Path(d) / "ExpertPlus.dat").read_text())
        play_bpm = info["_beatsPerMinute"]
        # the stored beat time MUST be t reconverted through the SAME BPM the
        # player reads (Info's) — exact, so it catches the mismatch at any
        # duration, not just where accumulated drift crosses a threshold
        assert dat["_notes"][0]["_time"] == round(t_ms / (60000.0 / play_bpm), 5), \
            (dat["_notes"][0]["_time"], play_bpm)


def test_slug_exotic_whitespace():
    # \xa0 in a filename once crashed hydra's override lexer (TANZNEID remix)
    import run
    assert run._clean("12. TANZNEID (Paolo Ferrara Remix\xa0)") == "12 TANZNEID Paolo Ferrara Remix"
    assert run._slug("a\xa0b (x)") == "a_b_x"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
    print("all passed")
