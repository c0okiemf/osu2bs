"""Export the paired playtest artifacts for a doubles-cap request:

- <out>/production/  — the unmodified production decode (convert.main).
- <out>/capped/      — the ADMITTED winner of apply_doubles_cap (cap="max-1")
  for the same song, exported through the same write_map path (lead shift,
  export checks, tail), plus cap_request.json provenance.

Artifacts land in experiments/ (gitignored); shipped weights and production
defaults untouched. This is the hand-off for the review-recommended paired
playtest (capped vs production).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(song="still_waiting", out_root=None):
    import torch
    torch.set_num_threads(4)
    import convert
    from convert import (parse_osu, diff_spec, grid_steps, write_map, check,
                         LEAD_MS, MIN_LEAD_MS, TAIL_MS, NPS_CAP)
    from eval.phrase_plan import apply_doubles_cap
    panel = json.loads((ROOT / "eval/clean_rhythm_panel.json").read_text())
    e = next(x for x in panel["entries"] if x["song"] == song)
    osu, audio = ROOT / e["osu"], ROOT / e["audio"]
    out_root = Path(out_root or ROOT / "experiments" / f"cap-playtest-{song}")
    meta, objects, bpm, offset = parse_osu(osu)

    # arm A: production, end to end through the normal exporter
    rc = convert.main(osu, audio, out_root / "production", diffs="ExpertPlus")

    # arm B: the admitted capped winner, same write path
    r = apply_doubles_cap(objects, bpm, offset, str(audio), "max-1")
    assert r["feasible"], f"cap request infeasible: {r['per_candidate']}"
    spec = diff_spec("ExpertPlus")
    _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
    notes = [{"t": grid.time(s), "hand": h, "col": c, "layer": l, "dir": d}
             for s, h, c, l, d in r["notes"]]
    walls = [{"t": grid.time(s0), "dur": grid.time(s0 + ln) - grid.time(s0),
              "col": col} for s0, ln, col in r["walls"]]
    t0 = min((n["t"] for n in notes), default=0)
    shift = LEAD_MS - t0 if (t0 < MIN_LEAD_MS or t0 > LEAD_MS) else 0
    if shift:
        for x in notes + walls:
            x["t"] += shift
        walls = [w for w in walls if w["t"] + w["dur"] > 0]
        for w in walls:
            if w["t"] < 0:
                w["dur"] += w["t"]
                w["t"] = 0.0
    end = max([n["t"] for n in notes] + [w["t"] + w["dur"] for w in walls],
              default=0)
    problems = check(notes, bpm, walls, cap=NPS_CAP * spec["scale"])
    assert not problems, ("capped winner failed export checks:\n  "
                          + "\n  ".join(problems))
    write_map({"ExpertPlus": (notes, walls, spec)}, bpm, meta, audio,
              out_root / "capped", shift, end + TAIL_MS)
    (out_root / "capped" / "cap_request.json").write_text(json.dumps({
        "song": song, "cap_8beat": r["cap"], "winner": r["winner_key"],
        "admitted": r["admitted"], "attempts": r["attempts"]}, indent=1))
    print(f"playtest pair -> {out_root}/production vs {out_root}/capped "
          f"(cap={r['cap']}, winner {r['winner_key']}, rc_prod={rc})")


if __name__ == "__main__":
    main(*sys.argv[1:])
