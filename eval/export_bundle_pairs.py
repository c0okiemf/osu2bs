"""Q5 playtest export: six frozen-rule B0-vs-bundle comparison pairs for
the final playtest (spec Q5). FROZEN RULE: the first six D panel entries
in dev order (still_waiting, numb, rather_be, FENT_TG, if_i_lose_myself,
rap_god). Both arms come from the STORED q5-composition partials (the exact
evaluated outputs), exported through the same write_map path as production
(lead shift, tail, full check).

  python -m eval.export_bundle_pairs
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "experiments" / "quality-v1" / "q5-composition"
OUT = ROOT / "experiments" / "bundle-playtest"
N_PAIRS = 6


def _export(arm_name, raw, walls_steps, grid, bpm, meta, audio, out_dir):
    from convert import (LEAD_MS, MIN_LEAD_MS, NPS_CAP, TAIL_MS, check,
                         diff_spec, write_map)
    spec = diff_spec("ExpertPlus")
    notes = [{"t": grid.time(s), "hand": h, "col": c, "layer": l, "dir": d}
             for s, h, c, l, d in raw]
    walls = [{"t": grid.time(s0), "dur": grid.time(s0 + ln) - grid.time(s0),
              "col": col} for s0, ln, col in walls_steps]
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
    assert not problems, f"{arm_name} failed export checks: {problems}"
    write_map({"ExpertPlus": (notes, walls, spec)}, bpm, meta, audio,
              out_dir, shift, end + TAIL_MS)


def main():
    from convert import parse_osu
    from eval.quality_panel import dev_entries
    entries = dev_entries()[:N_PAIRS]
    for e in entries:
        fam = e["family"]
        part = RUN / "partial" / (fam.replace(":", "_").replace("/", "_")
                                  + ".json")
        row = json.loads(part.read_text())
        osu = ROOT / e["osu"]
        audio = Path(e["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        meta, objects, bpm, offset = parse_osu(osu)
        if meta["Title"].startswith("Unknown"):
            meta["Title"] = e["song"].replace("_", " ").title()
        from convert import grid_steps
        _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
        base = OUT / e["song"]
        for arm in ("b0", "bundle"):
            raw = [tuple(int(v) for v in n) for n in row[arm]["notes"]]
            walls = [tuple(int(v) for v in w) for w in row[arm]["walls"]]
            _export(arm, raw, walls, grid, bpm, dict(meta), audio,
                    base / arm)
        (base / "pair.json").write_text(json.dumps({
            "song": e["song"], "family": fam, "source": row["source"],
            "note": "b0 = production winner; bundle = the evaluated scoped-"
                    "release output (motion+pairs claims only)"}, indent=1))
        print(f"  pair -> {base}/b0 vs {base}/bundle ({row['source']})")
    print(f"{len(entries)} playtest pairs under {OUT}")


if __name__ == "__main__":
    sys.exit(main())
