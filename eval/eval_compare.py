"""Convert the held-out gems with current models and diff against the
hand-mapped originals: style metrics + rest-placement F1 on absolute time."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home()) + "/app/osu2bs")
sys.path.insert(0, str(Path(__file__).parent))
import convert
from style_profile import profile
from eval_gen import SONGS, EVAL

HUMAN = {
    "rap_god": str(Path.home()) + "/app/beat-saber-map-gen/input/bytrius/"
               "19909 (Rap God V2 - Ryger)",
    "reality_check": str(Path.home()) + "/app/beat-saber-map-gen/input/input/"
                     "25f (Reality Check Through The Skull - DM DOKURO)",
    "spaceman": str(Path.home()) + "/app/beat-saber-map-gen/input/input/"
                "24e5e (SPACEMAN - oegoe)",
}


def bpm_of(d):
    info = json.loads((Path(d) / "info.dat").read_text(encoding="utf-8-sig"))
    return info.get("_beatsPerMinute") or info["audio"]["bpm"]


def rest_set(dat_path, bpm, win=2.0):
    """Set of 2s windows (absolute audio time) holding <=1 swing, plus span."""
    d = json.loads(Path(dat_path).read_text(encoding="utf-8-sig"))
    notes = d.get("_notes") or [
        {"_time": n.get("b", 0), "_cutDirection": n.get("d", 8),
         "_type": n.get("c")} for n in d.get("colorNotes", [])]
    times = sorted(n["_time"] * 60 / bpm for n in notes
                   if n.get("_type") in (0, 1)
                   and 0 <= n.get("_cutDirection", -1) <= 8
                   and n["_cutDirection"] != 8)
    end = times[-1]
    nw = int(end / win) + 1
    counts = [0] * nw
    for t in times:
        counts[int(t / win)] += 1
    return {i for i, c in enumerate(counts) if c <= 1}, nw, len(times), end


def gen_dat(name, audio):
    """Full convert (groomed) of the eval osu, written as a plain dat on the
    ORIGINAL audio timeline (no lead shift), for comparison."""
    meta, objects, bpm, offset = convert.parse_osu(EVAL / name / "gen.osu")
    notes, walls = convert.convert_groomed(objects, bpm, offset, audio)
    beat = 60000.0 / bpm
    dat = {"_version": "2.0.0", "_obstacles": [], "_notes": [
        {"_time": round(n["t"] / beat, 5), "_lineIndex": n["col"],
         "_lineLayer": n["layer"], "_type": n["hand"],
         "_cutDirection": n["dir"]} for n in notes]}
    p = EVAL / name / "gen.dat"
    p.write_text(json.dumps(dat))
    return p, bpm


if __name__ == "__main__":
    for name, audio in SONGS.items():
        gp, gbpm = gen_dat(name, audio)
        hd = Path(HUMAN[name]) / "ExpertPlus.dat"
        hbpm = bpm_of(HUMAN[name])
        gr, gnw, gn, gend = rest_set(gp, gbpm)
        hr, hnw, hn, hend = rest_set(hd, hbpm)
        nw = min(gnw, hnw)
        g, h = {i for i in gr if i < nw}, {i for i in hr if i < nw}
        f1 = 2 * len(g & h) / max(1, len(g) + len(h))
        print(f"== {name}  (gen {gn} swings/{gend:.0f}s vs human {hn}/{hend:.0f}s)")
        print(f"  human {profile([hd])}")
        print(f"  gen   {profile([gp])}")
        print(f"  rest windows: gen {len(g)}/{nw} human {len(h)}/{nw} "
              f"placement F1 {f1:.2f}")
