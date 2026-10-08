"""Frozen-model dense-vs-thinned evidence comparison (Packet D, review-approved).

For each affected-genre pilot family, decode the FROZEN rhythm+flow model on
thinned vs dense (unthinned) evidence — same 6 paired seeds, equal candidate
budget, replay off, frozen models/rules/critic/ladder, grid/audio/duration/
difficulty held constant. Records not just final scheduling but the MECHANISM
(schedule-suppression trace: thin/rest/nps skips, force-keeps, threshold
crossings, mono/double demotes, calibration natural rate) for every candidate
and the selected winner. A shared-calibration probe (calibrate on vs off) runs
too, since disabling calibration alone changes rest behavior.

Scope : all 9 electronic/punk-rock families + one pop + one rap control.

  .venv/bin/python -m eval.pilot_evidence_decode
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import convert
from eval.map_reader import metrics
import motion

MI = Path(__file__).parent.parent / "experiments" / "pilot-mi"
OUT = Path(__file__).parent.parent / "experiments" / "pilot-evidence-decode"
AFFECTED = {"electronic", "punk_rock"}


def _cand_summary(objects, bpm, offset, timing, audio, thin, calibrate):
    """All candidates + selected winner for one evidence/calibration setting."""
    coll = []
    convert.convert_groomed(objects, bpm, offset, audio, "ExpertPlus",
                            replay_mode="off", collect=coll, timing=timing,
                            thin=thin, calibrate=calibrate)
    beat_ms = 60000.0 / bpm
    grid = convert.grid_steps(objects, bpm, offset, timing, thin=thin)[4]

    def one(e):
        notes = sorted((round(grid.time(s) / beat_ms, 5), h, c, l, d)
                       for s, h, c, l, d in e["notes"])
        m = metrics(notes, beat_ms)
        rep = e["motion"]
        return {"seed": e["seed"], "selected": e["selected"], "ok": e["ok"],
                "sps": m["raw_sps"], "doubles_pct": m["raw_double_pct"],
                "rest2s_pct": m["raw_rest2s_pct"], "hand_ratio": m["raw_hand_ratio"],
                "dup8_pct": m["raw_dup8_pct"],
                "narrow_lr": m["lr_outward_row_doubles"],
                "flags_per_1k": rep["flags"]["flags_per_1000"],
                "longest_run": rep["workload"]["longest_run"]["count"],
                "schedule": e["trace"].get("schedule"),
                "calibration": e["trace"].get("calibration")}
    return [one(e) for e in coll]


def _winner(cands):
    return next((c for c in cands if c["selected"]), cands[0] if cands else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fams = []
    for d in sorted(MI.iterdir()):
        prov = d / "provenance.json"
        if prov.exists():
            fams.append(json.loads(prov.read_text()))
    affected = [f for f in fams if f["genre"] in AFFECTED]
    controls = ([next(f for f in fams if f["genre"] == "pop")]
                + [next(f for f in fams if f["genre"] == "rap_hiphop")])
    panel = affected + controls
    results = []
    for f in panel:
        fam_dir = f["family"].replace(":", "_")
        cache = OUT / f"{fam_dir}.json"
        if cache.exists():
            results.append(json.loads(cache.read_text()))
            print(f"  [cached] {f['genre']:11s} {str(f['song'])[:26]}")
            continue
        meta, objects, bpm, offset = convert.parse_osu(MI / fam_dir / "gen.osu")
        audio = str(next(q for q in Path(f["corpus_dir"]).iterdir()
                         if q.suffix.lower() in (".egg", ".ogg")))
        timing = meta.get("_timing")
        row = {"family": f["family"], "genre": f["genre"], "song": f["song"],
               "control": f["genre"] not in AFFECTED}
        row["thinned"] = _cand_summary(objects, bpm, offset, timing, audio, True, True)
        row["dense"] = _cand_summary(objects, bpm, offset, timing, audio, False, True)
        # shared-calibration probe: dense with calibration OFF
        row["dense_nocalib"] = _cand_summary(objects, bpm, offset, timing, audio,
                                             False, False)
        cache.write_text(json.dumps(row, indent=1))
        results.append(row)
        wt, wd = _winner(row["thinned"]), _winner(row["dense"])
        print(f"  {f['genre']:11s} {str(f['song'])[:24]:24s} "
              f"sps {wt['sps']}->{wd['sps']} dbl {wt['doubles_pct']}->{wd['doubles_pct']} "
              f"rest {wt['rest2s_pct']}->{wd['rest2s_pct']} flags "
              f"{wt['flags_per_1k']}->{wd['flags_per_1k']} <> {wt['narrow_lr']}->{wd['narrow_lr']}")

    # aggregate winner deltas; separate affected vs control
    def agg(rows):
        keys = ["sps", "doubles_pct", "rest2s_pct", "hand_ratio",
                "flags_per_1k", "longest_run", "narrow_lr"]
        out = {}
        for k in keys:
            ds = [abs(_winner(r["dense"])[k] - _winner(r["thinned"])[k])
                  for r in rows]
            out[k] = round(sum(ds) / max(1, len(ds)), 2)
        return out
    summary = {"affected_mean_abs_winner_delta": agg([r for r in results if not r["control"]]),
               "control_mean_abs_winner_delta": agg([r for r in results if r["control"]]),
               "n_affected": sum(1 for r in results if not r["control"]),
               "n_control": sum(1 for r in results if r["control"])}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\naffected winner |dense-thinned|: {summary['affected_mean_abs_winner_delta']}")
    print(f"control  winner |dense-thinned|: {summary['control_mean_abs_winner_delta']}")
    print(f"-> {OUT}/summary.json")


if __name__ == "__main__":
    main()
