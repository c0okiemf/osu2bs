"""E1 Task 5: blind player feedback — the first real acceptance (spec §E1).

Export B0-vs-checkpoint pairs with a deterministic hidden side order; the
arm-to-side mapping lives in a separate file the player does not open.
Ratings import validates pair identity; wins resolve against the NEW model
only after import; a B0 fallback can never be a new-checkpoint win. The
early gate: at least 4 of 6 wins, at most 1 loss, no new discomfort — ties
are not wins; no ratings = WAITING_FOR_PLAYTEST, never auto-pass.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E1 = ROOT / "experiments" / "expressive-v1" / "e1"


def blind_order(pair_id, seed):
    """Deterministic hidden side assignment per pair."""
    h = hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()[0]
    return "swapped" if h % 2 else "normal"


def resolve_wins(ratings, pairs):
    """Resolve overall {left,tie,right} against the NEW model. Convention:
    under 'normal' order the LEFT export is B0 and the RIGHT is the new
    model; 'swapped' inverts. Fallback pairs resolve to 'fallback'."""
    seen = set()
    out = []
    for r in ratings:
        rid = r.get("rating_id")
        if rid in seen:
            raise ValueError(f"duplicate rating id {rid!r}")
        seen.add(rid)
        p = pairs.get(r["pair_id"])
        if p is None:
            raise ValueError(f"unknown pair {r['pair_id']!r}")
        if p.get("fallback"):
            resolved = "fallback"
        elif r["overall"] == "tie":
            resolved = "tie"
        else:
            new_side = "right" if p["sides"] == "normal" else "left"
            resolved = "win" if r["overall"] == new_side else "loss"
        out.append({"pair_id": r["pair_id"], "resolved": resolved,
                    "discomfort": bool(r.get("discomfort"))})
    return out


def decide_early(resolved):
    """The frozen early gate. Requires SIX resolved comparisons."""
    if not resolved:
        return {"advance": False, "status": "WAITING_FOR_PLAYTEST",
                "wins": 0, "losses": 0, "ties": 0}
    wins = sum(1 for r in resolved if r["resolved"] == "win")
    losses = sum(1 for r in resolved if r["resolved"] == "loss")
    ties = sum(1 for r in resolved if r["resolved"] == "tie")
    discomfort = any(r.get("discomfort") for r in resolved)
    complete = len(resolved) >= 6
    advance = (complete and wins >= 4 and losses <= 1 and not discomfort)
    return {"advance": advance,
            "status": ("ADVANCE" if advance else
                       "INCOMPLETE" if not complete else
                       "DISCOMFORT_VETO" if discomfort else "FAIL"),
            "wins": wins, "losses": losses, "ties": ties,
            "discomfort": discomfort, "n": len(resolved)}


# ---------------- export / import ----------------

def export_pairs(arm, blind_seed=20260923):
    """Blind playable exports for one arm's six songs: <batch>/<pair>/
    {left,right}/ map dirs. The mapping file (arm_map.json) stays out of the
    listing the player opens. Fallback songs export B0 on both sides and
    are marked ineligible for wins."""
    from convert import (LEAD_MS, MIN_LEAD_MS, NPS_CAP, TAIL_MS, check,
                         diff_spec, grid_steps, parse_osu, write_map)
    from eval.expressive_manifest import load_run, partial_completed
    run = load_run(E1)
    cfg = run["config"]
    batch = E1 / f"playtest-arm{arm}"
    pairs, mapping = {}, {}
    for song in cfg["songs"]:
        sid = song["song"]
        arm_key = f"{sid}:arm{arm}"
        if not partial_completed(E1, arm_key):
            raise RuntimeError(f"{sid}: arm {arm} evaluation incomplete")
        armrec = json.loads(
            (E1 / "partial" / (f"{sid}:arm{arm}".replace("/", "_") + ".json"))
            .read_text())
        b0rec = json.loads(
            (E1 / "partial" / (f"{sid}:b0".replace("/", "_") + ".json"))
            .read_text())
        osu = ROOT / song["osu"]
        audio = Path(song["audio"])
        if not audio.is_absolute():
            audio = ROOT / audio
        meta, objects, bpm, offset = parse_osu(osu)
        if meta["Title"].startswith("Unknown"):
            meta["Title"] = sid.replace("_", " ").title()
        _s, _T, _sm, _off, grid = grid_steps(objects, bpm, offset, thin=True)
        raw0 = [tuple(int(v) for v in n) for n in b0rec["notes"]]
        walls0 = [tuple(int(v) for v in w) for w in b0rec["walls"]]
        fallback = armrec["selection"]["id"] is None
        if fallback:
            raw1, walls1 = raw0, walls0
        else:
            att = json.loads((E1 / "partial" /
                              (f"{sid}:arm{arm}:{armrec['selection']['id']}"
                               .replace("/", "_") + ".json")).read_text())
            raw1 = [tuple(int(v) for v in n) for n in att["notes"]]
            walls1 = [tuple(int(v) for v in w) for w in att["walls"]]
        pid = f"{sid}"
        order = blind_order(pid, blind_seed)
        left, right = ((raw0, walls0), (raw1, walls1)) \
            if order == "normal" else ((raw1, walls1), (raw0, walls0))
        spec = diff_spec("ExpertPlus")
        for side, (raw, walls_steps) in (("left", left), ("right", right)):
            notes = [{"t": grid.time(s), "hand": h, "col": c, "layer": l,
                      "dir": d} for s, h, c, l, d in raw]
            walls = [{"t": grid.time(s0),
                      "dur": grid.time(s0 + ln) - grid.time(s0),
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
            end = max([n["t"] for n in notes]
                      + [w["t"] + w["dur"] for w in walls], default=0)
            problems = check(notes, bpm, walls, cap=NPS_CAP * spec["scale"])
            assert not problems, f"{sid}/{side}: {problems}"
            m = dict(meta)
            m["Title"] = f"{m['Title']} [{side.upper()}]"
            write_map({"ExpertPlus": (notes, walls, spec)}, bpm, m, audio,
                      batch / pid / side, shift, end + TAIL_MS)
        pairs[pid] = {"sides": order, "fallback": fallback}
        mapping[pid] = {"sides": order, "fallback": fallback,
                        "checkpoint": armrec["checkpoint_sha256"][:12]}
    (batch / "pairs_public.json").write_text(json.dumps(
        {pid: {"song": pid} for pid in pairs}, indent=1))
    (E1 / f"arm_map-{arm}.json").write_text(json.dumps(
        {"blind_seed": blind_seed, "pairs": mapping}, indent=1))
    (batch / "RATING_TEMPLATE.json").write_text(json.dumps(
        [{"rating_id": f"r-{pid}", "pair_id": pid,
          "overall": "left|tie|right", "musical_flow": "",
          "arm_path_enjoyment": "", "doubles_enjoyment": "",
          "discomfort": False, "comment": ""} for pid in sorted(pairs)],
        indent=1))
    print(f"exported {len(pairs)} blind pairs -> {batch} "
          f"(mapping sealed in arm_map-{arm}.json; fill "
          f"RATING_TEMPLATE.json)")
    return {"batch": str(batch), "n_pairs": len(pairs)}


def import_ratings(arm, ratings_path):
    """Import the completed template, resolve against the sealed mapping,
    apply the early gate, and persist the decision."""
    ratings = json.loads(Path(ratings_path).read_text())
    mapping = json.loads((E1 / f"arm_map-{arm}.json").read_text())["pairs"]
    resolved = resolve_wins(ratings, mapping)
    decision = decide_early(resolved)
    out = {"arm": arm, "resolved": resolved, "decision": decision}
    (E1 / f"playtest_decision-{arm}.json").write_text(
        json.dumps(out, indent=1))
    print(f"arm {arm}: {decision['status']} (W{decision['wins']} "
          f"L{decision['losses']} T{decision['ties']})")
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "export":
        export_pairs(sys.argv[2])
    else:
        import_ratings(sys.argv[2], sys.argv[3])
