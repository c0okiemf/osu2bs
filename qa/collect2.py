"""QA collection packet 2 (review allocator +  quotas +
 protocol): expand telemetry to the four-way hostile split.

Quotas (families): GEN 30 total (10 existing kept), QA-TRAIN 16,
QA-CALIB 4, SEAL 6 total (2 existing unread kept). Hard caps <=600 replays
/ 5 GB INCLUDING the packet-1 collection.

Allocation: (1) bounded ELIGIBILITY-ONLY inventory (one leaderboard probe
per family — no score details, no downloads); (2) deterministic family
rank-split with EXACT quotas over usable families (sha256 of
"qa-v2-alloc:<family>"), FROZEN to allocation.json before any download or
kinematics; existing packet-1 assignments are preserved and count toward
quotas. Cross-side players keep their first side; removal reduces replay
counts, never reassigns families.

Sides and roots:
  gen      -> experiments/qa-v1/gen        (kinematics derived; generator may train)
  qa_train -> experiments/qa-v1-train      (QA-model development data)
  qa_calib -> experiments/qa-v1-calib      (threshold calibration, frozen sep.)
  seal     -> experiments/qa-v1-sealed     (sanitized+census only, UNREAD)
qa_* and seal roots are outside every generator dataset/cache path.
"""
import hashlib
import json
from pathlib import Path

from qa.bsor import PARSER_VERSION, parse_bsor
from qa.replays import (ACQ_ROOT, API, BANDS, GEN_ROOT, SEALED_ROOT, _get,
                        _save, _secret_key, align_notes, level_hash,
                        pseudonymize, swing_kinematics, _chart_notes_for)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_ROOT = ROOT / "experiments" / "qa-v1-train"
CALIB_ROOT = ROOT / "experiments" / "qa-v1-calib"
ALLOC_SEED = "qa-v2-alloc"
QUOTAS = {"gen": 30, "qa_train": 16, "qa_calib": 4, "seal": 6}
REPLAY_CAP, BYTE_CAP = 600, 5 * 10 ** 9
MIN_SCORES = 20                    # usability floor at inventory time
ROOTS = {"gen": GEN_ROOT, "qa_train": TRAIN_ROOT, "qa_calib": CALIB_ROOT,
         "qa_calib2": CALIB_ROOT,
         "qa_validate_latent": ROOT / "experiments" / "qa-v3-validate",
         "qa_validate_comparator": ROOT / "experiments" / "qa-v3-validate",
         "seal": SEALED_ROOT}
KINEMATICS = {"gen": True, "qa_train": True, "qa_calib": True,
              "qa_calib2": True, "qa_validate_latent": True,
              "qa_validate_comparator": True, "seal": False}
ALLOC_P = ROOT / "experiments" / "qa-v1" / "allocation.json"
STATE_P = ROOT / "experiments" / "qa-v1" / "collect2_state.json"


def _existing_assignments():
    """Packet-1 assignments are immutable: fetched gen fams stay gen,
    sealed fams stay sealed. Probe-error/no-leaderboard fams are excluded
    from future inventory (known unusable)."""
    p = ROOT / "experiments" / "qa-v1" / "pilot_state.json"
    s = json.loads(p.read_text())
    fixed, unusable = {}, set()
    for fam, c in s["charts"].items():
        if c.get("replays"):
            fixed[fam] = "gen" if c["side"] == "gen" else "seal"
        else:
            unusable.add(fam)
    return fixed, unusable, s


def inventory_and_allocate(max_probe=160):
    """Stage 1+2: eligibility-only inventory then frozen rank-split.
    Idempotent — an existing allocation.json is returned verbatim."""
    if ALLOC_P.exists():
        return json.loads(ALLOC_P.read_text())
    from eval import corpus
    fixed, unusable, pilot = _existing_assignments()
    m = corpus._load_validated()
    cands = sorted(
        ({"dir": r["dir"], "family": r["family"]}
         for r in m["maps"]
         if "/beatsaver/" in r["dir"] and r["eligible"] == "ok"
         and r.get("family_rep") and r["family"] not in fixed
         and r["family"] not in unusable),
        key=lambda r: hashlib.sha256(
            f"{ALLOC_SEED}:{r['family']}".encode()).hexdigest())
    need = sum(QUOTAS.values()) - len(fixed)
    usable, census = [], []
    for cand in cands:
        if len(usable) >= need + 8 or len(census) >= max_probe:
            break
        fam = cand["family"]
        try:
            h = level_hash(cand["dir"])
            lb = _get(f"{API}/leaderboards/hash/{h}")
            std = {l["difficulty"]["difficultyName"]: l
                   for l in (lb.get("leaderboards") or [])
                   if l["difficulty"]["modeName"] == "Standard"}
            if not std:
                census.append({"family": fam, "usable": False,
                               "reason": "no_standard_lb"})
                continue
            # HARDEST difficulty with an adequate leaderboard AND a locally
            # resolvable chart (the modelled tier is Expert+, and alignment
            # needs the chart) — never easiest-first (packet-2 defect fix)
            chosen = None
            for diff_name in ("ExpertPlus", "Expert", "Hard", "Normal",
                              "Easy"):
                l = std.get(diff_name)
                if l is None:
                    continue
                notes, _bpm = _chart_notes_for(cand["dir"], diff_name)
                if not notes:
                    continue
                d = _get(f"{API}/leaderboard/{l['id']}?page=1&count=50")
                n_ok = sum(1 for s in (d.get("scores") or [])
                           if not s.get("modifiers"))
                if n_ok >= MIN_SCORES:
                    chosen = (diff_name, l["id"], n_ok)
                    break
            if chosen is None:
                census.append({"family": fam, "usable": False,
                               "reason": "no_adequate_difficulty"})
                continue
            census.append({"family": fam, "usable": True,
                           "difficulty": chosen[0],
                           "n_modifier_free": chosen[2]})
            usable.append({"family": fam, "dir": cand["dir"], "hash": h,
                           "leaderboard": chosen[1],
                           "difficulty": chosen[0]})
        except Exception as e:
            census.append({"family": fam, "usable": False,
                           "reason": f"probe_error:{type(e).__name__}"})
    # exact-quota rank assignment over usable, minus fixed counts
    remaining = dict(QUOTAS)
    for side in set(fixed.values()):
        remaining[side] -= sum(1 for s in fixed.values() if s == side)
    assign = {}
    order = [s for s in ("gen", "qa_train", "qa_calib", "seal")
             for _ in range(max(0, remaining[s]))]
    for slot, rec in zip(order, usable):
        assign[rec["family"]] = {"side": slot, **rec}
    shortfall = len(order) - len(assign)
    alloc = {"version": 1, "seed": ALLOC_SEED, "quotas": QUOTAS,
             "fixed_from_pilot": fixed, "assigned": assign,
             "inventory_census": census, "shortfall": shortfall,
             "parser_version": PARSER_VERSION}
    ALLOC_P.write_text(json.dumps(alloc, indent=1, sort_keys=True))
    print(f"allocation frozen: {len(assign)} new families "
          f"(+{len(fixed)} fixed), shortfall {shortfall}, "
          f"probed {len(census)}")
    return alloc


def banded_picks(eligible):
    """Packet-2 default: two players per accuracy band."""
    n = len(eligible)
    picks, seen = [], set()
    for lo, hi in BANDS:
        band = eligible[int(lo * n):max(int(hi * n), int(lo * n) + 1)]
        got = 0
        for s in band:
            pid = s.get("playerId") or (s.get("player") or {}).get("id")
            if not pid or pid in seen or got >= 2:
                continue
            picks.append((s, pid))
            seen.add(pid)
            got += 1
    return picks


def _fetch_family(entry, side, state, key, picks_fn=banded_picks,
                  max_picks=14, max_replays=10):
    """Eligible replay fetch for one allocated family (mirrors the
    packet-1 loop; per-replay atomic; kinematics only where the side
    allows). Player selection is pluggable (banded default; the
    expansion packet passes performance-thirds)."""
    fam, lid = entry["family"], entry["leaderboard"]
    root = ROOTS[side]
    rec = {"side": side, "leaderboard": lid, "hash": entry["hash"],
           "difficulty": entry["difficulty"], "replays": [],
           "rejections": []}
    scores, page = [], 1
    while page <= 10:
        d = _get(f"{API}/leaderboard/{lid}?page={page}&count=100")
        ss = d.get("scores") or []
        scores += ss
        if len(ss) < 100:
            break
        page += 1
    eligible = [s for s in scores if not s.get("modifiers")]
    picks = picks_fn(eligible)
    total = sum(len(c.get("replays", []))
                for c in state["charts"].values()) + state["pilot_replays"]
    for s, pid in picks[:max_picks]:
        if len(rec["replays"]) >= max_replays or total + len(rec["replays"]) \
                >= REPLAY_CAP:
            break
        tok = pseudonymize(pid, key)
        owner = state["player_side"].setdefault(tok, side)
        if owner != side:
            rec["rejections"].append({"reason": "cross_side_player"})
            continue
        try:
            det = _get(f"{API}/score/{s['id']}")
        except Exception as e:
            rec["rejections"].append(
                {"reason": f"detail_error:{type(e).__name__}"})
            continue
        url = det.get("replay")
        if not url:
            rec["rejections"].append({"reason": "no_replay"})
            continue
        fname = hashlib.sha256(url.encode()).hexdigest()[:20] + ".bsor"
        raw_p = root / fam.replace(":", "_") / fname
        if raw_p.exists():
            blob = raw_p.read_bytes()
        else:
            try:
                blob = _get(url, binary=True)
            except Exception as e:
                rec["rejections"].append(
                    {"reason": f"download_error:{type(e).__name__}"})
                continue
            if state["bytes"] + len(blob) > BYTE_CAP:
                rec["rejections"].append({"reason": "byte_cap"})
                break
            raw_p.parent.mkdir(parents=True, exist_ok=True)
            tmp = raw_p.with_suffix(".tmp")
            tmp.write_bytes(blob)
            tmp.replace(raw_p)
            state["bytes"] += len(blob)
        try:
            parsed = parse_bsor(blob)
        except Exception as e:
            rec["rejections"].append(
                {"reason": f"parse_error:{type(e).__name__}"})
            continue
        info = parsed["info"]
        if info["modifiers"].strip() or info["failTime"] > 0 \
                or info["startTime"] != 0 or info["speed"] not in (0.0, 1.0) \
                or info["mode"] != "Standard":
            rec["rejections"].append({"reason": "ineligible_bsor"})
            continue
        with open(ACQ_ROOT / "acquisition.jsonl", "a") as f:
            f.write(json.dumps({"score_id": s["id"], "player": pid,
                                "token": tok, "family": fam,
                                "packet": 2}) + "\n")
        e2 = {"file": str(raw_p.relative_to(ROOT)), "player_token": tok,
              "score_id": s["id"],
              "sha256": hashlib.sha256(blob).hexdigest(),
              "height": info["height"], "left_handed": info["leftHanded"],
              "hmd": info["hmd"], "accuracy": s.get("accuracy"),
              "misses": s.get("missedNotes"), "bad_cuts": s.get("badCuts"),
              "n_frames": len(parsed["frames"]),
              "n_note_events": len(parsed["notes"]),
              "parser_version": PARSER_VERSION}
        if KINEMATICS[side]:
            chart_notes, _bpm = _chart_notes_for(entry["dir"],
                                                 entry["difficulty"])
            if chart_notes:
                al = align_notes(chart_notes, parsed["notes"], tol_s=0.2)
                e2["alignment"] = {"n_aligned": len(al["aligned"]),
                                   "ambiguous": al["ambiguous"],
                                   "unmatched": al["unmatched_events"]}
                kin = swing_kinematics(parsed, al["aligned"], chart_notes)
                kp = raw_p.with_suffix(".kinematics.json")
                kp.write_text(json.dumps({"swings": kin}))
                e2["kinematics"] = str(kp.relative_to(ROOT))
        rec["replays"].append(e2)
    rec["status"] = "done" if len(rec["replays"]) >= 4 else "thin"
    return rec


def run(max_families=10 ** 6):
    key = _secret_key()
    for p in ROOTS.values():
        p.mkdir(parents=True, exist_ok=True)
    alloc = inventory_and_allocate()
    pilot = json.loads((ROOT / "experiments" / "qa-v1" /
                        "pilot_state.json").read_text())
    if STATE_P.exists():
        state = json.loads(STATE_P.read_text())
    else:
        state = {"charts": {}, "player_side": dict(pilot["player_side"]),
                 "bytes": pilot["bytes"],
                 "pilot_replays": sum(len(c.get("replays", []))
                                      for c in pilot["charts"].values())}
    new = 0
    for fam, entry in sorted(alloc["assigned"].items()):
        if fam in state["charts"] or new >= max_families:
            continue
        rec = _fetch_family(entry, entry["side"], state, key)
        state["charts"][fam] = rec
        new += 1
        _save(STATE_P, state)
        print(f"  [{entry['side']:>8}] {fam:<12} {rec['status']:<5} "
              f"replays {len(rec['replays'])}", flush=True)
    per_side = {}
    for fam, c in state["charts"].items():
        per_side.setdefault(c["side"], [0, 0])
        per_side[c["side"]][0] += 1
        per_side[c["side"]][1] += len(c.get("replays", []))
    tot = sum(v[1] for v in per_side.values()) + state["pilot_replays"]
    print(f"collect2: {per_side}; total replays incl. pilot {tot}, "
          f"{state['bytes'] / 1e9:.2f} GB")
    return state


if __name__ == "__main__":
    run()
