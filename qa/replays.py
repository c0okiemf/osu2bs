"""QA packet 1: BeatLeader replay telemetry pilot .

FROZEN SCOPE: 12 corpus charts x up to 10 distinct players = 120 replays,
5 GB cap, percentile-BANDED sampling (never top-10-only), <=1 req/s with
Retry-After honored, atomic per-replay progress.

GENERATOR/QA SPLIT AT INGESTION (the "almost hostile systems" boundary):
each chart FAMILY is deterministically assigned gen|qa by the PUBLIC
assignment seed; families are song-disjoint by corpus family ids and
players are made disjoint (a cross-side player keeps the side of their
first-fetched replay; other-side records are dropped and reported).
- GEN half  -> experiments/qa-v1/gen/      (swing kinematics derived now)
- QA half   -> experiments/qa-v1-sealed/   (OUTSIDE generator dataset roots;
  sanitized raw storage + integrity census ONLY — no descriptor work until
  the independent evaluator is frozen; this is a CONFIRMATION set, not QA
  training data)

PSEUDONYMIZATION: durable records carry only hmac-style keyed tokens
(secret key in experiments/qa-v1-acquisition/key.txt, generated once);
playerID/name/platform live only in the restricted acquisition log.
Eligibility: replay-bearing, modifier-free, completed (failTime<=0),
non-practice (startTime==0, speed in {0,1}), Standard, exact map hash.
Misses/bad cuts within completed runs are retained as labeled outcomes.
"""
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from qa.bsor import PARSER_VERSION, parse_bsor

ROOT = Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "experiments" / "qa-v1" / "gen"
SEALED_ROOT = ROOT / "experiments" / "qa-v1-sealed"
ACQ_ROOT = ROOT / "experiments" / "qa-v1-acquisition"
API = "https://api.beatleader.xyz"
ASSIGN_SEED = "qa-v1-seal"            # PUBLIC assignment seed (not a secret)
N_CHARTS, N_PLAYERS = 12, 10
BYTE_CAP = 5 * 10 ** 9
BANDS = ((0.0, 0.10), (0.10, 0.30), (0.30, 0.60), (0.60, 0.90),
         (0.90, 1.0))                  # 2 players per band
NORMALIZATION_VERSION = 1

_last_req = [0.0]


def _get(url, binary=False):
    wait = 1.0 - (time.time() - _last_req[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent":
                                               "osu2bs-qa-pilot/1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                _last_req[0] = time.time()
                data = r.read()
                return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(float(e.headers.get("Retry-After", 5)))
                continue
            raise
    raise RuntimeError(f"gave up on {url}")


def assign_side(family, seed=ASSIGN_SEED):
    h = hashlib.sha256(f"{seed}:{family}".encode()).digest()[0]
    return "gen" if h % 2 == 0 else "qa"


def _secret_key():
    ACQ_ROOT.mkdir(parents=True, exist_ok=True)
    kp = ACQ_ROOT / "key.txt"
    if not kp.exists():
        kp.write_text(hashlib.sha256(os.urandom(32)).hexdigest())
        kp.chmod(0o600)
    return kp.read_text().strip()


def pseudonymize(player_id, key):
    return hashlib.sha256(f"{key}:{player_id}".encode()).hexdigest()[:16]


def level_hash(map_dir):
    d = Path(map_dir)
    info_p = next(p for p in d.iterdir() if p.name.lower() == "info.dat")
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    h = hashlib.sha1(info_p.read_bytes())
    for s in info["_difficultyBeatmapSets"]:
        for b in s["_difficultyBeatmaps"]:
            f = d / b["_beatmapFilename"]
            if not f.exists():
                raise FileNotFoundError(f)
            h.update(f.read_bytes())
    return h.hexdigest().upper()


def align_notes(chart_notes, events, tol_s=0.2):
    """Attribute + timing + order alignment; noteID attributes are NOT a
    unique index, so any ambiguity is rejected, never guessed.
    chart_notes: [(time_s, line_index, line_layer, color, cut_direction)].
    events: replay note events with decoded attribute fields."""
    used = set()
    aligned, ambiguous, unmatched = [], 0, 0
    for ei, ev in enumerate(events):
        cand = [ci for ci, (t, li, ll, col, cd) in enumerate(chart_notes)
                if ci not in used and li == ev["line_index"]
                and ll == ev["line_layer"] and col == ev["color"]
                and cd == ev["cut_direction"]
                and abs(t - ev["event_time"]) <= tol_s]
        if len(cand) == 1:
            aligned.append((cand[0], ei))
            used.add(cand[0])
        elif len(cand) > 1:
            ambiguous += 1
        else:
            unmatched += 1
    return {"aligned": aligned, "ambiguous": ambiguous,
            "unmatched_events": unmatched}


def swing_kinematics(parsed, aligned, chart_notes):
    """GEN-half only: per aligned note, cut telemetry + a +-250 ms frame
    window summary (controller path length, peak speed, head sway). Head/
    controller trajectories — never claimed elbow/shoulder biomechanics."""
    frames = parsed["frames"]
    times = [f[0] for f in frames]
    import bisect
    out = []
    for ci, ei in aligned:
        ev = parsed["notes"][ei]
        t = ev["event_time"]
        lo = bisect.bisect_left(times, t - 0.25)
        hi = bisect.bisect_right(times, t + 0.25)
        win = frames[lo:hi]
        # frame tuple: (t, fps, head(pos,rot), left(pos,rot), right(pos,rot))
        idx = 4 if (ev.get("cut") or {}).get("saberType", 1) == 1 else 3
        path, peak = 0.0, 0.0
        sway = 0.0
        for a, b in zip(win, win[1:]):
            dt = max(b[0] - a[0], 1e-6)
            pa, pb = a[idx][0], b[idx][0]
            d = sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
            path += d
            peak = max(peak, d / dt)
            ha, hb = a[2][0], b[2][0]
            sway += sum((x - y) ** 2 for x, y in zip(ha, hb)) ** 0.5
        rec = {"chart_index": ci, "event_time": t,
               "event_type": ev["event_type"],
               "window_frames": len(win), "path_m": round(path, 4),
               "peak_speed_mps": round(peak, 3),
               "head_sway_m": round(sway, 4)}
        if ev.get("cut"):
            c = ev["cut"]
            rec["cut"] = {k: c[k] for k in
                          ("saberSpeed", "timeDeviation", "cutDirDeviation",
                           "cutDistanceToCenter", "cutAngle",
                           "beforeCutRating", "afterCutRating", "speedOK",
                           "directionOK", "wasCutTooSoon")}
        out.append(rec)
    return out


def _chart_notes_for(map_dir, difficulty):
    """Chart notes in replay-attribute terms from the local .dat."""
    d = Path(map_dir)
    info_p = next(p for p in d.iterdir() if p.name.lower() == "info.dat")
    info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    bpm = float(info["_beatsPerMinute"])
    for s in info["_difficultyBeatmapSets"]:
        if s.get("_beatmapCharacteristicName") != "Standard":
            continue
        for b in s.get("_difficultyBeatmaps", []):
            if b.get("_difficulty") == difficulty:
                raw = json.loads((d / b["_beatmapFilename"])
                                 .read_text(encoding="utf-8-sig"))
                notes = []
                for n in raw.get("_notes", []):
                    if n.get("_type") in (0, 1):
                        notes.append((n["_time"] * 60.0 / bpm,
                                      n["_lineIndex"], n["_lineLayer"],
                                      n["_type"], n["_cutDirection"]))
                return sorted(notes), bpm
    return None, None


def _sanitize(parsed, token):
    """Durable record: identity replaced by the keyed token; raw identity
    NEVER stored outside the restricted acquisition log."""
    out = dict(parsed)
    out.pop("identity", None)
    out["player_token"] = token
    return out


def run_pilot(max_new=10 ** 6):
    """Chunk-safe pilot: probe corpus beatsaver charts in deterministic
    hash order; each family lands on its frozen side; fetch banded eligible
    replays until 12 charts / caps. Resume-safe; every replay atomic."""
    from eval import corpus
    key = _secret_key()
    for p in (GEN_ROOT, SEALED_ROOT, ACQ_ROOT):
        p.mkdir(parents=True, exist_ok=True)
    m = corpus._load_validated()
    cands = sorted(
        ({"dir": r["dir"], "family": r["family"]}
         for r in m["maps"]
         if "/beatsaver/" in r["dir"] and r["eligible"] == "ok"
         and r.get("family_rep")),
        key=lambda r: hashlib.sha256(
            f"{ASSIGN_SEED}:{r['family']}".encode()).hexdigest())
    state_p = ROOT / "experiments" / "qa-v1" / "pilot_state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else \
        {"charts": {}, "player_side": {}, "bytes": 0, "probed": []}
    n_done = sum(1 for c in state["charts"].values()
                 if c.get("status") == "done")
    new = 0
    for cand in cands:
        if n_done >= N_CHARTS or new >= max_new \
                or state["bytes"] >= BYTE_CAP:
            break
        fam = cand["family"]
        if fam in state["probed"]:
            continue
        side = assign_side(fam)
        root = GEN_ROOT if side == "gen" else SEALED_ROOT
        try:
            h = level_hash(cand["dir"])
            lb = _get(f"{API}/leaderboards/hash/{h}")
        except Exception as e:
            state["probed"].append(fam)
            state["charts"][fam] = {"status": f"probe_error:{type(e).__name__}"}
            _save(state_p, state)
            continue
        std = [l for l in (lb.get("leaderboards") or [])
               if l["difficulty"]["modeName"] == "Standard"]
        if not std:
            state["probed"].append(fam)
            state["charts"][fam] = {"status": "no_standard_leaderboard"}
            _save(state_p, state)
            continue
        best = std[0]
        lid = best["id"]
        difficulty = best["difficulty"]["difficultyName"]
        # page the full score list (count<=100/page) to band it
        scores = []
        page = 1
        while page <= 10:
            d = _get(f"{API}/leaderboard/{lid}?page={page}&count=100")
            ss = d.get("scores") or []
            scores += ss
            if len(ss) < 100:
                break
            page += 1
        eligible = [s for s in scores if not s.get("modifiers")]
        n = len(eligible)
        picks = []
        seen_players = set()
        for lo, hi in BANDS:
            band = eligible[int(lo * n):max(int(hi * n), int(lo * n) + 1)]
            got = 0
            for s in band:
                pid = s.get("playerId") or (s.get("player") or {}).get("id")
                if not pid or pid in seen_players or got >= 2:
                    continue
                picks.append((s, pid))
                seen_players.add(pid)
                got += 1
        kept, dropped_cross = [], 0
        rec = {"side": side, "leaderboard": lid, "difficulty": difficulty,
               "hash": h, "n_scores": len(scores),
               "replays": [], "rejections": []}
        for s, pid in picks[:N_PLAYERS + 4]:
            if len(kept) >= N_PLAYERS:
                break
            tok = pseudonymize(pid, key)
            owner = state["player_side"].setdefault(tok, side)
            if owner != side:
                dropped_cross += 1
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
                    or info["startTime"] != 0 \
                    or info["speed"] not in (0.0, 1.0) \
                    or info["mode"] != "Standard" \
                    or info["hash"].upper()[:40] != h[:40]:
                rec["rejections"].append({"reason": "ineligible_bsor",
                                          "modifiers": info["modifiers"],
                                          "fail": info["failTime"] > 0})
                continue
            with open(ACQ_ROOT / "acquisition.jsonl", "a") as f:
                f.write(json.dumps({"score_id": s["id"], "player": pid,
                                    "token": tok, "family": fam}) + "\n")
            entry = {"file": str(raw_p.relative_to(ROOT)),
                     "player_token": tok, "score_id": s["id"],
                     "sha256": hashlib.sha256(blob).hexdigest(),
                     "height": info["height"],
                     "left_handed": info["leftHanded"],
                     "hmd": info["hmd"], "accuracy": s.get("accuracy"),
                     "misses": s.get("missedNotes"),
                     "bad_cuts": s.get("badCuts"),
                     "n_frames": len(parsed["frames"]),
                     "n_note_events": len(parsed["notes"]),
                     "parser_version": PARSER_VERSION}
            if side == "gen":
                chart_notes, _bpm = _chart_notes_for(cand["dir"], difficulty)
                if chart_notes:
                    al = align_notes(chart_notes,
                                     parsed["notes"], tol_s=0.2)
                    entry["alignment"] = {
                        "n_aligned": len(al["aligned"]),
                        "ambiguous": al["ambiguous"],
                        "unmatched": al["unmatched_events"]}
                    kin = swing_kinematics(parsed, al["aligned"],
                                           chart_notes)
                    kp = raw_p.with_suffix(".kinematics.json")
                    kp.write_text(json.dumps(
                        {"normalization_version": NORMALIZATION_VERSION,
                         "swings": kin}))
                    entry["kinematics"] = str(kp.relative_to(ROOT))
            kept.append(entry)
        rec["replays"] = kept
        rec["dropped_cross_side"] = dropped_cross
        rec["status"] = "done" if len(kept) >= 4 else "thin"
        state["charts"][fam] = rec
        state["probed"].append(fam)
        if rec["status"] == "done":
            n_done += 1
        new += 1
        _save(state_p, state)
        print(f"  [{side:>3}] {fam:<12} {rec['status']:<5} "
              f"replays {len(kept)} (cross-dropped {dropped_cross})",
              flush=True)
    total = sum(len(c.get("replays", [])) for c in state["charts"].values())
    print(f"pilot: {n_done}/{N_CHARTS} charts done, {total} replays, "
          f"{state['bytes'] / 1e9:.2f} GB")
    return state


def _save(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(path)


if __name__ == "__main__":
    run_pilot()
