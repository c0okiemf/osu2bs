"""Fresh-generation confirmation of E2 (review; prereg addendum in
docs/specs/e1-machine-ab-prereg.md). One frozen 24-family packet:

  select   metadata-only fixed hash order over held-out (test-split)
           families minus every excluded set; identities frozen before any
           generation; < 24 stops the packet, no substitutions ever.
  generate MI v32 / difficulty 5.5 / seed 20260921 per family (serialized).
  run      per song: B0 production (convert_groomed, production timing),
           E2 u3600 and shipped-flow control, six seeds each on B0's
           schedule; every chart exported (notes+walls+Info) and scored as
           the exported scene; admission = comparator-v2 MACHINE_PASS and a
           passing export check; unchanged B0 selector; persisted per song.
  report   P1-P3 >= 20/24 vs both comparators, P4 guard + protections 24/24,
           frozen family-bootstrap 95% CI of the mean oriented delta > 0.
"""
import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "expressive-v1" / "fresh"
SALT = "expressive-fresh-confirm-v1"
N = 24
PASS = "MACHINE_PASS_QUALITY_NOT_EVALUATED"
NEED = 20


def _atomic(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def _hash(fam):
    return hashlib.sha256(f"{fam}:{SALT}".encode()).hexdigest()


def _excluded():
    """Every family that is QA-owned, touched by QA telemetry, or was
    previously evaluated / trained on by an expressive arm."""
    import glob
    import os
    from eval.e3_turning import qa_owned_families
    panel = json.loads((ROOT / "eval/quality_panel.json").read_text())
    qa_tel = set()
    for name in ("collect2_state.json", "pilot_state.json"):
        p = ROOT / "experiments/qa-v1" / name
        if p.exists():
            qa_tel |= set(json.loads(p.read_text())["charts"])
    return {
        "qa_owned_all43": sorted(qa_owned_families()),
        "qa_telemetry_any_side": sorted(qa_tel),
        "q5_confirm_C": sorted(e["family"]
                               for e in panel["confirm"]["families"]),
        "q5_dev_D": sorted(e["family"] for e in panel["dev"]),
        "q3_mi_generated": sorted(
            "fam:" + os.path.basename(d)[4:] for d in
            glob.glob(str(ROOT / "experiments/quality-v1/q3-mi/fam_*"))),
    }


def select():
    """Frozen once; never reshuffled."""
    sel_p = OUT / "selection.json"
    if sel_p.exists():
        return json.loads(sel_p.read_text())
    man = json.loads((ROOT / "eval/corpus_manifest.json").read_text())
    excl = _excluded()
    banned = set().union(*excl.values())
    fams = {}
    for m in man["maps"]:
        if not (m.get("family_rep") and m.get("eligible") == "ok"
                and "/beatsaver/" in m["dir"]):
            continue
        fams.setdefault(m["family"], m)
    heldout = {f: m for f, m in fams.items() if m["split"] == "test"}
    pool = sorted((f for f in heldout if f not in banned), key=_hash)
    picked, skipped = [], []
    for f in pool:
        if len(picked) == N:
            break
        d = Path(heldout[f]["dir"])
        aud = next((q for q in d.iterdir()
                    if q.suffix.lower() in (".egg", ".ogg")), None) \
            if d.exists() else None
        if aud is None:
            skipped.append({"family": f, "reason": "no_audio"})
            continue
        picked.append({"family": f, "split": "test", "genre": "unknown",
                       "dir": heldout[f]["dir"], "hash": _hash(f)})
    sel = {"salt": SALT, "n": N, "heldout_split": "test",
           "heldout_families": len(heldout), "pool_after_exclusions": len(pool),
           "exclusions": {k: len(v) for k, v in excl.items()},
           "excluded_heldout": {k: sorted(set(v) & set(heldout))
                                for k, v in excl.items()},
           "skipped": skipped, "families": picked,
           "status": "ok" if len(picked) == N else "INSUFFICIENT_INVENTORY"}
    _atomic(sel_p, sel)
    return sel


def generate():
    """MI per frozen family; a failure stops the packet (no substitution)."""
    import eval.quality_mi as qm
    sel = select()
    if sel["status"] != "ok":
        raise SystemExit(sel["status"])
    log = []
    with mock.patch.object(qm, "OUTROOT", OUT / "mi"), \
            mock.patch.object(qm, "SALT", SALT):
        for e in sel["families"]:
            status, dur = qm.run_one(e)
            log.append({"family": e["family"], "status": status,
                        "runtime_s": round(dur, 1)})
            print(f"  [{status:7s}] {e['family']} {dur:5.0f}s", flush=True)
            if status not in ("ok", "done"):
                _atomic(OUT / "mi_log.json", log)
                raise SystemExit(f"STOP: MI {status} for {e['family']}")
    _atomic(OUT / "mi_log.json", log)


# ---------------- per-song evaluation ----------------

def _export(raw, wall_runs, grid, bpm, meta, spec, d):
    """Production export of one ExpertPlus chart (lead shift, wall clip,
    final check) -> (exported scene, check problems)."""
    from convert import LEAD_MS, MIN_LEAD_MS, NPS_CAP, check, export_charts
    from qa.scene import read_scene
    notes = [{"t": grid.time(s), "hand": h, "col": c, "layer": l, "dir": dd}
             for s, h, c, l, dd in raw]
    walls = [{"t": grid.time(s0), "dur": grid.time(s0 + ln) - grid.time(s0),
              "col": c} for s0, ln, c in wall_runs]
    t0 = notes[0]["t"] if notes else 0
    shift = LEAD_MS - t0 if (t0 < MIN_LEAD_MS or t0 > LEAD_MS) else 0
    if shift:
        for x in notes + walls:
            x["t"] += shift
        walls = [w for w in walls if w["t"] + w["dur"] > 0]
        for w in walls:
            if w["t"] < 0:
                w["dur"] += w["t"]
                w["t"] = 0.0
    problems = check(notes, bpm, walls, cap=NPS_CAP * spec["scale"])
    export_charts({"ExpertPlus": (notes, walls, spec)}, bpm, meta, d)
    return read_scene(d / "ExpertPlus.dat", d / "Info.dat"), problems


def _measure(scene, problems, bank, thr, warn):
    from eval.e1_machine_ab import GUARD, PROPS, _get
    from eval.expression_profile import profile
    from qa.certificates import contradictions
    from qa.comparator_support import measure_support
    from qa.comparator_v2 import chart_verdict
    ns = scene["notes"]
    prof = profile([(c, li, ll, d) for _t, li, ll, c, d in ns],
                   [t for t, *_ in ns])
    contr = contradictions(scene, {"speed_warning": warn})
    sup = measure_support(scene, bank, thr, None)
    verdict = chart_verdict(contr, sup)
    return {"props": {k: _get(prof, p) for k, (p, _d) in PROPS.items()},
            "guard": _get(prof, GUARD[0]), "profile_status": prof["status"],
            "contradiction_status": contr["status"],
            "share_supported": sup["share_supported"], "verdict": verdict,
            "export_problems": problems, "chart_sha256": scene["chart_sha256"],
            "settings": scene["settings"], "n_walls": len(scene["walls"]),
            "admitted": verdict == PASS and not problems}


def eval_song(e, e2_model, e2_tag, bank, thr, warn):
    import torch
    torch.set_num_threads(4)
    import critic as critic_mod
    import groom
    import motion
    from convert import convert_groomed, diff_spec, grid_steps, parse_osu
    from eval.e2_context import shim_for_schedule
    from eval.expressive_eval import select_arm
    from flow_decode import EventSchedule, decode_geometry
    fam = e["family"]
    key = fam.replace(":", "_")
    done_p = OUT / "songs" / f"{key}.json"
    if done_p.exists():
        return json.loads(done_p.read_text())
    part = OUT / "partial" / key
    osu = OUT / "mi" / key / "gen.osu"
    d = Path(e["dir"])
    audio = next(q for q in d.iterdir() if q.suffix.lower() in (".egg", ".ogg"))
    meta, objects, bpm, offset = parse_osu(osu)
    timing = meta.get("_timing")
    steps, T, step_ms, off2, grid = grid_steps(objects, bpm, offset, timing,
                                               thin=True)
    afeat = groom.cached_audio_features(
        str(audio), [grid.time(s) for s in range(T)])
    spec = diff_spec("ExpertPlus")

    b0_p = part / "b0.json"
    if b0_p.exists():
        b0 = json.loads(b0_p.read_text())
    else:
        recs = []
        convert_groomed(objects, bpm, offset, audio_path=str(audio),
                        diff="ExpertPlus", replay_mode="off", collect=recs,
                        thin=True, calibrate=True, timing=timing)
        selr = next(r for r in recs if r["selected"])
        scene, probs = _export(selr["notes"], selr["walls"], grid, bpm, meta,
                               spec, part / "export" / "b0")
        b0 = {"seed": selr["seed"], "temp": selr["temp"], "ok": selr["ok"],
              "notes": selr["notes"], "walls": selr["walls"],
              "measure": _measure(scene, probs, bank, thr, warn)}
        _atomic(b0_p, b0)
    raw0 = [tuple(int(v) for v in n) for n in b0["notes"]]
    walls0 = [tuple(int(v) for v in w) for w in b0["walls"]]
    sched = EventSchedule.from_raw(raw0, walls0, grid, T, step_ms)
    p = decode_geometry(sched, steps, T, step_ms, off2, grid, afeat, spec,
                        flow_model=None, seed=b0["seed"], temp=b0["temp"],
                        checkpoint="shipped-inject")
    if p.raw != raw0 or not p.honored:
        raise RuntimeError(f"{fam}: shipped-weight injection did not replay "
                           "the B0 winner bit-exactly")

    arms = {}
    for arm in ("E2", "ctrl"):
        pool, cands = [], {}
        for sd in range(6):
            cp = part / f"{arm}.s{sd}.json"
            if cp.exists():
                c = json.loads(cp.read_text())
            else:
                fm = shim_for_schedule(e2_model, sched, step_ms) \
                    if arm == "E2" else None
                g = decode_geometry(sched, steps, T, step_ms, off2, grid,
                                    afeat, spec, flow_model=fm, seed=sd,
                                    temp=groom.TEMPS[sd % len(groom.TEMPS)],
                                    checkpoint=e2_tag if arm == "E2"
                                    else "shipped-ctrl")
                if not g.honored or g.infeasible:
                    c = {"ok": False, "reason": "not_honored"
                         if not g.honored else "infeasible"}
                else:
                    scene, probs = _export(g.raw, g.walls, grid, bpm, meta,
                                           spec, part / "export" / f"{arm}.s{sd}")
                    mrep = motion.report([(grid.time(s), h, c_, l, dd)
                                          for s, h, c_, l, dd in g.raw])
                    crit = float(critic_mod.score(g.raw, g.walls, T, afeat))
                    c = {"ok": True, "notes": g.raw, "walls": g.walls,
                         "critic": crit,
                         "select_key": crit - motion.select_cost(mrep),
                         "measure": _measure(scene, probs, bank, thr, warn)}
                _atomic(cp, c)
            cands[f"s{sd}"] = {k: v for k, v in c.items()
                               if k not in ("notes", "walls")}
            if c["ok"] and c["measure"]["admitted"]:
                pool.append({"id": f"s{sd}", "production_key":
                             (True, c["select_key"])})
        choice = select_arm(pool, selector="b0") if pool else \
            {"id": None, "status": "fallback_b0", "selector": "b0"}
        chosen = cands[choice["id"]]["measure"] if choice["id"] else None
        arms[arm] = {"selection": choice, "chosen": chosen,
                     "candidates": cands, "n_admitted": len(pool)}
    rec = {"family": fam, "b0": b0["measure"], "b0_seed": b0["seed"],
           "b0_ok": b0["ok"], "arms": arms,
           "osu_sha256": hashlib.sha256(osu.read_bytes()).hexdigest()}
    _atomic(done_p, rec)
    print(f"  {fam}: E2 {arms['E2']['n_admitted']}/6 -> "
          f"{arms['E2']['selection']['id']}  ctrl {arms['ctrl']['n_admitted']}"
          f"/6 -> {arms['ctrl']['selection']['id']}", flush=True)
    return rec


def run(shard=0, nshards=1):
    from eval.e2_context import load_selected
    from qa.certificates import load_speed_warning
    from qa.neighbours import bank_from_role
    sel = select()
    model, s = load_selected()
    thr = {"T_support": json.loads(
        (ROOT / "experiments/qa-v4/comparator-v2/evaluator_freeze.json")
        .read_text())["threshold"]}
    bank, _r = bank_from_role(identity="qa-train-v2")
    warn = load_speed_warning()
    for i, e in enumerate(sel["families"]):
        if i % nshards == shard:
            eval_song(e, model, f"e2@{s['update']}", bank, thr, warn)


# ---------------- frozen report ----------------

def _oriented(base, arm, k):
    """Improvement-oriented delta; fallback/unknown = 0 (no improvement)."""
    from eval.e1_machine_ab import PROPS
    if arm is None or arm["props"][k] is None or base["props"][k] is None:
        return 0.0
    dlt = arm["props"][k] - base["props"][k]
    return dlt if PROPS[k][1] == "up" else -dlt


def report():
    from eval.e1_machine_ab import decide
    from eval.quality_eval import _boot_ci
    sel = select()
    recs = []
    for e in sel["families"]:
        p = OUT / "songs" / (e["family"].replace(":", "_") + ".json")
        if not p.exists():
            raise SystemExit(f"INCOMPLETE: {e['family']} missing")
        recs.append(json.loads(p.read_text()))
    rep = {"prereg": "docs/specs/e1-machine-ab-prereg.md", "n": len(recs),
           "comparators": {}, "per_family": {}}
    for comp in ("vs_b0", "vs_ctrl"):
        rows, deltas = [], {k: [] for k in ("P1", "P2", "P3")}
        for r in recs:
            e2 = r["arms"]["E2"]["chosen"]
            base = r["b0"] if comp == "vs_b0" else \
                (r["arms"]["ctrl"]["chosen"] or r["b0"])
            dec = decide(base, e2)
            rows.append(dec)
            for k in deltas:
                deltas[k].append(_oriented(base, e2, k))
            rep["per_family"].setdefault(r["family"], {})[comp] = {
                **dec, **{f"d_{k}": round(deltas[k][-1], 6) for k in deltas}}
        counts = {k: sum(x[k] for x in rows)
                  for k in ("P1", "P2", "P3", "guard", "protect")}
        ci = {k: _boot_ci(v) for k, v in deltas.items()}
        mean = {k: sum(v) / len(v) for k, v in deltas.items()}
        ok = (all(counts[k] >= NEED for k in ("P1", "P2", "P3"))
              and counts["guard"] == len(recs) and counts["protect"] == len(recs)
              and all(ci[k][0] > 0 for k in ci))
        rep["comparators"][comp] = {"pass": ok, "counts": counts,
                                    "mean_oriented_delta": mean,
                                    "boot_ci95": ci}
    rep["verdict"] = ("IMPROVEMENT_CONFIRMED" if all(
        c["pass"] for c in rep["comparators"].values()) else "NOT_CONFIRMED")
    rep["fallbacks"] = {a: sum(r["arms"][a]["selection"]["id"] is None
                               for r in recs) for a in ("E2", "ctrl")}
    _atomic(OUT / "report.json", rep)
    print(rep["verdict"], json.dumps(rep["comparators"], indent=1))
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(*(int(x) for x in sys.argv[2:4]))
    else:
        {"select": select, "generate": generate, "report": report}[cmd]()
