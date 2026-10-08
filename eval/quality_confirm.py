"""Q5 C CONFIRMATION: the sealed 24-family holdout,
opened ONCE. The amended progression rule: D's point estimates, safety and
runtime QUALIFIED the candidate; C keeps the original thresholds and
two-sided 95% CI requirements unchanged. Candidate AND analysis are frozen
here BEFORE the panel opens — no burden-subset selection, alternative tests
or retuning afterward. Promote only if C passes.

Stages (each idempotent/chunk-safe; MI generation is SERIALIZED — never run
it concurrently with a CPU eval, 15GB box):
  1. open  — verify token, freeze candidate+analysis, resolve C dirs.
  2. mi    — generate C MI inputs once (frozen upstream config v32/5.5/seed).
  3. eval  — the same run_q5 composition per C song, chunked.

Usage:
  python -m eval.quality_confirm open
  python -m eval.quality_confirm mi   [--n N]
  python -m eval.quality_confirm eval [--max-songs 6]
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CROOT = ROOT / "experiments" / "quality-v1" / "c-confirm"
CMI = CROOT / "mi"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def open_and_freeze():
    """One-time unseal: freeze the candidate + analysis, resolve C family
    dirs deterministically (family_rep if flagged, else first eligible-ok
    dir in manifest order). Idempotent: a second call verifies the freeze
    and refuses any drift."""
    from eval.quality_panel import UNSEAL_TOKEN, confirm_families
    from eval import corpus
    CROOT.mkdir(parents=True, exist_ok=True)
    marker = CROOT / "c_open.json"
    d_cfg = json.loads((ROOT / "experiments/quality-v1/q5-composition/"
                        "config.json").read_text())
    families = confirm_families(unseal=UNSEAL_TOKEN)["families"]
    m = corpus._load_validated()
    by_fam = {}
    for r in m["maps"]:
        if r["split"] == "test" and r.get("eligible") == "ok":
            by_fam.setdefault(r["family"], []).append(r)
    entries = []
    for f in families:
        maps = by_fam.get(f["family"])
        if not maps:
            entries.append({"family": f["family"], "status": "unresolvable"})
            continue
        rec = next((x for x in maps if x.get("family_rep")), maps[0])
        entries.append({"family": f["family"], "status": "ok",
                        "dir": rec["dir"], "audio_file": rec["audio_file"]})
    freeze = {
        "version": 1,
        "opened_utc": None,                       # set on first write only
        "authorized_by": "review  (amended progression rule)",
        "candidate": d_cfg,                       # exact D-composition shas
        "analysis": {
            "gates": "identical to release_contract d_composition gates",
            "contract_sha256": d_cfg["contract_sha256"],
            "bootstrap": {"n": 10000, "seed": 20260921,
                          "two_sided_ci": 0.95,
                          "criterion": "all three positive-delta CIs "
                                       "exclude zero"},
            "no_posthoc": "no burden-subset selection, alternative tests "
                          "or retuning after open"},
        "mi_upstream": {"config": "v32", "difficulty": 5.5,
                        "seed": 20260921},
        "entries": entries,
    }
    if marker.exists():
        prior = json.loads(marker.read_text())
        cmp_prior = {k: v for k, v in prior.items() if k != "opened_utc"}
        cmp_new = {k: v for k, v in freeze.items() if k != "opened_utc"}
        if cmp_prior != cmp_new:
            raise RuntimeError("C already opened with a DIFFERENT frozen "
                               "candidate/analysis — refusing")
        print(f"C already open ({prior['opened_utc']}); freeze verified")
        return prior
    freeze["opened_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
    marker.write_text(json.dumps(freeze, indent=1, sort_keys=True))
    ok = sum(1 for e in entries if e["status"] == "ok")
    print(f"C OPENED: {ok}/{len(entries)} families resolved; freeze written")
    return freeze


def generate_mi(limit=10 ** 6):
    """C MI inputs, once, frozen upstream config. NEVER run concurrently
    with a convert/eval pipeline (15GB box)."""
    import eval.quality_mi as qmi
    freeze = json.loads((CROOT / "c_open.json").read_text())
    CMI.mkdir(parents=True, exist_ok=True)
    # reuse run_one by pointing its OUTROOT at the C mi dir
    qmi.OUTROOT = CMI
    n = 0
    for e in freeze["entries"]:
        if e["status"] != "ok":
            continue
        if (CMI / e["family"].replace(":", "_") / "gen.osu").exists():
            continue
        if n >= limit:
            break
        status, dur = qmi.run_one({"family": e["family"], "split": "test",
                                   "genre": "unknown", "dir": e["dir"]})
        print(f"  [{status:7s}] {e['family']:16s} {dur:5.0f}s", flush=True)
        n += 1
    done = sum(1 for e in freeze["entries"] if e["status"] == "ok"
               and (CMI / e["family"].replace(":", "_") / "gen.osu").exists())
    print(f"C MI: {done} families ready")
    return done


def run_eval(max_songs=6):
    from eval.quality_eval import run_q5
    freeze = json.loads((CROOT / "c_open.json").read_text())
    entries = []
    for e in freeze["entries"]:
        if e["status"] != "ok":
            continue
        gen = CMI / e["family"].replace(":", "_") / "gen.osu"
        if not gen.exists():
            raise RuntimeError(f"C MI missing for {e['family']} — run "
                               "`python -m eval.quality_confirm mi` first")
        entries.append({"family": e["family"],
                        "song": e["family"],
                        "osu": str(gen),
                        "audio": str(Path(e["dir"]) / e["audio_file"])})
    return run_q5(CROOT / "eval", panel="confirm", max_songs=max_songs,
                  entries=entries, confirm=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("open")
    mi = sub.add_parser("mi")
    mi.add_argument("--n", type=int, default=10 ** 6)
    ev = sub.add_parser("eval")
    ev.add_argument("--max-songs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "open":
        open_and_freeze()
    elif a.cmd == "mi":
        generate_mi(a.n)
    else:
        run_eval(a.max_songs)


if __name__ == "__main__":
    main()
