"""Quality-v1 panels: development set D and sealed confirmation set C (Q1.1).

D = the deduplicated union of the 8-song clean-rhythm panel (generation
sources, frozen hashes) and the 18 pilot-subset families using their SAVED MI
outputs (experiments/pilot-mi/fam_*/gen.osu) with provenance. Any missing or
hash-deviating artifact FAILS the build/load — membership never silently
shrinks. D contains training-family songs; it is a development set and must
never be called a held-out quality test.

C = 24 prospective-holdout families hash-selected from the corpus test split
(deterministic sha256 of family id + salt) BEFORE any model output is
inspected; unusable families advance in the same order with the exclusion
recorded. C is SEALED: development code may list its metadata but must never
decode or report per-song results until Q5 opens it once.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = Path(__file__).parent / "quality_panel.json"
CLEAN_PANEL = Path(__file__).parent / "clean_rhythm_panel.json"
MANIFEST = Path(__file__).parent / "corpus_manifest.json"
PILOT = ROOT / "experiments" / "pilot-subset.json"
MI_DIR = ROOT / "experiments" / "pilot-mi"
SALT = "quality-v1-confirm"
N_CONFIRM = 24
UNSEAL_TOKEN = "q5-open-confirm-once"


class PanelError(RuntimeError):
    pass


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _norm_song(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def build_panel():
    """Build and write quality_panel.json. Fail-closed everywhere."""
    entries = []
    cp = json.loads(CLEAN_PANEL.read_text())
    for e in cp["entries"]:
        osu, audio = ROOT / e["osu"], ROOT / e["audio"]
        for p, want, kind in ((osu, e["osu_sha256"], "osu"),
                              (audio, e["audio_sha256"], "audio")):
            if not p.exists():
                raise PanelError(f"D missing {kind} for {e['song']}: {p}")
            if _sha(p) != want:
                raise PanelError(f"D {kind} hash deviates for {e['song']}")
        entries.append({
            "id": f"panel:{e['song']}", "family": f"panel:{e['song']}",
            "song": e["song"], "source": "clean_rhythm_panel",
            "osu": e["osu"], "audio": e["audio"],
            "osu_sha256": e["osu_sha256"], "audio_sha256": e["audio_sha256"],
            "split": "panel", "cohort": e.get("cohort"),
        })
    manifest = json.loads(MANIFEST.read_text())
    by_dir = {m["dir"]: m for m in manifest["maps"]}
    pilot = json.loads(PILOT.read_text())["chosen"]
    for c in pilot:
        famdir = MI_DIR / ("fam_" + c["family"].split(":", 1)[1])
        gen, prov = famdir / "gen.osu", famdir / "provenance.json"
        if not gen.exists() or not prov.exists():
            raise PanelError(f"D missing saved MI artifacts for {c['family']}")
        rec = by_dir.get(c["dir"])
        if rec is None:
            raise PanelError(f"D pilot dir not in manifest: {c['dir']}")
        audio = Path(c["dir"]) / rec["audio_file"]
        if not audio.exists():
            raise PanelError(f"D missing audio for {c['family']}: {audio}")
        entries.append({
            "id": c["family"], "family": c["family"], "song": c["song"],
            "source": "pilot_mi",
            "osu": str(gen.relative_to(ROOT)),
            "audio": (str(audio.relative_to(ROOT))
                      if audio.is_relative_to(ROOT) else str(audio)),
            "osu_sha256": _sha(gen), "audio_sha256": _sha(audio),
            "mi_provenance_sha256": _sha(prov),
            "split": c["split"], "genre": c.get("genre", "unknown"),
            "genre_confidence": c.get("genre_confidence"),
            "genre_provenance": c.get("genre_provenance"),
        })
    fams = [e["family"] for e in entries]
    if len(set(fams)) != len(fams):
        raise PanelError("duplicate family in D")
    names = [_norm_song(e["song"]) for e in entries]
    if len(set(names)) != len(names):
        raise PanelError("duplicate song (post-normalization) in D")

    # C: deterministic hash order over distinct test families, chosen before
    # any model output; unusable families advance in order, recorded.
    fam_maps = {}
    for m in manifest["maps"]:
        if m["split"] == "test":
            fam_maps.setdefault(m["family"], []).append(m)
    order = sorted(fam_maps,
                   key=lambda f: hashlib.sha256(
                       f"{f}:{SALT}".encode()).hexdigest())
    chosen, excluded = [], []
    for f in order:
        ok = [m for m in fam_maps[f]
              if m.get("eligible") == "ok" and Path(m["dir"]).exists()
              and (Path(m["dir"]) / m["audio_file"]).exists()]
        if ok:
            # membership only: dirs are resolved from the corpus manifest by
            # family at Q5 open time (keeps committed paths out of the panel)
            chosen.append({"family": f, "n_maps": len(ok)})
            if len(chosen) == N_CONFIRM:
                break
        else:
            reason = ("not_eligible_ok"
                      if not any(m.get("eligible") == "ok" for m in fam_maps[f])
                      else "missing_dir_or_audio")
            excluded.append({"family": f, "reason": reason})
    if len(chosen) < N_CONFIRM:
        raise PanelError(f"only {len(chosen)} usable confirm families")
    payload = {"version": 1, "salt": SALT,
               "note": ("D is a DEVELOPMENT set (contains training-family "
                        "songs); C is a sealed prospective policy holdout, "
                        "opened once in Q5."),
               "dev": entries,
               "confirm": {"sealed": True, "families": chosen,
                           "excluded": excluded}}
    PANEL.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload


def load_panel(verify=True):
    """Load the frozen panel; verify=True re-hashes every D source and fails
    closed on any deviation or missing file."""
    if not PANEL.exists():
        raise PanelError("quality_panel.json missing — run build_panel()")
    p = json.loads(PANEL.read_text())
    if verify:
        for e in p["dev"]:
            osu = ROOT / e["osu"]
            audio = Path(e["audio"]) if Path(e["audio"]).is_absolute() \
                else ROOT / e["audio"]
            if not osu.exists() or not audio.exists():
                raise PanelError(f"D artifact missing for {e['id']}")
            if _sha(osu) != e["osu_sha256"] or _sha(audio) != e["audio_sha256"]:
                raise PanelError(f"D artifact hash deviates for {e['id']}")
    return p


def dev_entries(verify=True):
    return load_panel(verify)["dev"]


def confirm_families(unseal=None):
    """C membership. SEALED: raises unless called with the explicit Q5 token.
    Development commands must never decode or report C."""
    if unseal != UNSEAL_TOKEN:
        raise PanelError("C is sealed until Q5; refusing to expose membership "
                         "for decoding/reporting")
    return load_panel(verify=False)["confirm"]


if __name__ == "__main__":
    p = build_panel()
    print(f"D: {len(p['dev'])} entries; C: {len(p['confirm']['families'])} "
          f"families ({len(p['confirm']['excluded'])} excluded) -> {PANEL}")
