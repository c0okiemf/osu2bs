"""Feature-parity repair Task 1 (review spec 2026-09-24 §1): ONE shared
evidence provider for training, calibration, diagnostics and serving.

Historical defect this replaces: qa.train.assemble_role passed real audio
evidence to map_features while qa.calibrate and qa.diagnose passed
audio=None — a train/serve skew that zeroed 24 audio channels against a
scaler fitted on the training distribution. All callers now consume
records_for_role + family_evidence + features_for. Evidence is explicit:
it names its family and audio policy and can never silently become None.
Eligibility (supported + outcome=="good") is decided HERE, once.

reconcile() proves row-for-row equality between this provider and the
actual cached matrices/scalers the frozen models were fitted on, and
persists the window-identity mapping; a mismatch is
FEATURE_PROVENANCE_UNRESOLVED and stops dependent work.
"""
import gzip
import hashlib
import json
from pathlib import Path

import torch

from qa.audio import AUDIO_EVIDENCE_VERSION, audio_cache_key, evidence
from qa.contract import FROZEN_ONLY, ROLE_PURPOSES
from qa.model import FEATURE_NAMES, TARGET_NAMES, map_features
from qa.telemetry_v2 import TELEMETRY_VERSION, _family_chart

ROOT = Path(__file__).resolve().parent.parent
AUDIO_CACHE = ROOT / "experiments" / "qa-v1" / "audio-cache"
PARITY_DIR = ROOT / "experiments" / "qa-v2" / "feature-parity"


def feature_schema_sha():
    """Pins channel order + target order into cache/freeze identities."""
    return hashlib.sha256(",".join(FEATURE_NAMES).encode() + b"|"
                          + ",".join(TARGET_NAMES).encode()).hexdigest()


def _state():
    return json.loads((ROOT / "experiments/qa-v1/collect2_state.json")
                      .read_text())


def _manifest():
    return json.loads((ROOT / "eval/corpus_manifest.json").read_text())


def _audio_path(manifest, fam):
    """Exact replica of the training-time audio resolution."""
    audio_dir = Path(next(r["dir"] for r in manifest["maps"]
                          if r["family"] == fam and r.get("family_rep")
                          and "/beatsaver/" in r["dir"]))
    return next((q for q in audio_dir.iterdir()
                 if q.suffix.lower() in (".egg", ".ogg", ".mp3")), None)


def family_evidence(fam, manifest=None):
    """Explicit audio evidence for one family, matching the policy the
    training caches were built under (missing/unsupported audio trained as
    zeros — recorded here as a named policy, never a silent None)."""
    manifest = manifest or _manifest()
    audio_p = _audio_path(manifest, fam)
    if audio_p is None:
        return {"family": fam, "policy": "no_audio:missing", "audio": None,
                "audio_path": None, "audio_sha256": None}
    aud = evidence(audio_p, cache_dir=AUDIO_CACHE)
    if not aud.get("supported", True):
        return {"family": fam, "policy": "no_audio:unsupported",
                "audio": None, "audio_path": str(audio_p),
                "audio_sha256": _sha_file(audio_p)}
    return {"family": fam, "policy": "audio", "audio": aud,
            "audio_path": str(audio_p), "audio_sha256": _sha_file(audio_p),
            "audio_config": {"version": AUDIO_EVIDENCE_VERSION,
                             "cache_key": audio_cache_key(
                                 audio_p, AUDIO_CACHE).name}}


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replay_view(scene, m_scene, entry, left_handed):
    """Scene view + body profile for one replay (mirror-aware)."""
    profile = {"height": entry.get("height"),
               "height_known": (entry.get("height") or 0) > 0.2,
               "left_handed": bool(left_handed)}
    return (m_scene if left_handed else scene), profile


def role_families(role):
    """Families with usable charts for a role, in canonical order."""
    from qa.scene import read_scene
    state, manifest = _state(), _manifest()
    out = []
    for fam, chart_rec in sorted(state["charts"].items()):
        if chart_rec.get("side") != role or not chart_rec.get("replays"):
            continue
        dat_p, info_p = _family_chart(manifest, fam,
                                      chart_rec["difficulty"])
        if dat_p is None:
            continue
        if read_scene(dat_p, info_p)["scope"] is not None:
            continue
        out.append(fam)
    return out


def records_for_role(role, purpose, families=None):
    """Iterator of eligible WindowRecords for one role under an allowed
    purpose. Enforces the frozen role×purpose matrix; the provider never
    serves frozen-only purposes (seal confirmation has its own runner)."""
    allowed = ROLE_PURPOSES.get(role, set())
    if purpose not in allowed:
        raise PermissionError(f"role {role!r} does not permit {purpose!r}")
    if purpose in FROZEN_ONLY:
        raise PermissionError(
            f"{purpose!r} is frozen-only and never served by the shared "
            "provider")
    from qa.scene import read_scene
    from qa.train import mirror_scene
    state, manifest = _state(), _manifest()
    for fam, chart_rec in sorted(state["charts"].items()):
        if chart_rec.get("side") != role or not chart_rec.get("replays"):
            continue
        if families is not None and fam not in families:
            continue
        dat_p, info_p = _family_chart(manifest, fam,
                                      chart_rec["difficulty"])
        if dat_p is None:
            continue
        scene = read_scene(dat_p, info_p)
        if scene["scope"] is not None:
            continue
        m_scene = mirror_scene(scene)
        for entry in chart_rec["replays"]:
            spt = ROOT / entry["file"].replace(".bsor", ".sanitized.pt")
            wp = spt.with_suffix(".windows_v2.json")
            if not wp.exists():
                continue
            rec = json.loads(wp.read_text())
            lh = bool(torch.load(spt)["info"].get("leftHanded"))
            sc, profile = replay_view(scene, m_scene, entry, lh)
            for wi, w in enumerate(rec["windows"]):
                if not w.get("supported") or w.get("outcome") != "good":
                    continue
                yield {"window_id": f"{fam}:{entry['sha256'][:16]}:"
                                    f"{w['chart_index']}:{wi}",
                       "family": fam, "player_token": rec["player_token"],
                       "replay_sha": entry["sha256"],
                       "chart_index": w["chart_index"], "window_index": wi,
                       "scene": sc, "profile": profile, "window": w,
                       "accuracy": entry.get("accuracy"),
                       "chart_sha256": scene["chart_sha256"],
                       "info_sha256": scene["info_sha256"]}


def features_for(record, ev):
    """The single feature path. Evidence must be explicit and family-bound;
    audio=None is only legal under a named no-audio policy."""
    if ev is None:
        raise ValueError("evidence must be explicit; use family_evidence()")
    if ev.get("family") != record["family"]:
        raise ValueError(f"evidence family {ev.get('family')!r} does not "
                         f"match record family {record['family']!r}")
    policy = ev.get("policy")
    if policy == "audio" and ev.get("audio") is None:
        raise ValueError("policy 'audio' with audio=None violates the "
                         "evidence contract")
    if policy not in ("audio",) and not str(policy).startswith("no_audio"):
        raise ValueError(f"unknown evidence policy {policy!r}")
    return map_features(record["scene"], ev.get("audio"),
                        record["chart_index"], record["profile"])


# thin caller adapters — shared provider, not independent implementations
training_features = features_for
calibration_features = features_for


def family_identity(fam, role, manifest=None):
    """Complete cache/freeze identity for one family (spec §1: chart+Info+
    audio hashes, replay hashes, config versions, channel schema)."""
    state = _state()
    chart_rec = state["charts"][fam]
    manifest = manifest or _manifest()
    dat_p, info_p = _family_chart(manifest, fam, chart_rec["difficulty"])
    ev = family_evidence(fam, manifest)
    return {"family": fam, "role": role,
            "chart_sha256": _sha_file(dat_p),
            "info_sha256": _sha_file(info_p),
            "audio_sha256": ev["audio_sha256"],
            "audio_policy": ev["policy"],
            "replay_sha256s": sorted(e["sha256"]
                                     for e in chart_rec["replays"]),
            "telemetry_version": TELEMETRY_VERSION,
            "audio_evidence_version": AUDIO_EVIDENCE_VERSION,
            "feature_schema_sha256": feature_schema_sha()}


def reconcile(role="qa_train"):
    """Row-for-row parity proof: provider features vs the actual cached
    matrices the frozen models were fitted on, plus the final checkpoint
    scaler. Persists the identity mapping; any mismatch stops as
    FEATURE_PROVENANCE_UNRESOLVED."""
    from qa.model import target_from_window
    from qa.train import DATA, DATA_VERSION, FITS, _standardize
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    report = {"role": role, "status": "OK", "families": {},
              "feature_schema_sha256": feature_schema_sha()}
    mapping = {}
    manifest = _manifest()
    packs = {}
    for fam in role_families(role):
        key = hashlib.sha256(
            f"{fam}:{TELEMETRY_VERSION}:{DATA_VERSION}".encode()
        ).hexdigest()[:16]
        cp = DATA / role / f"{key}.pt"
        if not cp.exists():
            report["families"][fam] = {"status": "missing_cache"}
            report["status"] = "FEATURE_PROVENANCE_UNRESOLVED"
            continue
        pack = torch.load(cp)
        ev = family_evidence(fam, manifest)
        xs, ys, ms, players, ids = [], [], [], [], []
        for r in records_for_role(role, "fit" if role == "qa_train"
                                  else "calibrate", families={fam}):
            y, mask = target_from_window(r["window"])
            xs.append(features_for(r, ev))
            ys.append(y)
            ms.append(mask)
            players.append(r["player_token"])
            ids.append(r["window_id"])
        ok = (len(xs) == len(pack["x"])
              and torch.equal(torch.stack(xs), pack["x"])
              and torch.equal(torch.stack(ys), pack["y"])
              and torch.equal(torch.stack(ms), pack["mask"])
              and players == pack["players"])
        fam_rep = {"status": "match" if ok else "MISMATCH",
                   "n_rows": len(xs),
                   "cache_file": cp.name,
                   "cache_sha256": _sha_file(cp),
                   "ids_sha256": hashlib.sha256(
                       "\n".join(ids).encode()).hexdigest(),
                   "identity": family_identity(fam, role, manifest)}
        if not ok:
            report["status"] = "FEATURE_PROVENANCE_UNRESOLVED"
        report["families"][fam] = fam_rep
        mapping[fam] = ids
        packs[fam] = pack
        print(f"  [{fam}] {fam_rep['status']} ({len(xs)} rows)", flush=True)
    if role == "qa_train" and report["status"] == "OK":
        ck = torch.load(FITS / "motion_mixture_final.pt")
        allx = torch.cat([packs[f]["x"] for f in packs])
        mu, sd = _standardize(allx)
        s_ok = (torch.equal(mu, ck["scaler"][0])
                and torch.equal(sd, ck["scaler"][1]))
        report["final_scaler_match"] = bool(s_ok)
        if not s_ok:
            report["status"] = "FEATURE_PROVENANCE_UNRESOLVED"
    out = PARITY_DIR / f"provenance_{role}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, sort_keys=True))
    tmp.replace(out)
    with gzip.open(PARITY_DIR / f"window_ids_{role}.json.gz", "wt") as f:
        json.dump(mapping, f)
    print(f"[{role}] provenance: {report['status']} -> {out}")
    return report


if __name__ == "__main__":
    import sys
    r = reconcile(sys.argv[1] if len(sys.argv) > 1 else "qa_train")
    if r["status"] != "OK":
        sys.exit(1)
