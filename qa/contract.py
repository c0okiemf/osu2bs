"""Independent-QA Task 1: frozen roles, access policy, safe identity
storage (spec §2, plan Task 1).

- open_role enforces the role×purpose matrix and hash-verifies every file
  it serves; confirm on the seal requires a FROZEN contract. gen is never a
  QA-readable role.
- sanitized_record strips identity AND score ids from a parsed replay while
  keeping keyed player tokens and source-hash provenance; parsing alone
  never anonymized the original bytes (recorded fact).
- scan_imports proves evaluator modules import no generator module — an
  application-level guard layered under (not replacing) launch-time
  filesystem isolation.
"""
import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ROLE_PURPOSES = {"qa_train": {"fit", "reference"},
                 "qa_calib": {"calibrate"},
                 "qa_calib2": {"calibrate"},
                 "qa_validate_latent": {"validate"},
                 "qa_validate_comparator": {"validate"},
                 "seal": {"confirm"}}
FROZEN_ONLY = {"confirm"}

EVALUATOR_MODULES = ("qa/contract.py", "qa/telemetry_v2.py", "qa/scene.py",
                     "qa/physics.py", "qa/model.py", "qa/train.py",
                     "qa/neighbours.py", "qa/calibrate.py",
                     "qa/mutations.py", "qa/evidence.py",
                     "qa/adjudication.py", "qa/run.py")
FORBIDDEN_IMPORTS = ("groom", "convert", "critic", "cond_flow",
                     "quality_repair", "flow_decode", "phrase_planner",
                     "quality_policy", "pref", "motion", "parity")

IDENTITY_KEYS = ("identity", "score_id", "playerID", "playerName",
                 "platform")


@dataclass
class Contract:
    path: Path
    frozen: bool
    roles: dict


def load_contract(path):
    d = json.loads(Path(path).read_text())
    roles = d["roles"]
    seen_f, seen_p = {}, {}
    for role, rec in roles.items():
        for f in rec.get("families", []):
            if f in seen_f and seen_f[f] != role:
                raise ValueError(f"family overlap: {f} in {seen_f[f]} "
                                 f"and {role}")
            seen_f[f] = role
        for p in rec.get("players", []):
            if p in seen_p and seen_p[p] != role:
                raise ValueError(f"player overlap: {p} in {seen_p[p]} "
                                 f"and {role}")
            seen_p[p] = role
    return Contract(path=Path(path), frozen=bool(d.get("frozen")),
                    roles=roles)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def open_role(contract, role, purpose):
    """Hash-verified records for one role under an allowed purpose."""
    allowed = ROLE_PURPOSES.get(role, set())
    if purpose not in allowed:
        raise PermissionError(f"role {role!r} does not permit {purpose!r}")
    if purpose in FROZEN_ONLY and not contract.frozen:
        raise PermissionError(
            f"{purpose!r} requires a FROZEN contract (evaluator freeze "
            "before the seal opens)")
    rec = contract.roles[role]
    root = Path(rec["root"]).resolve()
    out = []
    for rel, want in sorted(rec.get("files", {}).items()):
        p = (root / rel)
        if ".." in Path(rel).parts or p.is_symlink():
            raise ValueError(f"unsafe path {rel!r}")
        rp = p.resolve()
        if not str(rp).startswith(str(root) + os.sep):
            raise ValueError(f"unsafe path escape {rel!r}")
        data = rp.read_bytes()
        if _sha_bytes(data) != want:
            raise ValueError(f"content hash mismatch for {rel!r}")
        out.append({"path": str(rp), "data": data})
    return out


def canonical_bytes(record):
    return json.dumps(record, sort_keys=True, default=str).encode()


def sanitized_record(parsed, player_token=None, source_sha256=None):
    """Identity-stripped derived record: keeps poses/cut events/info,
    replaces identity with the keyed token, retains source-hash provenance.
    The ORIGINAL bytes are NOT anonymous and belong in restricted storage."""
    out = {k: v for k, v in parsed.items() if k not in IDENTITY_KEYS}
    info = dict(out.get("info") or {})
    for k in IDENTITY_KEYS:
        info.pop(k, None)
    out["info"] = info
    if player_token is not None:
        out["player_token"] = player_token
    if source_sha256 is not None:
        out["source_sha256"] = source_sha256
    # leak check against STRING fields only — numeric identity digits can
    # coincide with float sequences in pose arrays (false positive class)
    strings = []

    def _walk(x):
        if isinstance(x, str):
            strings.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
    _walk(out)
    ident = parsed.get("identity") or {}
    pid = ident.get("playerID")
    if pid and any(s == pid for s in strings):
        raise ValueError("identity value survived sanitization: playerID")
    pname = ident.get("playerName")
    # a player who mapped their own chart makes playerName == mapper a
    # legitimate public-field collision, not a leak
    public = {info.get("songName"), info.get("mapper")}
    if pname and pname not in public and any(s == pname for s in strings):
        raise ValueError("identity value survived sanitization: playerName")
    return out


def scan_imports(modules, forbidden):
    """Static import scan of evaluator modules; returns violations."""
    root = Path(__file__).resolve().parent.parent
    bad = []
    for rel in modules:
        p = root / rel
        if not p.exists():
            continue
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top in forbidden:
                    bad.append(f"{rel}: import {n}")
    return bad


def migrate_originals():
    """One-time safe-storage migration (plan Task 1): every original .bsor
    (identity-bearing bytes) moves into restricted acquisition storage keyed
    by content hash; gen/train/calib get identity-stripped .sanitized.pt
    derived records in place; SEAL originals move UNPARSED (sanitized
    materialization happens post-freeze in the seal runner). Historical
    collection manifests stay intact; a migration map records provenance.
    Idempotent."""
    import torch
    from qa.bsor import parse_bsor
    root = Path(__file__).resolve().parent.parent
    acq = root / "experiments" / "qa-v1-acquisition"
    originals = acq / "originals"
    originals.mkdir(parents=True, exist_ok=True)
    key = (acq / "key.txt").read_text().strip()
    map_p = acq / "migration_map.json"
    mig = json.loads(map_p.read_text()) if map_p.exists() else {}
    n_moved = n_sane = 0
    for state_name in ("pilot_state.json", "collect2_state.json"):
        sp = root / "experiments" / "qa-v1" / state_name
        if not sp.exists():
            continue
        state = json.loads(sp.read_text())
        for fam, chart in state["charts"].items():
            side = chart.get("side")
            for entry in chart.get("replays", []):
                rel = entry["file"]
                src = root / rel
                if rel in mig:
                    continue
                if not src.exists():
                    mig[rel] = {"status": "missing"}
                    continue
                blob = src.read_bytes()
                sha = _sha_bytes(blob)
                dst = originals / (sha + ".bsor")
                if not dst.exists():
                    tmp = dst.with_suffix(".tmp")
                    tmp.write_bytes(blob)
                    tmp.replace(dst)
                    dst.chmod(0o600)
                if side in ("qa", "seal"):
                    src.unlink()             # move only, never parsed here
                    mig[rel] = {"status": "seal_deferred",
                                "original_sha256": sha}
                else:
                    parsed = parse_bsor(blob)
                    san = sanitized_record(
                        parsed, player_token=entry["player_token"],
                        source_sha256=sha)
                    sp_out = src.with_suffix(".sanitized.pt")
                    tmpp = sp_out.with_suffix(".tmp")
                    torch.save(san, tmpp)
                    tmpp.replace(sp_out)
                    src.unlink()
                    n_sane += 1
                    mig[rel] = {"status": "sanitized",
                                "original_sha256": sha,
                                "sanitized": str(sp_out.relative_to(root))}
                n_moved += 1
                if n_moved % 25 == 0:        # incremental persistence
                    tmp = map_p.with_suffix(".tmp")
                    tmp.write_text(json.dumps(mig, indent=1, sort_keys=True))
                    tmp.replace(map_p)
    mig["_note"] = ("originals were NOT previously anonymous; identity "
                    "lives only here (restricted) and in acquisition.jsonl")
    tmp = map_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(mig, indent=1, sort_keys=True))
    tmp.replace(map_p)
    print(f"migrated {n_moved} originals ({n_sane} sanitized in place, "
          f"rest seal-deferred)")
    return {"moved": n_moved, "sanitized": n_sane}


def atomic_record(root, key, payload, identity):
    """Atomic identity-bound record; an existing key with a different
    identity refuses."""
    p = Path(root) / (key.replace("/", "_") + ".json")
    if p.exists():
        prior = json.loads(p.read_text())
        if prior.get("identity") != identity:
            raise ValueError(f"identity mismatch for existing key {key!r}")
    rec = dict(payload)
    rec["identity"] = identity
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return rec
