"""E1 Task 1: run identity, approved provenance, split/panel reservation,
budgets and resume records (expressive-flow plan, spec 2026-09-23).

- A family is APPROVED when ANY of its map dirs lies under a hand-approved
  root, regardless of which directory is the canonical representative;
  EXPLICITLY REJECTED when a dir lies under the excluded root (provenance =
  the path itself). Everything else is GENERAL (unlabeled, never negative).
- freeze_run/save_partial: atomic (tmp -> fsync -> os.replace), identity =
  sha256 of the canonical config JSON; a run dir refuses a different
  config; a completed partial key is never recomputed.
- reserve_families: 12 fresh eligible test families by sha256(salt:family),
  excluding every prior panel/feedback family; the first 8 are the frozen
  playtest set, chosen before any outcome exists. A shortage is reported,
  never padded.
"""
import hashlib
import json
import os
from pathlib import Path

APPROVED_ROOTS = ("/beat-saber-map-gen/input/bytrius/",
                  "/beat-saber-map-gen/input/input/")
REJECTED_ROOTS = ("/beat-saber-map-gen/input/excluded/",)
_REQUIRED = ("dir", "family", "split", "eligible")


def _validate(corpus):
    maps = corpus.get("maps")
    if not maps:
        raise ValueError("corpus manifest has no maps")
    for m in maps:
        missing = [k for k in _REQUIRED if k not in m]
        if missing:
            raise ValueError(f"map record missing fields {missing}: "
                             f"{m.get('dir', '?')}")
    return maps


def family_strata(corpus):
    """{'approved': [...], 'general': [...], 'rejected': [...]} over ALL
    families in the manifest (callers intersect with a split). Approval and
    rejection are family-level provenance from ANY member directory."""
    maps = _validate(corpus)
    approved, rejected, everyone = set(), set(), set()
    for m in maps:
        d = m["dir"]
        everyone.add(m["family"])
        if any(r in d for r in APPROVED_ROOTS):
            approved.add(m["family"])
        if any(r in d for r in REJECTED_ROOTS):
            rejected.add(m["family"])
    approved -= rejected                     # explicit rejection dominates
    general = everyone - approved - rejected
    return {"approved": sorted(approved), "general": sorted(general),
            "rejected": sorted(rejected)}


def _canon(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _identity(config):
    return hashlib.sha256(_canon(config).encode()).hexdigest()


def _atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def freeze_run(root, config):
    """Create (or verify) a run's frozen identity. A different config for an
    existing run dir is an identity error, never an overwrite."""
    root = Path(root)
    run_p = root / "run.json"
    ident = _identity(config)
    if run_p.exists():
        prior = json.loads(run_p.read_text())
        if prior["identity"] != ident:
            raise ValueError(
                f"run identity mismatch: dir frozen as {prior['identity'][:12]}"
                f", got {ident[:12]} — start a fresh run dir")
        return prior
    rec = {"identity": ident, "config": config}
    _atomic_write(run_p, json.dumps(rec, indent=1, sort_keys=True))
    return rec


def load_run(root):
    return json.loads((Path(root) / "run.json").read_text())


def _partial_path(root, key):
    safe = key.replace("/", "_").replace(":", "_") if False else \
        key.replace("/", "_")
    return Path(root) / "partial" / (safe + ".json")


def save_partial(root, key, record):
    """Atomic per-key partial bound to the run identity. A COMPLETED key is
    immutable; an incomplete key may be resumed/finished."""
    run = load_run(root)
    p = _partial_path(root, key)
    if p.exists():
        prior = json.loads(p.read_text())
        if prior.get("completed"):
            raise ValueError(f"partial {key!r} already completed — "
                             "never recomputed")
    rec = dict(record)
    rec["key"] = key
    rec["config_sha"] = run["identity"]
    _atomic_write(p, json.dumps(rec, indent=1, sort_keys=True))
    return rec


def partial_completed(root, key):
    p = _partial_path(root, key)
    if not p.exists():
        return False
    return bool(json.loads(p.read_text()).get("completed"))


def reserve_families(corpus, exclude, n=12, salt="20260923", n_playtest=8):
    """Deterministic fresh-family reservation from eligible test families.
    Order = sha256(f"{salt}:{family}"); exclusions are every prior panel or
    feedback family. Returns the frozen playtest subset (first n_playtest)
    BEFORE any outcome exists. Shortage is an explicit status."""
    maps = _validate(corpus)
    pool = sorted({m["family"] for m in maps
                   if m["split"] == "test" and m["eligible"] == "ok"
                   and m["family"] not in exclude},
                  key=lambda f: hashlib.sha256(
                      f"{salt}:{f}".encode()).hexdigest())
    if len(pool) < n:
        return {"status": "SHORTAGE", "available": len(pool),
                "requested": n, "families": pool, "playtest": []}
    fams = pool[:n]
    return {"status": "ok", "families": fams,
            "playtest": fams[:n_playtest], "salt": salt}


# ---------------- the actual E1 freeze ----------------

ROOT = Path(__file__).resolve().parent.parent
E1_ROOT = ROOT / "experiments" / "expressive-v1" / "e1"
E1_SEED = 20260923
E1_SONGS = ("still_waiting", "numb", "FENT_TG", "spaceman", "rather_be",
            "rap_god")
PROTECTED = ("groom.pt", "flow.pt", "critic.pt", "ladder.json")


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _prior_panel_families():
    """Every family that has appeared in a panel or feedback set: quality D
    (panel + fam entries), the opened C confirmation, and the playtest
    exports (subsets of D)."""
    from eval.quality_panel import UNSEAL_TOKEN, confirm_families, load_panel
    p = load_panel(verify=False)
    prior = {e["family"] for e in p["dev"]}
    prior |= {f["family"] for f in
              confirm_families(unseal=UNSEAL_TOKEN)["families"]}
    return prior


def build_e1_run():
    """Freeze the E1 run: strata inventory, fresh-family reservation, six
    frozen song identities, protected hashes, budgets. Idempotent."""
    from eval import corpus
    from eval.quality_panel import dev_entries
    m = corpus._load_validated()
    strata = family_strata(m)
    by_split = {}
    for r in m["maps"]:
        by_split.setdefault(r["split"], set()).add(r["family"])
    inv = {}
    for split in ("train", "val"):
        fams = by_split.get(split, set())
        inv[split] = {
            "approved": sorted(set(strata["approved"]) & fams),
            "general_n": len(set(strata["general"]) & fams),
            "rejected_n": len(set(strata["rejected"]) & fams)}
    reserved = reserve_families(m, exclude=_prior_panel_families(),
                                n=12, salt=str(E1_SEED))
    songs = []
    for e in dev_entries():
        if e["song"] in E1_SONGS:
            songs.append({"song": e["song"], "family": e["family"],
                          "osu": e["osu"], "osu_sha256": e["osu_sha256"],
                          "audio": e["audio"],
                          "audio_sha256": e["audio_sha256"]})
    assert len(songs) == len(E1_SONGS), \
        f"panel is missing E1 songs: {[s['song'] for s in songs]}"
    config = {
        "packet": "e1", "seed": E1_SEED,
        "arms": {"A": 0.50, "B": 0.75},
        "train": {"lr": 3e-5, "batch": 8, "max_updates": 3600,
                  "kl_weight": 0.05, "snapshots": [0, 600, 1800, 3600]},
        "selection": "family-macro val CE, approved/general equally "
                     "weighted; fallback overall family-macro CE if "
                     "approved val families < 5 (frozen NOW, pre-training)",
        "approved_val_families_n": len(inv["val"]["approved"]),
        "strata_inventory": {
            "train_approved_n": len(inv["train"]["approved"]),
            "train_general_n": inv["train"]["general_n"],
            "train_rejected_n": inv["train"]["rejected_n"],
            "val_approved_n": len(inv["val"]["approved"]),
            "val_general_n": inv["val"]["general_n"]},
        "songs": songs,
        "attempt_budget": {"per_checkpoint_per_song": 6, "checkpoints": 2,
                           "max_new_attempts": 72, "baseline_max": 36},
        "ceilings": {"flags_per_extent": "max(1.25*B0, B0+10)",
                     "narrow_or_conv": "B0+2 each",
                     "window_4s_flags": "B0+2"},
        "reserved_confirmation": reserved,
        "protected": {f: _sha_file(ROOT / f) for f in PROTECTED},
        "policy_state": json.loads((ROOT / "quality_policy.json").read_text()),
        "shipped_flow_source": _sha_file(ROOT / "flow.pt"),
    }
    E1_ROOT.mkdir(parents=True, exist_ok=True)
    run = freeze_run(E1_ROOT, config)
    print(f"E1 frozen: identity {run['identity'][:12]}; "
          f"train approved/general/rejected = "
          f"{config['strata_inventory']['train_approved_n']}/"
          f"{config['strata_inventory']['train_general_n']}/"
          f"{config['strata_inventory']['train_rejected_n']}; "
          f"val approved = {config['approved_val_families_n']}; "
          f"reserved {reserved['status']} "
          f"({len(reserved['families'])} fams, playtest 8 frozen)")
    return run


if __name__ == "__main__":
    build_e1_run()
