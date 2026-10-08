"""Q4: durable generation-job context (spec §7 resolve_style bullet).

A job manifest is created in the OUTPUT directory before the first song and
records every per-song resolution (style, seed, difficulties) keyed by the
song's STABLE audio content hash (sha256 of the audio file — never Python's
process-randomized hash(), never a worker-local variable). Restarting,
appending, reordering or retrying a playlist re-reads the manifest and
returns the identical resolution; two concurrent creators each own their
manifest because it lives in their own output dir.

The style vocabulary resolved to neutral K=1 (see style.py), so resolve_style currently always resolves
"neutral" — the binding machinery is what ships; exposed styles slot in via
the same manifest without schema change.
"""
import hashlib
import json
import os
from pathlib import Path

MANIFEST_NAME = "osu2bs_job.json"
MANIFEST_VERSION = 1


def audio_key(audio_path):
    """Stable content identity of a song: sha256 of the audio bytes."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(key, salt="osu2bs-q4"):
    """Deterministic per-song base seed from the audio content hash."""
    return int.from_bytes(hashlib.sha256(
        f"{salt}:{key}".encode()).digest()[:4], "big")


class JobManifest:
    """Durable per-job resolution store. Atomic writes; safe to re-open."""

    def __init__(self, out_dir):
        self.path = Path(out_dir) / MANIFEST_NAME
        if self.path.exists():
            d = json.loads(self.path.read_text())
            if d.get("version") != MANIFEST_VERSION:
                raise ValueError(
                    f"job manifest v{d.get('version')} != {MANIFEST_VERSION}")
            self.data = d
        else:
            self.data = {"version": MANIFEST_VERSION, "songs": {}}
            self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def resolve(self, audio_path, *, difficulties=("ExpertPlus",),
                explicit_seed=None, explicit_style=None):
        """The one entry point: returns {style, seed, difficulties, key}.
        First resolution binds and persists; every later call (restart,
        append, reorder, retry) returns the SAME binding. An explicit seed or
        style only applies on first binding — a rebind would silently change
        already-exported charts, so a conflicting explicit value raises."""
        key = audio_key(audio_path)
        songs = self.data["songs"]
        if key in songs:
            rec = songs[key]
            if explicit_seed is not None and explicit_seed != rec["seed"]:
                raise ValueError(
                    f"song {key[:12]} already bound to seed {rec['seed']}; "
                    "start a new output dir to change it")
            if explicit_style is not None and explicit_style != rec["style"]:
                raise ValueError(
                    f"song {key[:12]} already bound to style {rec['style']}")
            missing = [d for d in difficulties
                       if d not in rec["difficulties"]]
            if missing:                      # additional tiers extend the set
                rec["difficulties"] = sorted(set(rec["difficulties"])
                                             | set(difficulties))
                self._write()
            return dict(rec, key=key)
        rec = {"style": explicit_style or resolve_style(key),
               "seed": explicit_seed if explicit_seed is not None
               else stable_seed(key),
               "difficulties": sorted(difficulties)}
        songs[key] = rec
        self._write()
        return dict(rec, key=key)

    def cleanup_zip_only(self, out_dir, keep_zips=True):
        """zip-only mode removes exported map DIRS but never the manifest —
        the bindings must survive so a re-run reproduces the zips."""
        out = Path(out_dir)
        removed = []
        for p in sorted(out.iterdir()):
            if p.is_dir() and (p / "Info.dat").exists():
                import shutil
                shutil.rmtree(p)
                removed.append(p.name)
        return removed


def resolve_style(audio_key_hex):
    """Stable style for a song. The fitted vocabulary is neutral-only
    (K=1); when exposed styles exist this maps the audio key onto them
    deterministically via the SAME stable hash — never process-randomized
    hash(), never a worker-local variable."""
    vocab_p = (Path(__file__).resolve().parent / "experiments" / "quality-v1"
               / "q4-style" / "style_vocab.json")
    if not vocab_p.exists():
        return "neutral"
    vocab = json.loads(vocab_p.read_text())
    styles = sorted(vocab.get("styles") or {})
    if not styles:
        return "neutral"
    idx = int.from_bytes(hashlib.sha256(
        f"style:{audio_key_hex}".encode()).digest()[:4], "big")
    return styles[idx % len(styles)]
