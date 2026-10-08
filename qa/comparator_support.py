"""Comparator Task 3 (spec §4): raw-reference execution support.

Support means: five hash-valid raw human references with the existing
diversity rules (>=3 families, >=3 player tokens, query family/players
excluded) at median descriptor distance <= T_support. Distance measures
SUPPORT, never quality; unavailable/corrupt references and insufficient
diversity are UNSUPPORTED, not zero distance. Unfamiliar maps are
unknown, not "bad mapping".

freeze_support computes the ONE cutoff formula: per development family,
require >=90% of its frozen windows to have valid diverse retrieval,
then p95 over valid finite distances; T_support = max family p95 across
all eight. Anything less stops DATA_SUPPORT_INSUFFICIENT. The historical
7.52 constant is never reused without this recipe's own provenance.

measure_support serves EVERY scoreable head (same-hand same-time groups)
in bounded chunks with resumable progress: per-head support, origin-zero
2 s reporting bins, the complete unsupported-run list (a run violates
when last-first > 2 s AND it contains >= 4 heads; initial/terminal runs
count), evidence errors, and machine_support_pass (>=90% supported, no
violating run). Diagnostic mixture values cannot enter this report.
"""
import hashlib
import json
from pathlib import Path

from qa.calibrate import _p95

ROOT = Path(__file__).resolve().parent.parent
SHARE_MIN = 0.90
DIVERSE_MIN_SHARE = 0.90
RUN_SPAN_S = 2.0
RUN_MIN_HEADS = 4
BIN_S = 2.0
CHUNK = 200


def run_gate(supported_flags, times):
    """Production unsupported-run reducer. Returns runs + gate verdict.
    A run is maximal consecutive unsupported heads in time order."""
    runs, cur = [], []
    for ok, t in sorted(zip(supported_flags, times), key=lambda x: x[1]):
        if ok:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        runs.append(cur)
    out = [{"start": r[0], "end": r[-1], "n_heads": len(r),
            "violating": (r[-1] - r[0]) > RUN_SPAN_S
            and len(r) >= RUN_MIN_HEADS} for r in runs]
    return {"runs": out,
            "run_gate": not any(r["violating"] for r in out)}


def freeze_support(dev_rows, bank_identity):
    """dev_rows: [{family, window_id, distance|None, diverse: bool}].
    ONE max-family-p95 formula; complete denominators; no reuse of a
    bare historical constant."""
    if not dev_rows:
        raise ValueError("no development rows: cannot freeze a support "
                         "cutoff from nothing")
    fams = {}
    for r in dev_rows:
        fams.setdefault(r["family"], []).append(r)
    per_family = {}
    for fam, rows in sorted(fams.items()):
        valid = [r["distance"] for r in rows
                 if r.get("diverse") and r.get("distance") is not None]
        share = len(valid) / len(rows)
        if share < DIVERSE_MIN_SHARE:
            raise RuntimeError(
                f"DATA_SUPPORT_INSUFFICIENT: {fam} has only "
                f"{share:.3f} valid diverse retrieval (need >= "
                f"{DIVERSE_MIN_SHARE})")
        per_family[fam] = {"n_windows": len(rows), "n_valid": len(valid),
                           "valid_share": round(share, 4),
                           "p95": _p95(valid)}
    t_support = max(v["p95"] for v in per_family.values())
    return {"T_support": t_support, "per_family": per_family,
            "recipe": "max-family-p95 over valid finite diverse "
                      "distances; >=90% valid diverse retrieval required "
                      "per family",
            "bank_identity": bank_identity,
            "provenance": hashlib.sha256(json.dumps(
                {"bank": bank_identity,
                 "fams": sorted(per_family)}, sort_keys=True).encode())
            .hexdigest()}


class _RefValidator:
    """Validation of raw trajectory references (content-hash bound to the
    restricted-source sha). Keeps only a compact permanent record per replay
    (source sha + per-window "observed trajectory" flag), backed by a disk
    cache keyed on both files' size+mtime, so each replay is unpickled at
    most once instead of on every LRU miss. Verdicts are memoized per
    reference; outcomes are identical to reading the full files."""

    CACHE_P = ROOT / "experiments" / "qa-v4" / "ref_validation_cache.json"

    def __init__(self):
        self.recs = {}
        self.verdicts = {}
        try:
            self.disk = json.loads(self.CACHE_P.read_text())
        except (OSError, ValueError):
            self.disk = {}

    @staticmethod
    def _stamp(p):
        st = p.stat()
        return [st.st_size, st.st_mtime_ns]

    def _load(self, rel):
        if rel in self.recs:
            return self.recs[rel]
        spt = ROOT / rel.replace(".bsor", ".sanitized.pt")
        wp = spt.with_suffix(".windows_v2.json")
        entry = None
        if spt.exists() and wp.exists():
            stamp = [self._stamp(spt), self._stamp(wp)]
            hit = self.disk.get(rel)
            if hit and hit["stamp"] == stamp:
                entry = {"source_sha256": hit["sha"], "flags": hit["flags"]}
            else:
                import torch
                try:
                    entry = {"source_sha256":
                             torch.load(spt).get("source_sha256"),
                             "flags": [bool(w.get("supported")
                                            and w.get("rel_path_24"))
                                       for w in json.loads(wp.read_text())
                                       ["windows"]]}
                except Exception:
                    entry = None
                if entry is not None:
                    self.disk[rel] = {"stamp": stamp,
                                      "sha": entry["source_sha256"],
                                      "flags": entry["flags"]}
                    self._flush()
        self.recs[rel] = entry
        return entry

    def _flush(self):
        tmp = self.CACHE_P.with_suffix(".tmp")
        self.CACHE_P.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.disk))
        tmp.replace(self.CACHE_P)

    def valid(self, row):
        """A neighbour's raw reference must exist, be hash-bound and
        carry an observed trajectory."""
        ref = row.get("traj_ref") or ""
        memo_key = (ref, row.get("content_hash"))
        if memo_key in self.verdicts:
            return self.verdicts[memo_key]
        out = self._valid(ref, row)
        self.verdicts[memo_key] = out
        return out

    def _valid(self, ref, row):
        if "#" not in ref:
            return False, "malformed_ref"
        rel, wi = ref.rsplit("#", 1)
        rec = self._load(rel)
        if rec is None or rec.get("source_sha256") is None:
            return False, "missing_source"
        want = hashlib.sha256(
            f"{rec['source_sha256']}:{wi}".encode()).hexdigest()[:16]
        if row.get("content_hash") != want:
            return False, "hash_mismatch"
        try:
            observed = rec["flags"][int(wi)]
        except (IndexError, ValueError):
            return False, "missing_window"
        if not observed:
            return False, "unsupported_reference_tracking"
        return True, None


def _heads(scene):
    groups = {}
    for i, (t, li, ll, c, d) in enumerate(scene["notes"]):
        groups.setdefault((round(float(t), 6), c), []).append((li, ll, i))
    heads = []
    for (t, hand), notes in groups.items():
        notes.sort()
        heads.append({"t": t, "hand": hand, "rep_index": notes[0][2]})
    heads.sort(key=lambda h: (h["t"], h["hand"]))
    return heads


def measure_support(scene, bank, threshold, exclusions, progress_dir=None,
                    validator=None):
    """Per-head support for one candidate scene."""
    from qa.neighbours import descriptor, retrieve
    validator = validator or _RefValidator()
    heads = _heads(scene)
    per_head, errors = [], []
    prog_p = (Path(progress_dir) / "support_progress.json") \
        if progress_dir else None
    done = 0
    if prog_p and prog_p.exists():
        prior = json.loads(prog_p.read_text())
        per_head = prior["per_head"]
        errors = prior["evidence_errors"]
        done = len(per_head)
    desc_cache = {}
    for idx in range(done, len(heads)):
        h = heads[idx]
        key = h["rep_index"]
        if key not in desc_cache:
            desc_cache[key] = descriptor(scene, key)
        nn = retrieve(bank, desc_cache[key], exclude=exclusions)
        rec = {"t": h["t"], "hand": h["hand"], "supported": False,
               "distance": None, "reason": None}
        if nn["status"] != "ok":
            rec["reason"] = "insufficient_diversity"
        else:
            bad = None
            for nb in nn["neighbours"]:
                ok, why = validator.valid(nb)
                if not ok:
                    bad = why
                    break
            if bad:
                rec["reason"] = f"reference_{bad}"
                errors.append({"t": h["t"], "error": bad})
            elif nn["support_distance"] <= threshold["T_support"]:
                rec["supported"] = True
                rec["distance"] = nn["support_distance"]
            else:
                rec["reason"] = "distance_above_cutoff"
                rec["distance"] = nn["support_distance"]
        per_head.append(rec)
        if prog_p and (idx + 1) % CHUNK == 0:
            tmp = prog_p.with_suffix(".tmp")
            prog_p.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"per_head": per_head,
                                       "evidence_errors": errors}))
            tmp.replace(prog_p)
    flags = [r["supported"] for r in per_head]
    times = [r["t"] for r in per_head]
    rg = run_gate(flags, times)
    bins = {}
    for r in per_head:
        b = int(r["t"] // BIN_S)
        cell = bins.setdefault(b, [0, 0])
        cell[1] += 1
        cell[0] += int(r["supported"])
    share = (sum(flags) / len(flags)) if flags else None
    report = {"heads_total": len(per_head),
              "heads_supported": sum(flags),
              "share_supported": round(share, 4)
              if share is not None else None,
              "unsupported_runs": rg["runs"],
              "run_gate": rg["run_gate"],
              "bins_2s": {str(k): {"supported": v[0], "total": v[1]}
                          for k, v in sorted(bins.items())},
              "per_head": per_head, "evidence_errors": errors,
              "threshold": {"T_support": threshold["T_support"],
                            "provenance": threshold.get("provenance")},
              "machine_support_pass": bool(
                  share is not None and share >= SHARE_MIN
                  and rg["run_gate"])}
    if prog_p:
        tmp = prog_p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"per_head": per_head,
                                   "evidence_errors": errors}))
        tmp.replace(prog_p)
    return report
