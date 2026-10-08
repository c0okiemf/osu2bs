"""Verified, reusable prepared training dataset for the clean-rhythm pilot.

Persists the EXACT post-_split (tr, val, split_rng_state) so train can skip a
second ~12-min live load. Reuse is fail-closed: an identity manifest binds the
corpus snapshot, consumed sources, code/config and consumed feature values, and
a payload checksum guards the tensors. Bit-identical to the serial path — no
tolerance, no silent rebuild. See docs/specs/2026-09-21-pipeline-
speed-design.md.
"""
import hashlib
import json
import struct
from pathlib import Path

import torch

SCHEMA = 1
_PAYLOAD = "dataset.pt"
_IDENTITY = "dataset.identity.json"       # written LAST = completion marker


def _enc(v, out):
    """Deterministic, bit-stable encoding of the (tr,val,rng_state) payload:
    identical content -> identical bytes, even for separately-allocated tensors."""
    if isinstance(v, torch.Tensor):
        t = v.detach().cpu().contiguous()
        out.append(b"T" + str(t.dtype).encode() + b":"
                   + repr(tuple(t.shape)).encode() + b":"
                   + t.numpy().tobytes() + b";")
    elif isinstance(v, bool):
        out.append(b"B1" if v else b"B0")
    elif isinstance(v, int):
        out.append(b"i" + str(v).encode() + b";")
    elif isinstance(v, float):
        out.append(b"f" + struct.pack("<d", v))
    elif isinstance(v, str):
        out.append(b"s" + v.encode() + b";")
    elif isinstance(v, (bytes, bytearray)):
        out.append(b"y" + str(len(v)).encode() + b":" + bytes(v))
    elif v is None:
        out.append(b"N")
    elif isinstance(v, tuple):
        out.append(b"(")
        for x in v:
            _enc(x, out)
        out.append(b")")
    elif isinstance(v, list):
        out.append(b"[")
        for x in v:
            _enc(x, out)
        out.append(b"]")
    elif isinstance(v, dict):
        out.append(b"{")
        for k, val in v.items():        # insertion order (deterministic)
            _enc(k, out)
            _enc(val, out)
        out.append(b"}")
    else:
        raise TypeError(f"uncanonicalizable type {type(v)!r}")


def canonical_payload_bytes(value):
    out = []
    _enc(value, out)
    return b"".join(out)


def dataset_digest(tr, val, split_rng_state):
    return hashlib.sha256(
        canonical_payload_bytes((tr, val, split_rng_state))).hexdigest()


def build_identity(snapshot_sha256, inventory, consumed_features, code_hashes,
                   extra=None):
    """Semantic identity that must match for reuse. Excludes worker count, run
    dir, PIDs and timing (execution metadata, kept in a sidecar). `inventory`
    binds train/val source files/hashes/order + absent-file sentinels;
    `consumed_features` is {key: value_digest} for the features actually used."""
    ident = {
        "schema": SCHEMA,
        "manifest_snapshot_sha256": snapshot_sha256,
        "inventory": inventory,
        "consumed_features": dict(sorted(consumed_features.items())),
        "code_hashes": dict(sorted(code_hashes.items())),
        "torch_version": torch.__version__,
    }
    try:
        import numpy as np
        ident["numpy_version"] = np.__version__
    except Exception:
        ident["numpy_version"] = None
    return ident


def write_prepared(run, tr, val, split_rng_state, identity):
    """Atomically persist the payload, then publish the identity marker LAST."""
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    payload_sha = dataset_digest(tr, val, split_rng_state)
    tmp = run / (_PAYLOAD + ".tmp")
    torch.save((tr, val, split_rng_state), tmp)
    tmp.replace(run / _PAYLOAD)
    marker = {"schema": SCHEMA, "identity": identity, "payload_sha256": payload_sha}
    mtmp = run / (_IDENTITY + ".tmp")
    mtmp.write_text(json.dumps(marker, indent=1, sort_keys=True))
    mtmp.replace(run / _IDENTITY)
    return payload_sha


class PreparedDatasetError(Exception):
    pass


def read_prepared(run, expected_identity):
    """Load (tr, val, split_rng_state) only if the completion marker exists, the
    identity matches expected_identity exactly, and the payload checksum holds.
    Never rebuilds silently."""
    run = Path(run)
    marker_p, payload_p = run / _IDENTITY, run / _PAYLOAD
    if not marker_p.exists():
        raise PreparedDatasetError("no completion marker (incomplete/absent run)")
    if not payload_p.exists():
        raise PreparedDatasetError("payload missing")
    marker = json.loads(marker_p.read_text())
    if marker.get("schema") != SCHEMA:
        raise PreparedDatasetError("schema mismatch")
    if marker.get("identity") != expected_identity:
        raise PreparedDatasetError("identity mismatch (stale/changed sources or code)")
    tr, val, state = torch.load(payload_p, map_location="cpu")
    if dataset_digest(tr, val, state) != marker.get("payload_sha256"):
        raise PreparedDatasetError("payload checksum mismatch (corrupt/truncated)")
    return tr, val, state
