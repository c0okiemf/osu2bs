"""Shared chart-level scope predicate for the fixed-time v2 audits.

Both the wall-corridor and E01 visibility audits assume a supported, fixed-time
v2 chart with fixed lane geometry. This one helper decides whether a chart meets
that assumption, so the two audits cannot diverge on the tricky cases (review
review 2026-09-20).
"""
from math import isfinite


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if isfinite(v) else None


def fixed_v2_scope_reason(raw):
    """Return a reason string if `raw` is NOT a supported fixed-time v2 chart
    with fixed lanes, else None. Rejects:
    - non-v2 schema;
    - a nonempty editor `_customData._BPMChanges` (timing unverified);
    - ANY lane-rotation event (v2 `_events` type 14/15) — value 0 is a real
      60-degree left rotation in the legacy enum, NOT a no-op, so zero and
      malformed values are unsupported too;
    - legacy type-100 BPM-change events (outside the fixed-time model).
    Ordinary lighting events, empty editor arrays and bookmarks are fine."""
    ver = str(raw.get("_version") or raw.get("version") or "")
    if not ver.startswith("2"):
        return f"unsupported_schema:{ver or 'none'}"
    cd = raw.get("_customData")
    if isinstance(cd, dict):
        bpmc = cd.get("_BPMChanges") or cd.get("_bpmChanges")
        if isinstance(bpmc, list) and bpmc:
            return "ambiguous_timing:editor_bpm_changes"
    evs = raw.get("_events")
    if isinstance(evs, list):
        for e in evs:
            if not isinstance(e, dict):
                continue
            t = e.get("_type")
            if t in (14, 15):
                return "unsupported_transform:lane_rotation"
            if t == 100:
                return "unsupported_transform:legacy_bpm_event"
    return None
