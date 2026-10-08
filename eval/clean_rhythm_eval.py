"""A/B measurement + frozen acceptance for the clean-rhythm pilot.

Compares a new (B) rhythm checkpoint against the shipped (A) one under the
UNCHANGED production decode policy, per-song, with non-regression gates from
docs/specs/2026-09-21-clean-rhythm-pilot-design.md. Descriptive
development tolerances, not human limits; a failed gate never promotes anything.
"""

REST_BIN_MS = 2000.0


def shared_rest_bins(notes_ms, start_ms, end_ms):
    """Fixed 2s bins anchored at start_ms; a bin is quiet if it holds <=1
    directional head (dir != 8). Trailing partial bins (ending past end_ms) are
    excluded. Both A and B use the SAME start/end so quiet bins are comparable.
    notes_ms: [{"t": ms, "dir": d}]."""
    if end_ms - start_ms < REST_BIN_MS:
        return {"counts": [], "quiet_ids": [], "n_bins": 0}
    n_bins = int((end_ms - start_ms) // REST_BIN_MS)
    counts = [0] * n_bins
    for note in notes_ms:
        if note["dir"] == 8:
            continue                      # dot follower, not a directional head
        k = int((note["t"] - start_ms) // REST_BIN_MS)
        if 0 <= k < n_bins:
            counts[k] += 1
    quiet_ids = [k for k, c in enumerate(counts) if c <= 1]
    return {"counts": counts, "quiet_ids": quiet_ids, "n_bins": n_bins}


def rest_pass(a_quiet, b_quiet, n_bins):
    """B's quiet share within 5pp of A, and >=90% of A's quiet-bin identities
    retained (when A has any)."""
    if n_bins <= 0:
        return False
    share_ok = abs(len(a_quiet) - len(b_quiet)) / n_bins <= 0.05 + 1e-12
    retained = (not a_quiet
                or len(set(a_quiet) & set(b_quiet)) / len(a_quiet) >= 0.90 - 1e-12)
    return share_ok and retained


def _dup_ok(a, b):
    return b <= max(a, 1.0) + 1e-12


def _flags_ok(a, b):
    return b <= a + max(5.0, 0.10 * a) + 1e-12


def gate_song(a, b):
    """Per-song non-regression verdict from two metric dicts (winner + mean of
    the initial six). Each dict: valid, in_band, dup8 (winner, six_mean),
    flags_by_ext {ext: (winner, six_mean)}, narrow_winner, narrow_six_mean,
    conv_winner, conv_six_mean, longest_run_winner, longest_run_six_mean,
    quiet (a_ids, b_ids, n_bins)."""
    fails = []
    if not (a["valid"] and b["valid"]):
        fails.append("validity")
    if not b["in_band"]:
        fails.append("rate_band")
    if not (_dup_ok(a["dup8_winner"], b["dup8_winner"])
            and _dup_ok(a["dup8_six_mean"], b["dup8_six_mean"])):
        fails.append("dup8")
    for ext in a["flags_by_ext"]:
        aw, am = a["flags_by_ext"][ext]
        bw, bm = b["flags_by_ext"][ext]
        if not (_flags_ok(aw, bw) and _flags_ok(am, bm)):
            fails.append(f"repositioning@{ext}")
    if not (b["narrow_winner"] <= max(a["narrow_winner"], 1) + 1e-12
            and b["conv_winner"] <= max(a["conv_winner"], 1) + 1e-12
            and b["narrow_six_mean"] <= a["narrow_six_mean"] + 0.5 + 1e-12
            and b["conv_six_mean"] <= a["conv_six_mean"] + 0.5 + 1e-12):
        fails.append("pair_exposure")
    if not (b["longest_run_winner"] <= max(a["longest_run_winner"], 12)
            and b["longest_run_six_mean"] <= max(a["longest_run_six_mean"], 12) + 1e-12):
        fails.append("hand_monopoly")
    q = b["quiet"]
    if not rest_pass(q["a_ids"], q["b_ids"], q["n_bins"]):
        fails.append("rest_preservation")
    return fails


def evaluate_acceptance(run_record):
    """Reduce the per-song A/B record to a status + failures. run_record:
    {learned: bool, integrity_ok: bool, songs: {song: {a, b} | {inconclusive}}}.
    Status: AB_ACCEPTED / AB_NOT_ACCEPTED / INCOMPLETE."""
    if not run_record.get("integrity_ok") or not run_record.get("learned"):
        return {"status": "INCOMPLETE", "failures": ["integrity_or_learning"],
                "per_song_gates": {}}
    per_song, any_fail, any_inconc = {}, False, False
    for song, rec in run_record["songs"].items():
        if rec.get("inconclusive"):
            per_song[song] = ["inconclusive_baseline"]
            any_inconc = True
            continue
        fails = gate_song(rec["a"], rec["b"])
        per_song[song] = fails
        any_fail = any_fail or bool(fails)
    if any_inconc:
        return {"status": "INCOMPLETE", "failures": ["inconclusive_baseline"],
                "per_song_gates": per_song}
    status = "AB_NOT_ACCEPTED" if any_fail else "AB_ACCEPTED"
    return {"status": status, "per_song_gates": per_song,
            "failures": sorted({f for fs in per_song.values() for f in fs})}
