"""Independent-QA Task 5: empirical raw-replay comparator (spec §6).

The bank indexes QA-TRAIN ONLY (any other role raises), stores explicit
scene descriptors + raw trajectory references + target vectors — never
model embeddings. Retrieval returns five neighbours spanning >=3 families
and >=3 player tokens (excluding the query's family/player for controls),
ties broken by content hash; support distance = the median descriptor
distance. Insufficient diversity is UNKNOWN support, not a stretched
answer. The comparison is a conditional human reference, not a measured
trajectory for the candidate.
"""
import hashlib
from dataclasses import dataclass

import torch

N_NEIGHBOURS = 5
MIN_FAMS = 3
MIN_PLAYERS = 3
CTX = 4


@dataclass
class Bank:
    identity: str
    desc: torch.Tensor            # [N, D] standardized
    mu: torch.Tensor
    sd: torch.Tensor
    rows: list                    # metadata per row
    hash_arr: object = None       # lazy numpy hash array for fast ordering


def descriptor(scene, note_index):
    """Explicit scene descriptor for one note: same-hand prev/next gaps,
    relative positions, cut directions, opposite-hand timing/position,
    doubles opportunity, approach scenario. Independent of the model's
    feature builder by construction (frozen field list)."""
    from qa.model import _dir_feat
    notes = scene["notes"]
    t, li, ll, color, d = notes[note_index]
    ds, dc, dot = _dir_feat(d)
    out = [li / 3.0, ll / 2.0, ds, dc, dot, 1.0 if color == 1 else 0.0]
    same = [n for n in notes if n[3] == color]
    pos = next(k for k, n in enumerate(same)
               if n[0] == t and n[1] == li and n[2] == ll)
    for direction in (-1, +1):
        for k in range(1, CTX + 1):
            j = pos + k * direction
            ok = 0 <= j < len(same)
            n2 = same[j] if ok else None
            out.append(1.0 if ok else 0.0)
            out.append(max(-2.0, min(2.0, n2[0] - t)) if ok else 0.0)
            out.append(((n2[1] - li) / 3.0) if ok else 0.0)
            out.append(((n2[2] - ll) / 2.0) if ok else 0.0)
            s2, c2, _ = _dir_feat(n2[4]) if ok else (0.0, 0.0, 0.0)
            out += [s2, c2]
    other = [n for n in notes if n[3] != color]
    prev_o = max((n for n in other if n[0] <= t), key=lambda n: n[0],
                 default=None)
    next_o = min((n for n in other if n[0] > t), key=lambda n: n[0],
                 default=None)
    for nb in (prev_o, next_o):
        ok = nb is not None
        out.append(1.0 if ok else 0.0)
        out.append(max(-2.0, min(2.0, nb[0] - t)) if ok else 0.0)
        out.append(((nb[1] - li) / 3.0) if ok else 0.0)
        out.append(((nb[2] - ll) / 2.0) if ok else 0.0)
    simul = next((n for n in other if abs(n[0] - t) <= 1e-3), None)
    out.append(1.0 if simul else 0.0)
    st = scene.get("settings") or {}
    njs = st.get("njs")
    out.append((float(njs) / 20.0) if isinstance(njs, (int, float)) else 0.0)
    return torch.tensor(out, dtype=torch.float32)


def build_bank(records, identity):
    """QA-train-only bank; continuous fields standardized on the bank."""
    for r in records:
        if r.get("role") != "qa_train":
            raise PermissionError(
                f"bank accepts qa_train records only, got {r.get('role')!r}")
    desc = torch.stack([r["desc"] for r in records])
    mu = desc.mean(dim=0)
    sd = desc.std(dim=0).clamp(min=1e-6)
    return Bank(identity=identity, desc=(desc - mu) / sd, mu=mu, sd=sd,
                rows=[{k: r[k] for k in ("family", "player", "traj_ref",
                                         "content_hash", "targets")
                       if k in r} | ({"mask": r["mask"]} if "mask" in r
                                     else {})
                      for r in records])


PREFILTER_K = 512


def _scan(bank, order, exf, exp):
    """The unchanged selection walk over `order`. Returns exhausted=True
    when a loop ran out of rows without its own stopping condition (the
    only case where a longer order could change the result)."""
    picked = []
    fams, players = set(), set()
    exhausted = True
    for i in order:
        r = bank.rows[i]
        if r["family"] in exf or r["player"] in exp:
            continue
        if len(picked) >= N_NEIGHBOURS:
            exhausted = False
            break
        picked.append((i, r))
        fams.add(r["family"])
        players.add(r["player"])
    if len(picked) >= N_NEIGHBOURS and (len(fams) < MIN_FAMS
                                        or len(players) < MIN_PLAYERS):
        exhausted = True
        for i in order:
            r = bank.rows[i]
            if r["family"] in exf or r["player"] in exp:
                continue
            if all(i != j for j, _ in picked) and \
                    (r["family"] not in fams or r["player"] not in players):
                for k in range(len(picked) - 1, -1, -1):
                    _, pr = picked[k]
                    fam_count = sum(1 for _, x in picked
                                    if x["family"] == pr["family"])
                    if fam_count > 1:
                        picked[k] = (i, r)
                        fams = {x["family"] for _, x in picked}
                        players = {x["player"] for _, x in picked}
                        break
                if len(fams) >= MIN_FAMS and len(players) >= MIN_PLAYERS:
                    exhausted = False
                    break
    return picked, fams, players, exhausted


def retrieve(bank, query_desc, exclude=None):
    """Five diverse nearest neighbours with deterministic hash ties."""
    exclude = exclude or {}
    exf = set(exclude.get("families") or ())
    exp = set(exclude.get("players") or ())
    q = (query_desc - bank.mu) / bank.sd
    dist = torch.linalg.norm(bank.desc - q[None], dim=1)
    # deterministic (rounded distance, content hash) order. Exact top-K
    # prefilter: every row with rounded distance <= the K-th smallest is
    # sorted, which is exactly the prefix of the full order; the scan falls
    # back to the full order only if it runs off that prefix.
    import numpy as np
    if bank.hash_arr is None:
        bank.hash_arr = np.array([r["content_hash"] for r in bank.rows])
    dr = np.round(dist.numpy().astype(np.float64), 9)
    if len(dr) > PREFILTER_K:
        thresh = np.partition(dr, PREFILTER_K - 1)[PREFILTER_K - 1]
        cand = np.nonzero(dr <= thresh)[0]
        prefix = cand[np.lexsort((bank.hash_arr[cand], dr[cand]))]
        picked, fams, players, exhausted = _scan(bank, prefix, exf, exp)
    else:
        exhausted = True
    if exhausted:
        order = np.lexsort((bank.hash_arr, dr))
        picked, fams, players, _ex = _scan(bank, order, exf, exp)
    if len(picked) < N_NEIGHBOURS or len(fams) < MIN_FAMS \
            or len(players) < MIN_PLAYERS:
        return {"status": "insufficient_support", "neighbours": [],
                "support_distance": None}
    ds = sorted(float(dist[i]) for i, _ in picked)
    return {"status": "ok",
            "neighbours": [{**r, "distance": round(float(dist[i]), 6)}
                           for i, r in picked],
            "support_distance": ds[len(ds) // 2]}


def bank_from_role(identity="qa-train-v1"):
    """Assemble the real bank from qa_train window records + scenes."""
    import json
    from pathlib import Path
    from qa.scene import read_scene
    from qa.telemetry_v2 import _family_chart
    from qa.train import mirror_scene
    from qa.model import target_from_window
    root = Path(__file__).resolve().parent.parent
    state = json.loads((root / "experiments/qa-v1/collect2_state.json")
                       .read_text())
    manifest = json.loads((root / "eval/corpus_manifest.json").read_text())
    records = []
    for fam, chart_rec in sorted(state["charts"].items()):
        if chart_rec.get("side") != "qa_train" \
                or not chart_rec.get("replays"):
            continue
        dat_p, info_p = _family_chart(manifest, fam,
                                      chart_rec["difficulty"])
        if dat_p is None:
            continue
        scene = read_scene(dat_p, info_p)
        if scene["scope"] is not None:
            continue
        m_scene = mirror_scene(scene)
        desc_cache = {}
        for entry in chart_rec["replays"]:
            import torch as _t
            spt = root / entry["file"].replace(".bsor", ".sanitized.pt")
            wp = spt.with_suffix(".windows_v2.json")
            if not wp.exists():
                continue
            rec = json.loads(wp.read_text())
            lh = bool(_t.load(spt)["info"].get("leftHanded")) \
                if spt.exists() else False
            sc = m_scene if lh else scene
            for wi, w in enumerate(rec["windows"]):
                if not w.get("supported") or w.get("outcome") != "good":
                    continue
                ck = (lh, w["chart_index"])
                if ck not in desc_cache:
                    desc_cache[ck] = descriptor(sc, w["chart_index"])
                y, mask = target_from_window(w)
                records.append({
                    "role": "qa_train", "family": fam,
                    "player": rec["player_token"],
                    "desc": desc_cache[ck], "targets": y, "mask": mask,
                    "traj_ref": f"{entry['file']}#{wi}",
                    "content_hash": hashlib.sha256(
                        f"{entry['sha256']}:{wi}".encode()).hexdigest()[:16]})
    return build_bank(records, identity), records
