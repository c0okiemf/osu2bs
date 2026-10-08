"""One experimental joint-event model; separate from shipped flow checkpoints.

python -m eval.joint_model prepare|train [--deadline-seconds 2700]
"""
import argparse
import hashlib
import json
import inspect
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from eval.joint_phrase import (OUT as READINESS, ROOT, PROTECTED, _atomic, _sha,
                               decode_events, verify_sources)

OUT = ROOT / "experiments/joint-phrase-v1/model"
SEED = 20260930
GAPS = (0., 1/48, 1/32, 1/24, 1/16, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2,
        2/3, 3/4, 1., 1.5, 2., 3., 4., 8.)
N_GAP = len(GAPS) + 1  # 0 = advance to window boundary; gap entries start at1
EMPTY = 108
AUDIO_DIM = 27
CONTEXT_DIM = 33 * AUDIO_DIM + 4 + 3 + 12
SNAPSHOTS = (0, 1000, 3000, 6000)


def _torch_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def literal_slots(event):
    result = [EMPTY] * 6
    for hand, notes in enumerate(event["hands"]):
        for slot, (c, l, d) in enumerate(notes):
            result[hand * 3 + slot] = (c * 3 + l) * 9 + d
    return result

def encode_gap(gap):
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("invalid gap")
    if gap == 0:
        return 1, 0.
    i = min(range(1, len(GAPS)), key=lambda j: abs(math.log(gap / GAPS[j])))
    return i + 1, math.log(gap / GAPS[i])


def decode_gap(category, residual):
    if not 1 <= category < N_GAP or not math.isfinite(residual):
        raise ValueError("invalid gap category/residual")
    return 0. if category == 1 else GAPS[category - 1] * math.exp(residual)


def actions(source):
    """Teacher actions retaining float64 exact beats plus log-gap targets."""
    events = source["events"]
    result, cursor, i = [], 0., 0
    prev_slots, prev_kind, prev_gap = [EMPTY] * 6, 2, 0.
    for start in np.arange(0., source["duration_beats"], 8.):
        end = min(float(start + 8), source["duration_beats"])
        while i < len(events) and events[i]["beat"] < end:
            event = events[i]
            delta = event["beat"] - cursor
            category, residual = encode_gap(delta)
            slots = literal_slots(event)
            counts = len(event["hands"][0]) * 4 + len(event["hands"][1])
            result.append({"cursor": cursor, "target_beat": event["beat"], "end": end,
                           "prev_slots": prev_slots, "prev_kind": prev_kind,
                           "prev_gap": prev_gap, "gap": category, "residual": residual,
                           "count": counts, "slots": slots})
            prev_slots, prev_kind, prev_gap = slots, 1, delta
            cursor, i = event["beat"], i + 1
        result.append({"cursor": cursor, "target_beat": end, "end": end,
                       "prev_slots": prev_slots, "prev_kind": prev_kind,
                       "prev_gap": prev_gap, "gap": 0, "residual": 0.,
                       "count": 0, "slots": [EMPTY] * 6})
        # REST has its own type; keep the last literal event available as context.
        prev_kind, prev_gap = 0, end - cursor
        cursor = end
    if i != len(events):
        raise ValueError("events beyond audio duration")
    return result


def action_notes(rows):
    """Check representation precision independently of tensor training targets."""
    events = []
    for row in rows:
        if row["gap"] == 0:
            continue
        b = row["cursor"] + decode_gap(row["gap"], row["residual"])
        if not math.isclose(b, row["target_beat"], rel_tol=0, abs_tol=1e-10):
            raise ValueError("gap reconstruction changed timing")
        hands = [[], []]
        for k, token in enumerate(row["slots"]):
            if token != EMPTY:
                cell, direction = divmod(token, 9)
                c, l = divmod(cell, 3)
                hands[k // 3].append((c, l, direction))
        events.append({"beat": row["target_beat"], "hands": hands})
    return decode_events(events)


def audio_features(audio_path):
    """Audio-only full-song features; no target notes, beatmap or identity input."""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    if len(y) == 0:
        raise ValueError("empty audio")
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    power = stft ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=power, sr=sr)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr,
                                        hop_length=512)[None]
    rms = librosa.feature.rms(S=stft)[0:1]
    n = min(x.shape[1] for x in (mfcc, chroma, onset, rms))
    x = np.concatenate([a[:, :n] for a in (mfcc, chroma, onset, rms)], axis=0).T
    raw_onset, raw_rms = x[:, -2].copy(), x[:, -1].copy()
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    return {"x": x.astype(np.float32), "times": np.arange(n) * 512 / sr,
            "onset": raw_onset, "rms": raw_rms, "duration_s": len(y) / sr}


def context(audio, beat, bpm, rate, njs, walls, bombs):
    """Shared train/serve context; never accesses target notes or future geometry."""
    times = (beat + np.arange(-8., 8.01, .5)) * 60 / bpm
    sampled = np.stack([np.interp(times, audio["times"], audio["x"][:, j],
                                 left=0, right=0) for j in range(AUDIO_DIM)], axis=-1)
    occupancy = np.zeros(12)
    for b, duration, col, width, y, height in walls:
        if b <= beat < b + duration:
            for c in range(col, min(col + width, 4)):
                for layer in range(3):
                    # Grid-cell geometry context, not a physical collision proof.
                    if y <= layer < y + height:
                        occupancy[c * 3 + layer] = 1
    for b, col, layer in bombs:
        if abs(b - beat) * 60 / bpm <= .1:
            occupancy[col * 3 + layer] = 1
    phase = [math.sin(2*math.pi*beat), math.cos(2*math.pi*beat),
             math.sin(math.pi*beat/2), math.cos(math.pi*beat/2)]
    return np.concatenate([sampled.reshape(-1), phase,
                           [60/bpm, rate/6, njs/20], occupancy]).astype(np.float32)


def source_rate(source):
    return sum(bool(h) for e in source["events"] for h in e["hands"]) \
        / (source["duration_beats"] * 60 / source["bpm"])


class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Embedding(109, 16)
        self.kind = nn.Embedding(3, 8)
        self.inp = nn.Linear(CONTEXT_DIM + 96 + 8 + 1, 128)
        self.rnn = nn.GRU(128, 128, num_layers=2, batch_first=True)
        self.gap = nn.Linear(128, N_GAP)
        self.residual = nn.Linear(128, N_GAP)
        self.gap_embed = nn.Embedding(N_GAP, 16)
        self.count = nn.Linear(128 + 16, 16)
        self.count_embed = nn.Embedding(16, 16)
        self.position = nn.Embedding(6, 16)
        self.slot = nn.Sequential(nn.Linear(128 + 16 * 4, 128), nn.GELU(), nn.Linear(128, 108))

    def hidden(self, ctx, prev, kind, gap, state=None):
        x = torch.cat([ctx, self.token(prev).flatten(-2), self.kind(kind),
                       torch.log1p(gap)[..., None] / 4], -1)
        return self.rnn(F.gelu(self.inp(x)), state)

    def count_logits(self, hidden, category):
        return self.count(torch.cat([hidden, self.gap_embed(category)], -1))

    def slot_logits(self, hidden, category, count, targets):
        # Exclusive prefix sum: current/future targets cannot leak into a slot.
        embeddings = self.token(targets)
        prefix = torch.cat([torch.zeros_like(embeddings[..., :1, :]),
                            embeddings[..., :-1, :]], -2).cumsum(-2)
        shape = (*hidden.shape[:-1], 6, -1)
        parts = [hidden.unsqueeze(-2).expand(shape),
                 self.gap_embed(category).unsqueeze(-2).expand(shape),
                 self.count_embed(count).unsqueeze(-2).expand(shape),
                 self.position(torch.arange(6, device=hidden.device)).expand(*hidden.shape[:-1], 6, 16),
                 prefix]
        return self.slot(torch.cat(parts, -1))

    def forward(self, batch, state=None):
        h, state = self.hidden(batch["context"], batch["prev"], batch["kind"],
                               batch["prev_gap"], state)
        return {"gap": self.gap(h), "residual": self.residual(h),
                "count": self.count_logits(h, batch["gap"]),
                "slots": self.slot_logits(h, batch["gap"], batch["count"], batch["slots"])}, state


def losses(out, batch, warmup=32):
    valid = torch.ones_like(batch["gap"], dtype=torch.bool)
    valid[:, :warmup] = False
    if not valid.any():
        raise ValueError("no target steps after warmup")
    event = valid & (batch["gap"] > 0)
    positive = valid & (batch["gap"] > 1)
    loss = F.cross_entropy(out["gap"][valid], batch["gap"][valid])
    if positive.any():
        pred = out["residual"].gather(-1, batch["gap"].unsqueeze(-1)).squeeze(-1)
        loss = loss + F.smooth_l1_loss(pred[positive], batch["residual"][positive])
    if event.any():
        loss = loss + F.cross_entropy(out["count"][event], batch["count"][event])
    slots = event[..., None] & (batch["slots"] != EMPTY)
    if slots.any():
        loss = loss + F.cross_entropy(out["slots"][slots], batch["slots"][slots])
    return loss


def freeze():
    from eval.expressive_manifest import freeze_run
    readiness = json.loads((READINESS / "run.json").read_text())
    report = json.loads((READINESS / "report.json").read_text())
    supported = {r["family"] for r in report["families"] if r["status"] == "supported"}
    train = [r for r in readiness["config"]["records"] if r["role"] == "train"
             and r["family"] in supported]
    train.sort(key=lambda r: hashlib.sha256(("joint-model-val-v1:" + r["family"]).encode()).hexdigest())
    if len(train) != 348:
        raise ValueError("readiness inventory changed")
    records = [{**r, "fit_role": "validation" if i < 32 else "train"}
               for i, r in enumerate(train)]
    records += [{**r, "fit_role": "development"} for r in readiness["config"]["records"]
                if r["role"] == "development" and r["family"] in supported]
    for r in records:
        name = r["family"].replace(":", "_") + ".json"
        part = json.loads((READINESS / "partial" / name).read_text())
        r["artifact_sha256"] = part["artifact_sha256"]
    protected = {p: _sha(ROOT / p) for p in PROTECTED}
    if protected != readiness["config"]["protected"]:
        raise ValueError("protected artifacts changed")
    import librosa
    recipe = "\n".join(inspect.getsource(fn) for fn in
                       (literal_slots, encode_gap, decode_gap, actions, action_notes,
                        audio_features, context, source_rate))
    return freeze_run(OUT, {"seed": SEED,
        "data_recipe_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
        "librosa_version": librosa.__version__, "readiness": readiness["identity"],
        "spec_sha256": _sha(ROOT / "docs/specs/2026-09-30-joint-model-pilot-design.md"),
        "records": records, "protected": protected, "gaps": GAPS,
        "training": {"updates": 6000, "batch": 16, "sequence": 128, "warmup": 32,
                     "lr": 3e-4, "weight_decay": .01, "snapshots": SNAPSHOTS}})


def prepare(deadline_seconds=2700):
    torch.set_num_threads(4)
    run = freeze()
    deadline = time.monotonic() + min(deadline_seconds, 2700)
    completed = 0
    for record in run["config"]["records"]:
        if time.monotonic() >= deadline:
            return {"status": "INCOMPLETE", "completed": completed, "required": 356}
        verify_sources(record)
        name = record["family"].replace(":", "_")
        sp = READINESS / "sources" / (name + ".json")
        if _sha(sp) != record["artifact_sha256"]:
            raise ValueError("readiness artifact changed")
        path, receipt = OUT / "data" / (name + ".pt"), OUT / "data" / (name + ".json")
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["identity"] != run["identity"] or _sha(path) != saved["sha256"]:
                raise ValueError("prepared data identity changed")
            completed += 1
            continue
        source = json.loads(sp.read_text())
        audio = audio_features(record["sources"]["audio"]["path"])
        rows = actions(source)
        if action_notes(rows) != list(map(tuple, source["notes"])):
            raise ValueError("action codec failed exact roundtrip")
        rate, njs = source_rate(source), source["authored"]["njs"]
        ctx = np.stack([context(audio, r["cursor"], source["bpm"], rate, njs,
                                source["walls"], source["bombs"]) for r in rows])
        data = {"family": record["family"], "fit_role": record["fit_role"],
                "source": source, "audio": audio, "rate": rate,
                "context": torch.from_numpy(ctx),
                "prev": torch.tensor([r["prev_slots"] for r in rows]),
                "kind": torch.tensor([r["prev_kind"] for r in rows]),
                "prev_gap": torch.tensor([r["prev_gap"] for r in rows], dtype=torch.float32),
                "gap": torch.tensor([r["gap"] for r in rows]),
                "residual": torch.tensor([r["residual"] for r in rows], dtype=torch.float32),
                "count": torch.tensor([r["count"] for r in rows]),
                "slots": torch.tensor([r["slots"] for r in rows])}
        _torch_save(path, data)
        _atomic(receipt, {"identity": run["identity"], "sha256": _sha(path),
                          "source_sha256": record["artifact_sha256"], "actions": len(rows),
                          "code_sha256": _sha(__file__)})
        completed += 1
        print(f"prepared {completed}/356 {name}: {len(rows)} actions", flush=True)
    return {"status": "DATA_READY", "families": completed}


KEYS = ("context", "prev", "kind", "prev_gap", "gap", "residual", "count", "slots")


def train(deadline_seconds=2700):
    """Single resumable fit; selection is external full-rollout validation."""
    run = freeze()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for r in run["config"]["records"]:
        if r["fit_role"] != "train":
            continue
        path = OUT / "data" / (r["family"].replace(":", "_") + ".pt")
        rec = json.loads(path.with_suffix(".json").read_text())
        if rec["identity"] != run["identity"] or _sha(path) != rec["sha256"]:
            raise ValueError("data identity mismatch before training")
        d = torch.load(path, weights_only=False)
        if len(d["gap"]) < 128:
            raise ValueError(f"frozen family too short for crop: {r['family']}")
        rows.append({k: d[k] for k in KEYS})
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = JointModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    cursor, history = 0, []
    resume_path = OUT / "train_state.pt"
    if resume_path.exists():
        s = torch.load(resume_path, map_location=device, weights_only=False)
        if s["identity"] != run["identity"] or s["code_sha256"] != _sha(__file__):
            raise ValueError("training recipe identity mismatch")
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["optimizer"])
        random.setstate(s["python_rng"])
        np.random.set_state(s["numpy_rng"])
        torch.set_rng_state(s["torch_rng"].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda_rng"]])
        cursor, history = s["update"], s["history"]
    if cursor == 0 and not (OUT / "snapshot-0.pt").exists():
        _torch_save(OUT / "snapshot-0.pt", model.cpu().state_dict())
        model.to(device)
    deadline = time.monotonic() + min(deadline_seconds, 2700)
    model.train()
    while cursor < 6000 and time.monotonic() < deadline:
        crops = []
        for _ in range(16):
            row = random.choice(rows)
            start = random.randrange(len(row["gap"]) - 128 + 1)
            crops.append({k: row[k][start:start + 128] for k in KEYS})
        batch = {k: torch.stack([c[k] for c in crops]).to(device) for k in KEYS}
        optimizer.zero_grad()
        outputs, _state = model(batch)
        loss = losses(outputs, batch)
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite training loss at {cursor}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        cursor += 1
        if cursor % 100 == 0:
            history.append({"update": cursor, "loss": float(loss)})
            print(f"update {cursor}/6000 loss {float(loss):.4f}", flush=True)
        if cursor in SNAPSHOTS:
            _torch_save(OUT / f"snapshot-{cursor}.pt", {k: v.cpu() for k, v in model.state_dict().items()})
        if cursor % 100 == 0 or cursor == 6000 or time.monotonic() >= deadline:
            _torch_save(resume_path, {"identity": run["identity"], "code_sha256": _sha(__file__),
                "update": cursor, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [], "history": history})
            _atomic(OUT / "history.json", {"update": cursor, "history": history})
    return {"status": "FIT_COMPLETE" if cursor == 6000 else "INCOMPLETE", "update": cursor}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train"])
    parser.add_argument("--deadline-seconds", type=float, default=2700)
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "train": train}[args.command](args.deadline_seconds), indent=1))
