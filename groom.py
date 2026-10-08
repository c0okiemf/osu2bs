"""Learned mapper: rhythm TCN + autoregressive flow model, trained on
hand-mapped BS maps.

Rhythm (Groomer): which 1/4-beat steps each hand swings on — denoising task,
so it inserts follow-through notes the osu stream lacks.
Flow (Flow): each note's direction/column/layer, conditioned on both hands'
previous notes (swing arcs are ~1st-order Markov per hand — a per-step
independent predictor provably collapses to the marginal; this doesn't).
Parity machine masks direction choices, so output is legal by construction.

  .venv/bin/python groom.py check   # dataset + invariant sanity (~10s)
  .venv/bin/python groom.py train   # ~3 min, writes groom.pt + flow.pt
convert.py auto-uses the models once groom.pt exists.
"""
import json
import math
import random
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import motion
from parity import (HandParity, DIR_VEC, FOREHAND, BACKHAND, LATERAL,
                    ang_dist, family, is_hammer, violations)

# two-tier corpus: the big scrape teaches general conversion; the hand-
# approved maps are the style/pattern/density authority (weighted training,
# critic positives, and the TARGET_BAND difficulty calibration)
APPROVED_DIRS = [Path.home() / "app/beat-saber-map-gen/input/bytrius",
                 Path.home() / "app/beat-saber-map-gen/input/input"]
MAPS_DIRS = APPROVED_DIRS + [Path(__file__).parent / "beatsaver"]
APPROVED_W = 4.0  # ~175 approved x4 vs ~700 general: half the gradient mass
# eval holdout: the reference gems the pipeline is tuned to reproduce — kept
# OUT of all training (rhythm/flow/critic) so the comparison is honest.
# Rap God (19909, 46d4), Reality Check (25f, 3741), Spaceman (24e5e).
HOLDOUT_IDS = {"19909", "46d4", "25f", "3741", "24e5e"}


def held_out(d):
    return d.name.split(" ")[0].lower() in HOLDOUT_IDS
MODEL_PT = Path(__file__).parent / "groom.pt"
FLOW_PT = Path(__file__).parent / "flow.pt"
STEPS_PER_BEAT = 4  # 1/4 grid; go 1/8 if fast streams come out mushy
CROP = 512          # training crop, steps (128 beats)
POS_WEIGHT = 3.0    # ~15% of steps carry a note
WALL_POS_WEIGHT = 15.0  # walls cover only ~3.5% of steps; without this the
                        # model learns "no wall anywhere" (wall f1 ~0)
OFFGRID_SKIP = 0.10  # skip swing/triplet maps (>10% notes off the 1/4 grid)
MIN_GAP = 2         # per-hand minimum gap, steps (1/2 beat)
THIN_MS = 90        # legacy STREAM_MS port — steps shorter than this
                    # never run back-to-back (2:1 thin), so full streams above
                    # ~167bpm stay under the peak-nps cap
NPS_WIN = 3500      # rolling window (ms), mirrors convert.check's density cap
NPS_WIN_MAX = 38    # 11 swings/s * 3.5s floored — decode cannot exceed the cap
                    # (thinning alone missed double-streams: 2 swings/2 steps)
TEMP = 0.9          # flow sampling temperature; 0.7 locked the AR transformer
                    # into single-mode songs (all-flip180 or all-chains)
TEMPS = (0.85, 1.0, 1.15)  # best-of-N ladder: peaked seeds keep stream
                    # styles (Rap God), hot seeds escape the vertical
                    # attractor (spaceman); gates + critic pick per song
# wrist physics at speed: human fast transitions are 99% at >=135deg from the
# previous cut (180 flip or the smooth 45-off arc); sharper deltas are jank
FAST_STEPS = 2      # per-hand gap (steps) at or below which the rule applies
FAST_MIN_ANG = 135
CHAIN_BOOST = 2.5   # decode boost on multi-note swings, governed below:
CHAIN_COOL = 32     # boost fades with chains in the last N events, so chains
                    # arrive as occasional accents — AR feedback snowballed a
                    # constant boost into 24% dots (numb) or 0% without it
# pattern ergonomics (decode-time logit penalties, calibrated vs human maps):
SH_UP, SH_DOWN = {0, 4, 5}, {1, 6, 7}
SH_STEPS = 4    # same-hand gap (steps) counting as "in combo" (1 beat)
SH_PEN = 4.0    # up@bottom-lane -> down@top-lane same hand: shoulder killer;
                # measured 61/map generated vs 1/map human — rare accent only
LAT_STEPS = 4
LAT_PEN = 0.2   # laterals mid-stream ('>^'): parity-neutral escape hatch the
                # decoder leans on (0.1 + the run breaker collapsed numb into
                # 73% left-right spam; the axis breaker below handles runs)
VRUN_MAX = 5    # axis-flip runs longer than this consume the run budget
VRUN_PEN = 2.0  # full penalty once the budget is blown
# self-regulating: long runs are free while they stay under RUN_SHARE_LO of
# recent transitions (structured streams, Rap God style: ~39%), and the
# penalty ramps to full by RUN_SHARE_HI (mode collapse: spaceman decoded 80%)
RUN_SHARE_LO, RUN_SHARE_HI, RUN_WIN = 0.35, 0.55, 64
VRUN_HARD = 8   # absolute cap, history-free: the budget can't govern the
                # song's FIRST section (empty history), and replays clone it
# density: quality over quantity — model trained to also DELETE onsets
# (insertion noise in corruption), decode keeps only confident notes
PRES_T = 0.65       # presence threshold (0.5 kept too many marginal notes)
KEEP_MIN = 0.40     # input onsets below this get dropped (phrasing deletions;
                    # 0.35 kept too much of the over-dense osu input on
                    # sparse gems — RCTTS decoded 4.8 swings/s vs human 3.0)
# no completely dead sections: sprinkle sparse notes into long empty zones —
# but 1-4 bar rests are PHRASING (gem maps rest in 22% of 4-beat windows,
# generated had 3%): only zones past 16 beats get sprinkled, and sparsely
SPRINKLE_GAP = 64    # zones emptier than 16 beats get sprinkles
SPRINKLE_EVERY = 16  # candidate spacing at neutral energy (steps)
REST_PRES = 0.15  # breathers are LEARNED: a 4-beat window whose mean model
                  # presence (max over hands) is below this rests fully —
                  # force-keep/sprinkles may not override. Replaces energy-
                  # carved rests, which landed randomly; tuned against the
                  # held-out gems (HOLDOUT_IDS)
SPRINKLE_MIN_E = -0.5  # onset-strength z gate: actual silence stays silent
DENSITY_W = 0.05    # energy tilts the presence/keep thresholds: loud sections
                    # need less model confidence to place a note, quiet more.
                    # 0.04 measured too flat (density CV 0.36 vs 0.69 in the
                    # reference gems); 0.10 + carved rests over-emptied
                    # variable songs (neckhurts 67% empty windows). REST_Q
                    # owns the breathers, the tilt owns section contrast.
# cross-song difficulty: clamp each song's scheduled swing rate into the
# hand-approved corpus IQR (170 maps in ../beat-saber-map-gen: p25 4.5,
# median 5.2, p75 5.9 swings/s) via a uniform threshold offset — kills the
# sleeper->Camellia spread while section dynamics keep the model's shape
TARGET_BAND = (3.5, 6.0)  # ceiling = Rap God, the hardest approved gem;
                          # 6.5 let mild songs (rizzstag) out-camellia it

# flow transformer: phrase-level context replaces the per-note MLP and all
# the hand-written section-style logit biases it needed
CTX = 256           # event context window (~30-60 beats of both hands)
D_MODEL, N_LAYER, N_HEAD = 192, 4, 4
NTOK = 64           # token feature size (prev event attrs + current context)
ENERGY_BINS = [-0.5, 0.0, 0.5, 1.0]  # section-RMS z -> 5 buckets
# positions are predicted as DISPLACEMENT from the hand's previous note —
# the inductive bias for smooth paths (a sweep = "repeat delta +1")
DCOL, DLAY = 7, 5   # delta classes: col -3..+3, layer -2..+2
ANCHOR = {0: (1, 0), 1: (2, 0)}  # a hand's start position before its 1st note
N_DECODES = 6       # best-of-N: decodes sampled, model-likelihood picks one
ROT_BAND = (0.15, 0.55)  # candidates outside this rotation share lose the vote

MIRROR_DIR = [0, 1, 3, 2, 5, 4, 7, 6, 8]
VFLIP_DIR = [1, 0, 2, 3, 6, 7, 4, 5, 8]  # vertical mirror: swaps fore/back parity
MAX_CHAIN = 3  # notes cut in a single swing (head + up to 2 dot followers)
WALL_MIN_STEPS = 8   # decoded wall runs shorter than 2 beats are dropped
WALL_MAX_GAP = 3     # sub-beat holes in a predicted wall run get bridged
WALL_THRESH = 0.3    # wall channel is under-confident (heavy pos_weight);
                     # 0.3 matches corpus wall coverage, run filter cuts noise

# structural repeats (chorus reuse): match 8-beat blocks into runs >= 16 beats
BLOCK = 32          # steps per block (8 beats)
MAX_HAM = 4         # per-block rhythm mismatch tolerance, steps
MIN_RUN = 2         # blocks; shorter matches are motifs, not sections


# ---------------- rhythm model ----------------

N_AUDIO = 4   # full-mix onsets, percussive onsets, local rms, section rms
IN_CH = 9 + N_AUDIO + 1  # +1: difficulty conditioning (swings/s / 6)
DIFF_NAMES = ("Easy", "Normal", "Hard", "Expert", "ExpertPlus")
# Full-mix and percussive (HPSS) onsets go in as SEPARATE channels; training
# on human maps decides which to follow where (drum sections vs vocal/solo
# focus). Replaces the old PERC_ONSETS hard blend that locked notes to drums.
_FEAT_CACHE = Path(__file__).parent / "feats_cache4.pt"


def audio_features(path, step_times_ms):
    """Per-step audio energy: the signal that lets the rhythm model learn
    section dynamics (half-time verses, dense drops) from human maps."""
    import numpy as np
    import librosa
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    yp = librosa.effects.percussive(y, margin=3.0)
    operc = librosa.onset.onset_strength(y=yp, sr=sr, hop_length=hop)
    n = min(len(oenv), len(operc))
    # perc channel keeps the old max-blend form so pre-split caches merge
    operc = np.maximum(operc[:n], 0.6 * oenv[:n])
    oenv = oenv[:n]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    idx = np.clip((np.asarray(step_times_ms) / 1000 * sr / hop).astype(int),
                  0, min(n, len(rms)) - 1)
    o, p, r = oenv[idx], operc[idx], rms[idx]
    sec = np.convolve(r, np.ones(33) / 33, mode="same")  # ~8-beat section energy
    feats = np.stack([o, p, r, sec], axis=1)
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-6)
    return torch.tensor(feats, dtype=torch.float32)


_FEATS = {}  # in-process view of _FEAT_CACHE; reloading 40MB per song is real minutes
FEAT_VERSION = 2  # bump when audio_features' math changes -> old cache ignored


def _feat_key(path, step_times_ms):
    """Content-addressed cache key (Packet C item d): audio bytes + the exact
    time grid + feature version. Replaces the old str(path)+length key, which
    silently reused stale features after a BPM/offset/audio/normalization
    change (the grid is what audio_features indexes into)."""
    import hashlib
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    g = step_times_ms if isinstance(step_times_ms, (list, tuple)) \
        else list(step_times_ms)
    h.update(repr([round(float(x), 3) for x in g]).encode())  # exact grid
    return f"v{FEAT_VERSION}:{h.hexdigest()[:32]}"


def cached_audio_features(path, step_times_ms):
    if not _FEATS and _FEAT_CACHE.exists():
        _FEATS.update(torch.load(_FEAT_CACHE))
    key = _feat_key(path, step_times_ms)
    if key not in _FEATS or len(_FEATS[key]) != len(step_times_ms):
        _FEATS[key] = audio_features(path, step_times_ms)
        torch.save(_FEATS, _FEAT_CACHE)
    return _FEATS[key]


def make_input(merged, afeat=None, drate=0.0, pos=None, bar=None):
    """merged: bool [T] single-stream note presence -> [T, IN_CH] features.
    drate: difficulty conditioning — the map's swings/s (training: measured
    per difficulty; decode: the requested tier's rate), /6 = the pinned
    Expert+ ceiling, so the axis is continuous and open-ended upward.
    pos/bar: per-step beat phase (Packet C, from the TimeGrid). Default
    s%4 / (s//4)%4 — identical to a single-tempo grid, so training and
    single-BPM inference are unchanged; a tempo-changing song gets correct
    LOCAL phase instead of a globally drifting one."""
    T = len(merged)
    steps = torch.arange(T)
    if afeat is None:
        afeat = torch.zeros(T, N_AUDIO)
    pos = steps % 4 if pos is None else torch.as_tensor(pos[:T])
    bar = (steps // 4) % 4 if bar is None else torch.as_tensor(bar[:T])
    return torch.cat([merged[:, None].float(),
                      F.one_hot(pos, 4).float(),                # pos in beat
                      F.one_hot(bar, 4).float(),                # beat in bar
                      afeat[:T],
                      torch.full((T, 1), drate / 6.0)], dim=1)


class Groomer(nn.Module):
    """Tiny non-causal TCN, ~75k params, receptive field ~±16 beats."""

    def __init__(self, ch=64, layers=6):
        super().__init__()
        self.inp = nn.Conv1d(IN_CH, ch, 1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(layers))
        self.out = nn.Conv1d(ch, 4, 1)

    def forward(self, x):  # [B, T, IN_CH] -> [B, T, 4]: noteL, noteR, wallL, wallR
        h = self.inp(x.transpose(1, 2))
        for c in self.blocks:
            h = h + F.gelu(c(h))
        return self.out(h).transpose(1, 2)


def corrupt(inp, drop, gen=None, add=0.1):
    """Drop AND insert merged-stream bits. Drops teach the model to insert
    follow-throughs; spurious insertions teach it to SKIP onsets — without
    them it maps everything and maps come out wall-to-wall dense."""
    inp = inp.clone()
    keep = torch.rand(len(inp), generator=gen) >= drop
    spur = torch.rand(len(inp), generator=gen) < add
    inp[:, 0] = ((inp[:, 0] > 0) & keep | spur).float()
    return inp


# ---------------- flow model ----------------

def _bucket(z):
    return sum(z > b for b in ENERGY_BINS)


def token_vec(prev, hand, s, dt_same, dt_any, is_double, wl, wr, energy_z,
              pos=None, bar=None):
    """One event token: full attributes of the PREVIOUS event + the current
    event's known context (its outputs are what the model predicts).
    pos/bar: local beat phase (Packet C); default s%4 / (s//4)%4 so a
    single-tempo grid is identical to before."""
    def oh(v, n):
        return F.one_hot(torch.tensor(v), n).float()
    ph, pd, pc, pl, pk = prev if prev else (2, 9, 4, 3, 0)
    pk_idx = 3 if prev is None else min(pk, MAX_CHAIN) - 1
    return torch.cat([
        oh(ph, 3), oh(pd, 10), oh(pc, 5), oh(pl, 4), oh(pk_idx, 4),
        oh(hand, 2), oh(dt_same, 10), oh(dt_any, 10),
        oh(s % 4 if pos is None else pos, 4),
        oh((s // 4) % 4 if bar is None else bar, 4), oh(_bucket(energy_z), 5),
        torch.tensor([float(is_double), float(wl), float(wr)])])


def flow_time_features(grid, step, previous_same, previous_any):
    """Shared train/decode timing inputs; missing history has explicit flags.

    Log-scaled milliseconds retain long gaps instead of clipping them to two
    beats. The last duration is four local grid steps, independent of geometry.
    """
    gaps = [grid.gap_ms(p, step) if p is not None else 0.0
            for p in (previous_same, previous_any)]
    durations = gaps + [4 * grid.local_dt(step)]
    if any(not math.isfinite(v) or v < 0 for v in durations) or durations[2] <= 0:
        raise ValueError("invalid Flow timing context")
    return [math.log1p(v / 250.0) for v in durations] + [
        float(previous_same is not None), float(previous_any is not None)]


class Flow(nn.Module):
    """Small causal transformer over the note-event sequence: phrase-level
    context, so patterns (rotations, chains, spacing, section character) are
    generated by the model rather than by decode-time biases."""

    def __init__(self, width=D_MODEL, timing=False):
        super().__init__()
        self.width, self.timing = width, timing
        self.inp = nn.Linear(NTOK + (5 if timing else 0), width)
        self.pos = nn.Embedding(CTX, width)
        layer = nn.TransformerEncoderLayer(
            width, N_HEAD, 4 * width, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, N_LAYER)
        self.dir_head = nn.Linear(width, 9)
        self.col_head = nn.Linear(width + 9, DCOL)          # delta col
        self.lay_head = nn.Linear(width + 9 + DCOL, DLAY)   # delta layer
        self.chain_head = nn.Linear(width + 9, MAX_CHAIN)

    def hidden(self, x):  # [B, N, NTOK] -> [B, N, D_MODEL], causal
        n = x.shape[1]
        h = self.inp(x) + self.pos(torch.arange(n, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(n).to(x.device)
        return self.tr(h, mask=mask, is_causal=True)

    def forward(self, x, dir_oh, dcol_oh):  # teacher-forced training
        h = self.hidden(x)
        return (self.dir_head(h),
                self.col_head(torch.cat([h, dir_oh], -1)),
                self.lay_head(torch.cat([h, dir_oh, dcol_oh], -1)),
                self.chain_head(torch.cat([h, dir_oh], -1)))


def events_to_xy(events, wallL, wallR, energy=None, *, grid=None):
    """events: [(step, hand, dir, col, layer, chain)] sorted by (step, hand)
    -> token matrix X [N, NTOK], target matrix Y [N, 4]:
    (dir, dcol+3, dlay+2, k-1), deltas from the hand's previous position
    (ANCHOR before its first note). Walks exactly like the decoder does."""
    lastn, last_any, prev = {0: None, 1: None}, None, None
    T = len(wallL)
    categories, flags, ys, timing_rows = [], [], [], []
    for i, (s, h, d, c, l, chain) in enumerate(events):
        dbl = any(e[0] == s and e[1] != h
                  for e in events[max(0, i - 1):i + 2])
        si = min(s, T - 1)
        dt_same = min(s - lastn[h][0], 8) if lastn[h] else 9
        dt_any = min(s - last_any, 8) if last_any is not None else 9
        ez = float(energy[si]) if energy is not None else 0.0
        ph, pd, pc, pl, pk = prev if prev else (2, 9, 4, 3, 0)
        categories.append((ph, pd, pc, pl, 3 if prev is None else min(pk, MAX_CHAIN)-1,
                           h, dt_same, dt_any, s % 4, (s // 4) % 4, _bucket(ez)))
        flags.append((float(dbl), float(bool(wallL[si])), float(bool(wallR[si]))))
        if grid is not None:
            timing_rows.append(flow_time_features(
                grid, s, lastn[h][0] if lastn[h] else None, last_any))
        ac, al = lastn[h][2:4] if lastn[h] else ANCHOR[h]
        ys.append((d, c - ac + 3, l - al + 2, chain - 1))
        prev = (h, d, c, l, chain)
        lastn[h], last_any = (s, d, c, l), s
    # Same token_vec encoding, batched to avoid millions of tiny one-hot calls
    # when preparing thousands of human charts.
    cats = torch.tensor(categories)
    widths = (3, 10, 5, 4, 4, 2, 10, 10, 4, 4, 5)
    x = torch.cat([F.one_hot(cats[:, i], width).float()
                   for i, width in enumerate(widths)] + [torch.tensor(flags)], dim=1)
    if grid is not None:
        x = torch.cat([x, torch.tensor(timing_rows)], dim=1)
    return x, torch.tensor(ys)


def mirror_events(events):
    return sorted((s, 1 - h, MIRROR_DIR[d], 3 - c, l, k)
                  for s, h, d, c, l, k in events)


# ---------------- dataset ----------------

def grid_shift(step_times):
    """Constant grid phase (in steps) a map was saved with — maps often bake
    in an audio offset. Shared by load_map_all and the eligibility census so
    the off-grid test matches the loader exactly (review v2-3)."""
    ang = [2 * math.pi * (t % 1) for t in step_times]
    return math.atan2(sum(map(math.sin, ang)),
                      sum(map(math.cos, ang))) / (2 * math.pi)


def offgrid_fraction(step_times):
    """Fraction of step times more than 0.2 off the nearest 1/4 line (apply
    grid_shift first). >OFFGRID_SKIP means swing/triplet — skipped by the
    1/4-grid loader."""
    if not step_times:
        return 0.0
    return sum(abs(t - round(t)) > 0.2 for t in step_times) / len(step_times)


class FeatureMiss(Exception):
    """A read-only feature provider (parallel loader worker) signalling that a
    needed audio-feature key is absent, so the PARENT computes it serially. Never
    raised on the default disk-writing path."""

    def __init__(self, path, step_times):
        super().__init__(f"feature miss: {path}")
        self.path = str(path)
        self.step_times = list(step_times)


def load_map_all(d, *, feature_provider=None, with_timing=False):
    """One hand-mapped dir -> {difficulty: (inp, pres, events, wallL, wallR)}
    for every usable Standard difficulty; audio features computed once on the
    longest difficulty's grid and sliced per diff. inp carries the diff's own
    measured swings/s as the conditioning channel.
    feature_provider overrides cached_audio_features (parallel loader passes a
    read-only provider that raises FeatureMiss instead of computing/writing)."""
    info_p = next((p for p in d.iterdir() if p.name.lower() == "info.dat"), None)
    if not info_p:
        return {}
    try:
        info = json.loads(info_p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    files = {}
    for s in info.get("_difficultyBeatmapSets", []):
        if s.get("_beatmapCharacteristicName", "Standard") != "Standard":
            continue
        for dm in s.get("_difficultyBeatmaps", []):
            if dm.get("_difficulty") in DIFF_NAMES:
                files[dm["_difficulty"]] = dm.get("_beatmapFilename")
    for dm in info.get("difficultyBeatmaps", []):  # v4 info
        if (dm.get("characteristic") == "Standard"
                and dm.get("difficulty") in DIFF_NAMES):
            files[dm["difficulty"]] = dm.get("beatmapDataFilename")
    if "ExpertPlus" not in files or not (d / files["ExpertPlus"]).exists():
        # info.dat often names ExpertPlusStandard.dat while the file is ExpertPlus.dat
        p = next((q for q in d.iterdir() if q.suffix.lower() == ".dat"
                  and q.name.lower().startswith("expertplus")), None)
        if p is not None:
            files["ExpertPlus"] = p.name
    bpm = info.get("_beatsPerMinute") or info.get("audio", {}).get("bpm", 120)
    parsed = {}
    for name, fname in files.items():
        p = d / fname if fname else None
        if p is None or not p.exists():
            continue
        try:
            dat = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        notes = dat.get("_notes") or [  # v3 beatmaps use colorNotes
            {"_time": n.get("b", 0), "_type": n.get("c"),
             "_lineIndex": n.get("x", -1), "_lineLayer": n.get("y", -1),
             "_cutDirection": n.get("d", -1)} for n in dat.get("colorNotes", [])]
        notes = [n for n in notes if n.get("_type") in (0, 1)
                 and 0 <= n.get("_lineIndex", -1) <= 3
                 and 0 <= n.get("_lineLayer", -1) <= 2
                 and 0 <= n.get("_cutDirection", -1) <= 8]
        if len(notes) < 100:
            continue
        times = [n["_time"] * STEPS_PER_BEAT for n in notes]
        shift = grid_shift(times)  # remove baked-in constant audio offset
        times = [t - shift for t in times]
        if offgrid_fraction(times) > OFFGRID_SKIP:
            continue
        parsed[name] = (dat, notes, times, shift)
    if not parsed:
        return {}
    song = info.get("_songFilename") or info.get("audio", {}).get("songFilename", "")
    song_p = d / song if song and (d / song).exists() else next(
        (q for q in d.iterdir() if q.suffix.lower() in (".egg", ".ogg")), None)
    if song_p is None:
        return {}
    Tmax = max(int(max(times)) + 2 for _, _, times, _ in parsed.values())
    shift0 = (parsed.get("ExpertPlus") or next(iter(parsed.values())))[3]
    step_times = [(s + shift0) / STEPS_PER_BEAT * 60000 / bpm
                  for s in range(Tmax)]
    provider = cached_audio_features if feature_provider is None else feature_provider
    try:
        afeat = provider(song_p, step_times)
    except FeatureMiss:
        raise                                  # parent computes it serially
    except Exception as e:
        print(f"  audio failed for {d.name[:40]}: {e}")
        return {}
    out = {}
    for name, (dat, notes, times, shift) in parsed.items():
        T = int(max(times)) + 2
        pres = torch.zeros(T, 2)
        seen, cnt = {}, {}
        for n, t in zip(notes, times):
            s, h = min(round(t), T - 1), n["_type"]
            pres[s, h] = 1
            cnt[(s, h)] = cnt.get((s, h), 0) + 1
            seen.setdefault((s, h), (s, h, n["_cutDirection"],
                                     n["_lineIndex"], n["_lineLayer"]))
        # same-step same-hand clusters = one swing through several notes
        events = sorted(e + (min(cnt[e[:2]], MAX_CHAIN),)
                        for e in seen.values())
        span_s = (events[-1][0] - events[0][0]) / STEPS_PER_BEAT * 60 / bpm
        if span_s < 30:
            continue
        rate = min(12.0, len(events) / span_s)  # the conditioning label
        # lean-wall targets: full-height side walls only (crouch walls
        # excluded, so the model can never learn to emit them)
        wallL = torch.zeros(T, dtype=torch.bool)
        wallR = torch.zeros(T, dtype=torch.bool)
        obst = dat.get("_obstacles") or dat.get("obstacles") or []
        for o in obst:
            if "_type" in o:                       # v2
                if o["_type"] != 0:
                    continue
                b, dur, x, w = (o["_time"], o["_duration"], o["_lineIndex"],
                                o.get("_width", 1))
            else:                                  # v3
                if o.get("y", 0) != 0 or o.get("h", 5) < 3:
                    continue
                b, dur, x, w = (o.get("b", 0), o.get("d", 0), o.get("x", 0),
                                o.get("w", 1))
            if dur <= 0:
                continue
            side = None
            if x + w <= 2:
                side = wallL
            elif x >= 2:
                side = wallR
            if side is not None:
                s0 = max(0, round(b * STEPS_PER_BEAT - shift))
                s1 = min(T, round((b + dur) * STEPS_PER_BEAT - shift))
                side[s0:s1] = True
        out[name] = (make_input(pres.any(dim=1), afeat, rate), pres, events,
                     wallL, wallR)
        if with_timing:
            from timing import TimeGrid
            out[name] += (TimeGrid.from_beatmap(bpm, dat, shift, T),)
    return out


def load_map(d):
    """ExpertPlus sample only — the critic/eval path."""
    return load_map_all(d).get("ExpertPlus")


def load_dataset(mapdirs, split=None, *, map_executor=None):
    """[(inp, pres, events, wallL, wallR, sample_weight)] per usable map.
    split: when given ('train'), restrict to that corpus-manifest split so a
    learned artifact never sees dev/test (phase 5A). Default None keeps the
    legacy all-dirs-minus-held-out behavior. Packet D training passes
    split='train'.
    map_executor(dirs)->[samples-dict...] in the SAME order optionally loads maps
    concurrently; None keeps the serial path. Held-out filtering and all weight
    logic stay here so the dataset is byte-identical to serial."""
    approved = {Path(p).resolve() for p in APPROVED_DIRS}
    if split is not None:
        # family-representative dirs only: duplicate folders must not multiply
        # a song's gradient mass, and split_dirs fails closed on a stale/absent
        # manifest (review findings 5, 6)
        import eval.corpus as corpus
        dirs = [Path(p) for p in corpus.train_families_rep(split)]
    else:
        dirs = [d for mapdir in mapdirs if Path(mapdir).exists()
                for d in sorted(Path(mapdir).iterdir()) if d.is_dir()]
    data, skipped = [], 0
    eligible = [d for d in dirs if not held_out(d)]
    results = (list(map_executor(eligible)) if map_executor is not None
               else [load_map_all(d) for d in eligible])
    for d, samples in zip(eligible, results):
        w = APPROVED_W if d.parent.resolve() in approved else 1.0
        if not samples:
            skipped += 1
            continue
        # family/difficulty budget: a song's TOTAL sampling mass is w
        # regardless of how many difficulty charts it has, so a 5-tier map
        # doesn't out-influence a 1-tier map (review v2-2). split=None keeps
        # the legacy per-difficulty weighting for reproducing old checkpoints.
        # The 7th element is the family key so the trainer can RE-normalize
        # after its CROP/CTX length filter drops some charts (review v2 final).
        if split is not None:
            pw = w / len(samples)             # family-normalized; [6]=family key
            for m in samples.values():
                data.append(m + (pw, str(d)))
        else:
            for m in samples.values():        # legacy: per-chart weight, no key
                data.append(m + (w,))
    napp = sum(1 for m in data if m[5] >= 1)
    print(f"dataset: {len(data)} difficulty-maps ({skipped} dirs skipped"
          + (f", split={split}, family-normalized weights" if split else "")
          + ")")
    return data


def _renorm_by_family(samples, base_by_fam):
    """After a length filter, rescale surviving samples so each family again
    sums to its base weight (review v2 final: normalize AFTER CROP/CTX, not
    only at load). samples: rows whose [5]=weight, [6]=family key."""
    from collections import defaultdict
    surv = defaultdict(float)
    for m in samples:
        surv[m[6]] += m[5]
    return [m[:5] + (m[5] * base_by_fam[m[6]] / surv[m[6]],) + m[6:]
            for m in samples]


# ---------------- training ----------------

def _targets(pres, wallL, wallR):
    return torch.cat([pres, wallL[:, None].float(), wallR[:, None].float()], 1)


@torch.no_grad()
def eval_rhythm(model, data, dev, lossf):
    model.eval()
    tot = 0.0
    tp = torch.zeros(2)
    fp = torch.zeros(2)
    fn = torch.zeros(2)
    gen = torch.Generator().manual_seed(0)
    for inp, pres, _, wl, wr, *_ in data:
        lg = model(corrupt(inp, 0.25, gen)[None].to(dev))[0]
        t = _targets(pres, wl, wr).to(dev)
        tot += lossf(lg, t).item()
        pred, tb = (lg.sigmoid() > 0.5).cpu(), t.bool().cpu()
        for i, sl in enumerate((slice(0, 2), slice(2, 4))):  # notes, walls
            tp[i] += (pred[:, sl] & tb[:, sl]).sum()
            fp[i] += (pred[:, sl] & ~tb[:, sl]).sum()
            fn[i] += (~pred[:, sl] & tb[:, sl]).sum()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return tot / len(data), float(f1[0]), float(f1[1])


@torch.no_grad()
def eval_flow(model, val, dev, *, window="first"):
    """Mean 4-way CE + direction accuracy on the flow val windows, built exactly
    as train_flow does (both hands + L/R mirror, first CTX steps). Used to verify
    a reloaded flow checkpoint reproduces its training val loss."""
    seqs = []
    for m in val:
        inp, _, events, wl, wr, w = m[:6]
        energy = inp[:, 12]
        for ev, a, b in ((events, wl, wr), (mirror_events(events), wr, wl)):
            x, y = events_to_xy(ev, a, b, energy,
                                grid=m[7] if getattr(model, "timing", False) else None)
            seqs.append((x, y))
    if window not in ("first", "middle", "end"):
        raise ValueError("unknown validation window")
    vas = []
    for x, y in seqs:
        if len(x) < CTX:
            continue
        start = {"first": 0, "middle": (len(x)-CTX)//2, "end": len(x)-CTX}[window]
        vas.append((x[start:start+CTX], y[start:start+CTX]))
    xva = torch.stack([x for x, _ in vas]).to(dev)
    yva = torch.stack([y for _, y in vas]).to(dev)
    model.eval()
    d_lg, c_lg, l_lg, k_lg = model(xva, F.one_hot(yva[..., 0], 9).float(),
                                   F.one_hot(yva[..., 1], DCOL).float())
    loss = (F.cross_entropy(d_lg.reshape(-1, 9), yva[..., 0].reshape(-1))
            + F.cross_entropy(c_lg.reshape(-1, DCOL), yva[..., 1].reshape(-1))
            + F.cross_entropy(l_lg.reshape(-1, DLAY), yva[..., 2].reshape(-1))
            + F.cross_entropy(k_lg.reshape(-1, MAX_CHAIN), yva[..., 3].reshape(-1)))
    acc = (d_lg.argmax(-1) == yva[..., 0]).float().mean().item()
    return loss.item(), acc


def train_rhythm(tr, val, dev, rng, *, out_path=None, history_path=None,
                 max_epochs=200, steps_per_epoch=30):
    """Train the rhythm Groomer. Defaults reproduce the shipped behavior (save
    to groom.pt, 200x30 updates). The clean-rhythm experiment passes an explicit
    out_path (never groom.pt) + history_path; all training math is unchanged.
    Returns the run history dict."""
    out = Path(out_path) if out_path is not None else MODEL_PT
    model = Groomer().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [POS_WEIGHT, POS_WEIGHT, WALL_POS_WEIGHT, WALL_POS_WEIGHT], device=dev))
    split_mode = bool(tr) and len(tr[0]) > 6  # 7-tuple => family key present
    if split_mode:
        from collections import defaultdict
        base = defaultdict(float)      # full family mass before the CROP filter
        for m in tr:
            base[m[6]] += m[5]
    tr = [m for m in tr if len(m[0]) > CROP]
    if split_mode:
        tr = _renorm_by_family(tr, base)  # constant family mass post-filter
    wts = [m[5] for m in tr]
    init_vloss, init_f1n, init_f1w = eval_rhythm(model, val, dev, lossf)
    history = {"initial_val_loss": init_vloss, "epochs": [], "updates": 0,
               "best_epoch": None, "n_train_after_crop": len(tr)}
    best, best_state, best_epoch, patience = 1e9, None, None, 0
    for epoch in range(max_epochs):
        model.train()
        ep_losses = []
        for _ in range(steps_per_epoch):
            xs, ys = [], []
            for inp, pres, _, wl, wr, *_ in rng.choices(tr, weights=wts, k=16):
                i = rng.randrange(len(inp) - CROP)
                x = corrupt(inp[i:i + CROP], rng.uniform(0.1, 0.4))
                y = _targets(pres, wl, wr)[i:i + CROP]
                if rng.random() < 0.5:
                    y = y[:, [1, 0, 3, 2]]  # L/R mirror: swap hands and wall sides
                xs.append(x)
                ys.append(y)
            loss = lossf(model(torch.stack(xs).to(dev)), torch.stack(ys).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
            history["updates"] += 1
        vloss, f1n, f1w = eval_rhythm(model, val, dev, lossf)
        history["epochs"].append({"epoch": epoch,
                                  "train_loss": sum(ep_losses) / len(ep_losses),
                                  "val_loss": vloss, "note_f1": f1n, "wall_f1": f1w})
        if vloss < best - 1e-4:
            best, patience, best_epoch, best_state = vloss, 0, epoch, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 10 == 0 or patience > 25:
            print(f"[rhythm] epoch {epoch}: val loss {vloss:.4f} "
                  f"note f1 {f1n:.3f} wall f1 {f1w:.3f}")
        if patience > 25:
            break
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    if history_path is not None:
        Path(history_path).write_text(json.dumps(history, indent=1))
    print(f"saved {out} (best val loss {best:.4f})")
    return history


def train_flow(tr, val, dev, rng, *, out_path=None, history_path=None,
               max_epochs=600, steps_per_epoch=30, model_kwargs=None,
               snapshot_updates=(), early_stop_patience=30):
    """Train the Flow geometry model. Defaults reproduce the shipped behavior
    (save to flow.pt, 600x30 updates). The clean-flow experiment passes an
    explicit out_path (never flow.pt) + history_path; the training math and RNG
    stream are unchanged. Returns the run history dict."""
    if early_stop_patience is not None and early_stop_patience < 0:
        raise ValueError("early_stop_patience must be nonnegative or None")
    split_mode = bool(tr) and len(tr[0]) > 6  # family key at [6]

    def build(maps):
        seqs = []
        for m in maps:
            inp, _, events, wl, wr, w = m[:6]
            fam = m[6] if len(m) > 6 else id(m)  # unique key if legacy
            energy = inp[:, 12]  # section-RMS z from make_input layout
            for ev, a, b in ((events, wl, wr),
                             (mirror_events(events), wr, wl)):
                x, y = events_to_xy(ev, a, b, energy,
                    grid=m[7] if (model_kwargs or {}).get("timing", False) else None)
                seqs.append((x, y, w, fam))       # [3] = family key
        return seqs

    all_tr = build(tr)
    trs = [s for s in all_tr if len(s[0]) > CTX]
    if split_mode:  # constant family mass after the CTX filter (review v2)
        from collections import defaultdict
        base = defaultdict(float)
        for s in all_tr:
            base[s[3]] += s[2]
        surv = defaultdict(float)
        for s in trs:
            surv[s[3]] += s[2]
        trs = [(x, y, w * base[f] / surv[f], f) for x, y, w, f in trs]
    del all_tr
    tws = [s[2] for s in trs]
    vas = [(x[:CTX], y[:CTX]) for x, y, *_ in build(val) if len(x) >= CTX]
    print(f"[flow] {sum(len(s[0]) for s in trs)} train notes, "
          f"{len(vas)} val windows")
    xva = torch.stack([x for x, _ in vas]).to(dev)
    yva = torch.stack([y for _, y in vas]).to(dev)
    model = Flow(**(model_kwargs or {})).to(dev)
    opt = torch.optim.Adam(model.parameters(), 3e-4)

    def loss_of(x, y):
        d_lg, c_lg, l_lg, k_lg = model(x, F.one_hot(y[..., 0], 9).float(),
                                       F.one_hot(y[..., 1], DCOL).float())
        return (F.cross_entropy(d_lg.reshape(-1, 9), y[..., 0].reshape(-1))
                + F.cross_entropy(c_lg.reshape(-1, DCOL), y[..., 1].reshape(-1))
                + F.cross_entropy(l_lg.reshape(-1, DLAY), y[..., 2].reshape(-1))
                + F.cross_entropy(k_lg.reshape(-1, MAX_CHAIN),
                                  y[..., 3].reshape(-1)))

    out = Path(out_path) if out_path is not None else FLOW_PT
    with torch.no_grad():
        init_vloss = loss_of(xva, yva).item()      # no rng: stream unchanged
    history = {"initial_val_loss": init_vloss, "epochs": [], "updates": 0,
               "best_epoch": None, "n_train": len(trs)}
    best, best_state, best_epoch, patience = 1e9, None, None, 0
    for epoch in range(max_epochs):
        model.train()
        ep_losses = []
        for _ in range(steps_per_epoch):
            xs, ys = [], []
            for x, y, *_ in rng.choices(trs, weights=tws, k=8):
                i = rng.randrange(len(x) - CTX)
                xs.append(x[i:i + CTX])
                ys.append(y[i:i + CTX])
            loss = loss_of(torch.stack(xs).to(dev), torch.stack(ys).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
            history["updates"] += 1
        model.eval()
        with torch.no_grad():
            vloss = loss_of(xva, yva).item()
            d_lg, c_lg, _, _ = model(xva, F.one_hot(yva[..., 0], 9).float(),
                                     F.one_hot(yva[..., 1], DCOL).float())
            acc = (d_lg.argmax(-1) == yva[..., 0]).float().mean().item()
            cacc = (c_lg.argmax(-1) == yva[..., 1]).float().mean().item()
        history["epochs"].append({"epoch": epoch,
                                  "train_loss": sum(ep_losses) / len(ep_losses),
                                  "val_loss": vloss, "dir_acc": acc,
                                  "dcol_acc": cacc})
        if vloss < best - 1e-4:
            best, patience, best_epoch, best_state = vloss, 0, epoch, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if history["updates"] in snapshot_updates:
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, out.with_name(f"flow-{history['updates']}.pt"))
            out.with_name(f"history-{history['updates']}.json").write_text(json.dumps(
                {**history, "best_epoch": best_epoch, "best_val_loss": best}, indent=1))
        stop = early_stop_patience is not None and patience > early_stop_patience
        if epoch % 10 == 0 or stop:
            print(f"[flow] epoch {epoch}: val loss {vloss:.4f} "
                  f"dir acc {acc:.2f} dcol acc {cacc:.2f}")
        if stop:
            break
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out)
    if history_path is not None:
        Path(history_path).write_text(json.dumps(history, indent=1))
    print(f"saved {out} (best val loss {best:.4f})")
    return history


def _split(mapdir, manifest=True, *, map_executor=None):
    """Training/validation data. manifest=True (Packet D default) draws train
    from the corpus 'train' split and validation from the song-disjoint 'val'
    split — never a shuffle of difficulties from the same song (review 5A
    finding 5). manifest=False keeps the legacy all-dirs random 10% val for
    reproducing the pre-5A checkpoints.
    map_executor (opt-in) loads maps concurrently; None keeps the serial path
    and byte-identical output."""
    rng = random.Random(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if manifest:
        tr = load_dataset(mapdir, split="train", map_executor=map_executor)
        val = load_dataset(mapdir, split="val", map_executor=map_executor)
        assert tr and val, f"train={len(tr)} val={len(val)} — build the manifest"
        rng.shuffle(tr)
        return tr, val, dev, rng
    data = load_dataset(mapdir)
    rng.shuffle(data)
    nval = max(2, len(data) // 10)
    val, tr = data[:nval], data[nval:]
    assert tr, f"only {len(data)} usable maps — not enough to train"
    return tr, val, dev, rng


def train(mapdir):
    tr, val, dev, rng = _split(mapdir)
    train_rhythm(tr, val, dev, rng)
    train_flow(tr, val, dev, rng)


def train_rhythm_only(mapdir):
    train_rhythm(*_split(mapdir))


# ---------------- inference ----------------

def find_replays(merged):
    """Detect structural repeats in the rhythm stream.
    -> {target block index: (source block index, run length in blocks)}.
    Greedy left-to-right, earliest source wins (mappers copy the FIRST chorus).
    Runs of identical blocks (steady beat) don't count as sections."""
    nb = len(merged) // BLOCK
    blocks = [merged[i * BLOCK:(i + 1) * BLOCK] for i in range(nb)]

    def match(a, b):
        return int((a ^ b).sum()) <= MAX_HAM

    replays, i = {}, 0
    while i < nb:
        found = None
        for j in range(i):
            if match(blocks[i], blocks[j]):
                k = 1
                while i + k < nb and j + k < i and match(blocks[i + k], blocks[j + k]):
                    k += 1
                variety = len({tuple(blocks[i + m].tolist()) for m in range(k)})
                if k >= MIN_RUN and variety >= 2:
                    found = (j, k)
                    break
        if found:
            replays[i] = found
            i += found[1]
        else:
            i += 1
    return replays


def _decode_walls(active):
    """Boolean per-step wall prediction -> [(start, length)] runs:
    bridge sub-beat holes, drop runs shorter than WALL_MIN_STEPS."""
    a = active.clone()
    idx = a.nonzero().flatten().tolist()
    for p, q in zip(idx, idx[1:]):
        if 1 < q - p <= WALL_MAX_GAP + 1:
            a[p:q] = True
    runs, i = [], 0
    while i < len(a):
        if a[i]:
            j = i
            while j < len(a) and a[j]:
                j += 1
            if j - i >= WALL_MIN_STEPS:
                runs.append((i, j - i))
            i = j
        else:
            i += 1
    return runs


def _sample(logits, allowed, gen, temp=TEMP):
    mask = torch.full_like(logits, float("-inf"))
    mask[list(allowed)] = 0
    p = F.softmax((logits + mask) / temp, dim=0)
    return int(torch.multinomial(p, 1, generator=gen))


def _axis(d):
    """0 = vertical cut, 1 = lateral, None = diagonal/dot."""
    return 0 if d in (0, 1) else 1 if d in (2, 3) else None


def _sched_windows(presl, thrl, keepl, c, merg, spr, restl, win=16):
    """Per-4-beat-window swing counts the decode loop would schedule at
    threshold offset c — geometry-free mirror of the hands logic (replays/
    THIN/budget ignored, close enough to calibrate on)."""
    last = [-MIN_GAP, -MIN_GAP]
    counts = [0] * ((len(merg) + win - 1) // win)
    for s in range(len(merg)):
        if restl[s]:
            continue
        p, t, k = presl[s], thrl[s] + c, keepl[s] + c
        hands = [h for h in (0, 1) if p[h] > t and s - last[h] >= MIN_GAP]
        if merg[s] and not hands:
            h = int(p[1] > p[0])
            if s - last[h] < MIN_GAP:
                h = 1 - h
            if (p[h] >= k or spr[s]) and s - last[h] >= MIN_GAP:
                hands = [h]
        for h in hands:
            last[h] = s
            counts[s // win] += 1
    return counts


def _sched_count(presl, thrl, keepl, c, merg, spr, restl):
    return sum(_sched_windows(presl, thrl, keepl, c, merg, spr, restl))


def run_stats(out):
    """(vertical-cut share, share of in-combo transitions inside same-axis
    flip runs >= 6) — decode-selection gates against flow mode collapse."""
    heads = sorted((s, h, d) for s, h, _, _, d in out if d != 8)
    vert = sum(1 for _, _, d in heads if d in (0, 1))
    last, run, trans, inrun = {}, {}, 0, 0
    for s, h, d in heads:
        if h in last:
            ps, pd = last[h]
            ax = _axis(d)
            if s - ps <= SH_STEPS and ax is not None and _axis(pd) == ax:
                trans += 1
                run[h] = run.get(h, 1) + 1
            else:
                if s - ps <= SH_STEPS:
                    trans += 1
                if run.get(h, 0) >= 6:
                    inrun += run[h]
                run[h] = 1
        last[h] = (s, d)
    inrun += sum(r for r in run.values() if r >= 6)
    return vert / max(1, len(heads)), inrun / max(1, trans)


_CACHE = {}


def _load(cls, path):
    if path not in _CACHE:
        m = cls()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        _CACHE[path] = m
    return _CACHE[path]


def out_to_events(out):
    """Decoded notes [(s, h, col, lay, dir)] -> event list with chain counts,
    the representation events_to_xy expects."""
    dots = {}
    for s, h, c, l, d in out:
        if d == 8:
            dots[(s, h)] = dots.get((s, h), 0) + 1
    return sorted((s, h, d, c, l, 1 + dots.get((s, h), 0))
                  for s, h, c, l, d in out if d != 8)


def score_notes(out, wall_runs, T, afeat, flow=None, *, grid=None):
    """Mean log-likelihood of a decoded map under the flow model — the
    best-of-N selector. Delta targets are always in range (grid is 4x3)."""
    if flow is None:
        flow = _load(Flow, FLOW_PT)
    if getattr(flow, "timing", False) and grid is None:
        raise ValueError("timing-conditioned Flow requires an explicit time grid")
    wl = torch.zeros(T, dtype=torch.bool)
    wr = torch.zeros(T, dtype=torch.bool)
    for s0, ln, col in wall_runs:
        (wl if col == 0 else wr)[s0:s0 + ln] = True
    x, y = events_to_xy(out_to_events(out), wl, wr,
                        afeat[:, 3] if afeat is not None else None,
                        grid=grid if getattr(flow, "timing", False) else None)
    lp, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), CTX):
            xb, yb = x[i:i + CTX], y[i:i + CTX]
            d_lg, c_lg, l_lg, k_lg = flow(
                xb[None], F.one_hot(yb[None, :, 0], 9).float(),
                F.one_hot(yb[None, :, 1], DCOL).float())
            for lg, t in ((d_lg, 0), (c_lg, 1), (l_lg, 2), (k_lg, 3)):
                lp -= F.cross_entropy(lg[0], yb[:, t], reduction="sum").item()
            n += len(yb)
    return lp / max(1, n)


def rot_share(out):
    last, rot, tot = {}, 0, 0
    for s, h, c, l, d in sorted(out):
        if d == 8:
            continue
        if h in last:
            tot += 1
            rot += ang_dist(d, last[h]) in (90, 135)
        last[h] = d
    return rot / max(1, tot)


def _chain_fit(col, lay, d, k, used, s, wl_on, wr_on):
    """True iff k pinned dot followers fit along d from (col, lay): in-grid,
    unoccupied (incl. the simultaneous other hand's cells already in `used`),
    outside the centre ban and not wall-blocked. Schedule-mode masking only
    (adaptive mechanism revision, Q2 6+6); unpinned decodes never call this."""
    vx, vy = DIR_VEC[d]
    cc, ll = col, lay
    for _ in range(k):
        cc, ll = cc + vx, ll + vy
        if not (0 <= cc <= 3 and 0 <= ll <= 2) or (cc, ll) in used:
            return False
        if ll == 1 and cc in (1, 2):
            return False
        if (cc == 0 and wl_on[s]) or (cc == 3 and wr_on[s]):
            return False
    return True


def groom_notes(note_steps, T, step_ms, offset=0.0, model=None, flow=None,
                afeat=None, seed=0, temp=TEMP, rate_scale=1.0,
                drate=4.9, band_scale=1.0, band=None,
                replay_mode="legacy", calibrate=True, trace=None, grid=None,
                rest_mask_in=None, rest_policy=None, thr_vectors=None,
                density_adjust=None, doubles_cap_8b=None, schedule_in=None,
                schedule_mask=False, geometry_policy="legacy"):
    """note_steps: step indices of the merged converted stream.
    -> [(step, hand, col, layer, dir)] with parity-legal directions.
    Repeated sections (chorus etc.) replay the first occurrence's patterns.
    Outside replays no input step is dropped; inside a replay the source
    pattern wins over small rhythm mismatches (that's the reuse).
    replay_mode: 'legacy' copies detected repeats (current behavior); 'off'
    decodes everything fresh AND drops the calibration replay-remap, so an
    off-vs-legacy pair isolates literal copying (plan phase 1).
    calibrate=False freezes the density thresholds (no band targeting) for
    replay-only comparisons. trace, if a dict, receives detected/executed
    replay runs and calibration details for diagnostics.
    Rest protection and density calibration are DECOUPLED (review diagnosis):
    - rest_mask_in: a fixed [T] bool mask to use verbatim (freeze rests).
    - rest_policy: 'compute' (learned breathers) or 'off' (no rests).
    - thr_vectors: fixed (thr, keep) [T] tensors to use verbatim (freeze
      density thresholds — skips the calibration bisection AND energy tilt).
    - density_adjust: run the rate-band calibration offset (True) or not.
    `calibrate` is the back-compat alias: rest_policy and density_adjust each
    default to it, so calibrate=True/False is unchanged. The final rest_mask
    and thr/keep vectors are written to trace for baseline derivation.
    grid: a timing.TimeGrid giving absolute ms + local beat phase per step
    (Packet C). Default None -> a uniform grid from step_ms/offset, so every
    single-tempo call and self-check is byte-identical; all physical times,
    gaps, and beat-phase features read the grid so a tempo-changing song is
    correct on its LOCAL tempo.
    doubles_cap_8b (opt-in, phrase-plan): max FINAL doubles per aligned 8-beat
    window. Past the cap, the staler hand's note of a non-accent double is
    SUPPRESSED (not emitted) while its scheduling state updates as if it had
    played, so every later scheduling decision — and the set of occupied
    timestamps — replicates the uncapped decode; accent doubles are never
    suppressed but still count, so a window whose accents alone exceed the cap
    is INFEASIBLE and reported in trace["doubles_cap"]["windows_over"] instead
    of silently broken. None (default) changes nothing.
    schedule_in (opt-in, Q2 flow_decode): {step: {"hands": (h...), "k": {h:
    followers}}} fixes the COMPLETE event schedule — every scheduling decision
    (thin/rest/forcekeep/mono/nps/cap) is bypassed and the geometry loop runs
    verbatim on the given hands. Chain samples are still drawn (identical RNG
    stream) but pinned follower counts are used; a pinned follower that cannot
    be placed legally is counted in trace["schedule_infeasible"] — the caller
    REJECTS such proposals, never ships them shortened. Requires replay off;
    incompatible with doubles_cap_8b. None (default) changes nothing.
    geometry_policy='learned' is an experimental replay-off path: sample complete
    model-scored cuts under grid/wall/occupancy/parity/clearance constraints, with
    geometry style penalties, center/head-side bans and posthoc nudges disabled.
    The default 'legacy' preserves the current production sampling policy."""
    assert geometry_policy in ("legacy", "learned"), geometry_policy
    if geometry_policy == "learned":
        assert replay_mode == "off", "learned geometry requires replay off"
    assert replay_mode in ("off", "legacy"), replay_mode
    if schedule_in is not None:
        assert replay_mode == "off" and doubles_cap_8b is None, \
            "schedule_in requires replay off and no doubles cap"
    if rest_policy is None:
        rest_policy = "compute" if calibrate else "off"
    if density_adjust is None:
        density_adjust = calibrate
    if grid is None:
        from timing import TimeGrid
        grid = TimeGrid.uniform(step_ms, offset, T)
    if model is None:
        model = _load(Groomer, MODEL_PT)
    if flow is None:
        flow = _load(Flow, FLOW_PT)
    model.eval()
    flow.eval()
    merged = torch.zeros(T, dtype=torch.bool)
    merged[list(note_steps)] = True
    rest_mask = torch.zeros(T, dtype=torch.bool)  # filled by the calibration
    # sprinkle candidates into dead zones so no section is completely empty;
    # spacing follows section energy (loud -> 1/beat, quiet -> 1/4 beats)
    sprinkle = torch.zeros(T, dtype=torch.bool)
    idx = sorted(note_steps)
    for a, b in zip([0] + idx, idx + [T]):
        if b - a < SPRINKLE_GAP:
            continue
        for s0 in range(a + 4, b - 4):
            z = float(afeat[s0, 3]) if afeat is not None else 0.0
            every = 4 if z > 0.5 else (SPRINKLE_EVERY if z > -0.5 else 16)
            if s0 % every == 0 and (
                    afeat is None or float(afeat[s0, 0]) > SPRINKLE_MIN_E):
                sprinkle[s0] = True
    merged |= sprinkle
    # energy-tilted density thresholds
    if afeat is not None:
        zc = afeat[:, 3].clamp(-1.5, 1.5)
        thr, keep = PRES_T - DENSITY_W * zc, KEEP_MIN - DENSITY_W * zc
    else:
        thr = torch.full((T,), PRES_T)
        keep = torch.full((T,), KEEP_MIN)
    npsmax = max(6, int(NPS_WIN_MAX * band_scale))
    with torch.no_grad():
        lg = model(make_input(merged, afeat, drate,
                              grid.pos, grid.bar)[None])[0].sigmoid()
    pres = lg[:, :2]
    span = ((grid.time(idx[-1]) - grid.time(idx[0])) / 1000
            if len(idx) > 1 else 0.0)
    # REST protection — built independently of density calibration.
    if rest_mask_in is not None:
        rest_mask = rest_mask_in.clone() if hasattr(rest_mask_in, "clone") \
            else torch.as_tensor(rest_mask_in, dtype=torch.bool)
    elif (rest_policy == "compute" and afeat is not None and len(idx) > 1
          and span > 30 and T // 16 >= 8):
        # learned breathers: windows where the rhythm model itself goes quiet
        # (trained on human phrasing) rest fully — with the onset veto (a
        # densely-filled window is being actively mapped, never rest it)
        mp = pres.max(dim=1).values
        for i in range(T // 16):
            if (float(mp[i * 16:(i + 1) * 16].mean()) < REST_PRES
                    and int(merged[i * 16:(i + 1) * 16].sum()) <= 4):
                rest_mask[i * 16:(i + 1) * 16] = True
    # DENSITY calibration (see TARGET_BAND) — separable and injectable.
    if thr_vectors is not None:
        thr, keep = (torch.as_tensor(thr_vectors[0]).clone(),
                     torch.as_tensor(thr_vectors[1]).clone())
    elif density_adjust and afeat is not None and len(idx) > 1 and span > 30:
        if True:
            presl, thrl, keepl = pres.tolist(), thr.tolist(), keep.tolist()
            merg, spr = merged.tolist(), sprinkle.tolist()
            restl = rest_mask.tolist()
            # replayed blocks copy their SOURCE's notes, ignoring thresholds
            # — remap their windows onto the source counts or the proxy
            # undershoots by ~0.5-1 swings/s on replay-heavy songs.
            # replay off: no copying happens, so no remap either — otherwise
            # the off arm of the experiment would calibrate inconsistently
            wpb = BLOCK // 16  # windows per replay block
            remap = {}
            if replay_mode == "legacy":
                for tb, (sb, k) in find_replays(merged).items():
                    for m in range(k * wpb):
                        remap[tb * wpb + m] = sb * wpb + m

            def sched_total(c):
                wc = _sched_windows(presl, thrl, keepl, c, merg, spr, restl)
                return sum(wc[remap[i]] if remap.get(i, len(wc)) < len(wc)
                           else w for i, w in enumerate(wc))

            natural = sched_total(0.0) / span
            band_lo, band_hi = band or (TARGET_BAND[0] * band_scale,
                                        TARGET_BAND[1] * band_scale)
            want = min(max(natural, band_lo), band_hi) * rate_scale
            c = 0.0
            if abs(want - natural) > 0.05:
                lo, hi = -0.35, 0.35
                for _ in range(12):
                    c = (lo + hi) / 2
                    if sched_total(c) > want * span:
                        lo = c
                    else:
                        hi = c
                c = (lo + hi) / 2
                thr, keep = thr + c, keep + c
                if seed == 0:
                    print(f"density calib: {natural:.1f} swings/s natural, "
                          f"thr {c:+.2f} -> target {want:.1f}")
            if trace is not None:
                trace["calibration"] = {
                    "natural": round(natural, 3), "want": round(want, 3),
                    "thr_offset": round(c, 4), "remap_windows": len(remap)}
    if trace is not None:  # baseline mask/vectors + rhythm probs (step trace)
        trace["rest_mask"] = rest_mask.clone()
        trace["thr"] = thr.clone()
        trace["keep"] = keep.clone()
        trace["pres"] = pres.clone()          # [T,2] per-hand rhythm probs
        trace["merged"] = merged.clone()      # input evidence + sprinkles
        trace["sprinkle"] = sprinkle.clone()
        trace["rest_policy"] = rest_policy
        trace["density_adjust"] = bool(density_adjust) and thr_vectors is None
    wl_runs = _decode_walls(lg[:, 2] > WALL_THRESH)
    wr_runs = _decode_walls(lg[:, 3] > WALL_THRESH)
    wl_on = torch.zeros(T, dtype=torch.bool)
    wr_on = torch.zeros(T, dtype=torch.bool)
    for s0, ln in wl_runs:
        wl_on[s0:s0 + ln] = True
    for s0, ln in wr_runs:
        wr_on[s0:s0 + ln] = True
    gen = torch.Generator().manual_seed(seed)
    machines = {0: HandParity(), 1: HandParity()}
    lastn, last_any = {0: None, 1: None}, None
    out, last = [], {0: -MIN_GAP, 1: -MIN_GAP}
    swings = []  # swing times inside the trailing NPS_WIN, both hands
    lr_x = motion.LrExposure()  # <> exposure, destination timeline (C01/F05)
    hrec = {0: deque(), 1: deque()}  # per-hand swing times, rolling window
    dbl_hist = deque()  # scheduled events: (t_ms, was_double), rolling

    def hrec_add(h, t):
        hrec[h].append(t)
        for q in hrec.values():
            while q and t - q[0] > motion.WORK_WIN_MS:
                q.popleft()
    vrun = {0: (None, 0), 1: (None, 0)}  # per hand: (axis, in-combo run count)
    rhist = []  # recent transitions: True = continued a long axis run
    toks, prev_ev = [], None  # transformer event history
    ez = afeat[:, 3] if afeat is not None else None

    def _ctx_tok(h, s, dbl):
        dt_same = min(s - lastn[h][0], 8) if lastn[h] else 9
        dt_any = min(s - last_any, 8) if last_any is not None else 9
        ezv = float(ez[min(s, len(ez) - 1)]) if ez is not None else 0.0
        si = min(s, grid.n - 1)
        tok = token_vec(prev_ev, h, s, dt_same, dt_any, dbl,
                         bool(wl_on[min(s, T - 1)]), bool(wr_on[min(s, T - 1)]),
                         ezv, grid.pos[si], grid.bar[si])
        if getattr(flow, "timing", False):
            tok = torch.cat([tok, torch.tensor(flow_time_features(
                grid, s, lastn[h][0] if lastn[h] else None, last_any))])
        return tok

    replays = find_replays(merged)
    if trace is not None:  # detection is always recorded, even when off
        blocks = [merged[i * BLOCK:(i + 1) * BLOCK]
                  for i in range(len(merged) // BLOCK)]
        trace["replay_mode"] = replay_mode
        trace["detected"] = [
            {"target_block": tb, "source_block": sb, "blocks": k,
             "target_ms": [round(grid.time(tb * BLOCK)),
                           round(grid.time((tb + k) * BLOCK))],
             "source_ms": [round(grid.time(sb * BLOCK)),
                           round(grid.time((sb + k) * BLOCK))],
             "hamming": sum(int((blocks[tb + m] ^ blocks[sb + m]).sum())
                            for m in range(k)),
             "variety": len({tuple(blocks[tb + m].tolist())
                             for m in range(k)})}
            for tb, (sb, k) in sorted(replays.items())]
        trace["executed"] = []
    if replays and seed == 0:
        cov = sum(k for _, k in replays.values()) * BLOCK / T
        print(f"structural reuse: {len(replays)} repeated sections, "
              f"{cov * 100:.0f}% of song "
              + ("replayed" if replay_mode == "legacy" else
                 "matched (replay off: decoding fresh)"))
    # schedule-suppression mechanism counters (review frozen-evidence probe):
    # where input onsets become / fail to become scheduled swings
    sched = {"thin_skips": 0, "rest_skips": 0, "both_hand_cross": 0,
             "forcekeep": 0, "forcekeep_dropped": 0, "mono_reassign": 0,
             "double_demote": 0, "nps_dropped": 0, "sprinkle_added": int(sprinkle.sum()),
             "cap_demote": 0}
    cap_win, cap_over = {}, {}   # per-8-beat FINAL doubles / accent-kept overflow
    s = 0
    while s < T:
        if replay_mode == "legacy" and s % BLOCK == 0 \
                and s // BLOCK in replays:
            j, k = replays[s // BLOCK]
            delta = s - j * BLOCK
            src = [n for n in out if j * BLOCK <= n[0] < (j + k) * BLOCK]
            rec = {"target_block": s // BLOCK, "source_block": j, "blocks": k,
                   "copied": 0, "followers": 0, "skipped_gap": 0,
                   "skipped_rest": 0, "skipped_nps": 0, "flipped": 0}
            if trace is not None:
                trace["executed"].append(rec)
            # parity: enforced per kept note (a rest window or budget skip mid-
            # copy removes swings, so a one-time seam flip can fall out of sync)
            kept_heads = set()
            for ss, h, col, lay, d in src:
                ns = ss + delta
                if d == 8:  # chain follower: rides its head, no swing of its own
                    if (ns, h) in kept_heads:
                        out.append((ns, h, col, lay, 8))
                        rec["followers"] += 1
                    continue
                if ns - last[h] < MIN_GAP:
                    rec["skipped_gap"] += 1
                    continue
                if rest_mask[min(ns, T - 1)]:
                    rec["skipped_rest"] += 1
                    continue  # replayed copies respect carved breathers too
                t_ms = grid.time(ns)
                while swings and t_ms - swings[0] > NPS_WIN:
                    swings.pop(0)
                if len(swings) >= npsmax:
                    rec["skipped_nps"] += 1
                    continue
                req = machines[h].required(t_ms)
                if req is not None and family(d) != "lateral" \
                        and family(d) != req:
                    d = VFLIP_DIR[d]
                    rec["flipped"] += 1
                machines[h].commit(d, t_ms)
                dbl = any(m[0] == ss and m[1] != h and m[4] != 8 for m in src)
                toks.append(_ctx_tok(h, ns, dbl))  # keep transformer context
                out.append((ns, h, col, lay, d))
                rec["copied"] += 1
                hrec_add(h, t_ms)
                if d == 3 and h == 1:
                    # copied outward pairs count destination exposure too —
                    # replaying an approved <> cannot reset the budget
                    o = next((m for m in out[-6:] if m[0] == ns
                              and m[1] == 0 and m[4] == 2), None)
                    if o is not None and o[3] == lay and o[2] < col:
                        lr_x.add(t_ms)
                swings.append(t_ms)
                kept_heads.add((ns, h))
                kf = sum(1 for m in src
                         if m[0] == ss and m[1] == h and m[4] == 8)
                prev_ev = (h, d, col, lay, 1 + kf)
                ax = _axis(d)
                cont = (ax is not None and lastn[h]
                        and _axis(lastn[h][1]) == ax
                        and ns - lastn[h][0] <= SH_STEPS)
                vrun[h] = (ax, vrun[h][1] + 1 if cont else 0)
                if lastn[h] and ns - lastn[h][0] <= SH_STEPS:
                    rhist.append(cont and vrun[h][1] >= VRUN_MAX)
                lastn[h], last_any, last[h] = (ns, d, col, lay), ns, ns
            s = (s // BLOCK + k) * BLOCK
            continue
        if schedule_in is not None:
            # Q2 complete-schedule mode: the event schedule is fixed —
            # every scheduling decision is bypassed, the emission below
            # runs verbatim on the given hands
            ent = schedule_in.get(s)
            hands = list((ent or {}).get("hands", ()))
            k_pin = (ent or {}).get("k") or {}
            cap_drop = None
        else:
            k_pin = {}
            if rest_mask[s]:
                if merged[s]:
                    sched["rest_skips"] += 1
                s += 1
                continue
            if (last_any is not None and s - last_any == 1
                    and grid.time(s) - grid.time(last_any) < THIN_MS):
                sched["thin_skips"] += 1  # adjacent steps closer than THIN_MS
                s += 1
                continue
            hands = [h for h in (0, 1)
                     if pres[s, h] > thr[s] and s - last[h] >= MIN_GAP]
            if len(hands) == 2:
                sched["both_hand_cross"] += 1
            cap_drop = None      # hand whose note the doubles cap suppresses at s
            if merged[s] and not hands:
                # keep an input onset only if the model shows some conviction;
                # low-confidence onsets get dropped — mapper-style phrasing
                h = int(pres[s, 1] > pres[s, 0])
                if s - last[h] < MIN_GAP:
                    h = 1 - h
                if (pres[s, h] >= keep[s] or sprinkle[s]) \
                        and s - last[h] >= MIN_GAP:
                    hands = [h]
                    sched["forcekeep"] += 1
                else:
                    sched["forcekeep_dropped"] += 1
            if len(hands) == 1:
                # C09 hand monopoly: when the rolling window shows one hand
                # out-swinging the other by MONO_DIFF, hand single events to
                # the idle hand if its gap allows — reassign, never add
                h = hands[0]
                t_ms = grid.time(s)
                for q in hrec.values():
                    while q and t_ms - q[0] > motion.WORK_WIN_MS:
                        q.popleft()
                if (len(hrec[h]) - len(hrec[1 - h]) >= motion.MONO_DIFF
                        and s - last[1 - h] >= MIN_GAP):
                    hands = [1 - h]
                    sched["mono_reassign"] += 1
            elif len(hands) == 2:
                # C08 constant doubles: when the rolling window saturates with
                # two-hand events, demote non-accent doubles to the fresher
                # hand — the event time survives, saturation becomes interplay
                t_ms = grid.time(s)
                while dbl_hist and t_ms - dbl_hist[0][0] > motion.WORK_WIN_MS:
                    dbl_hist.popleft()
                share = (sum(x[1] for x in dbl_hist) / len(dbl_hist)
                         if len(dbl_hist) >= 8 else 0.0)
                accent = (afeat is not None
                          and float(afeat[min(s, T - 1), 0]) > 1.0)
                if share > motion.DBL_MAX and not accent:
                    hands = [max(hands, key=lambda h_: s - last[h_])]
                    sched["double_demote"] += 1
                # opt-in per-8-beat doubles cap (phrase-plan request): past the
                # cap, SUPPRESS the staler hand's note (accents never); the hand
                # list and its scheduling state stay baseline-identical so every
                # downstream decision — and thus occupancy — replicates baseline
                if doubles_cap_8b is not None and len(hands) == 2 and not accent \
                        and cap_win.get(s // (8 * STEPS_PER_BEAT), 0) >= doubles_cap_8b:
                    cap_drop = 1 - max(hands, key=lambda h_: s - last[h_])
            if hands:  # rolling density budget: never exceed the nps cap
                t_ms = grid.time(s)
                while swings and t_ms - swings[0] > NPS_WIN:
                    swings.pop(0)
                room = npsmax - len(swings)
                if room <= 0:
                    hands = []
                    sched["nps_dropped"] += 1
                elif len(hands) > room:
                    hands = [max(hands, key=lambda h_: float(pres[s, h_]))]
            if cap_drop is not None and len(hands) != 2:
                cap_drop = None       # nps budget already collapsed the double
            if hands:
                dbl_hist.append((grid.time(s), len(hands) == 2))
                if doubles_cap_8b is not None and len(hands) == 2 \
                        and cap_drop is None:                 # FINAL doubles only
                    w = s // (8 * STEPS_PER_BEAT)
                    if cap_win.get(w, 0) >= doubles_cap_8b:   # accent kept past cap
                        cap_over[w] = cap_over.get(w, 0) + 1  # -> infeasible, report
                    cap_win[w] = cap_win.get(w, 0) + 1
        used = set()
        first_d = None  # first hand's direction within a same-step double
        for h in hands:
            t_ms = grid.time(s)
            if h == cap_drop:
                # cap-suppressed note: no emission, no RNG, no parity commit —
                # but the SCHEDULING state updates exactly as an emission
                # would, so later thin/rest/forcekeep/mono/nps decisions (and
                # therefore occupied timestamps) replicate baseline
                last[h], last_any = s, s
                hrec_add(h, t_ms)
                swings.append(t_ms)
                sched["cap_demote"] += 1
                continue
            req = machines[h].required(t_ms)
            # dots (8) are follower-only: heads always carry a direction
            allowed = (range(8) if req is None else
                       (FOREHAND if req == "fore" else BACKHAND) | LATERAL)
            if geometry_policy == "legacy" and lastn[h] is not None and s - lastn[h][0] <= FAST_STEPS \
                    and lastn[h][1] != 8:
                fast_ok = [a for a in allowed
                           if ang_dist(a, lastn[h][1]) >= FAST_MIN_ANG]
                if fast_ok:
                    allowed = fast_ok
            ac, al = lastn[h][2:4] if lastn[h] else ANCHOR[h]
            # legality masked BEFORE sampling so clamps never yank the
            # model's path: grid bounds, walls, cross-body
            dcols = [i for i in range(DCOL) if 0 <= ac + i - 3 <= 3
                     and not (ac + i - 3 == 0 and (wl_on[s] or h == 1))
                     and not (ac + i - 3 == 3 and (wr_on[s] or h == 0))]
            # follower-aware legality masks (schedule mode with pinned chains):
            # keep only options whose WHOLE pinned chain fits, falling back to
            # unmasked sampling when nothing fits (the mismatch is recorded and
            # the proposal rejected — never silently shortened)
            pin = k_pin.get(h, 0) \
                if (schedule_in is not None and schedule_mask) else 0
            if pin and geometry_policy == "legacy":
                cand_cols = [ac + i - 3 for i in dcols]
                ma = [a for a in allowed
                      if any(_chain_fit(c2, l2, a, pin, used, s, wl_on, wr_on)
                             for c2 in cand_cols for l2 in range(3)
                             if not (l2 == 1 and c2 in (1, 2)))]
                allowed = ma or allowed
            tok = _ctx_tok(h, s, len(hands) == 2)
            gap_h = s - lastn[h][0] if lastn[h] else None
            with torch.no_grad():
                x = torch.stack(toks[-(CTX - 1):] + [tok])
                hid = flow.hidden(x[None])[0, -1]
                if geometry_policy == "learned":
                    from learned_geometry import sample_geometry
                    other = [n for n in out[-2*MAX_CHAIN:] if n[0] == s]
                    wall_cols = {c for c, active in ((0, wl_on[s]), (3, wr_on[s])) if active}
                    d, col, lay, k = sample_geometry(flow, hid, h, s, (ac, al), allowed,
                        other, wall_cols, gen, temp, k_pin.get(h) if schedule_in is not None else None)
                else:
                    d_lg = flow.dir_head(hid)
                    if gap_h is not None and gap_h <= LAT_STEPS:
                        for a in allowed:
                            if a in (2, 3):
                                d_lg[a] -= LAT_PEN
                    if first_d is not None and h == 1:
                        # C01/C02 joint-pair cost: the direction COMPLETING an
                        # outward <> pays the current exposure price; a
                        # converging >< a milder one (never a blind swap)
                        if first_d[0] == 2 and 3 in allowed:
                            d_lg[3] -= lr_x.cost(t_ms)
                        elif first_d[0] == 3 and 2 in allowed:
                            d_lg[2] -= motion.LR_IN_SCALE * lr_x.cost(t_ms)
                    pax, pn = vrun[h]
                    if pax is not None and pn >= VRUN_MAX:
                        share = sum(rhist[-RUN_WIN:]) / max(1, len(rhist[-RUN_WIN:]))
                        ramp = min(1.0, max(0.0, (share - RUN_SHARE_LO)
                                            / (RUN_SHARE_HI - RUN_SHARE_LO)))
                        if ramp >= 1.0 or pn >= VRUN_HARD:
                            # deep collapse: soft penalties lose to a peaked AR
                            # distribution — force off the axis, diagonals first
                            off = ([a for a in allowed if _axis(a) is None]
                                   or [a for a in allowed if _axis(a) != pax])
                            allowed = off or allowed
                        elif ramp > 0:
                            for a in allowed:  # budget filling: arc off the axis
                                if _axis(a) == pax:
                                    d_lg[a] -= VRUN_PEN * ramp
                    d = _sample(d_lg, allowed, gen, temp)
                    ax = _axis(d)
                    cont = (ax is not None and lastn[h]
                            and _axis(lastn[h][1]) == ax
                            and gap_h is not None and gap_h <= SH_STEPS)
                    vrun[h] = (ax, vrun[h][1] + 1 if cont else 0)
                    if gap_h is not None and gap_h <= SH_STEPS:
                        rhist.append(cont and vrun[h][1] >= VRUN_MAX)
                    d_oh = F.one_hot(torch.tensor(d), 9).float()
                    c_lg = flow.col_head(torch.cat([hid, d_oh]))
                    gap_ms = (grid.time(s) - grid.time(lastn[h][0])
                              if gap_h is not None else None)
                    if gap_ms is not None:
                        # B04: fast large repositioning costs per grid step
                        # beyond 1, per axis (layers below)
                        for i in dcols:
                            c_lg[i] -= motion.repo_penalty(gap_ms, abs(i - 3))
                    if pin:                      # exact per-direction column mask
                        mc = [i for i in dcols
                              if any(_chain_fit(ac + i - 3, l2, d, pin, used, s,
                                                wl_on, wr_on)
                                     for l2 in range(3)
                                     if not (l2 == 1 and ac + i - 3 in (1, 2)))]
                        dcols = mc or dcols
                    dc = _sample(c_lg, dcols, gen, temp)
                    col = ac + dc - 3
                    dlays = [j for j in range(DLAY) if 0 <= al + j - 2 <= 2
                             and not (al + j - 2 == 1 and col in (1, 2))]
                    l_lg = flow.lay_head(torch.cat(
                        [hid, d_oh, F.one_hot(torch.tensor(dc), DCOL).float()]))
                    if (gap_h is not None and gap_h <= SH_STEPS
                            and lastn[h][3] == 0 and lastn[h][1] in SH_UP
                            and d in SH_DOWN):
                        for j in dlays:
                            if al + j - 2 == 2:
                                l_lg[j] -= SH_PEN
                    if gap_ms is not None:
                        for j in dlays:
                            l_lg[j] -= motion.repo_penalty(gap_ms, abs(j - 2))
                    if pin:                      # exact chain-fit layer mask
                        ml = [j for j in dlays
                              if _chain_fit(col, al + j - 2, d, pin, used, s,
                                            wl_on, wr_on)]
                        dlays = ml or dlays
                    dl = _sample(l_lg, dlays, gen, temp)
                    lay = al + dl - 2
                    # temp 1.0 + governed boost (see CHAIN_COOL)
                    recent = sum(n[4] == 8 for n in out[-CHAIN_COOL:])
                    k_lg = flow.chain_head(torch.cat([hid, d_oh]))
                    k_lg[1:] = k_lg[1:] + CHAIN_BOOST * max(0.0, 1 - recent / 3)
                    k = 1 + _sample(k_lg, range(MAX_CHAIN), gen, temp=1.0)
                    if schedule_in is not None and h in k_pin:
                        k = 1 + k_pin[h]     # pinned count; the draw above is
                                             # burned to keep the RNG stream
            machines[h].commit(d, t_ms)
            if (col, lay) in used:           # double collision: nudge apart
                nc = max(0, col - 1) if h == 0 else min(3, col + 1)
                if ((nc == 0 and wl_on[s]) or (nc == 3 and wr_on[s])
                        or (nc, lay) in used):
                    lay = 2 if lay == 0 else 0
                else:
                    col = nc
            used.add((col, lay))
            toks.append(tok)
            lastn[h], last_any = (s, d, col, lay), s
            out.append((s, h, col, lay, d))
            hrec_add(h, t_ms)
            if len(hands) == 2 and h == 0:
                first_d = (d, col, lay)
            elif (first_d is not None and h == 1 and first_d[0] == 2
                    and d == 3 and lay == first_d[2] and first_d[1] < col):
                lr_x.add(t_ms)  # emitted a narrow <>: exposure recorded
            swings.append(t_ms)
            last[h] = s
            # chain followers: extra notes along the cut line, one swing (dots)
            vx, vy = DIR_VEC[d]
            cc, ll = col, lay
            nf = 0
            for _ in range(k - 1):
                cc, ll = cc + vx, ll + vy
                if not (0 <= cc <= 3 and 0 <= ll <= 2) or (cc, ll) in used:
                    break
                if geometry_policy == "legacy" and ll == 1 and cc in (1, 2):
                    break
                if (cc == 0 and wl_on[s]) or (cc == 3 and wr_on[s]):
                    break
                out.append((s, h, cc, ll, 8))
                used.add((cc, ll))
                nf += 1
            if schedule_in is not None and h in k_pin and nf != k_pin[h]:
                sched["schedule_infeasible"] = \
                    sched.get("schedule_infeasible", 0) + 1
            prev_ev = (h, d, col, lay, 1 + nf)
        s += 1
    if trace is not None:
        sched["scheduled_swings"] = sum(1 for n in out if n[4] != 8)
        sched["input_onsets"] = len(idx)
        trace["schedule"] = sched
        if schedule_in is not None:
            trace["schedule_infeasible"] = sched.get("schedule_infeasible", 0)
        if doubles_cap_8b is not None:
            trace["doubles_cap"] = {
                "cap": int(doubles_cap_8b),
                "per_window": {str(w): c for w, c in sorted(cap_win.items())},
                "windows_over": {str(w): c for w, c in sorted(cap_over.items())},
                "demoted": sched["cap_demote"]}
    if geometry_policy == "legacy":
        out = _fix_shoulders(_fix_hammers(out))
    # Shared hard style rule also covers copied phrases and all earlier nudges.
    # Translate whole hand groups; never shorten chains or change the rhythm.
    from swing_clearance import repair_approaches
    out, clearance = repair_approaches(out, [(s0, ln, col)
        for runs, col in ((wl_runs, 0), (wr_runs, 3)) for s0, ln in runs])
    if trace is not None:
        trace['cut_clearance'] = clearance
        trace['geometry_policy'] = geometry_policy
    # safety net: drop wall runs that collide with a note in their column
    # (replayed/nudged notes bypass the sampling mask)
    walls = []
    for runs, col in ((wl_runs, 0), (wr_runs, 3)):
        for s0, ln in runs:
            if not any(n[2] == col and s0 <= n[0] < s0 + ln for n in out):
                walls.append((s0, ln, col))
    return out, sorted(walls)


def _fix_shoulders(notes):
    """up@bottom-lane -> same-hand down@top-lane within a beat: drop the second
    note to a lower legal layer. Catches every source — sampling leaks, double
    nudges, and replay seams whose VFLIP turns down->up paths into up->down.
    Instances with no legal lower cell survive: that's the rare-accent budget."""
    by_step = {}
    for n in notes:
        by_step.setdefault(n[0], set()).add((n[2], n[3]))
    heads = {}
    for i, n in enumerate(notes):
        s, h, col, lay, d = n
        if d == 8:
            continue
        p = heads.get(h)
        if (p is not None and s - p[0] <= SH_STEPS and p[4] in SH_UP
                and p[3] == 0 and d in SH_DOWN and lay == 2
                and not any(m[0] == s and m[1] == h and m[4] == 8
                            for m in notes[i + 1:i + 4])):  # chained: leave
            for nl in (1, 0):
                if nl == 1 and col in (1, 2):
                    continue
                if (col, nl) in by_step[s]:
                    continue
                by_step[s].discard((col, lay))
                by_step[s].add((col, nl))
                notes[i] = n = (s, h, col, nl, d)
                break
        heads[h] = n
    return notes


def _fix_hammers(notes):
    """Nudge the second note of any same-step equal-direction in-line pair
    ('>>') perpendicular to the cut, keeping cell legal and un-stacked."""
    by_step = {}
    for i, n in enumerate(notes):
        by_step.setdefault(n[0], []).append(i)
    for idxs in by_step.values():
        if len(idxs) != 2:
            continue
        a, b = notes[idxs[0]], notes[idxs[1]]
        if a[4] != b[4] or not is_hammer(a[2], a[3], b[2], b[3], a[4]):
            continue
        s, h, col, lay, d = b
        cands = [(c, l) for c in (col, col - 1, col + 1) if 0 <= c <= 3
                 for l in (lay, 0, 2, 1) if 0 <= l <= 2]
        for c2, l2 in cands:
            if (c2, l2) in ((a[2], a[3]), (col, lay)):
                continue
            if l2 == 1 and c2 in (1, 2):                       # vision block
                continue
            if (h == 0 and c2 == 3) or (h == 1 and c2 == 0):   # cross-body
                continue
            if not is_hammer(a[2], a[3], c2, l2, d):
                notes[idxs[1]] = (s, h, c2, l2, d)
                break
    return notes


# ---------------- self-check ----------------

def check(mapdir):
    data = load_dataset(mapdir)
    assert data, "no usable maps"
    inp, pres, events, wl, wr = data[0][:5]
    both = (pres.sum(1) == 2).float().mean()
    wallcov = sum(float((m[3] | m[4]).float().mean()) for m in data) / len(data)
    print(f"first map: {len(inp)} steps, note rate {pres.mean():.3f}, "
          f"doubles rate {both:.3f}; dataset wall coverage {wallcov:.1%}")
    assert Groomer()(inp[None]).shape == (1, len(inp), 4)
    x, y = events_to_xy(events, wl, wr, inp[:, 12])
    assert x.shape == (len(events), NTOK) and y.shape == (len(events), 4)
    chains = (y[:, 3] > 0).float().mean()
    print(f"chain (multi-note swing) rate in first map: {chains:.1%}")
    n = min(len(events), CTX)
    assert int(y[:, 1].min()) >= 0 and int(y[:, 1].max()) < DCOL
    assert int(y[:, 2].min()) >= 0 and int(y[:, 2].max()) < DLAY
    d_lg, c_lg, l_lg, k_lg = Flow()(x[None, :n], F.one_hot(y[None, :n, 0], 9).float(),
                                    F.one_hot(y[None, :n, 1], DCOL).float())
    assert d_lg.shape == (1, n, 9) and k_lg.shape == (1, n, MAX_CHAIN)
    # wall run decoding: bridges sub-beat holes, drops short runs
    a = torch.zeros(40, dtype=torch.bool)
    a[0:6] = a[8:14] = True   # 6 + gap2 + 6 -> one 14-step run
    a[30:34] = True           # too short -> dropped
    assert _decode_walls(a) == [(0, 14)], _decode_walls(a)
    # decode invariants: rhythm model at exactly 0.5 presence -> below PRES_T
    # but above KEEP_MIN, so only the force-keep path fires (walls silenced);
    # zeroed flow transformer -> uniform sampling over parity-legal choices
    quiet = Groomer()
    nn.init.zeros_(quiet.out.weight)
    nn.init.zeros_(quiet.out.bias)
    with torch.no_grad():
        quiet.out.bias[2:] = -3.0
    zflow = Flow()
    for p in zflow.parameters():
        nn.init.zeros_(p)
    in_steps = {i for i in range(0, 400, 3)}
    out, walls = groom_notes(in_steps, 410, 200.0, model=quiet, flow=zflow)
    assert walls == [], "quiet model must emit no walls"
    assert {s for s, *_ in out} == in_steps, "decode dropped input steps"
    # dead zones get sprinkled: two onsets 60 beats apart must not stay empty
    out2, _ = groom_notes({0, 240}, 250, 200.0, model=quiet, flow=zflow)
    mid = {s for s, *_ in out2 if 0 < s < 240}
    assert mid, "dead zone stayed empty"
    assert all(s % SPRINKLE_EVERY == 0 for s in mid), mid
    assert len(mid) <= 240 // SPRINKLE_EVERY, "sprinkle too dense"
    lastd = {0: -MIN_GAP, 1: -MIN_GAP}
    for s, h, col, lay, d in out:
        assert 0 <= col <= 3 and 0 <= lay <= 2 and 0 <= d <= 8
        if d == 8:
            continue  # chain follower, not a swing
        assert s - lastd[h] >= MIN_GAP, f"hand {h} gap < {MIN_GAP} at step {s}"
        lastd[h] = s
    assert violations([(s * 200.0, h, d) for s, h, _, _, d in out]) == 0
    # replay invariants: an ABAB stream must replay AB for the second AB,
    # identical geometry, parity fixed at the seam by vertical mirror at most
    a0, a1 = list(range(0, 31, 3)), [0, 5, 10, 16, 21, 26]
    steps = {s for s in a0} | {32 + s for s in a1} \
        | {64 + s for s in a0} | {96 + s for s in a1}
    m = torch.zeros(128, dtype=torch.bool)
    m[list(steps)] = True
    assert find_replays(m) == {2: (0, 2)}, find_replays(m)
    out, _ = groom_notes(steps, 128, 200.0, model=quiet, flow=zflow)
    first = [n for n in out if n[0] < 64]
    second = [n for n in out if n[0] >= 64]
    assert [(n[0] + 64, n[1], n[2], n[3]) for n in first] == \
        [n[:4] for n in second], "replay changed structure or geometry"
    flips = {VFLIP_DIR[a[4]] == b[4] for a, b in zip(first, second)}
    sames = {a[4] == b[4] for a, b in zip(first, second)}
    assert flips == {True} or sames == {True}, "replay dirs neither copied nor mirrored"
    assert violations([(s * 200.0, h, d) for s, h, _, _, d in out]) == 0
    # phase-1 probe: that same ABAB stream is a steady ALTERNATING beat, not
    # a verified musical section — the detector nominating it is the known
    # false-positive the replay_mode toggle exists to measure.
    # legacy records detection + execution in the trace...
    tr = {}
    groom_notes(steps, 128, 200.0, model=quiet, flow=zflow, trace=tr)
    det = tr["detected"]
    assert [(r["target_block"], r["source_block"], r["blocks"]) for r in det] \
        == [(2, 0, 2)], det
    assert det[0]["hamming"] == 0 and det[0]["variety"] == 2, det
    assert tr["executed"][0]["copied"] > 0, tr["executed"]
    # ...off still detects but copies nothing, drops no input step, and
    # keeps parity legal; the calibration remap is also disabled (no afeat
    # here, so that path is covered by the replay_ab experiment)
    tr = {}
    out_off, _ = groom_notes(steps, 128, 200.0, model=quiet, flow=zflow,
                             replay_mode="off", trace=tr)
    assert tr["detected"] == det and tr["executed"] == [], "off must not copy"
    assert {s for s, *_ in out_off} == steps, "replay off dropped input steps"
    assert violations([(s * 200.0, h, d) for s, h, _, _, d in out_off]) == 0
    # hammer fix: '>>' (same step, same dir, in line) gets nudged perpendicular
    fixed = _fix_hammers([(0, 0, 1, 0, 3), (0, 1, 2, 0, 3)])
    a, b = fixed
    assert not is_hammer(a[2], a[3], b[2], b[3], 3), fixed
    down_pair = [(0, 0, 1, 0, 1), (0, 1, 2, 0, 1)]  # parallel down double: legal
    assert _fix_hammers(list(down_pair)) == down_pair
    # shoulder fix: up@bottom -> down@top within a beat drops to a lower layer
    fixed = _fix_shoulders([(0, 0, 1, 0, 0), (2, 0, 1, 2, 1)])
    assert fixed[1] == (2, 0, 1, 0, 1), fixed  # lay 1 vision-blocked -> lay 0
    slow = [(0, 0, 1, 0, 0), (8, 0, 1, 2, 1)]  # outside the beat window: kept
    assert _fix_shoulders(list(slow)) == slow
    # density: a wall-to-wall 187bpm input stream stays under the nps cap
    import convert as _c
    dense, _ = groom_notes(set(range(600)), 600, 80.0, model=quiet, flow=zflow)
    dtimes = sorted(80.0 * s for s, _, _, _, d in dense if d != 8)
    assert _c.peak_nps(dtimes) <= _c.NPS_CAP, _c.peak_nps(dtimes)
    print("check ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    mapdirs = sys.argv[2:] if len(sys.argv) > 2 else MAPS_DIRS
    {"train": train, "check": check,
     "train-rhythm": lambda md: train_rhythm_only(md),
     "train-flow": lambda md: train_flow(*_split(md))}[cmd](mapdirs)
