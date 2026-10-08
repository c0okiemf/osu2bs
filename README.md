# osu2bs

Turn any song into Beat Saber maps — locally, on your own GPU (or CPU).

An AI pipeline: [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)
listens to the audio and decides *what* to map; osu2bs's own models — trained
on top-rated and ranked community maps — turn that into parity-correct Beat
Saber swing flow across the difficulty ladder (Easy … Expert+, and
custom Expert++/+++ tiers).

## The app

Grab the [latest release](../../releases/latest): `osu2bs.exe` (Windows,
WebView2 preinstalled on 10/11) or `osu2bs.AppImage` (Linux). First launch
detects your GPU, asks where to run inference, and downloads the engine
(Python, PyTorch for your hardware, ffmpeg, the AI mapper — ~2–3 GB) with
progress bars. After that: drop songs or folders, pick difficulties, hit
Generate. Maps land straight in your chosen folder (defaults to the game's
`CustomLevels`) as game-ready folders and/or shareable zips, with an optional
shared playlist.

Everything runs on your machine; nothing is uploaded anywhere.

## CLI

```sh
make song SONG=song.mp3 [DIFFS=Easy,Hard,ExpertPlus,ExpertPlus2]
make batch DIR=~/music [PLAYLIST=name] [FORCE=1]   # bulk, resumable
make regen                                         # re-convert out/ (no inference)
make test check                                    # self-checks
```

Needs a Python env with the deps from `requirements-win.txt` (any OS despite
the name) plus PyTorch, and the `Mapperatorinator` submodule.

## Pinned 36k generator and multiple difficulties

The app and `run.py` use the pinned **36k onset-flow checkpoint** in
`onset_flow.pt`. Osu owns circle/slider-head and slider-end timestamps and density;
the Beat Saber model chooses hands, positions, directions and block groups.
Parity, occupancy and cut-approach clearance remain constrained. There is no
rhythm TCN, snapping, thinning, density correction, critic selection, or audio
trimming in this route. No generated walls are added.

```sh
.venv/bin/python run.py song.mp3 out/song 5.5 --diffs ExpertPlus,ExpertPlus2,ExpertPlus3
# Convert one existing osu timeline explicitly with the pinned model:
.venv/bin/python convert.py source.osu song.ogg out/exact --onset-flow onset_flow.pt
```

Choose **up to five difficulties per map**, including combinations of Expert+,
Expert++, Expert+++, etc. Each gets a distinct native Beat Saber slot and a custom
label; none replaces another selected tier. For example +/++/+++ occupy
Hard/Expert/ExpertPlus internally while displaying Expert+/Expert++/Expert+++ in
SongCore. Without custom-label support a client may show the native slot names.

Each selected tier requests its **own upstream osu chart**, so generation takes
longer with more tiers. The source-intensity setting is the Expert+ osu-star
anchor (default5.5); each easier/harder tier requests one star below/above it.
Thus +/++/+++ request5.5/6.5/7.5. This is a request to the upstream model, not a
certified Beat Saber difficulty calibration or a guarantee of perfectly ordered
physical demand. Learned doubles can still affect difficulty.

`generation.json` records the checkpoint hash, per-tier source hashes and requested
osu stars. `source-onsets.json` records each immutable source clock. Batch mode
needs `--force` to regenerate an existing ZIP. An explicit `--onset-flow PATH`
overrides the pinned checkpoint; `run.py --legacy-flow` selects the old generator.
`make regen` remains a legacy converter workflow, not multi-source onset generation.

All playable exports validate real Ogg Vorbis audio and fully decode it before
publishing a ZIP; a `.egg` filename alone is not accepted as proof of format.
For review maps assembled from saved charts use `convert.write_audio` and
`convert.package_map`, which apply the same validation.

## Legacy generator (`--legacy-flow`)

- **Rhythm**: a small TCN predicts per-hand swing timing from the osu!-side
  onsets plus audio features (separate full-mix and percussive onset
  channels), conditioned on a continuous difficulty signal — one model
  serves every tier, trained on ~2,600 difficulty charts.
- **Geometry**: a causal transformer places direction/column/layer as motion
  deltas with a two-saber parity machine masking illegal swings; the default
  decodes repeated song sections afresh.
- **Selection**: each map is the best of several sampled decodes across a
  temperature ladder, gated by corpus-derived style bounds (density caps,
  run-length and lateral shares) and ranked by a learned critic trained with
  hand-approved maps as its only positives.
- **Difficulty**: per-song density calibrates into a band pinned to
  reference maps; the tier ladder itself (`ladder.py`) is fitted from ~2,000
  same-song difficulty pairs and extrapolates upward indefinitely.
- **Evaluation**: `eval/` regenerates held-out reference maps from raw audio
  and diffs them against the human originals (style profile + breather
  placement), which is how the decode knobs get tuned.

## Repo map

| path | what |
|---|---|
| `app/` | Tauri desktop app (see `app/README.md` for dev/build) |
| `run.py`, `convert.py` | pipeline: audio → osu! chart → Beat Saber map |
| `groom.py`, `critic.py`, `parity.py`, `ladder.py` | models + parity + tier ladder |
| `onset_flow.py`, `onset_flow.pt` | pinned36k hand/geometry generator |
| `groom.pt`, `flow.pt`, `critic.pt` | legacy inference models (committed) |
| `scrape.py`, `precompute_feats.py` | corpus growth + feature cache (training) |
| `eval/` | held-out evaluation harness |

Evaluation manifests use paths relative to the repository root. They describe
external corpora and historical outputs; those files are not bundled. Run the
evaluation tools from the repository root after supplying the referenced data.
Technical experiment specifications live in `docs/specs/`. Regenerate local
experiment receipts when their code, specification, or manifest hashes change.

## Credits & license

MIT — see [LICENSE](LICENSE). Built on Mapperatorinator (OliBomby) and the
Beat Saber mapping community's shared work: [CREDITS.md](CREDITS.md).
