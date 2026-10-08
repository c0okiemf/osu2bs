# Credits & attributions

osu2bs stands on a lot of other people's work:

- **[Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)** by
  **OliBomby** (MIT) — the osu!-side AI mapper whose source is bundled (as a
  pinned submodule) and whose `Mapperatorinator-v32` checkpoint the app
  downloads from Hugging Face on first run. This project would not exist
  without it. Also OliBomby's **[slider](https://github.com/OliBomby/slider)**
  beatmap library.
- **Beat Saber mapping community** — the rhythm/flow/critic models are
  trained on publicly shared custom maps from **BeatSaver** (top-rated and
  **ScoreSaber**-ranked maps), including the work of **Ryger**, **Bytrius**
  and hundreds of other mappers. Their craft is what the models learn.
- **[Tauri](https://tauri.app)** (MIT/Apache-2.0) — the desktop shell.
- **[PyTorch](https://pytorch.org)** (BSD-3), **[librosa](https://librosa.org)**
  (ISC), **[Transformers](https://github.com/huggingface/transformers)**
  (Apache-2.0) and the wider Python ecosystem installed at setup.
- **[light-the-torch](https://github.com/pmeier/light-the-torch)** (BSD-3) —
  hardware-aware PyTorch wheel selection during setup.
- **[FFmpeg](https://ffmpeg.org)** (GPL builds by [gyan.dev](https://www.gyan.dev/ffmpeg/builds/))
  — downloaded at setup for audio conversion.
- **Python** runtimes downloaded at setup: the python.org embeddable build
  on Windows (PSF license) and
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
  on Linux; Linux ffmpeg comes from
  [John Van Sickle's static builds](https://johnvansickle.com/ffmpeg/) (GPL).

Beat Saber is a trademark of Beat Games. This project is unaffiliated fan
tooling for creating custom content you play locally.
