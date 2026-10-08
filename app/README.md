# osu2bs desktop app

Fully native, inference-only consumer app: drop songs (or folders), pick
up to five difficulties (Easy … Expert+, plus custom Expert++/+++ tiers), bulk-generate
into a remembered output folder (defaults to Beat Saber's CustomLevels)
as map folders and/or zips, with one shared playlist. No WSL.

The default is the pinned36k onset-flow model (`onset_flow.pt`). Each selected
tier gets its own upstream osu timeline and a separate Beat Saber difficulty slot;
custom labels retain +/++/+++ names. Five is the per-mode limit. More tiers require
more upstream inference passes. The source-intensity setting anchors Expert+;
other tiers request one osu star per step. This is not calibrated physical difficulty.

The engine (Python inference code + 10MB models + Mapperatorinator source)
is embedded in the exe (`pack_engine.sh` → engine.zip) and unpacks to
`%LOCALAPPDATA%\osu2bs\engine` (Windows) or `~/.local/share/osu2bs/engine`
(Linux — both platforms run the identical setup). The wizard detects NVIDIA
GPUs, asks where to infer (any CUDA GPU or CPU), then downloads with
progress bars: embeddable Python 3.11, device-matched PyTorch, inference
deps (requirements-win.txt), static ffmpeg, and the 1.7GB Mapperatorinator
checkpoint (contained under the engine's hf/). Setup is marker-file
resumable and re-runnable from Settings.

## Dev (WSL, needs webkit2gtk-4.1)
    cd app && npm install && npm run dev

## Portable Windows exe (cross-compiled from WSL)
    sudo apt install clang lld                       # cc-rs wants clang-cl:
    ln -sf /usr/bin/clang ~/.local/bin/clang-cl      # driver-mode via argv[0]
    rustup target add x86_64-pc-windows-msvc && cargo install cargo-xwin
    ./app/pack_engine.sh   # embeds engine.zip (required before build)
    cd app && PATH="$HOME/.local/bin:$PATH" npx tauri build --runner cargo-xwin \
      --target x86_64-pc-windows-msvc --no-bundle
    # -> src-tauri/target/x86_64-pc-windows-msvc/release/osu2bs-app.exe
Needs the WebView2 runtime on the target machine (preinstalled on Win 10/11).

## Visual review harness
    app/shot.sh [dark]   # headless screenshot with stubbed backend
