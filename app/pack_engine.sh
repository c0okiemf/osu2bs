#!/bin/bash
# Pack the inference engine into src-tauri/engine.zip (wrapper; CI calls
# pack_engine.py directly with whatever python the runner has).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
command -v "$PY" >/dev/null 2>&1 || PY=python3
"$PY" app/pack_engine.py
