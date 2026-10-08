#!/bin/bash
# Re-convert every out/ map with current models (no Mapperatorinator) and
# rebuild the library playlist. Audio comes from the pristine song_orig.egg
# when present — features must align with the osu timeline, and song.egg may
# already be trimmed/padded by an earlier conversion.
set -e
cd "$(dirname "$0")"
V=.venv/bin/python
fails=0
for d in out/*/; do
  osu=$(ls "$d"*.osu 2>/dev/null | head -1)
  if [ -z "$osu" ]; then echo "skip ${d%/}: no saved .osu"; continue; fi
  audio="${d}song.egg"
  [ -f "${d}song_orig.egg" ] && audio="${d}song_orig.egg"
  echo "--- ${d%/}"
  $V convert.py "$osu" "$audio" "${d%/}" || { echo "REGEN FAILED: ${d%/}"; fails=$((fails+1)); }
done
export PLAYLIST="${PLAYLIST:-osu2bs}"
rm -f "out/$(echo "$PLAYLIST" | tr -cd 'A-Za-z0-9 _-' | tr ' ' '_').bplist"
$V - <<'EOF'
import json
import os
from pathlib import Path
import run
out = Path("out")
for d in sorted(out.iterdir()):
    if d.is_dir() and d.with_suffix(".zip").exists():
        run.add_to_playlist(os.environ.get("PLAYLIST", "osu2bs"), d, out)
import re
pl = json.load(open("out/" + re.sub(r"[^\w\s-]", "", os.environ.get("PLAYLIST", "osu2bs")).strip().replace(" ", "_") + ".bplist"))
print("playlist:", len(pl["songs"]), "songs")
EOF
echo "=== REGEN DONE (fails: $fails) ==="
