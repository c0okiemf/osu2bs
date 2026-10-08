#!/bin/bash
# Render the UI headless for visual review: shot.sh [dark] [outfile]
set -e
cd "$(dirname "$0")/ui"
tmp=$(mktemp -d)
cp main.js devstub.js "$tmp/"
if [ "$1" = "dark" ]; then
  sed 's/@media (prefers-color-scheme: dark)/@media all/' styles.css > "$tmp/styles.css"
else
  cp styles.css "$tmp/"
fi
sed 's#<script src="main.js">#<script src="devstub.js"></script><script src="main.js">#' \
  index.html > "$tmp/index.html"
out="${2:-/tmp/osu2bs-ui-${1:-light}.png}"
google-chrome --headless=new --disable-gpu --hide-scrollbars \
  --virtual-time-budget=2500 --window-size=980,1500 \
  --screenshot="$out" "file://$tmp/index.html${SHEET:+#sheet}" 2>/dev/null
rm -rf "$tmp"
echo "$out"
