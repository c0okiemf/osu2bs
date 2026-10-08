#!/bin/bash
# Difficulty-conditioning pipeline: features -> check -> train -> critic ->
# holdout eval -> regen library (EP) -> multi-difficulty sample
set -e
cd "$(dirname "$0")"
V=.venv/bin/python

echo "=== stage 1: feature precompute (multi-difficulty grids) ==="
$V precompute_feats.py

echo "=== stage 2: dataset + invariant check ==="
$V groom.py check

echo "=== stage 3: train rhythm + flow (difficulty-conditioned) ==="
$V groom.py train

echo "=== stage 4: critic (ExpertPlus approved positives) ==="
rm -f negatives.pt critic.pt
$V critic.py gen
$V critic.py train

echo "=== stage 5: holdout gem eval (ExpertPlus quality) ==="
$V eval/eval_compare.py

echo "=== stage 6: regen library ==="
bash regen_all.sh

echo "=== stage 7: multi-difficulty sample (neckhurts, Easy..Expert++) ==="
d=out/neckhurts
osu=$(ls $d/*.osu | head -1)
$V convert.py "$osu" "$d/song_orig.egg" "$d" "Easy,Normal,Hard,Expert,ExpertPlus"

echo "=== DIFF PIPELINE DONE ==="
