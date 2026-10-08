PY := .venv/bin/python

# mp3/ogg -> Beat Saber map:  make song SONG=path/to/song.mp3 [OUT=out/name] [PLAYLIST=name]
song:
	$(PY) run.py "$(SONG)" $(OUT) $(if $(PLAYLIST),--playlist "$(PLAYLIST)") $(if $(DIFFS),--diffs "$(DIFFS)")

# every audio file in a folder (resumable):  make batch DIR=path/to/folder [OUT=out] [PLAYLIST=name] [FORCE=1]
batch:
	$(PY) run.py "$(DIR)" $(OUT) $(if $(PLAYLIST),--playlist "$(PLAYLIST)") $(if $(FORCE),--force) $(if $(DIFFS),--diffs "$(DIFFS)")

# re-convert all maps in out/ (no Mapperatorinator) + rebuild the playlist
regen:
	bash regen_all.sh

test:
	$(PY) test_parity.py

check:
	$(PY) groom.py check
	$(PY) critic.py check

# retrain rhythm + flow on bytrius + beatsaver corpora (~30 min)
train:
	$(PY) groom.py train

# refresh the critic after generator changes (~25 min: re-decodes negatives)
critic:
	rm -f negatives.pt critic.pt
	$(PY) critic.py gen
	$(PY) critic.py train

# grow the corpus:  make scrape [LIMIT=500]
LIMIT := 500
scrape:
	$(PY) scrape.py --limit $(LIMIT)

.PHONY: song regen test check train critic scrape
