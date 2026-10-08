"""Generate Mapperatorinator .osu files for the held-out eval gems (once)."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(str(Path.home()) + "/app/osu2bs")
MAPPER = HERE / "Mapperatorinator"
PY = HERE / ".venv/bin/python"
EVAL = Path(__file__).parent  # gen.osu lives beside this script per song

SONGS = {
    "rap_god": str(Path.home()) + "/app/beat-saber-map-gen/input/bytrius/"
               "19909 (Rap God V2 - Ryger)/01 Rap God.egg",
    "reality_check": str(Path.home()) + "/app/beat-saber-map-gen/input/input/"
                     "25f (Reality Check Through The Skull - DM DOKURO)/RCTTS.egg",
    "spaceman": str(Path.home()) + "/app/beat-saber-map-gen/input/input/"
                "24e5e (SPACEMAN - oegoe)/song.egg",
}

if __name__ != "__main__":
    SONGS_ITER = ()
else:
    SONGS_ITER = SONGS.items()
for name, audio in SONGS_ITER:
    d = EVAL / name
    if (d / "gen.osu").exists():
        print(f"{name}: osu already generated")
        continue
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        work = Path(td)
        song = work / "song.ogg"
        song.symlink_to(audio)
        subprocess.run(
            [str(PY), "inference.py", "-cn", "v32",
             f"audio_path={song}", f"output_path={work}",
             "gamemode=0", "difficulty=5.5", "export_osz=false",
             "title=" + name, "hydra.run.dir=" + str(work / "logs")],
            cwd=MAPPER, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            check=True)
        osus = sorted(work.glob("*.osu"), key=lambda p: p.stat().st_mtime)
        shutil.copyfile(osus[-1], d / "gen.osu")
    print(f"{name}: gen.osu saved")
