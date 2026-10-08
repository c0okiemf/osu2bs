"""Pack the inference engine (code + models, no env) into src-tauri/engine.zip
for embedding in the app binary. Cross-platform (CI runners lack `zip`)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "src-tauri" / "engine.zip"

FILES = ["run.py", "convert.py", "groom.py", "critic.py", "parity.py",
         "motion.py", "swing_clearance.py", "onset_flow.py", "learned_geometry.py", "timing.py", "ladder.py", "ladder.json", "groom.pt",
         "flow.pt", "onset_flow.pt", "critic.pt", "requirements-win.txt",
         # Q5 scoped release: bundle policy + its modules and artifacts
         "quality_policy.py", "quality_policy.json", "release_manifest.json",
         "condflow.pt", "planner.pt", "pref_model.pt",
         "cond_flow.py", "phrase_planner.py", "flow_decode.py",
         "quality_repair.py", "pref.py",
         "eval/quality_metrics.py", "eval/quality_train.py",
         "eval/clean_rhythm.py", "eval/clean_rhythm_eval.py",
         "eval/phrase_plan.py", "eval/map_reader.py"]
SKIP_DIRS = {".git", "datasets", "__pycache__"}

OUT.unlink(missing_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in FILES:
        z.write(ROOT / f, f)
    for p in sorted((ROOT / "Mapperatorinator").rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or set(rel.parts) & SKIP_DIRS or rel.suffix == ".egg-info":
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        z.write(p, rel.as_posix())
print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
