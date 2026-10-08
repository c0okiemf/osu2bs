"""BeatSaver scraper: all of Ryger's Expert+ Standard maps, then top-rated
maps until --limit is reached. Resumable (existing folders count and skip).

  .venv/bin/python scrape.py [--limit 500]

~1 req/s politeness per the plan; extracts .dat + info + audio only.
"""
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

API = "https://api.beatsaver.com"
DEST = Path(__file__).parent / "beatsaver"
MIN_SCORE, MIN_VOTES = 0.85, 50
RYGER_MIN_SCORE = 0.7   # take nearly everything of his
SKIP_EXT = {".jpg", ".jpeg", ".png", ".gif", ".mp4", ".webm"}

_last = [0.0]


def get(url, **kw):
    wait = _last[0] + 1.0 - time.time()  # 1 req/s politeness
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    for attempt in (1, 2):
        r = requests.get(url, timeout=120,
                         headers={"User-Agent": "osu2bs-research/1.0"}, **kw)
        if r.status_code == 429 and attempt == 1:
            time.sleep(10)
            continue
        r.raise_for_status()
        return r


def ep_standard(doc):
    return any(d["characteristic"] == "Standard" and d["difficulty"] == "ExpertPlus"
               and not d.get("me") and not d.get("ne")
               for d in doc["versions"][0]["diffs"])


def want(doc):
    s = doc["stats"]
    return (ep_standard(doc) and s["score"] >= MIN_SCORE
            and s["upvotes"] + s["downvotes"] >= MIN_VOTES
            and 60 <= doc["metadata"]["duration"] <= 600)


def save_map(doc):
    """Download + extract one map. Returns True if it's now on disk."""
    name = re.sub(r"[^\w\s-]", "", f"{doc['metadata']['songName']} - "
                  f"{doc['uploader']['name']}")[:60].strip()
    folder = DEST / f"{doc['id']} ({name})"
    if folder.exists():
        return True
    try:
        z = zipfile.ZipFile(io.BytesIO(get(doc["versions"][0]["downloadURL"]).content))
        folder.mkdir(parents=True)
        for n in z.namelist():
            base = Path(n).name
            if base and Path(base).suffix.lower() not in SKIP_EXT:
                (folder / base).write_bytes(z.read(n))
        return True
    except Exception as e:
        print(f"  failed {doc['id']} {name[:30]}: {e}")
        return False


def scrape_ryger():
    uid = get(f"{API}/users/name/Ryger").json()["id"]
    got, page = 0, 0
    while True:
        docs = get(f"{API}/maps/uploader/{uid}/{page}").json()["docs"]
        if not docs:
            return got
        for doc in docs:
            if ep_standard(doc) and doc["stats"]["score"] >= RYGER_MIN_SCORE:
                got += save_map(doc)
        page += 1


def scrape_top(limit):
    got, page = 0, 0
    while got < limit and page < 500:
        docs = get(f"{API}/search/text/{page}",
                   params={"sortOrder": "Rating", "minRating": MIN_SCORE,
                           "noodle": "false", "me": "false"}).json().get("docs", [])
        if not docs:
            break
        for doc in docs:
            if got >= limit:
                break
            if want(doc):
                got += save_map(doc)
                if got % 25 == 0:
                    print(f"  {got}/{limit} top-rated maps")
        page += 1
    return got


def scrape_scoresaber(limit):
    """ScoreSaber-ranked EP standard maps: ranked status is a human quality
    gate, so no vote/score filter — used when the rating search runs dry
    (BeatSaver's search pagination caps out well before the catalog does)."""
    got, page, seen = 0, 1, set()
    while got < limit:
        lbs = get("https://scoresaber.com/api/leaderboards",
                  params={"ranked": "true", "page": page}).json()["leaderboards"]
        if not lbs:
            break
        for lb in lbs:
            h, d = lb["songHash"].lower(), lb["difficulty"]
            if (d["difficulty"] != 9 or d["gameMode"] != "SoloStandard"
                    or h in seen):
                continue
            seen.add(h)
            if got >= limit:
                break
            try:
                doc = get(f"{API}/maps/hash/{h}").json()
            except Exception:
                continue  # deleted from beatsaver
            if ep_standard(doc) and 60 <= doc["metadata"]["duration"] <= 600:
                got += save_map(doc)
                if got and got % 25 == 0:
                    print(f"  {got}/{limit} scoresaber-ranked maps")
        page += 1
    return got


if __name__ == "__main__":
    def _arg(name, default, cast):
        return cast(sys.argv[sys.argv.index(name) + 1]) \
            if name in sys.argv else default
    limit = _arg("--limit", 500, int)
    MIN_SCORE = _arg("--min-score", MIN_SCORE, float)
    MIN_VOTES = _arg("--min-votes", MIN_VOTES, int)
    DEST.mkdir(exist_ok=True)
    r = scrape_ryger()
    print(f"Ryger: {r} maps")
    t = scrape_top(max(0, limit - r))
    print(f"top-rated: {t} maps")
    have = sum(1 for d in DEST.iterdir() if d.is_dir())
    if have < limit:
        print(f"scoresaber-ranked: {scrape_scoresaber(limit - have)} maps")
    print(f"total on disk: {sum(1 for d in DEST.iterdir() if d.is_dir())}")
