"""Freeze the bounded MI-inference subset for the evidence premise check.

review's spec: 20 new families (12 train + 8 val) across four broad groups
(punk/rock, electronic, pop, rap/hip-hop), 3 train + 2 val each, >=3 punk in
punk/rock, contrasting tempo/density. Genre labels need PROVENANCE/confidence
and must not be silently guessed — so only families whose artist this curator
recognizes are labeled; the rest stay 'unknown' and are not eligible. If a
stratum can't be filled from confident labels, that's reported and the claim
reduced, never back-filled with guesses.

  .venv/bin/python -m eval.pilot_subset   # writes experiments/pilot-subset.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MANIFEST = Path(__file__).parent / "corpus_manifest.json"
OUT = Path(__file__).parent.parent / "experiments" / "pilot-subset.json"

# curated artist/title -> broad group, by this model's music knowledge
# (provenance recorded per pick; confidence: hi = well-known artist+genre,
# med = reasonably confident). Keys matched case-insensitively as substrings
# of "song_author" (preferred) or "song_name".
GENRE = {
    "punk_rock": {
        "linkin park": "hi", "bring me the horizon": "hi", "set it off": "hi",
        "5 seconds of summer": "hi", "unleash the archers": "hi",
        "men at work": "hi", "paramore": "hi", "fall out boy": "hi",
        "my chemical romance": "hi", "green day": "hi", "the offspring": "hi",
        "sum 41": "hi", "all time low": "hi", "a day to remember": "hi",
        "spiritbox": "hi", "electric callboy": "hi", "rise against": "hi",
    },
    "electronic": {
        "tiësto": "hi", "deorro": "hi", "porter robinson": "hi",
        "teminite": "hi", "mdk": "hi", "dabin": "hi", "kai wachi": "hi",
        "dirty palm": "hi", "fox stevenson": "hi", "pixel terror": "hi",
        "camellia": "hi", "virtual riot": "hi", "skrillex": "hi",
        "deadmau5": "hi", "madeon": "hi", "knife party": "hi", "wrld": "med",
        "sota": "med", "aviella": "med",
    },
    "pop": {
        "clean bandit": "hi", "charli xcx": "hi", "marina": "hi",
        "pitbull": "hi", "ray parker": "hi", "troye sivan": "hi",
        "dua lipa": "hi", "taylor swift": "hi", "the weeknd": "hi",
        "ed sheeran": "hi", "lady gaga": "hi", "katy perry": "hi",
        "carly rae": "hi", "kyary pamyu pamyu": "med",
    },
    "rap_hiphop": {
        "tech n9ne": "hi", "rittz": "hi", "bbno$": "hi", "eminem": "hi",
        "kendrick": "hi", "drake": "hi", "kanye": "hi", "tyler the creator": "hi",
        "denzel curry": "hi", "run the jewels": "hi", "childish gambino": "hi",
        "nf": "hi", "logic": "hi", "ken carson": "med", "playboi carti": "med",
    },
}


def _group(song, author):
    a = (author or "").lower()
    n = (song or "").lower()
    for grp, arts in GENRE.items():
        for key, conf in arts.items():
            if key in a or key in n:
                return grp, conf, f"artist/title match {key!r}"
    return None, None, None


def build():
    m = json.loads(MANIFEST.read_text())
    elig = [r for r in m["maps"] if r["split"] in ("train", "val")
            and r.get("family_rep") and r.get("eligible") == "ok"]
    labeled = {"punk_rock": [], "electronic": [], "pop": [], "rap_hiphop": []}
    for r in elig:
        grp, conf, why = _group(r["song_name"], r["song_author"])
        if grp:
            labeled[grp].append({
                "family": r["family"], "split": r["split"], "dir": r["dir"],
                "song": r["song_name"], "author": r["song_author"],
                "bpm": r["bpm"], "genre": grp, "genre_confidence": conf,
                "genre_provenance": "manual (model music knowledge): " + why})
    # pick 3 train + 2 val per group, preferring contrasting BPM
    chosen, gaps = [], []
    for grp, cands in labeled.items():
        tr = sorted([c for c in cands if c["split"] == "train"],
                    key=lambda c: c["bpm"] or 0)
        va = sorted([c for c in cands if c["split"] == "val"],
                    key=lambda c: c["bpm"] or 0)

        def spread(xs, k):  # pick k spread across the BPM range
            if len(xs) <= k:
                return xs
            idx = [round(i * (len(xs) - 1) / (k - 1)) for i in range(k)]
            return [xs[i] for i in sorted(set(idx))]
        pt, pv = spread(tr, 3), spread(va, 2)
        chosen += pt + pv
        if len(pt) < 3:
            gaps.append(f"{grp}: only {len(pt)}/3 train labeled")
        if len(pv) < 2:
            gaps.append(f"{grp}: only {len(pv)}/2 val labeled")
    subset = {
        "note": "Frozen MI-inference premise-check subset (review-approved <=20). "
                "Genre labels are manual with recorded confidence; unlabeled "
                "families are excluded, not guessed. Gaps reduce the claim.",
        "target": "12 train + 8 val across 4 groups (3 train + 2 val each)",
        "chosen": chosen, "count": len(chosen), "gaps": gaps,
        "labeled_pool": {g: len(v) for g, v in labeled.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(subset, indent=1))
    from collections import Counter
    print(f"pilot subset: {len(chosen)} families "
          f"({dict(Counter((c['genre'], c['split']) for c in chosen))})")
    print(f"  labeled pool per group: {subset['labeled_pool']}")
    for g in gaps:
        print(f"  GAP: {g}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    build()
