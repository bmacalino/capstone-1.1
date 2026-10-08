"""Build labelling round 8: sentences from ordinary reviews the defect models flag.

Rounds 1-7 came from enriched strata, so a defect model trained on them had
seen few plain remarks about a working product. On unused 4-5 star reviews of
our electronics scrapes the keywords flag 2% of sentences, the team-label
models (hazard_classifier_v2b) 4% and the model retrained on rounds 1-7
(hazard_classifier_v4) 7%. Round 8 samples those flags so they can be
labelled (research/ordinary_review_pool.py explains the two parts):

  mining part (listings used for training)
    mining_flagged_high   150  flagged by either model, not by a keyword, 4-5 stars
    mining_flagged_low    150  the same, 1-3 stars
    mining_unflagged       60  flagged by neither model, any stars
  live_check part (listings never used for training)
    live_flagged          all  flagged by either model (keywords included)
    live_unflagged        110  flagged by neither

The first tagging pass is cached in outputs/cache/ because it takes ten
minutes; delete the cache to recompute it.

    python research/build_ordinary_round8.py
"""

from __future__ import annotations

import csv
import io
import json
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ordinary_review_pool import checker, sentences_of, unused_reviews  # noqa: E402

OUT = ROOT / "outputs" / "labelling_queue_round_8"
CACHE = ROOT / "outputs" / "cache" / "ordinary_pool_v2b_v4.pkl"
SEED = 20261010
BUDGET = {"mining_flagged_high": 150, "mining_flagged_low": 150, "mining_unflagged": 60, "live_unflagged": 110}
FIELDS = [
    "row_id", "part", "stratum", "pool", "sampled", "rating", "product_id", "category", "review_id", "sentence_index",
    "sentence_text", "review_text_raw", "flagged_team_label_models", "flagged_v4", "keyword_flag",
]


def tagged_pool() -> list[dict]:
    if CACHE.is_file():
        with CACHE.open("rb") as source:
            return pickle.load(source)
    rows = sentences_of(unused_reviews())
    old, new = checker("hazard_classifier_v2b"), checker("hazard_classifier_v4")
    for row in rows:
        a, b = old.tag_sentence(row["sentence_text"]), new.tag_sentence(row["sentence_text"])
        row["old"], row["new"] = "defect" in a["tags"], "defect" in b["tags"]
        row["keyword"] = any(
            match["category"] == "defect" and not match["negated"] and not match.get("excluded") for match in b["terms"]
        )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as target:
        pickle.dump(rows, target)
    return rows


def stratum_of(row: dict) -> str | None:
    flagged = row["old"] or row["new"]
    if row["part"] == "live_check":
        return "live_flagged" if flagged else "live_unflagged"
    if not flagged:
        return "mining_unflagged"
    if row["keyword"]:
        return None  # a keyword flag is not what round 8 is about
    return "mining_flagged_high" if row["rating"] in (4, 5) else "mining_flagged_low"


def main() -> int:
    pools: dict[str, list[dict]] = {}
    for row in tagged_pool():
        name = stratum_of(row)
        if name:
            pools.setdefault(name, []).append(row)
    rng = random.Random(SEED)
    chosen = []
    for name in sorted(pools):
        pool = pools[name]
        take = pool if name not in BUDGET else rng.sample(pool, min(BUDGET[name], len(pool)))
        for row in take:
            chosen.append({**row, "stratum": name, "pool": len(pool), "sampled": len(take)})
    rng.shuffle(chosen)
    rows = []
    for index, row in enumerate(chosen, start=1):
        rows.append({
            **{key: row.get(key, "") for key in FIELDS},
            "row_id": f"r8-{index:05d}",
            "flagged_team_label_models": "yes" if row["old"] else "no",
            "flagged_v4": "yes" if row["new"] else "no",
            "keyword_flag": "yes" if row["keyword"] else "no",
        })
    OUT.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (OUT / "queue_master.csv").write_bytes(("﻿" + buffer.getvalue()).encode("utf-8"))
    manifest = {
        "round": 8, "seed": SEED,
        "purpose": "sentences the defect models flag in ordinary, unused gadget reviews; AI-labelled, not ground truth",
        "pools": {name: len(pool) for name, pool in sorted(pools.items())},
        "sampled": dict(Counter(row["stratum"] for row in rows)),
        "listings": {part: len({row["product_id"] for row in rows if row["part"] == part}) for part in ("mining", "live_check")},
    }
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
