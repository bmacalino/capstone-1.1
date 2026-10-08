"""Build labelling round 4: a counterfeit-only set, pre-labelled by an AI assistant.

Unlike rounds 1-3 this set is NOT human ground truth. The assistant labels every
sentence and marks its confidence; a person decides every uncertain one and
audits a random sample of the confident ones. It may be used to improve the
counterfeit check, never to score it.

Strata (sentence level, weights = pool / sampled):
  keyword_flag   sentences keyword checker 0.3.0 tags counterfeit (all)
  authenticity   other sentences with an authenticity word (sample)
  low_star       other sentences from 1-3 star reviews (sample)
No review from rounds 1-3 is used.

    python research/build_counterfeit_round4.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_labelling_queue import AUTHENTICITY, is_low_star, load_firecs, load_own_scrapes, sentences_for, text_key  # noqa: E402

OUT = ROOT / "outputs" / "labelling_queue_round_4"
SEED = 20261005
BUDGET = {"authenticity": 200, "low_star": 75}
FIELDS = ["row_id", "stratum", "sampling_weight", "rating", "sentence_text", "review_text_raw", "review_id"]


def main() -> int:
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    used = set()
    for number in (1, 2, 3):
        for name in ("queue_master.csv", "practice_set_master.csv"):
            with (ROOT / "outputs" / f"labelling_queue_round_{number}" / name).open(encoding="utf-8-sig", newline="") as source:
                used |= {text_key(row["review_text_raw"]) for row in csv.DictReader(source)}
    tagger = LexiconHazardTagger()
    pools: dict[str, list[dict]] = {"keyword_flag": [], "authenticity": [], "low_star": []}
    seen = set()
    for review in load_own_scrapes()[0] + load_firecs()[0]:
        key = text_key(review["review_text"])
        if key in used or key in seen:
            continue
        seen.add(key)
        for sentence in sentences_for(review["review_text"]):
            if "counterfeit" in tagger.tag_sentence(sentence)["tags"]:
                stratum = "keyword_flag"
            elif AUTHENTICITY.search(sentence):
                stratum = "authenticity"
            elif is_low_star(review):
                stratum = "low_star"
            else:
                continue
            pools[stratum].append({
                "stratum": stratum, "rating": review.get("rating", ""), "sentence_text": sentence,
                "review_text_raw": review["review_text"], "review_id": f"{review['cohort'][:4]}-{key[:12]}",
            })
    rng = random.Random(SEED)
    rows = []
    for stratum, pool in pools.items():
        take = pool if stratum not in BUDGET else rng.sample(pool, min(BUDGET[stratum], len(pool)))
        for row in take:
            row["sampling_weight"] = f"{len(pool) / len(take):.4f}"
        rows += take
    rng.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["row_id"] = f"r4-{index:05d}"
    OUT.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (OUT / "queue_master.csv").write_bytes(("﻿" + buffer.getvalue()).encode("utf-8"))
    manifest = {
        "round": 4, "seed": SEED, "purpose": "counterfeit only; AI-assisted labels, human-adjudicated; not ground truth",
        "pools": {name: len(pool) for name, pool in pools.items()},
        "sampled": {name: sum(1 for row in rows if row["stratum"] == name) for name in pools},
    }
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
