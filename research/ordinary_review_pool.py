"""Ordinary gadget reviews no labelling round has used, split by listing.

The labelling rounds were drawn from enriched strata (keyword hits, low
stars, model disagreements), so they hold few of the plain remarks that fill
a real listing: "Gumagana naman sya", "tried charging it for about 2hrs". A
defect model that looks good on the rounds can still quote such remarks as
defect reports (the Anker check of 7 October 2026).

This module gives the unused reviews of our own electronics scrapes, split by
listing into two parts that never share a product:

  mining      about four listings in five: where sentences the model wrongly
              flags are found and labelled for training
  live_check  the rest: left untouched by training, to see how a model behaves
              on whole listings

and tags every sentence with a checker, caching the sentiment readings.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

OUTPUTS = ROOT / "outputs"
LIVE_CHECK_SHARE = 5  # one listing in five


def used_review_keys(last_round: int) -> set[str]:
    from build_labelling_queue import text_key

    keys: set[str] = set()
    for number in range(1, last_round + 1):
        for name in ("queue_master.csv", "practice_set_master.csv"):
            path = OUTPUTS / f"labelling_queue_round_{number}" / name
            if not path.is_file():
                continue
            with path.open(encoding="utf-8-sig", newline="") as source:
                keys |= {text_key(row["review_text_raw"]) for row in csv.DictReader(source) if row.get("review_text_raw")}
    return keys


def part_of(product_id: str) -> str:
    digest = int(hashlib.sha1(product_id.encode("utf-8")).hexdigest(), 16)
    return "live_check" if digest % LIVE_CHECK_SHARE == 0 else "mining"


def unused_reviews(last_round: int = 7) -> list[dict[str, Any]]:
    """Unused reviews of our electronics scrapes, each marked mining or live_check."""
    from build_labelling_queue import load_own_scrapes, text_key
    from gadget_scope import GADGET_CATEGORIES

    used = used_review_keys(last_round)
    seen: set[str] = set()
    reviews = []
    for review in load_own_scrapes()[0]:
        key = text_key(review["review_text"])
        if key in used or key in seen or not review.get("product_id"):
            continue
        if str(review.get("category") or "").casefold() not in GADGET_CATEGORIES:
            continue
        seen.add(key)
        reviews.append({**review, "review_key": key, "part": part_of(review["product_id"])})
    return reviews


def sentences_of(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from build_labelling_queue import sentences_for

    rows = []
    for review in reviews:
        for index, sentence in enumerate(sentences_for(review["review_text"]), start=1):
            rows.append({
                "review_id": f"own_-{review['review_key'][:12]}", "product_id": review["product_id"],
                "product_name": review.get("product_name", ""), "category": review.get("category", ""),
                "rating": review.get("rating", ""), "part": review["part"], "sentence_index": index,
                "sentence_text": sentence, "review_text_raw": review["review_text"],
            })
    return rows


def checker(folder_name: str):
    """A served-style tagger for one model folder, with cached sentiment."""
    from defaketive_sentiment.hazard_learned import MODELS_ROOT, LearnedHazardScorer
    from defaketive_sentiment.hazard_model import LexiconHazardTagger
    from five_label_data import _classifier

    read, classifier, _ = _classifier()
    known: dict[str, str] = {}

    def label(text: str) -> str:
        if text not in known:
            known[text] = read(text)
        return known[text]

    scorer = LearnedHazardScorer(MODELS_ROOT / folder_name, sentiment_model_sha256=classifier.manifest["files"]["model.onnx"])
    return LexiconHazardTagger.for_scorer(scorer, sentiment_label=label)
