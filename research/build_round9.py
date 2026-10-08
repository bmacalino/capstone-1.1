"""Rounds 9, 10 and 11: more real sentences for the three weak labels.

Poor quality, possible fake and not as advertised are missed when they are
worded in ways the rules have not seen (research/RESULTS_TABLES.md, Table
11j, step 9). The reports we lack are in the unused reviews, but the rules
cannot find them. Here a pointer model picks likely ones, and the AI
assistant then labels every picked sentence (research/label_round9_ai.py and
its successors) without seeing why it was picked.

Each run builds one batch of about 1,000 sentences as its own round:

    python research/build_round9.py --round 9
    python research/build_round9.py --round 10      # after round 9 is labelled
    python research/build_round9.py --round 11

Pool: gadget reviews no earlier round used, from our own scans (training
listings only: the live-check listings stay untouched) and from the public
Shopee set. The held-out third of research/improve_label_rules.py is made of
earlier rounds, so no new sentence can join it.

Pointer model: the design in use (TF-IDF and keyword features, logistic
regression), one per label, trained on the dev two thirds of every labelled
round, on the rounds built here before this one, and on the written
sentences. It only chooses what gets read; it decides no label.

Strata, fixed here:
  model_<label>   sentences the rules do not flag for the label, half of them
                  the model's highest scores and half drawn from the next
                  band, so that not only its favourite wordings are read
  rules_<label>   a sample the rules do flag, to check the rules on new text
  aspect_words    size, capacity, photo or authenticity words, flagged by
                  nothing
  random          anything else
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

ROUND = int(sys.argv[sys.argv.index("--round") + 1]) if "--round" in sys.argv else 9
OUT = ROOT / "outputs" / f"labelling_queue_round_{ROUND}"
SEED = 20261000 + ROUND
TARGETS = ("not_as_described", "quality", "counterfeit")
MODEL_BUDGET = {"not_as_described": 250, "quality": 200, "counterfeit": 150}
RULES_BUDGET = 60
ASPECT_BUDGET, RANDOM_BUDGET = 150, 100
FIELDS = [
    "row_id", "part", "stratum", "cohort", "rating", "product_id", "category", "review_id", "sentence_index",
    "sentence_text", "review_text_raw", "score_not_as_described", "score_quality", "score_counterfeit",
]


def pool() -> list[dict[str, Any]]:
    """One row per unused gadget sentence, shaped like a labelled record."""
    from build_labelling_queue import load_shopee_hf, sentences_for, text_key
    from gadget_scope import is_gadget
    from ordinary_review_pool import unused_reviews, used_review_keys

    used = used_review_keys(ROUND - 1)
    reviews = [
        {**review, "cohort": "own_scrape"} for review in unused_reviews(ROUND - 1) if review["part"] == "mining"
    ]
    seen = {review["review_key"] for review in reviews}
    for review in load_shopee_hf(ROOT / "research" / "data" / "shopee_hf")[0]:
        key = text_key(review["review_text"])
        if key in used or key in seen or not is_gadget(review):
            continue
        seen.add(key)
        reviews.append({**review, "review_key": key, "part": "mining"})
    rows = []
    for review in reviews:
        prefix = "own_" if review["cohort"] == "own_scrape" else "shf_"
        for index, sentence in enumerate(sentences_for(review["review_text"]), start=1):
            rows.append({
                "review_id": f"{prefix}-{review['review_key'][:12]}", "cohort": review["cohort"], "part": "mining",
                "product_id": review.get("product_id", ""), "category": review.get("category", ""),
                "rating": review.get("rating", ""), "sentence_index": index, "sentence_text": sentence,
                "text": sentence, "review_text_raw": review["review_text"],
            })
    return rows


def main() -> int:
    from build_labelling_queue import ASPECT, AUTHENTICITY
    from defaketive_sentiment.hazard_learned import fit_category_model
    from five_label_data import keyword_view, load_five_label_records
    from improve_label_rules import held_out
    from synthetic_data import load_synthetic_records
    from train_hazard_classifier import SEED as MODEL_SEED

    labelled = [record for record in load_five_label_records() if not held_out(record)] + load_synthetic_records()
    rows = pool()
    view = keyword_view(labelled + rows)
    first = len(labelled)
    rng = random.Random(SEED)
    scores: dict[str, np.ndarray] = {}
    for label in TARGETS:
        train = np.array([i for i, record in enumerate(labelled) if record["labels"].get(label) is not None])
        y = np.array([int(bool(labelled[i]["labels"][label])) for i in train])
        model = fit_category_model("TK", list(view["text"][train]), view["kw"][train], None, y, category=label, seed=MODEL_SEED)
        scores[label] = model.probability(list(view["text"][first:]), view["kw"][first:], None)
        for row, score in zip(rows, scores[label]):
            row[f"score_{label}"] = round(float(score), 4)
    flagged = {label: view["flags"][label][first:].astype(bool) for label in TARGETS}

    chosen: dict[int, str] = {}
    pools: dict[str, int] = {}
    for label in TARGETS:
        ranked = [int(i) for i in np.argsort(-scores[label]) if not flagged[label][i] and i not in chosen]
        half = MODEL_BUDGET[label] // 2
        band = ranked[half: half + 8 * half]
        picks = ranked[:half] + rng.sample(band, min(half, len(band)))
        pools[f"model_{label}"] = half + len(band)
        for i in picks:
            chosen[i] = f"model_{label}"
    for label in TARGETS:
        hits = [i for i in range(len(rows)) if flagged[label][i] and i not in chosen]
        pools[f"rules_{label}"] = len(hits)
        for i in rng.sample(hits, min(RULES_BUDGET, len(hits))):
            chosen[i] = f"rules_{label}"
    unflagged = [i for i in range(len(rows)) if i not in chosen and not any(flagged[label][i] for label in TARGETS)]
    aspect = [i for i in unflagged if ASPECT.search(rows[i]["sentence_text"]) or AUTHENTICITY.search(rows[i]["sentence_text"])]
    pools["aspect_words"] = len(aspect)
    for i in rng.sample(aspect, min(ASPECT_BUDGET, len(aspect))):
        chosen[i] = "aspect_words"
    rest = [i for i in unflagged if i not in chosen]
    pools["random"] = len(rest)
    for i in rng.sample(rest, min(RANDOM_BUDGET, len(rest))):
        chosen[i] = "random"

    order = list(chosen)
    rng.shuffle(order)
    out_rows = [
        {**{key: rows[i].get(key, "") for key in FIELDS}, "row_id": f"r{ROUND}-{number:05d}", "stratum": chosen[i]}
        for number, i in enumerate(order, start=1)
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(out_rows)
    (OUT / "queue_master.csv").write_bytes(("﻿" + buffer.getvalue()).encode("utf-8"))
    manifest = {
        "round": ROUND, "seed": SEED,
        "purpose": "more real sentences for poor quality, possible fake and not as advertised; AI-labelled, training only",
        "pool_sentences": len(rows), "pool_by_cohort": dict(Counter(row["cohort"] for row in rows)),
        "pointer_model_trained_on": len(labelled), "pools": pools,
        "sampled": dict(Counter(row["stratum"] for row in out_rows)), "sentences": len(out_rows),
        "reviews": len({row["review_id"] for row in out_rows}),
    }
    (OUT / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
