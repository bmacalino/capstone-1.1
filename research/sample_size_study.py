"""Estimate risk-score sampling stability without mistaking it for model accuracy.

The study repeatedly resamples unique reviews from each product and compares the
subsample score with that product's full captured-review score. A final research
recommendation additionally requires human gold labels and broad product coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SIZES = (10, 20, 30, 50, 75, 100, 150, 200)


def risk_score(records: Iterable[dict[str, Any]]) -> float:
    rows = list(records)
    if not rows:
        return 0.0
    negative = sum(bool(row["negative"]) for row in rows) / len(rows)
    signalled = sum(bool(row["risk_detected"]) for row in rows) / len(rows)
    return 100 * ((0.30 * negative) + (0.70 * signalled))


def risk_level(score: float) -> str:
    if score >= 61:
        return "High"
    if score >= 31:
        return "Moderate"
    return "Low"


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def evaluate(
    products: list[dict[str, Any]],
    sizes: Iterable[int] = DEFAULT_SIZES,
    *,
    trials: int = 1000,
    seed: int = 20260910,
    max_p90_error: float = 5.0,
    minimum_agreement: float = 0.90,
    human_gold_labels: bool = False,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows = []
    for size in sizes:
        errors: list[float] = []
        agreements: list[bool] = []
        eligible = [product for product in products if len(product["reviews"]) >= size]
        for product in eligible:
            reviews = product["reviews"]
            reference = risk_score(reviews)
            reference_level = risk_level(reference)
            for _ in range(trials):
                # Bootstrap resampling estimates repeat-sample instability. It
                # does not correct a biased captured review pool.
                sample = [reviews[rng.randrange(len(reviews))] for _ in range(size)]
                sampled_score = risk_score(sample)
                errors.append(abs(sampled_score - reference))
                agreements.append(risk_level(sampled_score) == reference_level)
        agreement = sum(agreements) / len(agreements) if agreements else 0.0
        p90 = percentile(errors, 0.90)
        rows.append(
            {
                "sample_size": size,
                "eligible_products": len(eligible),
                "simulations": len(errors),
                "median_absolute_error": round(statistics.median(errors), 3) if errors else None,
                "p90_absolute_error": round(p90, 3) if errors else None,
                "p95_absolute_error": round(percentile(errors, 0.95), 3) if errors else None,
                "risk_level_agreement": round(agreement, 4) if agreements else None,
                "pilot_pass": bool(errors and p90 <= max_p90_error and agreement >= minimum_agreement),
            }
        )

    platforms = sorted({str(product.get("platform") or "unknown") for product in products})
    strata = sorted(
        {str(product.get("stratum") or "unknown") for product in products}
        - {"unknown", ""}
    )
    reference_counts = [len(product["reviews"]) for product in products]
    coverage_ok = len(products) >= 30 and "lazada" in platforms and len(strata) >= 5
    passing = [row["sample_size"] for row in rows if row["pilot_pass"]]
    validated_passing = [
        row["sample_size"]
        for row in rows
        if row["pilot_pass"] and row["eligible_products"] >= 30
    ]
    return {
        "status": "validation_ready" if coverage_ok and human_gold_labels else "pilot_only",
        "recommendation": (
            min(validated_passing)
            if coverage_ok and human_gold_labels and validated_passing
            else None
        ),
        "pilot_candidate": min(passing) if passing else None,
        "criteria": {
            "p90_absolute_error_at_most": max_p90_error,
            "risk_level_agreement_at_least": minimum_agreement,
            "minimum_products_for_recommendation": 30,
            "required_marketplace": "lazada",
            "minimum_electronics_strata": 5,
        },
        "coverage": {
            "products": len(products),
            "platforms": platforms,
            "electronics_strata": strata,
            "unique_reviews": sum(reference_counts),
            "reviews_per_product": reference_counts,
            "human_gold_labels": human_gold_labels,
        },
        "results": rows,
        "limitations": [
            "Machine labels test score stability, not classifier or construct validity.",
            "The captured review pool must be representative; more reviews do not remove selection bias.",
            "A final recommendation requires human gold labels and an unseen-product holdout.",
        ],
    }


def _products_from_json(paths: list[Path]) -> list[dict[str, Any]]:
    sentiment_root = Path(__file__).resolve().parents[1] / "sentiment-analysis"
    sys.path.insert(0, str(sentiment_root))
    from defaketive_sentiment.model import DefaketiveSentimentModel

    model = DefaketiveSentimentModel()
    products = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = payload if isinstance(payload, list) else payload.get("products", [payload])
        for index, product in enumerate(candidates):
            if not isinstance(product, dict):
                continue
            seen: set[str] = set()
            reviews = []
            for review in product.get("comments") or []:
                text = str(review.get("content") or review.get("comment") or "")
                fingerprint = model.fingerprint(text)
                if not fingerprint or fingerprint in seen:
                    continue
                seen.add(fingerprint)
                analysis = review.get("sentiment_analysis") or model.analyze(text, review.get("rating"))
                reviews.append(
                    {
                        "negative": analysis.get("label") == "negative",
                        "risk_detected": bool((analysis.get("risk") or {}).get("detected")),
                    }
                )
            if reviews:
                products.append(
                    {
                        "product_id": str(product.get("link") or f"{path.name}:{index}"),
                        "platform": product.get("platform") or product.get("market") or "unknown",
                        "stratum": product.get("research_stratum") or "unknown",
                        "reviews": reviews,
                    }
                )
    return products


def _products_from_annotations(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            product_id = row["product_id"].strip()
            product = grouped.setdefault(
                product_id,
                {
                    "product_id": product_id,
                    "platform": row["platform"].strip(),
                    "stratum": row.get("stratum", "").strip(),
                    "reviews": [],
                },
            )
            product["reviews"].append(
                {
                    "negative": row["sentiment_label"].strip().casefold() == "negative",
                    "risk_detected": row["risk_detected"].strip().casefold() in {"1", "true", "yes"},
                }
            )
    return list(grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", nargs="+", type=Path, help="Captured product JSON files")
    source.add_argument("--annotations", type=Path, help="Human annotation CSV")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    products = _products_from_json(args.json) if args.json else _products_from_annotations(args.annotations)
    result = evaluate(products, trials=args.trials, human_gold_labels=bool(args.annotations))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
