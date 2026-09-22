"""Run the current sentiment model against local annotated and scraped reviews."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from defaketive_sentiment.model import DefaketiveSentimentModel
from research_utils import evaluate_labels


def annotated_results(model: DefaketiveSentimentModel, path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    actual = [row["label"] for row in rows]
    predicted = [model.analyze(row["review"])["label"] for row in rows]
    return {
        "file": path.name,
        "metrics": evaluate_labels(actual, predicted),
        "predictions": [
            {
                "annotation_id": row["annotation_id"],
                "expected": row["label"],
                "predicted": prediction,
                "review": row["review"],
            }
            for row, prediction in zip(rows, predicted)
        ],
    }


def _products(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        products = payload.get("products")
        if isinstance(products, list):
            return [item for item in products if isinstance(item, dict)]
        return [payload]
    return []


def scraped_results(model: DefaketiveSentimentModel, root: Path) -> dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str]] = set()
    examples: list[dict[str, str]] = []
    file_count = 0
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_analyzed.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        file_count += 1
        fallback = next(
            (name for name in ("lazada", "shopee", "temu") if name in path.name),
            "unknown",
        )
        for product in _products(payload):
            platform = str(product.get("platform") or product.get("market") or fallback)
            for review in product.get("comments") or []:
                if not isinstance(review, dict):
                    continue
                text = str(review.get("content") or review.get("comment") or "").strip()
                key = (platform, text)
                if not text or key in seen:
                    continue
                seen.add(key)
                analysis = model.analyze(text, review.get("rating"))
                label = analysis["label"]
                counts[platform][label] += 1
                counts[platform]["risk"] += int(analysis["risk"]["detected"])
                if len(examples) < 12:
                    examples.append(
                        {
                            "platform": platform,
                            "label": label,
                            "risk": str(analysis["risk"]["detected"]),
                            "review": text,
                        }
                    )
    return {
        "file_count": file_count,
        "unique_review_count": len(seen),
        "by_platform": {platform: dict(values) for platform, values in sorted(counts.items())},
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--annotations", type=Path, default=SENTIMENT_ROOT / "data/firecs/manual_mixed_review_annotations.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = DefaketiveSentimentModel()
    result = {
        "model_version": model.model_version,
        "annotated": annotated_results(model, args.annotations),
        "scraped": scraped_results(model, args.root),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()