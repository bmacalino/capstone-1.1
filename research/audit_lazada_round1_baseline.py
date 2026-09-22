"""Create row-level error files for the locked Lazada Round 1 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_marketplace_v2 import (
    category_predictions,
    load_marketplace,
    uncertainty_predictions,
)
DEFAULT_DATASET = ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_all.csv"
DEFAULT_ARTIFACT = ROOT / "sentiment-analysis" / "models" / "lazada_round1_marketplace_v2_baseline.joblib"
DEFAULT_OUTPUT = ROOT / "sentiment-analysis" / "evaluation" / "lazada_round1_marketplace_v2_errors.csv"
DEFAULT_SUMMARY = ROOT / "sentiment-analysis" / "evaluation" / "lazada_round1_marketplace_v2_error_summary.json"
SENTIMENT_LABELS = {"negative", "neutral", "positive"}


def create_audit(dataset: Path, artifact_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_marketplace(dataset)
    test = [record for record in records if record.get("split") == "test"]
    artifact = joblib.load(artifact_path)
    errors: list[dict[str, Any]] = []

    def add(task: str, record: dict[str, Any], expected: object, predicted: object) -> None:
        if expected == predicted:
            return
        errors.append(
            {
                "task": task,
                "review_id": record["review_id"],
                "product_id": record["product_id"],
                "expected": expected,
                "predicted": predicted,
                "review_text": record["review"],
            }
        )

    uncertainty_threshold = artifact.get("sentiment_uncertainty_threshold")
    if uncertainty_threshold is None:
        sentiment_rows = [record for record in test if record["sentiment"] in SENTIMENT_LABELS]
        sentiment_predictions = artifact["sentiment_pipeline"].predict(
            [record["text"] for record in sentiment_rows]
        )
        sentiment_expected = [record["sentiment"] for record in sentiment_rows]
    else:
        sentiment_rows = test
        sentiment_pipeline = artifact["sentiment_pipeline"]
        sentiment_predictions = uncertainty_predictions(
            sentiment_pipeline.predict_proba([record["text"] for record in sentiment_rows]),
            sentiment_pipeline.classes_,
            float(uncertainty_threshold),
        )
        sentiment_expected = [
            record["sentiment"] if record["sentiment"] in {"negative", "positive"} else "uncertain"
            for record in sentiment_rows
        ]
    for record, expected, prediction in zip(
        sentiment_rows, sentiment_expected, sentiment_predictions
    ):
        add("sentiment", record, expected, str(prediction))

    risk_predictions = artifact["risk_detection_pipeline"].predict(
        [record["text"] for record in test]
    )
    for record, prediction in zip(test, risk_predictions):
        add("risk_detection", record, int(record["risk_detected"]), int(prediction))

    category_rows = [
        record for record in test
        if record["risk_detected"] == 1 and record["risk_categories"]
    ]
    binarizer = artifact["risk_category_binarizer"]
    known_category_rows = [
        record for record in category_rows
        if set(record["risk_categories"]).issubset(binarizer.classes_)
    ]
    probabilities = artifact["risk_category_pipeline"].predict_proba(
        [record["text"] for record in known_category_rows]
    )
    predicted_binary = category_predictions(probabilities, artifact["risk_category_threshold"])
    predicted_categories = binarizer.inverse_transform(predicted_binary)
    for record, prediction in zip(known_category_rows, predicted_categories):
        add("risk_category", record, "|".join(record["risk_categories"]), "|".join(prediction))

    severity_rows = [
        record for record in test
        if record["risk_detected"] == 1 and record["risk_severity"] is not None
    ]
    severity_predictions = artifact["risk_severity_pipeline"].predict(
        [record["text"] for record in severity_rows]
    )
    for record, prediction in zip(severity_rows, severity_predictions):
        add("risk_severity", record, int(record["risk_severity"]), int(prediction))

    summary = {
        "dataset": dataset.name,
        "artifact": artifact_path.name,
        "test_rows": len(test),
        "evaluated": {
            "sentiment": len(sentiment_rows),
            "risk_detection": len(test),
            "risk_category": len(known_category_rows),
            "risk_severity": len(severity_rows),
        },
        "error_counts": dict(sorted(Counter(error["task"] for error in errors).items())),
        "sentiment_confusions": dict(
            sorted(
                Counter(
                    f"{error['expected']} -> {error['predicted']}"
                    for error in errors if error["task"] == "sentiment"
                ).items()
            )
        ),
    }
    return errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    errors, summary = create_audit(args.dataset.resolve(), args.artifact.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "review_id", "product_id", "expected", "predicted", "review_text"],
        )
        writer.writeheader()
        writer.writerows(errors)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
