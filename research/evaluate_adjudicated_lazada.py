"""Evaluate the unchanged marketplace model on the adjudicated Lazada test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_utils import evaluate_labels
from train_marketplace_v2 import (
    _multilabel_metrics,
    _risk_metrics,
    category_predictions,
    load_marketplace,
    uncertainty_predictions,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "lazada_gold_test_v1"
DEFAULT_DATASET = DEFAULT_OUTPUT_DIR / "lazada_round1_gold_test_v1.csv"
DEFAULT_ARTIFACT = ROOT / "sentiment-analysis" / "models" / "lazada_round1_binary_uncertain.joblib"
DEFAULT_BASELINE = ROOT / "sentiment-analysis" / "evaluation" / "lazada_round1_binary_uncertain.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "final_model_evaluation.json"
DEFAULT_ERRORS = DEFAULT_OUTPUT_DIR / "final_model_errors.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def delta(after: float, before: float) -> float:
    return round(float(after) - float(before), 4)


def evaluate(dataset: Path, artifact_path: Path, baseline_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_marketplace(dataset)
    test = [record for record in records if record.get("split") == "test"]
    if not test:
        raise ValueError("The adjudicated dataset has no test rows.")

    artifact = joblib.load(artifact_path)
    errors: list[dict[str, Any]] = []

    def add_error(task: str, record: dict[str, Any], expected: object, predicted: object) -> None:
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

    sentiment_expected = [
        record["sentiment"] if record["sentiment"] in {"negative", "positive"} else "uncertain"
        for record in test
    ]
    sentiment_pipeline = artifact["sentiment_pipeline"]
    sentiment_threshold = float(artifact["sentiment_uncertainty_threshold"])
    sentiment_predicted = uncertainty_predictions(
        sentiment_pipeline.predict_proba([record["text"] for record in test]),
        sentiment_pipeline.classes_,
        sentiment_threshold,
    )
    sentiment_metrics = evaluate_labels(sentiment_expected, sentiment_predicted)
    for record, expected, predicted in zip(test, sentiment_expected, sentiment_predicted):
        add_error("sentiment", record, expected, predicted)

    risk_expected = [record["risk_detected"] for record in test]
    risk_predicted = artifact["risk_detection_pipeline"].predict(
        [record["text"] for record in test]
    ).tolist()
    risk_metrics = _risk_metrics(risk_expected, risk_predicted)
    for record, expected, predicted in zip(test, risk_expected, risk_predicted):
        add_error("risk_detection", record, int(expected), int(predicted))

    category_rows = [
        record for record in test if record["risk_detected"] == 1 and record["risk_categories"]
    ]
    binarizer = artifact["risk_category_binarizer"]
    known_category_rows = [
        record
        for record in category_rows
        if set(record["risk_categories"]).issubset(binarizer.classes_)
    ]
    category_actual = binarizer.transform(
        [record["risk_categories"] for record in known_category_rows]
    )
    category_probabilities = artifact["risk_category_pipeline"].predict_proba(
        [record["text"] for record in known_category_rows]
    )
    category_threshold = float(artifact["risk_category_threshold"])
    category_predicted = category_predictions(category_probabilities, category_threshold)
    category_metrics = _multilabel_metrics(
        category_actual,
        category_predicted,
        binarizer.classes_.tolist(),
    )
    category_metrics["unknown_category_rows_excluded"] = len(category_rows) - len(known_category_rows)
    category_metrics["probability_threshold"] = category_threshold
    predicted_categories = binarizer.inverse_transform(category_predicted)
    for record, predicted in zip(known_category_rows, predicted_categories):
        add_error(
            "risk_category",
            record,
            "|".join(record["risk_categories"]),
            "|".join(predicted),
        )

    severity_rows = [
        record
        for record in test
        if record["risk_detected"] == 1 and record["risk_severity"] is not None
    ]
    severity_expected = [record["risk_severity"] for record in severity_rows]
    severity_predicted = artifact["risk_severity_pipeline"].predict(
        [record["text"] for record in severity_rows]
    ).tolist()
    severity_metrics = {
        "sample_count": len(severity_expected),
        "accuracy": round(float(accuracy_score(severity_expected, severity_predicted)), 4),
        "macro_f1": round(
            float(f1_score(severity_expected, severity_predicted, average="macro", zero_division=0)),
            4,
        ),
        "weighted_kappa": round(
            float(cohen_kappa_score(severity_expected, severity_predicted, weights="quadratic")),
            4,
        ),
        "mean_absolute_error": round(
            float(np.mean(np.abs(np.asarray(severity_expected) - np.asarray(severity_predicted)))),
            4,
        ),
    }
    for record, expected, predicted in zip(severity_rows, severity_expected, severity_predicted):
        add_error("risk_severity", record, int(expected), int(predicted))

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["test"]
    comparison = {
        "sentiment_macro_f1": {
            "before": baseline["sentiment"]["f1"],
            "after": sentiment_metrics["f1"],
            "delta": delta(sentiment_metrics["f1"], baseline["sentiment"]["f1"]),
        },
        "uncertain_f1": {
            "before": baseline["sentiment"]["per_class"]["uncertain"]["f1"],
            "after": sentiment_metrics["per_class"]["uncertain"]["f1"],
            "delta": delta(
                sentiment_metrics["per_class"]["uncertain"]["f1"],
                baseline["sentiment"]["per_class"]["uncertain"]["f1"],
            ),
        },
        "risk_macro_f1": {
            "before": baseline["risk_detection"]["macro_f1"],
            "after": risk_metrics["macro_f1"],
            "delta": delta(risk_metrics["macro_f1"], baseline["risk_detection"]["macro_f1"]),
        },
        "risk_recall": {
            "before": baseline["risk_detection"]["risk"]["recall"],
            "after": risk_metrics["risk"]["recall"],
            "delta": delta(risk_metrics["risk"]["recall"], baseline["risk_detection"]["risk"]["recall"]),
        },
        "category_macro_f1": {
            "before": baseline["risk_category"]["macro_f1"],
            "after": category_metrics["macro_f1"],
            "delta": delta(category_metrics["macro_f1"], baseline["risk_category"]["macro_f1"]),
        },
        "severity_weighted_kappa": {
            "before": baseline["risk_severity"]["weighted_kappa"],
            "after": severity_metrics["weighted_kappa"],
            "delta": delta(
                severity_metrics["weighted_kappa"],
                baseline["risk_severity"]["weighted_kappa"],
            ),
        },
    }

    gates = {
        "sentiment_macro_f1_at_least_0_75": sentiment_metrics["f1"] >= 0.75,
        "uncertain_f1_at_least_0_60": sentiment_metrics["per_class"]["uncertain"]["f1"] >= 0.60,
        "risk_recall_at_least_0_80": risk_metrics["risk"]["recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": risk_metrics["macro_f1"] >= 0.75,
        "category_macro_f1_at_least_0_65": category_metrics["macro_f1"] >= 0.65,
        "severity_weighted_kappa_at_least_0_65": severity_metrics["weighted_kappa"] >= 0.65,
    }

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provisional": False,
            "evaluation_set_status": "final_locked",
            "model_retrained": False,
            "dataset": dataset.name,
            "dataset_sha256": sha256(dataset),
            "artifact": artifact_path.name,
            "artifact_sha256": sha256(artifact_path),
            "test_rows": len(test),
            "gates": gates,
        },
        "test": {
            "sentiment": sentiment_metrics,
            "risk_detection": risk_metrics,
            "risk_category": category_metrics,
            "risk_severity": severity_metrics,
        },
        "comparison_to_pre_adjudication_report": comparison,
        "error_counts": {
            task: sum(error["task"] == task for error in errors)
            for task in ("sentiment", "risk_detection", "risk_category", "risk_severity")
        },
    }
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    args = parser.parse_args()

    report, errors = evaluate(
        args.dataset.resolve(),
        args.artifact.resolve(),
        args.baseline.resolve(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.errors.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "review_id", "product_id", "expected", "predicted", "review_text"],
        )
        writer.writeheader()
        writer.writerows(errors)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
