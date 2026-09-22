"""Compare risk-category and severity candidates on validation data only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_marketplace_v2 import (
    _multilabel_metrics,
    category_predictions,
    classifier_pipeline,
    load_marketplace,
    locked_product_split,
    multilabel_pipeline,
)
from sklearn.preprocessing import MultiLabelBinarizer


DATASET = ROOT / "outputs" / "lazada_gold_test_v1" / "lazada_round1_all_adjudicated.csv"


def ratings_by_id(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["review_id"].strip(): row.get("rating", "").strip()
            for row in csv.DictReader(handle)
        }


def rating_text(record: dict[str, object], ratings: dict[str, str]) -> str:
    rating = ratings.get(str(record["review_id"]), "")
    token = f" __rating_{rating}" if rating in {"1", "2", "3", "4", "5"} else " __rating_unknown"
    return f"{record['text']}{token}"


def per_class_thresholds(actual: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = []
    for index in range(actual.shape[1]):
        if int(actual[:, index].sum()) == 0:
            thresholds.append(1.0)
            continue
        threshold = max(
            (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60),
            key=lambda candidate: (
                f1_score(
                    actual[:, index],
                    (probabilities[:, index] >= candidate).astype(int),
                    zero_division=0,
                ),
                candidate,
            ),
        )
        thresholds.append(threshold)
    return np.asarray(thresholds, dtype=float)


def category_candidate(records, train_indices, validation_indices) -> dict[str, object]:
    train = [
        index
        for index in train_indices
        if records[index]["risk_detected"] == 1 and records[index]["risk_categories"]
    ]
    validation = [
        index
        for index in validation_indices
        if records[index]["risk_detected"] == 1 and records[index]["risk_categories"]
    ]
    binarizer = MultiLabelBinarizer()
    train_actual = binarizer.fit_transform([records[index]["risk_categories"] for index in train])
    validation = [
        index
        for index in validation
        if set(records[index]["risk_categories"]).issubset(binarizer.classes_)
    ]
    validation_actual = binarizer.transform(
        [records[index]["risk_categories"] for index in validation]
    )
    model = multilabel_pipeline()
    model.fit([records[index]["text"] for index in train], train_actual)
    probabilities = model.predict_proba([records[index]["text"] for index in validation])
    thresholds = per_class_thresholds(validation_actual, probabilities)
    predicted = category_predictions(probabilities, thresholds)
    metrics = _multilabel_metrics(
        validation_actual, predicted, binarizer.classes_.tolist()
    )
    predicted_labels = binarizer.inverse_transform(predicted)
    metrics["thresholds"] = {
        str(label): float(threshold)
        for label, threshold in zip(binarizer.classes_, thresholds)
    }
    metrics["errors"] = [
        {
            "review_id": records[index]["review_id"],
            "expected": list(records[index]["risk_categories"]),
            "predicted": list(labels),
            "text": records[index]["review"],
        }
        for index, labels in zip(validation, predicted_labels)
        if set(labels) != set(records[index]["risk_categories"])
    ]
    return metrics


def ordinal_probabilities(model_ge2, model_ge3, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probability_ge2 = model_ge2.predict_proba(texts)[:, list(model_ge2.classes_).index(1)]
    probability_ge3 = model_ge3.predict_proba(texts)[:, list(model_ge3.classes_).index(1)]
    probability_ge3 = np.minimum(probability_ge2, probability_ge3)
    return probability_ge2, probability_ge3


def ordinal_predictions(
    probability_ge2: np.ndarray,
    probability_ge3: np.ndarray,
    threshold_ge2: float,
    threshold_ge3: float,
) -> np.ndarray:
    return np.where(
        probability_ge3 >= threshold_ge3,
        3,
        np.where(probability_ge2 >= threshold_ge2, 2, 1),
    )


def severity_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(np.mean(actual == predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
        "weighted_kappa": round(float(cohen_kappa_score(actual, predicted, weights="quadratic")), 4),
        "mean_absolute_error": round(float(np.mean(np.abs(actual - predicted))), 4),
    }


def severity_candidates(records, train_indices, validation_indices, ratings) -> dict[str, object]:
    train = [
        index
        for index in train_indices
        if records[index]["risk_detected"] == 1 and records[index]["risk_severity"] is not None
    ]
    validation = [
        index
        for index in validation_indices
        if records[index]["risk_detected"] == 1 and records[index]["risk_severity"] is not None
    ]
    actual = np.asarray([records[index]["risk_severity"] for index in validation], dtype=int)
    results: dict[str, object] = {}
    for name, text_builder in (
        ("text_only", lambda record: str(record["text"])),
        ("rating_aware", lambda record: rating_text(record, ratings)),
    ):
        train_texts = [text_builder(records[index]) for index in train]
        validation_texts = [text_builder(records[index]) for index in validation]

        baseline = classifier_pipeline()
        baseline.fit(train_texts, [records[index]["risk_severity"] for index in train])
        results[f"multiclass_{name}"] = severity_metrics(
            actual, np.asarray(baseline.predict(validation_texts), dtype=int)
        )

        model_ge2 = classifier_pipeline()
        model_ge3 = classifier_pipeline()
        model_ge2.fit(train_texts, [int(records[index]["risk_severity"] >= 2) for index in train])
        model_ge3.fit(train_texts, [int(records[index]["risk_severity"] >= 3) for index in train])
        probability_ge2, probability_ge3 = ordinal_probabilities(
            model_ge2, model_ge3, validation_texts
        )
        candidates = []
        for threshold_ge2 in np.arange(0.25, 0.76, 0.05):
            for threshold_ge3 in np.arange(0.25, 0.76, 0.05):
                predicted = ordinal_predictions(
                    probability_ge2,
                    probability_ge3,
                    float(threshold_ge2),
                    float(threshold_ge3),
                )
                metrics = severity_metrics(actual, predicted)
                candidates.append(
                    (
                        metrics["weighted_kappa"],
                        -metrics["mean_absolute_error"],
                        metrics["macro_f1"],
                        float(threshold_ge2),
                        float(threshold_ge3),
                        metrics,
                    )
                )
        best = max(candidates, key=lambda item: item[:3])
        results[f"ordinal_{name}"] = {
            **best[5],
            "threshold_ge2": round(best[3], 2),
            "threshold_ge3": round(best[4], 2),
        }
    return results


def main() -> None:
    records = load_marketplace(DATASET)
    split = locked_product_split(records)
    if split is None:
        raise ValueError("A locked split is required.")
    train_indices, validation_indices, _ = split
    ratings = ratings_by_id(DATASET)
    report = {
        "policy": "Validation-only comparison; locked test predictions are not computed.",
        "category": category_candidate(records, train_indices, validation_indices),
        "severity": severity_candidates(records, train_indices, validation_indices, ratings),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
