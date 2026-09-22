"""Nested product-group cross-validation for the hybrid-manuscript model.

This is a development estimate over previously reviewed data, not an
independent locked test. Every outer and inner split keeps complete products
together. Sentiment and risk thresholds are selected only from inner-fold
predictions, never from the corresponding outer test fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_firecs import load_firecs  # noqa: E402
from research.train_risk_scoring_v1 import (  # noqa: E402
    positive_probability,
    select_risk_threshold,
)
from research_utils import evaluate_labels  # noqa: E402
from train_marketplace_v2 import (  # noqa: E402
    _risk_metrics,
    classifier_pipeline,
    load_marketplace,
    normalized_text,
    select_uncertainty_threshold,
    uncertainty_predictions,
)


DEFAULT_DATASET = (
    ROOT
    / "outputs"
    / "lazada_gold_test_v1"
    / "lazada_round1_all_adjudicated.csv"
)
DEFAULT_FIRECS = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_train_set.csv"
DEFAULT_REPORT = (
    SENTIMENT_ROOT
    / "evaluation"
    / "hybrid_manuscript_grouped_cross_validation.json"
)
DEFAULT_PREDICTIONS = (
    SENTIMENT_ROOT
    / "evaluation"
    / "hybrid_manuscript_grouped_cross_validation_predictions.csv"
)
SENTIMENT_CLASSES = np.asarray(["negative", "positive"], dtype=object)
DIRECT_SENTIMENT_CLASSES = np.asarray(
    ["negative", "positive", "uncertain"], dtype=object
)
SENTIMENT_STRATEGIES = {
    "binary-abstention",
    "direct-three-class",
    "direct-three-class-tuned",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation_sentiment(label: object) -> str:
    value = str(label or "").strip().casefold()
    return value if value in {"negative", "positive"} else "uncertain"


def product_risk_score(sentiments: list[str], risks: list[int]) -> float:
    if not sentiments:
        return 0.0
    negative_ratio = sum(label == "negative" for label in sentiments) / len(sentiments)
    detected_risk_ratio = sum(int(value) == 1 for value in risks) / len(risks)
    return 100 * ((0.30 * negative_ratio) + (0.70 * detected_risk_ratio))


def risk_level(score: float) -> str:
    return "low" if score <= 30 else "moderate" if score <= 60 else "high"


def three_class_predictions(
    probabilities: np.ndarray, threshold: float
) -> list[str]:
    uncertain_index = int(np.flatnonzero(DIRECT_SENTIMENT_CLASSES == "uncertain")[0])
    binary_indices = [
        index
        for index, label in enumerate(DIRECT_SENTIMENT_CLASSES)
        if label != "uncertain"
    ]
    output: list[str] = []
    for row in probabilities:
        if row[uncertain_index] >= threshold:
            output.append("uncertain")
        else:
            strongest = binary_indices[int(np.argmax(row[binary_indices]))]
            output.append(str(DIRECT_SENTIMENT_CLASSES[strongest]))
    return output


def select_three_class_uncertain_threshold(
    actual: list[str], probabilities: np.ndarray
) -> float:
    candidates = tuple(round(value / 100, 2) for value in range(15, 61, 5))
    return max(
        candidates,
        key=lambda threshold: (
            evaluate_labels(
                actual, three_class_predictions(probabilities, threshold)
            )["f1"],
            -abs(threshold - (1 / 3)),
        ),
    )


def _probabilities_in_class_order(model: Any, texts: list[str]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(texts), dtype=float)
    positions = {str(label): index for index, label in enumerate(model.classes_)}
    return np.column_stack([raw[:, positions[str(label)]] for label in SENTIMENT_CLASSES])


def _fit_sentiment(
    records: list[dict[str, Any]],
    indices: np.ndarray,
    auxiliary_texts: list[str],
    auxiliary_labels: list[str],
    strategy: str,
) -> Any:
    if strategy in {"direct-three-class", "direct-three-class-tuned"}:
        local = [int(index) for index in indices]
        local_labels = [
            evaluation_sentiment(records[index]["sentiment"]) for index in local
        ]
    else:
        local = [
            int(index)
            for index in indices
            if records[int(index)]["sentiment"] in {"negative", "positive"}
        ]
        local_labels = [records[index]["sentiment"] for index in local]
    texts = auxiliary_texts + [records[index]["text"] for index in local]
    labels = auxiliary_labels + local_labels
    model = classifier_pipeline(large=True)
    model.fit(texts, labels)
    return model


def _inner_thresholds(
    records: list[dict[str, Any]],
    outer_train: np.ndarray,
    auxiliary_texts: list[str],
    auxiliary_labels: list[str],
    seed: int,
    sentiment_strategy: str,
) -> tuple[float | None, float, dict[str, Any]]:
    sentiments = np.asarray(
        [evaluation_sentiment(records[int(index)]["sentiment"]) for index in outer_train],
        dtype=object,
    )
    risks = np.asarray(
        [int(records[int(index)]["risk_detected"]) for index in outer_train],
        dtype=int,
    )
    groups = np.asarray(
        [records[int(index)]["product_id"] for index in outer_train], dtype=object
    )
    if sentiment_strategy == "binary-abstention":
        sentiment_probabilities: np.ndarray | None = np.zeros(
            (len(outer_train), 2), dtype=float
        )
    elif sentiment_strategy == "direct-three-class-tuned":
        sentiment_probabilities = np.zeros((len(outer_train), 3), dtype=float)
    else:
        sentiment_probabilities = None
    risk_probabilities = np.zeros(len(outer_train), dtype=float)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for inner_train_position, validation_position in inner.split(
        outer_train, sentiments, groups
    ):
        fit_indices = outer_train[inner_train_position]
        validation_indices = outer_train[validation_position]
        fit_products = {records[int(index)]["product_id"] for index in fit_indices}
        validation_products = {
            records[int(index)]["product_id"] for index in validation_indices
        }
        if fit_products & validation_products:
            raise RuntimeError("Product leakage detected in an inner fold.")

        validation_texts = [records[int(index)]["text"] for index in validation_indices]
        if sentiment_probabilities is not None:
            sentiment_model = _fit_sentiment(
                records,
                fit_indices,
                auxiliary_texts,
                auxiliary_labels,
                sentiment_strategy,
            )
            probability_classes = (
                SENTIMENT_CLASSES
                if sentiment_strategy == "binary-abstention"
                else DIRECT_SENTIMENT_CLASSES
            )
            raw_probabilities = np.asarray(
                sentiment_model.predict_proba(validation_texts), dtype=float
            )
            positions = {
                str(label): index
                for index, label in enumerate(sentiment_model.classes_)
            }
            sentiment_probabilities[validation_position] = np.column_stack(
                [
                    raw_probabilities[:, positions[str(label)]]
                    for label in probability_classes
                ]
            )

        risk_model = classifier_pipeline()
        risk_model.fit(
            [records[int(index)]["text"] for index in fit_indices],
            [int(records[int(index)]["risk_detected"]) for index in fit_indices],
        )
        risk_probabilities[validation_position] = positive_probability(
            risk_model, validation_texts
        )

    if sentiment_strategy == "binary-abstention":
        sentiment_threshold: float | None = select_uncertainty_threshold(
            sentiments.tolist(), sentiment_probabilities, SENTIMENT_CLASSES
        )
    elif sentiment_strategy == "direct-three-class-tuned":
        sentiment_threshold = select_three_class_uncertain_threshold(
            sentiments.tolist(), sentiment_probabilities
        )
    else:
        sentiment_threshold = None
    risk_threshold, threshold_metrics = select_risk_threshold(
        risks, risk_probabilities
    )
    return sentiment_threshold, risk_threshold, threshold_metrics


def _aggregate_fold_metrics(folds: list[dict[str, Any]], path: list[str]) -> dict[str, float]:
    values = []
    for fold in folds:
        value: Any = fold
        for key in path:
            value = value[key]
        values.append(float(value))
    return {
        "mean": round(float(np.mean(values)), 4),
        "standard_deviation": round(float(np.std(values, ddof=1)), 4),
        "minimum": round(float(np.min(values)), 4),
        "maximum": round(float(np.max(values)), 4),
    }


def evaluate(
    dataset: Path,
    firecs_path: Path,
    *,
    seed: int = 42,
    sentiment_strategy: str = "binary-abstention",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if sentiment_strategy not in SENTIMENT_STRATEGIES:
        raise ValueError(f"Unsupported sentiment strategy: {sentiment_strategy}")
    records = load_marketplace(dataset)
    actual_sentiment = np.asarray(
        [evaluation_sentiment(record["sentiment"]) for record in records], dtype=object
    )
    actual_risk = np.asarray(
        [int(record["risk_detected"]) for record in records], dtype=int
    )
    groups = np.asarray([record["product_id"] for record in records], dtype=object)
    indices = np.arange(len(records))

    local_texts = {record["text"] for record in records}
    firecs = load_firecs(firecs_path)
    allowed_auxiliary_labels = (
        {"negative", "neutral", "positive"}
        if sentiment_strategy in {"direct-three-class", "direct-three-class-tuned"}
        else {"negative", "positive"}
    )
    clean_firecs = [
        row
        for row in firecs
        if row["label"] in allowed_auxiliary_labels
        and normalized_text(row["review"]) not in local_texts
    ]
    auxiliary_texts = [normalized_text(row["review"]) for row in clean_firecs]
    auxiliary_labels = [evaluation_sentiment(row["label"]) for row in clean_firecs]

    predicted_sentiment = np.full(len(records), "", dtype=object)
    predicted_risk = np.full(len(records), -1, dtype=int)
    fold_assignment = np.full(len(records), -1, dtype=int)
    sentiment_confidence = np.zeros(len(records), dtype=float)
    risk_probability = np.zeros(len(records), dtype=float)
    folds: list[dict[str, Any]] = []

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_number, (outer_train, outer_test) in enumerate(
        outer.split(indices, actual_sentiment, groups), start=1
    ):
        train_products = {records[int(index)]["product_id"] for index in outer_train}
        test_products = {records[int(index)]["product_id"] for index in outer_test}
        if train_products & test_products:
            raise RuntimeError("Product leakage detected in an outer fold.")

        sentiment_threshold, risk_threshold, inner_risk = _inner_thresholds(
            records,
            outer_train,
            auxiliary_texts,
            auxiliary_labels,
            seed + (fold_number * 100),
            sentiment_strategy,
        )
        test_texts = [records[int(index)]["text"] for index in outer_test]

        sentiment_model = _fit_sentiment(
            records,
            outer_train,
            auxiliary_texts,
            auxiliary_labels,
            sentiment_strategy,
        )
        if sentiment_strategy == "binary-abstention":
            sentiment_probabilities = _probabilities_in_class_order(
                sentiment_model, test_texts
            )
            fold_sentiment_predictions = uncertainty_predictions(
                sentiment_probabilities,
                SENTIMENT_CLASSES,
                float(sentiment_threshold),
            )
            fold_sentiment_confidence = np.max(sentiment_probabilities, axis=1)
        elif sentiment_strategy == "direct-three-class-tuned":
            raw_probabilities = np.asarray(
                sentiment_model.predict_proba(test_texts), dtype=float
            )
            positions = {
                str(label): index
                for index, label in enumerate(sentiment_model.classes_)
            }
            ordered_probabilities = np.column_stack(
                [
                    raw_probabilities[:, positions[str(label)]]
                    for label in DIRECT_SENTIMENT_CLASSES
                ]
            )
            fold_sentiment_predictions = three_class_predictions(
                ordered_probabilities, float(sentiment_threshold)
            )
            fold_sentiment_confidence = np.max(ordered_probabilities, axis=1)
        else:
            raw_probabilities = np.asarray(
                sentiment_model.predict_proba(test_texts), dtype=float
            )
            fold_sentiment_predictions = sentiment_model.predict(test_texts).tolist()
            fold_sentiment_confidence = np.max(raw_probabilities, axis=1)

        risk_model = classifier_pipeline()
        risk_model.fit(
            [records[int(index)]["text"] for index in outer_train],
            actual_risk[outer_train],
        )
        fold_risk_probabilities = positive_probability(risk_model, test_texts)
        fold_risk_predictions = (fold_risk_probabilities >= risk_threshold).astype(int)

        predicted_sentiment[outer_test] = fold_sentiment_predictions
        predicted_risk[outer_test] = fold_risk_predictions
        fold_assignment[outer_test] = fold_number
        sentiment_confidence[outer_test] = fold_sentiment_confidence
        risk_probability[outer_test] = fold_risk_probabilities

        fold_sentiment = evaluate_labels(
            actual_sentiment[outer_test].tolist(), fold_sentiment_predictions
        )
        fold_risk = _risk_metrics(
            actual_risk[outer_test].tolist(), fold_risk_predictions.tolist()
        )
        folds.append(
            {
                "fold": fold_number,
                "train_rows": len(outer_train),
                "test_rows": len(outer_test),
                "train_products": len(train_products),
                "test_products": len(test_products),
                "test_product_ids": sorted(test_products),
                "sentiment_uncertainty_threshold": sentiment_threshold,
                "risk_detection_threshold": round(risk_threshold, 4),
                "inner_risk_threshold_selection": {
                    key: round(float(value), 4) for key, value in inner_risk.items()
                },
                "sentiment": fold_sentiment,
                "risk_detection": fold_risk,
            }
        )

    if np.any(fold_assignment < 0) or np.any(predicted_risk < 0):
        raise RuntimeError("Cross-validation failed to produce one prediction per review.")

    pooled_sentiment = evaluate_labels(
        actual_sentiment.tolist(), predicted_sentiment.tolist()
    )
    pooled_risk = _risk_metrics(actual_risk.tolist(), predicted_risk.tolist())

    product_rows: list[dict[str, Any]] = []
    for product_id in sorted(set(groups)):
        selected = np.flatnonzero(groups == product_id)
        gold_score = product_risk_score(
            actual_sentiment[selected].tolist(), actual_risk[selected].tolist()
        )
        predicted_score = product_risk_score(
            predicted_sentiment[selected].tolist(), predicted_risk[selected].tolist()
        )
        product_rows.append(
            {
                "product_id": str(product_id),
                "fold": int(fold_assignment[selected[0]]),
                "reviews": len(selected),
                "human_score": round(gold_score, 2),
                "predicted_score": round(predicted_score, 2),
                "absolute_error": round(abs(gold_score - predicted_score), 2),
                "human_level": risk_level(gold_score),
                "predicted_level": risk_level(predicted_score),
            }
        )
    gold_scores = np.asarray([row["human_score"] for row in product_rows], dtype=float)
    predicted_scores = np.asarray(
        [row["predicted_score"] for row in product_rows], dtype=float
    )
    product_score_metrics = {
        "products": len(product_rows),
        "mean_absolute_error_points": round(
            float(mean_absolute_error(gold_scores, predicted_scores)), 4
        ),
        "root_mean_squared_error_points": round(
            float(mean_squared_error(gold_scores, predicted_scores) ** 0.5), 4
        ),
        "pearson_correlation": round(float(pearsonr(gold_scores, predicted_scores).statistic), 4),
        "spearman_rank_correlation": round(
            float(spearmanr(gold_scores, predicted_scores).statistic), 4
        ),
        "risk_level_accuracy": round(
            sum(row["human_level"] == row["predicted_level"] for row in product_rows)
            / len(product_rows),
            4,
        ),
        "per_product": product_rows,
    }

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        predictions.append(
            {
                "review_id": record["review_id"],
                "product_id": record["product_id"],
                "fold": int(fold_assignment[index]),
                "actual_sentiment": actual_sentiment[index],
                "predicted_sentiment": predicted_sentiment[index],
                "sentiment_confidence": round(float(sentiment_confidence[index]), 6),
                "actual_risk": int(actual_risk[index]),
                "predicted_risk": int(predicted_risk[index]),
                "risk_probability": round(float(risk_probability[index]), 6),
                "review_text": record["review"],
            }
        )

    gates = {
        "sentiment_macro_f1_at_least_0_75": pooled_sentiment["f1"] >= 0.75,
        "uncertain_f1_at_least_0_60": (
            pooled_sentiment["per_class"].get("uncertain", {}).get("f1", 0) >= 0.60
        ),
        "risk_recall_at_least_0_80": pooled_risk["risk"]["recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": pooled_risk["macro_f1"] >= 0.75,
    }
    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_status": "development_product_grouped_cross_validation",
            "limitation": (
                "All reviews were previously available during development. This is a "
                "product-grouped cross-validation estimate, not an independent locked test."
            ),
            "dataset": str(dataset.relative_to(ROOT)),
            "dataset_sha256": sha256(dataset),
            "auxiliary_dataset": str(firecs_path.relative_to(ROOT)),
            "auxiliary_dataset_sha256": sha256(firecs_path),
            "random_state": seed,
            "outer_folds": 5,
            "inner_folds": 4,
            "rows": len(records),
            "products": len(set(groups)),
            "product_leakage": False,
            "sentiment_policy": "mixed, neutral, and uncertain map to uncertain",
            "sentiment_strategy": sentiment_strategy,
            "risk_policy": "binary yes/no; no categories or severity",
            "score_formula": "100 * (0.30 * negative_ratio + 0.70 * detected_risk_ratio)",
            "firecs_rows_loaded": len(firecs),
            "firecs_sentiment_rows_used": len(clean_firecs),
            "firecs_marketplace_overlap_rows_excluded": sum(
                row["label"] in allowed_auxiliary_labels
                and normalized_text(row["review"]) in local_texts
                for row in firecs
            ),
        },
        "dataset_distribution": {
            "sentiment": dict(Counter(actual_sentiment.tolist())),
            "risk": {
                "no_risk": int((actual_risk == 0).sum()),
                "risk": int((actual_risk == 1).sum()),
            },
        },
        "pooled_out_of_fold": {
            "sentiment": pooled_sentiment,
            "risk_detection": pooled_risk,
            "product_risk_score": product_score_metrics,
        },
        "fold_variability": {
            "sentiment_macro_f1": _aggregate_fold_metrics(folds, ["sentiment", "f1"]),
            "risk_macro_f1": _aggregate_fold_metrics(
                folds, ["risk_detection", "macro_f1"]
            ),
            "risk_recall": _aggregate_fold_metrics(
                folds, ["risk_detection", "risk", "recall"]
            ),
        },
        "folds": folds,
        "development_gates": gates,
        "passes_all_development_gates": all(gates.values()),
    }
    return report, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--firecs", type=Path, default=DEFAULT_FIRECS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sentiment-strategy",
        choices=sorted(SENTIMENT_STRATEGIES),
        default="binary-abstention",
    )
    args = parser.parse_args()
    report, predictions = evaluate(
        args.dataset.resolve(),
        args.firecs.resolve(),
        seed=args.seed,
        sentiment_strategy=args.sentiment_strategy,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
