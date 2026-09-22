"""Evaluate product-only risk detection on the adjudicated FiReCS sample.

The trained candidate uses nested, duplicate-grouped stratified cross-validation.
Exact normalized duplicate reviews stay within one fold. FiReCS has no product
identifier, so this is a review-level development estimate rather than a
product-disjoint validation.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.manuscript import (  # noqa: E402
    HybridManuscriptRiskModel,
    ManuscriptRiskModel,
)
from defaketive_sentiment.model import DefaketiveSentimentModel  # noqa: E402
from defaketive_sentiment.trained import feature_text  # noqa: E402
from research.train_risk_scoring_v1 import (  # noqa: E402
    positive_probability,
    select_risk_threshold,
)
from train_marketplace_v2 import classifier_pipeline  # noqa: E402


DEFAULT_DATASET = (
    ROOT
    / "outputs"
    / "firecs_risk_annotation_final"
    / "firecs_product_only_final.json"
)
DEFAULT_SOURCE_WORKBOOK = (
    ROOT
    / "outputs"
    / "firecs_risk_annotation_final"
    / "FiReCS_Risk_Annotation_Round_1_Product_Only_Final.xlsx"
)
DEFAULT_REPORT = (
    SENTIMENT_ROOT
    / "evaluation"
    / "firecs_product_only_risk_cross_validation.json"
)
DEFAULT_PREDICTIONS = (
    SENTIMENT_ROOT
    / "evaluation"
    / "firecs_product_only_risk_cross_validation_predictions.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("FiReCS risk dataset must be a non-empty JSON list.")
    required = {"review_id", "review_text", "risk_detected"}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"FiReCS risk row {position} is missing required fields.")
        review_id = str(row["review_id"] or "").strip()
        text = str(row["review_text"] or "").strip()
        label = str(row["risk_detected"] or "").strip().casefold()
        if not review_id or review_id in seen_ids:
            raise ValueError(f"Invalid or duplicate review_id at row {position}: {review_id!r}")
        if not text:
            raise ValueError(f"Empty review text at row {position}.")
        if label not in {"yes", "no"}:
            raise ValueError(f"Unsupported risk label at row {position}: {label!r}")
        seen_ids.add(review_id)
        records.append(
            {
                "review_id": review_id,
                "review_text": text,
                "risk_detected": 1 if label == "yes" else 0,
                "source_row": row.get("source_row"),
                "firecs_sentiment": str(row.get("firecs_sentiment") or "").strip().casefold(),
            }
        )
    return records


def duplicate_group(text: str) -> str:
    normalized = feature_text(DefaketiveSentimentModel.normalize(text))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def risk_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    result: dict[str, Any] = {
        "sample_count": int(len(actual)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(actual, predicted)), 4),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro", zero_division=0)), 4
        ),
        "no_risk": {
            "precision": round(float(precision[0]), 4),
            "recall": round(float(recall[0]), 4),
            "f1": round(float(f1[0]), 4),
            "support": int(support[0]),
        },
        "risk": {
            "precision": round(float(precision[1]), 4),
            "recall": round(float(recall[1]), 4),
            "f1": round(float(f1[1]), 4),
            "support": int(support[1]),
        },
        "confusion_matrix": {
            "labels": ["no_risk", "risk"],
            "values": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        },
    }
    if probabilities is not None:
        result.update(
            {
                "roc_auc": round(float(roc_auc_score(actual, probabilities)), 4),
                "average_precision": round(
                    float(average_precision_score(actual, probabilities)), 4
                ),
                "brier_score": round(float(brier_score_loss(actual, probabilities)), 4),
            }
        )
    return result


def inner_threshold(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    seed: int,
) -> tuple[float, dict[str, float]]:
    local_labels = labels[outer_train]
    local_groups = groups[outer_train]
    probabilities = np.zeros(len(outer_train), dtype=float)
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for inner_train_pos, validation_pos in inner.split(
        outer_train, local_labels, local_groups
    ):
        fit_indices = outer_train[inner_train_pos]
        validation_indices = outer_train[validation_pos]
        if set(groups[fit_indices]) & set(groups[validation_indices]):
            raise RuntimeError("Duplicate-text leakage detected in an inner fold.")
        model = classifier_pipeline()
        model.fit(texts[fit_indices], labels[fit_indices])
        probabilities[validation_pos] = positive_probability(
            model, texts[validation_indices]
        )
    threshold, metrics = select_risk_threshold(local_labels, probabilities)
    return threshold, {key: round(float(value), 4) for key, value in metrics.items()}


def evaluate(
    dataset: Path,
    source_workbook: Path,
    *,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_records(dataset)
    raw_texts = np.asarray(
        [record["review_text"] for record in records], dtype=object
    )
    texts = np.asarray(
        [feature_text(DefaketiveSentimentModel.normalize(text)) for text in raw_texts],
        dtype=object,
    )
    labels = np.asarray([record["risk_detected"] for record in records], dtype=int)
    groups = np.asarray([duplicate_group(text) for text in raw_texts], dtype=object)
    indices = np.arange(len(records))

    group_labels: dict[str, set[int]] = {}
    for group, label in zip(groups, labels):
        group_labels.setdefault(str(group), set()).add(int(label))
    conflicting_groups = sum(len(values) > 1 for values in group_labels.values())
    duplicate_groups = sum(list(groups).count(group) > 1 for group in set(groups))
    duplicate_rows = len(groups) - len(set(groups))

    out_of_fold_probability = np.zeros(len(records), dtype=float)
    out_of_fold_prediction = np.full(len(records), -1, dtype=int)
    fold_assignment = np.full(len(records), -1, dtype=int)
    folds: list[dict[str, Any]] = []

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_number, (outer_train, outer_test) in enumerate(
        outer.split(indices, labels, groups), start=1
    ):
        if set(groups[outer_train]) & set(groups[outer_test]):
            raise RuntimeError("Duplicate-text leakage detected in an outer fold.")
        threshold, threshold_metrics = inner_threshold(
            texts,
            labels,
            groups,
            outer_train,
            seed + (fold_number * 100),
        )
        model = classifier_pipeline()
        model.fit(texts[outer_train], labels[outer_train])
        probabilities = positive_probability(model, texts[outer_test])
        predictions = (probabilities >= threshold).astype(int)
        out_of_fold_probability[outer_test] = probabilities
        out_of_fold_prediction[outer_test] = predictions
        fold_assignment[outer_test] = fold_number
        fold_result = risk_metrics(labels[outer_test], predictions, probabilities)
        folds.append(
            {
                "fold": fold_number,
                "train_rows": int(len(outer_train)),
                "test_rows": int(len(outer_test)),
                "train_duplicate_groups": int(len(set(groups[outer_train]))),
                "test_duplicate_groups": int(len(set(groups[outer_test]))),
                "selected_threshold": round(float(threshold), 4),
                "inner_threshold_selection": threshold_metrics,
                "metrics": fold_result,
            }
        )

    if np.any(out_of_fold_prediction < 0) or np.any(fold_assignment < 0):
        raise RuntimeError("Cross-validation did not produce one prediction per review.")
    cross_validation = risk_metrics(
        labels, out_of_fold_prediction, out_of_fold_probability
    )

    current_model = HybridManuscriptRiskModel(allow_unqualified=True)
    current_probabilities = positive_probability(
        current_model.artifact["risk_detection_pipeline"], texts
    )
    current_threshold = float(
        current_model.artifact.get(
            "risk_detection_threshold",
            getattr(current_model.artifact["risk_detection_pipeline"], "threshold", 0.5),
        )
    )
    current_predictions = (current_probabilities >= current_threshold).astype(int)
    current_metrics = risk_metrics(labels, current_predictions, current_probabilities)
    current_metrics["threshold"] = round(current_threshold, 4)
    current_metrics["model_version"] = current_model.model_version

    lexicon_model = ManuscriptRiskModel()
    lexicon_predictions = np.asarray(
        [int(lexicon_model.analyze(text)["risk"]["detected"]) for text in raw_texts],
        dtype=int,
    )
    lexicon_metrics = risk_metrics(labels, lexicon_predictions)
    lexicon_metrics["model_version"] = lexicon_model.model_version

    majority_value = int(labels.mean() >= 0.5)
    majority_predictions = np.full(len(labels), majority_value, dtype=int)
    majority_metrics = risk_metrics(labels, majority_predictions)

    fold_macro_f1 = [fold["metrics"]["macro_f1"] for fold in folds]
    fold_risk_recall = [fold["metrics"]["risk"]["recall"] for fold in folds]
    gates = {
        "risk_recall_at_least_0_80": cross_validation["risk"]["recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": cross_validation["macro_f1"] >= 0.75,
    }
    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_status": "development_duplicate_grouped_nested_cross_validation",
            "limitation": (
                "FiReCS contains no product identifier. Exact normalized duplicate text "
                "is grouped to prevent direct duplicate leakage, but this is not a "
                "product-disjoint validation and must not support final production claims."
            ),
            "dataset": str(dataset.relative_to(ROOT)),
            "dataset_sha256": sha256(dataset),
            "source_workbook": str(source_workbook.relative_to(ROOT)),
            "source_workbook_sha256": sha256(source_workbook),
            "random_state": seed,
            "outer_folds": 5,
            "inner_folds": 4,
            "threshold_policy": (
                "Selected from inner-fold predictions to maximize macro F1 while "
                "prioritizing risk recall of at least 0.80."
            ),
            "model": "word-and-character TF-IDF with class-balanced logistic regression",
        },
        "dataset": {
            "rows": len(records),
            "risk": int(labels.sum()),
            "no_risk": int((labels == 0).sum()),
            "unique_normalized_reviews": len(set(groups)),
            "duplicate_groups": duplicate_groups,
            "duplicate_rows_beyond_first": duplicate_rows,
            "conflicting_duplicate_label_groups": conflicting_groups,
            "sentiment_distribution": dict(
                sorted(Counter(record["firecs_sentiment"] for record in records).items())
            ),
        },
        "nested_cross_validation": cross_validation,
        "fold_variability": {
            "macro_f1_mean": round(float(np.mean(fold_macro_f1)), 4),
            "macro_f1_standard_deviation": round(
                float(np.std(fold_macro_f1, ddof=1)), 4
            ),
            "risk_recall_mean": round(float(np.mean(fold_risk_recall)), 4),
            "risk_recall_standard_deviation": round(
                float(np.std(fold_risk_recall, ddof=1)), 4
            ),
        },
        "folds": folds,
        "current_hybrid_external_diagnostic": current_metrics,
        "manuscript_lexicon_baseline": lexicon_metrics,
        "majority_baseline": majority_metrics,
        "development_gates": gates,
        "passes_all_development_gates": all(gates.values()),
    }

    predictions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        predictions.append(
            {
                "review_id": record["review_id"],
                "source_row": record["source_row"],
                "fold": int(fold_assignment[index]),
                "actual_risk": int(labels[index]),
                "cross_validated_risk": int(out_of_fold_prediction[index]),
                "cross_validated_probability": round(
                    float(out_of_fold_probability[index]), 6
                ),
                "current_hybrid_risk": int(current_predictions[index]),
                "current_hybrid_probability": round(
                    float(current_probabilities[index]), 6
                ),
                "lexicon_risk": int(lexicon_predictions[index]),
                "review_text": record["review_text"],
            }
        )
    return report, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report, predictions = evaluate(
        args.dataset.resolve(), args.source_workbook.resolve(), seed=args.seed
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
