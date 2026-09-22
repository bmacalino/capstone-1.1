"""Build a validation-selected hybrid risk-category candidate without reading test labels."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.risk_heads import HybridRiskCategoryClassifier
from train_marketplace_v2 import (
    _multilabel_metrics,
    category_predictions,
    load_marketplace,
    multilabel_pipeline,
    sha256,
)


DEFAULT_TRAIN = (
    ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_train.csv"
)
DEFAULT_VALIDATION = (
    ROOT / "outputs" / "annotation_quality_round_1" / "lazada_round1_validation_adjudicated.csv"
)
DEFAULT_BASE = SENTIMENT_ROOT / "models" / "lazada_risk_heads_v3_candidate.joblib"
DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models" / "lazada_risk_heads_v4_candidate.joblib"
DEFAULT_REPORT = SENTIMENT_ROOT / "evaluation" / "lazada_risk_category_v4_validation.json"
CANONICAL_CLASSES = (
    "damage",
    "defect",
    "delivery",
    "other",
    "performance",
    "quality",
    "seller_service",
    "wrong_item",
)
THRESHOLD_CANDIDATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def supported_macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    support = actual.sum(axis=0) > 0
    return float(
        f1_score(actual[:, support], predicted[:, support], average="macro", zero_division=0)
    )


def per_class_f1_thresholds(actual: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds: list[float] = []
    for index in range(actual.shape[1]):
        thresholds.append(
            max(
                THRESHOLD_CANDIDATES,
                key=lambda threshold: (
                    f1_score(
                        actual[:, index],
                        probabilities[:, index] >= threshold,
                        zero_division=0,
                    ),
                    threshold,
                ),
            )
        )
    return np.asarray(thresholds, dtype=float)


def threshold_score(
    actual: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float, float, float, float]:
    predicted = category_predictions(probabilities, thresholds)
    supported_macro = supported_macro_f1(actual, predicted)
    micro = float(f1_score(actual, predicted, average="micro", zero_division=0))
    exact = float(accuracy_score(actual, predicted))
    hamming = float(hamming_loss(actual, predicted))
    # Balance category coverage with usable whole-review predictions.  Optimizing
    # each category independently heavily over-predicts rare categories.
    composite = supported_macro + micro + exact - hamming
    return composite, supported_macro, micro, exact, -hamming


def tune_thresholds(actual: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    starts = [
        per_class_f1_thresholds(actual, probabilities),
        *[
            np.full(actual.shape[1], threshold, dtype=float)
            for threshold in THRESHOLD_CANDIDATES
        ],
    ]
    candidates: list[np.ndarray] = []
    for start in starts:
        thresholds = start.copy()
        for _ in range(4):
            changed = False
            for index in range(actual.shape[1]):
                if actual[:, index].sum() == 0:
                    thresholds[index] = max(THRESHOLD_CANDIDATES)
                    continue
                best = max(
                    THRESHOLD_CANDIDATES,
                    key=lambda threshold: threshold_score(
                        actual,
                        probabilities,
                        np.concatenate(
                            (thresholds[:index], [threshold], thresholds[index + 1 :])
                        ),
                    ),
                )
                if best != thresholds[index]:
                    thresholds[index] = best
                    changed = True
            if not changed:
                break
        candidates.append(thresholds)
    return max(
        candidates,
        key=lambda thresholds: threshold_score(actual, probabilities, thresholds),
    )


def build(args: argparse.Namespace) -> dict[str, object]:
    train_records = load_marketplace(args.train_dataset.resolve())
    validation_records = load_marketplace(args.validation_dataset.resolve())
    train_rows = [
        row for row in train_records if row["risk_detected"] == 1 and row["risk_categories"]
    ]
    validation_rows = [
        row
        for row in validation_records
        if row["risk_detected"] == 1 and row["risk_categories"]
    ]

    training_classes = sorted(
        {category for row in train_rows for category in row["risk_categories"]}
    )
    if "damage" in training_classes:
        raise ValueError("The hybrid bridge is only intended when damage has no training labels.")
    unknown = set(training_classes).difference(CANONICAL_CLASSES)
    if unknown:
        raise ValueError(f"Unknown training categories: {sorted(unknown)}")

    training_binarizer = MultiLabelBinarizer(classes=training_classes)
    training_labels = training_binarizer.fit_transform(
        [row["risk_categories"] for row in train_rows]
    )
    base_model = multilabel_pipeline()
    base_model.fit([row["text"] for row in train_rows], training_labels)
    hybrid = HybridRiskCategoryClassifier(
        base_model, training_binarizer.classes_, CANONICAL_CLASSES
    )

    output_binarizer = MultiLabelBinarizer(classes=CANONICAL_CLASSES)
    output_binarizer.fit([[]])
    actual = output_binarizer.transform(
        [row["risk_categories"] for row in validation_rows]
    )
    probabilities = hybrid.predict_proba([row["text"] for row in validation_rows])
    thresholds = tune_thresholds(actual, probabilities)
    predicted = category_predictions(probabilities, thresholds)
    metrics = _multilabel_metrics(actual, predicted, list(CANONICAL_CLASSES))
    metrics["supported_macro_f1"] = round(supported_macro_f1(actual, predicted), 4)
    metrics["probability_thresholds"] = {
        category: round(float(threshold), 2)
        for category, threshold in zip(CANONICAL_CLASSES, thresholds)
    }
    metrics["unknown_category_rows_excluded"] = 0

    artifact = copy.deepcopy(joblib.load(args.base_artifact.resolve()))
    artifact["risk_category_pipeline"] = hybrid
    artifact["risk_category_binarizer"] = output_binarizer
    artifact["risk_category_threshold"] = thresholds
    metadata = dict(artifact.get("metadata", {}))
    created_at = datetime.now(timezone.utc).isoformat()
    metadata.update(
        {
            "model_version": "defaketive-marketplace-v4-hybrid-risk-category-candidate",
            "created_at": created_at,
            "qualified_for_integration": False,
            "qualification_reason": "Provisional validation-selected category candidate; frozen test remains unread.",
            "category_model": "multilabel_logistic_regression_plus_physical_damage_rule",
            "category_classes": list(CANONICAL_CLASSES),
            "category_probability_thresholds": metrics["probability_thresholds"],
            "selection_policy": "Category thresholds jointly selected on adjudicated validation for supported-macro F1, micro-F1, exact match, and hamming loss; locked test labels were not read.",
        }
    )
    artifact["metadata"] = metadata
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    report = {
        "metadata": {
            "created_at": created_at,
            "train_dataset": args.train_dataset.name,
            "train_dataset_sha256": sha256(args.train_dataset.resolve()),
            "validation_dataset": args.validation_dataset.name,
            "validation_dataset_sha256": sha256(args.validation_dataset.resolve()),
            "base_artifact": args.base_artifact.name,
            "base_artifact_sha256": sha256(args.base_artifact.resolve()),
            "candidate_artifact": args.artifact.name,
            "candidate_artifact_sha256": sha256(args.artifact.resolve()),
            "selection_policy": metadata["selection_policy"],
            "test_predictions_computed": False,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "training_classes": training_classes,
            "canonical_classes": list(CANONICAL_CLASSES),
        },
        "validation": {"risk_category": metrics},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-dataset", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2))
