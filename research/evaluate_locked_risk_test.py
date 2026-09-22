"""Run the selected risk candidate once on the locked product-disjoint test set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.risk_heads import rating_feature_text
from train_marketplace_v2 import (
    _multilabel_metrics,
    _risk_metrics,
    category_predictions,
    load_marketplace,
    sha256,
)


DEFAULT_TEST = (
    ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_test.csv"
)
DEFAULT_CURRENT = SENTIMENT_ROOT / "models" / "lazada_round1_binary_uncertain.joblib"
DEFAULT_CANDIDATE = SENTIMENT_ROOT / "models" / "lazada_risk_heads_v4_candidate.joblib"
DEFAULT_REPORT = SENTIMENT_ROOT / "evaluation" / "lazada_risk_models_locked_test.json"
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


def severity_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return {
        "sample_count": int(len(actual)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro", zero_division=0)), 4
        ),
        "weighted_kappa": round(
            float(cohen_kappa_score(actual, predicted, weights="quadratic")), 4
        ),
        "mean_absolute_error": round(float(np.mean(np.abs(actual - predicted))), 4),
    }


def aligned_category_probabilities(
    artifact: dict[str, Any], texts: list[str]
) -> np.ndarray:
    source_classes = artifact["risk_category_binarizer"].classes_.tolist()
    source_probabilities = np.asarray(
        artifact["risk_category_pipeline"].predict_proba(texts), dtype=float
    )
    aligned = np.zeros((len(texts), len(CANONICAL_CLASSES)), dtype=float)
    target_indices = {label: index for index, label in enumerate(CANONICAL_CLASSES)}
    for source_index, label in enumerate(source_classes):
        if label in target_indices:
            aligned[:, target_indices[label]] = source_probabilities[:, source_index]
    return aligned


def aligned_thresholds(artifact: dict[str, Any]) -> float | np.ndarray:
    threshold = artifact.get("risk_category_threshold", 0.5)
    values = np.asarray(threshold, dtype=float)
    if values.ndim == 0:
        return float(values)
    source_classes = artifact["risk_category_binarizer"].classes_.tolist()
    if len(values) != len(source_classes):
        raise ValueError("Category thresholds do not match artifact classes.")
    aligned = np.full(len(CANONICAL_CLASSES), 1.0, dtype=float)
    target_indices = {label: index for index, label in enumerate(CANONICAL_CLASSES)}
    for source_index, label in enumerate(source_classes):
        if label in target_indices:
            aligned[target_indices[label]] = values[source_index]
    return aligned


def evaluate(records: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    texts = [str(record["text"]) for record in records]

    risk_actual = [int(record["risk_detected"]) for record in records]
    risk_predicted = (
        artifact["risk_detection_pipeline"].predict(texts).astype(int).tolist()
    )

    category_rows = [
        record
        for record in records
        if record["risk_detected"] == 1 and record["risk_categories"]
    ]
    binarizer = MultiLabelBinarizer(classes=CANONICAL_CLASSES)
    binarizer.fit([[]])
    category_actual = binarizer.transform(
        [record["risk_categories"] for record in category_rows]
    )
    category_probabilities = aligned_category_probabilities(
        artifact, [record["text"] for record in category_rows]
    )
    category_predicted = category_predictions(
        category_probabilities, aligned_thresholds(artifact)
    )
    category = _multilabel_metrics(
        category_actual, category_predicted, list(CANONICAL_CLASSES)
    )
    supported = category_actual.sum(axis=0) > 0
    category["supported_macro_f1"] = round(
        float(
            f1_score(
                category_actual[:, supported],
                category_predicted[:, supported],
                average="macro",
                zero_division=0,
            )
        ),
        4,
    )
    category["unknown_category_rows_excluded"] = 0

    severity_rows = [
        record
        for record in records
        if record["risk_detected"] == 1 and record["risk_severity"] is not None
    ]
    severity_texts = [
        rating_feature_text(record["text"], record.get("rating"))
        if artifact.get("metadata", {}).get("severity_uses_rating_token")
        else record["text"]
        for record in severity_rows
    ]
    severity_actual = np.asarray(
        [record["risk_severity"] for record in severity_rows], dtype=int
    )
    severity_predicted = artifact["risk_severity_pipeline"].predict(severity_texts).astype(int)

    return {
        "artifact": artifact_path.name,
        "artifact_sha256": sha256(artifact_path),
        "model_version": artifact.get("metadata", {}).get("model_version"),
        "risk_detection": _risk_metrics(risk_actual, risk_predicted),
        "risk_category": category,
        "risk_severity": severity_metrics(severity_actual, severity_predicted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    paths = [args.test_dataset.resolve(), args.current.resolve(), args.candidate.resolve()]
    if args.report.exists():
        raise FileExistsError(
            f"Locked-test report already exists and will not be overwritten: {args.report}"
        )
    records = load_marketplace(paths[0])
    if len(records) != 133 or len({record["review_id"] for record in records}) != 133:
        raise ValueError("Locked test must contain exactly 133 unique reviews.")
    if any(record["risk_detected"] is None for record in records):
        raise ValueError("Locked test has missing risk labels.")

    current = evaluate(records, paths[1])
    candidate = evaluate(records, paths[2])
    comparison = {
        "risk_macro_f1_delta": round(
            candidate["risk_detection"]["macro_f1"]
            - current["risk_detection"]["macro_f1"],
            4,
        ),
        "risk_recall_delta": round(
            candidate["risk_detection"]["risk"]["recall"]
            - current["risk_detection"]["risk"]["recall"],
            4,
        ),
        "category_macro_f1_delta": round(
            candidate["risk_category"]["macro_f1"]
            - current["risk_category"]["macro_f1"],
            4,
        ),
        "category_supported_macro_f1_delta": round(
            candidate["risk_category"]["supported_macro_f1"]
            - current["risk_category"]["supported_macro_f1"],
            4,
        ),
        "category_micro_f1_delta": round(
            candidate["risk_category"]["micro_f1"]
            - current["risk_category"]["micro_f1"],
            4,
        ),
        "category_exact_match_delta": round(
            candidate["risk_category"]["exact_match_accuracy"]
            - current["risk_category"]["exact_match_accuracy"],
            4,
        ),
        "severity_weighted_kappa_delta": round(
            candidate["risk_severity"]["weighted_kappa"]
            - current["risk_severity"]["weighted_kappa"],
            4,
        ),
        "severity_macro_f1_delta": round(
            candidate["risk_severity"]["macro_f1"]
            - current["risk_severity"]["macro_f1"],
            4,
        ),
    }
    gates = {
        "risk_recall_at_least_0_80": candidate["risk_detection"]["risk"]["recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": candidate["risk_detection"]["macro_f1"] >= 0.75,
        "category_macro_f1_at_least_0_65": candidate["risk_category"]["macro_f1"] >= 0.65,
        "severity_weighted_kappa_at_least_0_65": candidate["risk_severity"]["weighted_kappa"] >= 0.65,
    }
    category_counts = Counter(
        category
        for record in records
        if record["risk_detected"] == 1
        for category in record["risk_categories"]
    )
    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_scope": "locked_product_disjoint_test_once",
            "test_predictions_computed": True,
            "test_dataset": paths[0].name,
            "test_dataset_sha256": sha256(paths[0]),
            "rows": len(records),
            "products": len({record["product_id"] for record in records}),
            "selection_statement": "Candidate architecture and thresholds were frozen before this test evaluation; no test label was used for fitting or tuning.",
            "canonical_category_classes": list(CANONICAL_CLASSES),
            "test_category_support": dict(sorted(category_counts.items())),
        },
        "current": current,
        "candidate": candidate,
        "comparison": comparison,
        "qualification_gates": gates,
        "qualified_for_integration": all(gates.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
