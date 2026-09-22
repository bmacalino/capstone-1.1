"""Compare current and rebuilt risk models on adjudicated validation labels only."""

from __future__ import annotations

import argparse
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
)


DEFAULT_DATASET = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "lazada_round1_validation_adjudicated.csv"
)
DEFAULT_CURRENT = SENTIMENT_ROOT / "models" / "lazada_round1_binary_uncertain.joblib"
DEFAULT_REBUILT = SENTIMENT_ROOT / "models" / "lazada_risk_heads_v3_candidate.joblib"
DEFAULT_REPORT = (
    SENTIMENT_ROOT
    / "evaluation"
    / "lazada_risk_models_adjudicated_validation.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def severity_metrics(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=int)
    predicted_array = np.asarray(predicted, dtype=int)
    return {
        "sample_count": len(actual),
        "accuracy": round(float(accuracy_score(actual_array, predicted_array)), 4),
        "macro_f1": round(
            float(f1_score(actual_array, predicted_array, average="macro", zero_division=0)),
            4,
        ),
        "weighted_kappa": round(
            float(cohen_kappa_score(actual_array, predicted_array, weights="quadratic")),
            4,
        ),
        "mean_absolute_error": round(
            float(np.mean(np.abs(actual_array - predicted_array))), 4
        ),
    }


def evaluate_model(records: list[dict[str, Any]], artifact_path: Path) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    texts = [str(record["text"]) for record in records]

    risk_actual = [int(record["risk_detected"]) for record in records]
    risk_predicted = artifact["risk_detection_pipeline"].predict(texts).astype(int).tolist()
    risk = _risk_metrics(risk_actual, risk_predicted)

    category_rows = [
        record
        for record in records
        if record["risk_detected"] == 1 and record["risk_categories"]
    ]
    binarizer = artifact["risk_category_binarizer"]
    known_category_rows = [
        record
        for record in category_rows
        if set(record["risk_categories"]).issubset(set(binarizer.classes_))
    ]
    category_actual = binarizer.transform(
        [record["risk_categories"] for record in known_category_rows]
    )
    category_probabilities = artifact["risk_category_pipeline"].predict_proba(
        [record["text"] for record in known_category_rows]
    )
    category_threshold = artifact.get("risk_category_threshold", 0.5)
    category_predicted = category_predictions(category_probabilities, category_threshold)
    category = _multilabel_metrics(
        category_actual,
        category_predicted,
        binarizer.classes_.tolist(),
    )
    supported_columns = category_actual.sum(axis=0) > 0
    category["supported_macro_f1"] = round(
        float(
            f1_score(
                category_actual[:, supported_columns],
                category_predicted[:, supported_columns],
                average="macro",
                zero_division=0,
            )
        ),
        4,
    )
    category["sample_count"] = len(known_category_rows)
    category["unknown_category_rows_excluded"] = len(category_rows) - len(
        known_category_rows
    )

    severity_rows = [
        record
        for record in records
        if record["risk_detected"] == 1 and record["risk_severity"] is not None
    ]
    severity_texts = [
        rating_feature_text(str(record["text"]), record.get("rating"))
        if artifact.get("metadata", {}).get("severity_uses_rating_token")
        else str(record["text"])
        for record in severity_rows
    ]
    severity_actual = [int(record["risk_severity"]) for record in severity_rows]
    severity_predicted = (
        artifact["risk_severity_pipeline"].predict(severity_texts).astype(int).tolist()
    )
    severity = severity_metrics(severity_actual, severity_predicted)

    return {
        "artifact": artifact_path.name,
        "artifact_sha256": sha256(artifact_path),
        "model_version": artifact.get("metadata", {}).get("model_version"),
        "risk_detection": risk,
        "risk_category": category,
        "risk_severity": severity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--rebuilt", type=Path, default=DEFAULT_REBUILT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    records = load_marketplace(args.dataset.resolve())
    if len(records) != 138:
        raise ValueError(f"Expected 138 validation rows, found {len(records)}.")
    current = evaluate_model(records, args.current.resolve())
    rebuilt = evaluate_model(records, args.rebuilt.resolve())

    comparison = {
        "risk_macro_f1_delta": round(
            rebuilt["risk_detection"]["macro_f1"]
            - current["risk_detection"]["macro_f1"],
            4,
        ),
        "risk_recall_delta": round(
            rebuilt["risk_detection"]["risk"]["recall"]
            - current["risk_detection"]["risk"]["recall"],
            4,
        ),
        "category_macro_f1_delta": round(
            rebuilt["risk_category"]["macro_f1"]
            - current["risk_category"]["macro_f1"],
            4,
        ),
        "category_supported_macro_f1_delta": round(
            rebuilt["risk_category"]["supported_macro_f1"]
            - current["risk_category"]["supported_macro_f1"],
            4,
        ),
        "severity_weighted_kappa_delta": round(
            rebuilt["risk_severity"]["weighted_kappa"]
            - current["risk_severity"]["weighted_kappa"],
            4,
        ),
        "severity_macro_f1_delta": round(
            rebuilt["risk_severity"]["macro_f1"]
            - current["risk_severity"]["macro_f1"],
            4,
        ),
    }
    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_scope": "adjudicated_validation_only",
            "test_predictions_computed": False,
            "dataset": args.dataset.name,
            "dataset_sha256": sha256(args.dataset.resolve()),
            "rows": len(records),
        },
        "current": current,
        "rebuilt": rebuilt,
        "comparison": comparison,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
