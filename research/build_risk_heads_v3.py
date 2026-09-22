"""Build a validation-selected ordinal severity candidate without reading test labels."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.risk_heads import OrdinalSeverityClassifier, rating_feature_text
from train_marketplace_v2 import classifier_pipeline, load_marketplace, sha256


DEFAULT_TRAIN = ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_train.csv"
DEFAULT_VALIDATION = (
    ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_validation.csv"
)
DEFAULT_BASE = SENTIMENT_ROOT / "models" / "lazada_round1_binary_uncertain.joblib"
DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models" / "lazada_risk_heads_v3_candidate.joblib"
DEFAULT_REPORT = SENTIMENT_ROOT / "evaluation" / "lazada_risk_heads_v3_validation.json"


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(
            float(f1_score(actual, predicted, average="macro", zero_division=0)), 4
        ),
        "weighted_kappa": round(
            float(cohen_kappa_score(actual, predicted, weights="quadratic")), 4
        ),
        "mean_absolute_error": round(float(np.mean(np.abs(actual - predicted))), 4),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    train_records = load_marketplace(args.train_dataset)
    validation_records = load_marketplace(args.validation_dataset)
    train = [
        index
        for index, record in enumerate(train_records)
        if record["risk_detected"] == 1 and record["risk_severity"] is not None
    ]
    validation = [
        index
        for index, record in enumerate(validation_records)
        if record["risk_detected"] == 1 and record["risk_severity"] is not None
    ]
    train_texts = [
        rating_feature_text(train_records[index]["text"], train_records[index]["rating"])
        for index in train
    ]
    validation_texts = [
        rating_feature_text(
            validation_records[index]["text"], validation_records[index]["rating"]
        )
        for index in validation
    ]
    train_severity = np.asarray(
        [train_records[index]["risk_severity"] for index in train], dtype=int
    )
    validation_severity = np.asarray(
        [validation_records[index]["risk_severity"] for index in validation], dtype=int
    )

    model_ge2 = classifier_pipeline()
    model_ge3 = classifier_pipeline()
    model_ge2.fit(train_texts, (train_severity >= 2).astype(int))
    model_ge3.fit(train_texts, (train_severity >= 3).astype(int))
    untuned = OrdinalSeverityClassifier(
        model_ge2, model_ge3, threshold_ge2=0.5, threshold_ge3=0.5
    )
    probability_ge2, probability_ge3 = untuned.cumulative_probabilities(validation_texts)

    candidates = []
    for threshold_ge2 in np.arange(0.25, 0.76, 0.05):
        for threshold_ge3 in np.arange(0.25, 0.76, 0.05):
            predicted = np.where(
                probability_ge3 >= threshold_ge3,
                3,
                np.where(probability_ge2 >= threshold_ge2, 2, 1),
            )
            score = metrics(validation_severity, predicted)
            candidates.append(
                (
                    score["weighted_kappa"],
                    -score["mean_absolute_error"],
                    score["macro_f1"],
                    float(threshold_ge2),
                    float(threshold_ge3),
                    score,
                )
            )
    best = max(candidates, key=lambda candidate: candidate[:3])
    severity_model = OrdinalSeverityClassifier(
        model_ge2,
        model_ge3,
        threshold_ge2=best[3],
        threshold_ge3=best[4],
    )

    artifact = joblib.load(args.base_artifact)
    artifact = copy.deepcopy(artifact)
    artifact["risk_severity_pipeline"] = severity_model
    metadata = dict(artifact["metadata"])
    metadata.update(
        {
            "model_version": "defaketive-marketplace-v3-risk-heads-candidate",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "qualified_for_integration": False,
            "qualification_reason": "Validation labels require clean human adjudication before final test evaluation.",
            "severity_model": "ordinal_cumulative_logistic_regression",
            "severity_uses_rating_token": True,
            "severity_threshold_ge2": round(best[3], 2),
            "severity_threshold_ge3": round(best[4], 2),
            "selection_policy": "Severity thresholds selected on validation only; locked test labels were not read.",
        }
    )
    artifact["metadata"] = metadata
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    report = {
        "metadata": {
            "created_at": metadata["created_at"],
            "train_dataset": args.train_dataset.name,
            "train_dataset_sha256": sha256(args.train_dataset),
            "validation_dataset": args.validation_dataset.name,
            "validation_dataset_sha256": sha256(args.validation_dataset),
            "base_artifact": args.base_artifact.name,
            "base_artifact_sha256": sha256(args.base_artifact),
            "candidate_artifact": args.artifact.name,
            "candidate_artifact_sha256": sha256(args.artifact),
            "selection_policy": metadata["selection_policy"],
            "test_predictions_computed": False,
            "train_rows": len(train),
            "validation_rows": len(validation),
        },
        "validation": {
            "severity": {
                **best[5],
                "threshold_ge2": round(best[3], 2),
                "threshold_ge3": round(best[4], 2),
            }
        },
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
