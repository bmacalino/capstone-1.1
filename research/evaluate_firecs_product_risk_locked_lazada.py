"""Evaluate the FiReCS risk candidate once on the locked Lazada test split.

Only the binary product-risk head is evaluated.  Risk categories and severity
are intentionally outside the current model design.  The per-product analysis
reports error in the 70-point risk contribution used by the manuscript score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research.cross_validate_firecs_product_risk import (  # noqa: E402
    DEFAULT_DATASET as FIRECS_DATASET,
    load_records as load_firecs_records,
    risk_metrics,
)
from research.train_firecs_product_risk import (  # noqa: E402
    DEFAULT_ARTIFACT as DEFAULT_CANDIDATE,
)
from research.train_risk_scoring_v1 import positive_probability  # noqa: E402
from train_marketplace_v2 import load_marketplace, normalized_text  # noqa: E402


DEFAULT_TEST = (
    ROOT / "outputs/lazada_gold_test_v1/lazada_round1_all_adjudicated.csv"
)
DEFAULT_TRAIN = (
    ROOT / "research/data/processed/lazada_round1/lazada_round1_train.csv"
)
DEFAULT_VALIDATION = (
    ROOT / "outputs/annotation_quality_round_1/lazada_round1_validation_adjudicated.csv"
)
DEFAULT_CURRENT = SENTIMENT_ROOT / "models/hybrid_manuscript_v1.joblib"
DEFAULT_REPORT = (
    SENTIMENT_ROOT
    / "evaluation/firecs_product_risk_candidate_lazada_locked_test.json"
)
RISK_CONTRIBUTION_WEIGHT = 0.70


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_artifact(
    artifact: dict[str, Any], texts: list[str]
) -> tuple[np.ndarray, np.ndarray, float]:
    pipeline = artifact["risk_detection_pipeline"]
    probabilities = positive_probability(pipeline, texts)
    threshold = float(
        artifact.get(
            "risk_detection_threshold", getattr(pipeline, "threshold", 0.5)
        )
    )
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities, threshold


def product_metrics(
    records: list[dict[str, Any]], predictions: np.ndarray
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record["product_id"])].append(index)

    rows: list[dict[str, Any]] = []
    for product_id, indices in sorted(grouped.items()):
        actual_rate = float(
            np.mean([int(records[index]["risk_detected"]) for index in indices])
        )
        predicted_rate = float(np.mean(predictions[indices]))
        rate_error = predicted_rate - actual_rate
        rows.append(
            {
                "product_id": product_id,
                "review_count": len(indices),
                "actual_risk_review_ratio": round(actual_rate, 4),
                "predicted_risk_review_ratio": round(predicted_rate, 4),
                "signed_ratio_error": round(rate_error, 4),
                "absolute_ratio_error": round(abs(rate_error), 4),
                "actual_risk_contribution_points": round(
                    100 * RISK_CONTRIBUTION_WEIGHT * actual_rate, 2
                ),
                "predicted_risk_contribution_points": round(
                    100 * RISK_CONTRIBUTION_WEIGHT * predicted_rate, 2
                ),
                "absolute_contribution_error_points": round(
                    100 * RISK_CONTRIBUTION_WEIGHT * abs(rate_error), 2
                ),
            }
        )

    ratio_errors = np.asarray(
        [row["predicted_risk_review_ratio"] - row["actual_risk_review_ratio"] for row in rows],
        dtype=float,
    )
    return {
        "product_count": len(rows),
        "unweighted_mean_absolute_ratio_error": round(
            float(np.mean(np.abs(ratio_errors))), 4
        ),
        "unweighted_root_mean_squared_ratio_error": round(
            float(np.sqrt(np.mean(np.square(ratio_errors)))), 4
        ),
        "unweighted_mean_absolute_risk_contribution_error_points": round(
            float(100 * RISK_CONTRIBUTION_WEIGHT * np.mean(np.abs(ratio_errors))),
            2,
        ),
        "products": rows,
    }


def evaluate(
    test_path: Path,
    train_path: Path,
    validation_path: Path,
    firecs_path: Path,
    candidate_path: Path,
    current_path: Path,
) -> dict[str, Any]:
    test_source = load_marketplace(test_path)
    test = [record for record in test_source if record["split"] == "test"]
    train = load_marketplace(train_path)
    validation = load_marketplace(validation_path)
    firecs = load_firecs_records(firecs_path)

    if len(test) != 133 or len({record["review_id"] for record in test}) != 133:
        raise ValueError("Locked Lazada test must contain 133 unique reviews.")
    if any(record["risk_detected"] is None for record in test):
        raise ValueError("Locked Lazada test contains missing risk labels.")

    test_products = {record["product_id"] for record in test}
    development_products = {
        record["product_id"] for record in train + validation
    }
    product_overlap = sorted(test_products & development_products)
    firecs_texts = {normalized_text(record["review_text"]) for record in firecs}
    text_overlap = sorted(record["review_id"] for record in test if record["text"] in firecs_texts)
    if product_overlap:
        raise ValueError(f"Locked Lazada products overlap development: {product_overlap}")
    if text_overlap:
        raise ValueError(f"Locked Lazada text overlaps FiReCS: {text_overlap}")

    texts = [str(record["text"]) for record in test]
    labels = np.asarray([int(record["risk_detected"]) for record in test], dtype=int)
    candidate_artifact = joblib.load(candidate_path)
    candidate_prediction, candidate_probability, candidate_threshold = predict_artifact(
        candidate_artifact, texts
    )
    candidate_review_metrics = risk_metrics(
        labels, candidate_prediction, candidate_probability
    )

    current_artifact = joblib.load(current_path)
    current_prediction, current_probability, current_threshold = predict_artifact(
        current_artifact, texts
    )
    current_review_metrics = risk_metrics(labels, current_prediction, current_probability)

    gates = {
        "risk_recall_at_least_0_80": candidate_review_metrics["risk"]["recall"]
        >= 0.80,
        "risk_macro_f1_at_least_0_75": candidate_review_metrics["macro_f1"] >= 0.75,
    }
    return {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_scope": "locked_lazada_product_disjoint_external_transfer_test",
            "test_predictions_computed": True,
            "test_dataset": str(test_path.relative_to(ROOT)),
            "test_dataset_sha256": sha256(test_path),
            "source_rows": len(test_source),
            "subset_rule": "split == test",
            "rows": len(test),
            "products": len(test_products),
            "risk_rows": int(labels.sum()),
            "no_risk_rows": int((labels == 0).sum()),
            "product_overlap_with_lazada_development": len(product_overlap),
            "normalized_text_overlap_with_firecs_training": len(text_overlap),
            "selection_statement": (
                "The FiReCS candidate architecture and threshold were frozen before "
                "this test. No Lazada test label was used to fit or tune it."
            ),
            "current_model_comparison_limitation": (
                "The current hybrid artifact was trained on the combined Lazada gold "
                "set, so its result here is an in-sample diagnostic and not a fair "
                "external comparison with the FiReCS candidate."
            ),
            "score_scope": (
                "Product metrics cover only the 70-point detected-risk contribution. "
                "The 30-point sentiment contribution is outside this risk-head test."
            ),
        },
        "candidate": {
            "artifact": str(candidate_path.relative_to(ROOT)),
            "artifact_sha256": sha256(candidate_path),
            "model_version": candidate_artifact["metadata"]["model_version"],
            "threshold": round(candidate_threshold, 4),
            "review_level": candidate_review_metrics,
            "product_level": product_metrics(test, candidate_prediction),
        },
        "current_in_sample_diagnostic": {
            "artifact": str(current_path.relative_to(ROOT)),
            "artifact_sha256": sha256(current_path),
            "model_version": current_artifact["metadata"]["model_version"],
            "threshold": round(current_threshold, 4),
            "review_level": current_review_metrics,
            "product_level": product_metrics(test, current_prediction),
        },
        "qualification_gates": gates,
        "qualified_for_integration": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--firecs", type=Path, default=FIRECS_DATASET)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(
            f"Locked-test report already exists and will not be overwritten: {args.report}"
        )
    report = evaluate(
        args.test.resolve(),
        args.train.resolve(),
        args.validation.resolve(),
        args.firecs.resolve(),
        args.candidate.resolve(),
        args.current.resolve(),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
