"""Train the final FiReCS product-risk development candidate.

The binary risk threshold is selected from five-fold out-of-fold predictions,
then one classifier is fitted on all adjudicated FiReCS reviews.  The artifact
reuses the deployed sentiment head and remains separate from the active model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.manuscript import (  # noqa: E402
    DEFAULT_HYBRID_ARTIFACT,
    HybridManuscriptRiskModel,
)
from defaketive_sentiment.model import DefaketiveSentimentModel  # noqa: E402
from defaketive_sentiment.trained import feature_text  # noqa: E402
from research.cross_validate_firecs_product_risk import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_REPORT as DEFAULT_CV_REPORT,
    DEFAULT_SOURCE_WORKBOOK,
    duplicate_group,
    load_records,
    risk_metrics,
)
from research.train_risk_scoring_v1 import (  # noqa: E402
    positive_probability,
    select_risk_threshold,
)
from train_marketplace_v2 import classifier_pipeline  # noqa: E402


MODEL_VERSION = "defaketive-firecs-product-risk-0.1.0-candidate"
DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models/firecs_product_risk_candidate.joblib"
DEFAULT_REPORT = (
    SENTIMENT_ROOT / "evaluation/firecs_product_risk_candidate_training.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(text: object) -> str:
    return feature_text(DefaketiveSentimentModel.normalize(text))


def grouped_oof_probabilities(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros(len(labels), dtype=float)
    folds = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, validation) in enumerate(
        splitter.split(texts, labels, groups), start=1
    ):
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("Duplicate-text leakage while selecting threshold.")
        model = classifier_pipeline()
        model.fit(texts[train].tolist(), labels[train])
        probabilities[validation] = positive_probability(
            model, texts[validation].tolist()
        )
        folds[validation] = fold
    if np.any(folds < 0):
        raise RuntimeError("Out-of-fold threshold selection missed one or more rows.")
    return probabilities, folds


def train(
    dataset: Path,
    source_workbook: Path,
    cv_report_path: Path,
    active_artifact_path: Path,
    output: Path,
    report_path: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    records = load_records(dataset)
    raw_texts = np.asarray([record["review_text"] for record in records], dtype=object)
    texts = np.asarray([normalized_text(text) for text in raw_texts], dtype=object)
    labels = np.asarray([record["risk_detected"] for record in records], dtype=int)
    groups = np.asarray([duplicate_group(text) for text in raw_texts], dtype=object)

    oof_probabilities, folds = grouped_oof_probabilities(
        texts, labels, groups, seed=seed
    )
    threshold, threshold_selection = select_risk_threshold(labels, oof_probabilities)
    oof_predictions = (oof_probabilities >= threshold).astype(int)
    threshold_metrics = risk_metrics(labels, oof_predictions, oof_probabilities)

    risk_model = classifier_pipeline()
    risk_model.fit(texts.tolist(), labels)

    active_artifact = joblib.load(active_artifact_path)
    if "sentiment_pipeline" not in active_artifact:
        raise ValueError("Active artifact has no sentiment pipeline to preserve.")
    cv_report = json.loads(cv_report_path.read_text(encoding="utf-8"))
    nested_metrics = cv_report["nested_cross_validation"]
    current_metrics = cv_report["current_hybrid_external_diagnostic"]
    development_gates = dict(cv_report["development_gates"])

    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "deployment_status": "development_candidate",
        "qualified_for_integration": False,
        "qualified_for_risk_scoring_development": bool(
            cv_report["passes_all_development_gates"]
        ),
        "qualification_note": (
            "Passed review-level duplicate-grouped nested cross-validation. "
            "FiReCS has no product identifier, so a locked product-disjoint test "
            "is still required before deployment."
        ),
        "sentiment_policy": "preserved from active hybrid artifact",
        "risk_policy": "binary product-only yes/no; no categories or severity",
        "score_formula": (
            "100 * (0.30 * negative_review_ratio + "
            "0.70 * detected_risk_review_ratio)"
        ),
        "random_state": seed,
        "risk_training_rows": len(records),
        "risk_distribution": {
            "no_risk": int((labels == 0).sum()),
            "risk": int(labels.sum()),
        },
        "sentiment_distribution": dict(
            sorted(Counter(record["firecs_sentiment"] for record in records).items())
        ),
        "unique_normalized_reviews": len(set(groups)),
        "risk_detection_threshold": float(threshold),
        "threshold_selection": (
            "five-fold stratified duplicate-text-group out-of-fold predictions"
        ),
        "sources": [
            {"path": str(dataset.relative_to(ROOT)), "sha256": sha256(dataset)},
            {
                "path": str(source_workbook.relative_to(ROOT)),
                "sha256": sha256(source_workbook),
            },
            {
                "path": str(cv_report_path.relative_to(ROOT)),
                "sha256": sha256(cv_report_path),
            },
            {
                "path": str(active_artifact_path.relative_to(ROOT)),
                "sha256": sha256(active_artifact_path),
                "use": "sentiment pipeline only",
            },
        ],
    }
    artifact = {
        "metadata": metadata,
        "sentiment_pipeline": active_artifact["sentiment_pipeline"],
        "risk_detection_pipeline": risk_model,
        "risk_detection_threshold": float(threshold),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    # Confirm that the candidate can be loaded by the same runtime wrapper used
    # by the application.  This does not activate or replace the current model.
    runtime_candidate = HybridManuscriptRiskModel(output, allow_unqualified=True)
    smoke_result = runtime_candidate.analyze(raw_texts[0])

    comparison = {
        "candidate_nested_cross_validation": {
            "accuracy": nested_metrics["accuracy"],
            "balanced_accuracy": nested_metrics["balanced_accuracy"],
            "macro_f1": nested_metrics["macro_f1"],
            "risk_precision": nested_metrics["risk"]["precision"],
            "risk_recall": nested_metrics["risk"]["recall"],
            "risk_f1": nested_metrics["risk"]["f1"],
            "false_positive_count": nested_metrics["confusion_matrix"]["values"][0][1],
            "false_negative_count": nested_metrics["confusion_matrix"]["values"][1][0],
        },
        "currently_deployed_external_diagnostic": {
            "accuracy": current_metrics["accuracy"],
            "balanced_accuracy": current_metrics["balanced_accuracy"],
            "macro_f1": current_metrics["macro_f1"],
            "risk_precision": current_metrics["risk"]["precision"],
            "risk_recall": current_metrics["risk"]["recall"],
            "risk_f1": current_metrics["risk"]["f1"],
            "false_positive_count": current_metrics["confusion_matrix"]["values"][0][1],
            "false_negative_count": current_metrics["confusion_matrix"]["values"][1][0],
        },
    }
    report = {
        "metadata": metadata,
        "artifact": str(output.relative_to(ROOT)),
        "artifact_sha256": sha256(output),
        "threshold_selection": {
            "threshold": round(float(threshold), 4),
            "selection_macro_f1": round(
                float(threshold_selection["macro_f1"]), 4
            ),
            "selection_risk_recall": round(
                float(threshold_selection["risk_recall"]), 4
            ),
            "oof_metrics_for_threshold_selection": threshold_metrics,
            "fold_counts": dict(sorted(Counter(folds.tolist()).items())),
            "note": (
                "These pooled out-of-fold predictions select the final threshold. "
                "Use the separate nested cross-validation metrics for an unbiased "
                "development estimate."
            ),
        },
        "side_by_side": comparison,
        "development_gates": development_gates,
        "passes_all_development_gates": bool(
            cv_report["passes_all_development_gates"]
        ),
        "runtime_smoke_test": {
            "loaded": True,
            "model_version": runtime_candidate.model_version,
            "risk_detected_type": type(smoke_result["risk"]["detected"]).__name__,
            "threshold": smoke_result["risk"]["detection_threshold"],
        },
        "deployment_changed": False,
        "next_gate": "Evaluate on a newly frozen product-disjoint locked test set.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK)
    parser.add_argument("--cv-report", type=Path, default=DEFAULT_CV_REPORT)
    parser.add_argument("--active-artifact", type=Path, default=DEFAULT_HYBRID_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(
        args.dataset.resolve(),
        args.source_workbook.resolve(),
        args.cv_report.resolve(),
        args.active_artifact.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
