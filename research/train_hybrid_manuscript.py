"""Train the deployable hybrid-manuscript sentiment and risk artifact.

Sentiment is a direct positive/negative/uncertain classifier trained with the
adjudicated Lazada reviews plus FiReCS training data. Risk is an independent
binary classifier trained only on the adjudicated Lazada reviews. Product
groups remain intact when selecting the risk threshold from out-of-fold
predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_firecs import load_firecs  # noqa: E402
from research.cross_validate_hybrid_manuscript import (  # noqa: E402
    evaluation_sentiment,
)
from research.train_risk_scoring_v1 import (  # noqa: E402
    positive_probability,
    select_risk_threshold,
)
from train_marketplace_v2 import (  # noqa: E402
    classifier_pipeline,
    load_marketplace,
    normalized_text,
)


MODEL_VERSION = "defaketive-hybrid-manuscript-1.1.0-development"
DEFAULT_DATASET = (
    ROOT / "outputs" / "lazada_gold_test_v1" / "lazada_round1_all_adjudicated.csv"
)
DEFAULT_FIRECS = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_train_set.csv"
DEFAULT_CV_REPORT = (
    SENTIMENT_ROOT
    / "evaluation"
    / "hybrid_manuscript_three_class_grouped_cross_validation.json"
)
DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models" / "hybrid_manuscript_v1.joblib"
DEFAULT_REPORT = (
    SENTIMENT_ROOT / "evaluation" / "hybrid_manuscript_training.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grouped_oof_risk_probabilities(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros(len(labels), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, validation in splitter.split(texts, labels, groups):
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError("Product leakage detected while selecting risk threshold.")
        model = classifier_pipeline()
        model.fit(texts[train].tolist(), labels[train])
        probabilities[validation] = positive_probability(
            model, texts[validation].tolist()
        )
    return probabilities


def train(
    dataset: Path,
    firecs_path: Path,
    cv_report_path: Path,
    output: Path,
    report_path: Path,
    *,
    seed: int = 42,
) -> dict:
    records = load_marketplace(dataset)
    texts = np.asarray([record["text"] for record in records], dtype=object)
    groups = np.asarray([record["product_id"] for record in records], dtype=object)
    risk_labels = np.asarray(
        [int(record["risk_detected"]) for record in records], dtype=int
    )

    local_texts = set(texts.tolist())
    firecs = load_firecs(firecs_path)
    clean_firecs = [
        row
        for row in firecs
        if normalized_text(row["review"]) not in local_texts
    ]
    sentiment_texts = [normalized_text(row["review"]) for row in clean_firecs]
    sentiment_labels = [evaluation_sentiment(row["label"]) for row in clean_firecs]
    sentiment_texts.extend(texts.tolist())
    sentiment_labels.extend(
        evaluation_sentiment(record["sentiment"]) for record in records
    )

    sentiment_model = classifier_pipeline(large=True)
    sentiment_model.fit(sentiment_texts, sentiment_labels)

    oof_risk = grouped_oof_risk_probabilities(
        texts, risk_labels, groups, seed=seed
    )
    risk_threshold, risk_threshold_metrics = select_risk_threshold(
        risk_labels, oof_risk
    )
    risk_model = classifier_pipeline()
    risk_model.fit(texts.tolist(), risk_labels)

    cv_report = json.loads(cv_report_path.read_text(encoding="utf-8"))
    cv_metrics = cv_report["pooled_out_of_fold"]
    risk_gates_pass = (
        cv_report["development_gates"]["risk_recall_at_least_0_80"]
        and cv_report["development_gates"]["risk_macro_f1_at_least_0_75"]
    )
    sentiment_gates_pass = (
        cv_report["development_gates"]["sentiment_macro_f1_at_least_0_75"]
        and cv_report["development_gates"]["uncertain_f1_at_least_0_60"]
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "deployment_status": "development_candidate",
        "qualified_for_integration": False,
        "qualified_for_risk_scoring_development": bool(risk_gates_pass),
        "sentiment_development_gates_passed": bool(sentiment_gates_pass),
        "qualification_note": (
            "Risk gates passed product-grouped cross-validation. Sentiment gates "
            "did not pass, and a newly collected locked product test is still required."
        ),
        "sentiment_policy": "positive, negative, or uncertain; mixed and neutral map to uncertain",
        "sentiment_strategy": "direct-three-class",
        "risk_policy": "binary yes/no; no risk categories or severity",
        "score_formula": "100 * (0.30 * negative_ratio + 0.70 * detected_risk_ratio)",
        "random_state": seed,
        "marketplace_rows": len(records),
        "marketplace_products": len(set(groups)),
        "firecs_rows_loaded": len(firecs),
        "firecs_rows_used": len(clean_firecs),
        "sentiment_training_rows": len(sentiment_texts),
        "risk_training_rows": len(records),
        "sentiment_distribution": dict(Counter(sentiment_labels)),
        "risk_distribution": {
            "no_risk": int((risk_labels == 0).sum()),
            "risk": int((risk_labels == 1).sum()),
        },
        "sources": [
            {"path": str(dataset.relative_to(ROOT)), "sha256": sha256(dataset)},
            {"path": str(firecs_path.relative_to(ROOT)), "sha256": sha256(firecs_path)},
            {
                "path": str(cv_report_path.relative_to(ROOT)),
                "sha256": sha256(cv_report_path),
            },
        ],
    }
    artifact = {
        "metadata": metadata,
        "sentiment_pipeline": sentiment_model,
        "risk_detection_pipeline": risk_model,
        "risk_detection_threshold": risk_threshold,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    report = {
        "metadata": metadata,
        "artifact": str(output.relative_to(ROOT)),
        "artifact_sha256": sha256(output),
        "risk_threshold_selection": {
            "method": "five-fold stratified product-group out-of-fold predictions",
            "threshold": round(float(risk_threshold), 4),
            **{
                key: round(float(value), 4)
                for key, value in risk_threshold_metrics.items()
            },
        },
        "cross_validation": {
            "report": str(cv_report_path.relative_to(ROOT)),
            "sentiment_macro_f1": cv_metrics["sentiment"]["f1"],
            "uncertain_f1": cv_metrics["sentiment"]["per_class"]["uncertain"]["f1"],
            "risk_macro_f1": cv_metrics["risk_detection"]["macro_f1"],
            "risk_recall": cv_metrics["risk_detection"]["risk"]["recall"],
            "product_score_mae_points": cv_metrics["product_risk_score"][
                "mean_absolute_error_points"
            ],
            "product_band_accuracy": cv_metrics["product_risk_score"][
                "risk_level_accuracy"
            ],
        },
        "next_gate": "Evaluate once on a newly frozen product-disjoint test set.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--firecs", type=Path, default=DEFAULT_FIRECS)
    parser.add_argument("--cv-report", type=Path, default=DEFAULT_CV_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(
        args.dataset.resolve(),
        args.firecs.resolve(),
        args.cv_report.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
