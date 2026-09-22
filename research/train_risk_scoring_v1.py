"""Train the category-free DeFaketive risk-scoring candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from defaketive_sentiment.risk_heads import (  # noqa: E402
    CalibratedBinaryClassifier,
    OrdinalSeverityClassifier,
    rating_feature_text,
)
from train_marketplace_v2 import classifier_pipeline, load_marketplace  # noqa: E402


MODEL_VERSION = "defaketive-risk-scoring-v1-0.1.0-candidate"
TRAIN_FILE = ROOT / "research/data/processed/lazada_round1/lazada_round1_train.csv"
VALIDATION_FILE = ROOT / "outputs/annotation_quality_round_1/lazada_round1_validation_adjudicated.csv"
ACTIVE_ARTIFACT = SENTIMENT_ROOT / "models/lazada_round1_binary_uncertain.joblib"
DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models/risk_scoring_v1_candidate.joblib"
DEFAULT_REPORT = SENTIMENT_ROOT / "evaluation/risk_scoring_v1_cross_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_probability(model, values) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(list(values)), dtype=float)
    return probabilities[:, list(model.classes_).index(1)]


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if mask.any():
            result += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(result) if total else 0.0


def select_risk_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict]:
    candidates = []
    for threshold in np.linspace(0.20, 0.80, 121):
        predicted = (probabilities >= threshold).astype(int)
        macro_f1 = f1_score(labels, predicted, average="macro", zero_division=0)
        recall = precision_recall_fscore_support(
            labels, predicted, labels=[1], average=None, zero_division=0
        )[1][0]
        candidates.append((float(recall >= 0.80), macro_f1, recall, -abs(threshold - 0.5), threshold))
    _passes, macro_f1, recall, _distance, threshold = max(candidates)
    return float(threshold), {"macro_f1": float(macro_f1), "risk_recall": float(recall)}


def select_severity_thresholds(labels: np.ndarray, p_ge2: np.ndarray, p_ge3: np.ndarray) -> tuple[float, float]:
    best = (-2.0, -2.0, 0.5, 0.5)
    for threshold_ge2 in np.linspace(0.25, 0.75, 21):
        for threshold_ge3 in np.linspace(0.25, 0.75, 21):
            predicted = np.where(
                p_ge3 >= threshold_ge3,
                3,
                np.where(p_ge2 >= threshold_ge2, 2, 1),
            )
            kappa = cohen_kappa_score(labels, predicted, weights="quadratic")
            macro_f1 = f1_score(labels, predicted, average="macro", zero_division=0)
            candidate = (float(kappa), float(macro_f1), -abs(threshold_ge2 - 0.5), -abs(threshold_ge3 - 0.5))
            if candidate > best:
                best = candidate
                selected = (float(threshold_ge2), float(threshold_ge3))
    return selected


def grouped_oof_risk(texts: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    raw = np.zeros(len(labels), dtype=float)
    folds = np.full(len(labels), -1, dtype=int)
    for fold, (train, test) in enumerate(splitter.split(texts, labels, groups)):
        model = classifier_pipeline()
        model.fit(texts[train], labels[train])
        raw[test] = positive_probability(model, texts[test])
        folds[test] = fold

    calibrated = np.zeros(len(labels), dtype=float)
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        calibrator = LogisticRegression(max_iter=2_000, solver="lbfgs")
        calibrator.fit(logit(raw[train]), labels[train])
        calibrated[test] = positive_probability(calibrator, logit(raw[test]))
    return raw, calibrated


def grouped_oof_severity(
    texts: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    p_ge2 = np.zeros(len(labels), dtype=float)
    p_ge3 = np.zeros(len(labels), dtype=float)
    for train, test in splitter.split(texts, labels, groups):
        model_ge2 = classifier_pipeline()
        model_ge3 = classifier_pipeline()
        model_ge2.fit(texts[train], (labels[train] >= 2).astype(int))
        model_ge3.fit(texts[train], (labels[train] >= 3).astype(int))
        p_ge2[test] = positive_probability(model_ge2, texts[test])
        p_ge3[test] = positive_probability(model_ge3, texts[test])
    p_ge3 = np.minimum(p_ge2, p_ge3)
    return p_ge2, p_ge3


def train(output: Path, report_path: Path, seed: int = 42) -> dict:
    records = load_marketplace(TRAIN_FILE) + load_marketplace(VALIDATION_FILE)
    review_ids = [record["review_id"] for record in records]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Development files contain duplicate review IDs.")

    texts = np.asarray([record["text"] for record in records], dtype=object)
    groups = np.asarray([record["product_id"] for record in records], dtype=object)
    risk_labels = np.asarray([record["risk_detected"] for record in records], dtype=int)
    raw_oof, calibrated_oof = grouped_oof_risk(texts, risk_labels, groups, seed)
    risk_threshold, risk_metrics = select_risk_threshold(risk_labels, calibrated_oof)
    risk_metrics.update(
        {
            "brier_score": float(brier_score_loss(risk_labels, calibrated_oof)),
            "expected_calibration_error_10_bins": expected_calibration_error(
                risk_labels, calibrated_oof
            ),
            "threshold": risk_threshold,
        }
    )

    final_base_risk = classifier_pipeline()
    final_base_risk.fit(texts, risk_labels)
    final_calibrator = LogisticRegression(max_iter=2_000, solver="lbfgs")
    final_calibrator.fit(logit(raw_oof), risk_labels)
    risk_model = CalibratedBinaryClassifier(
        final_base_risk, final_calibrator, threshold=risk_threshold
    )

    risk_indices = np.flatnonzero(risk_labels == 1)
    severity_labels = np.asarray(
        [records[index]["risk_severity"] for index in risk_indices], dtype=int
    )
    severity_texts = np.asarray(
        [rating_feature_text(texts[index], records[index]["rating"]) for index in risk_indices],
        dtype=object,
    )
    severity_groups = groups[risk_indices]
    p_ge2, p_ge3 = grouped_oof_severity(
        severity_texts, severity_labels, severity_groups, seed
    )
    threshold_ge2, threshold_ge3 = select_severity_thresholds(
        severity_labels, p_ge2, p_ge3
    )
    severity_predictions = np.where(
        p_ge3 >= threshold_ge3,
        3,
        np.where(p_ge2 >= threshold_ge2, 2, 1),
    )
    severity_metrics = {
        "weighted_kappa": float(
            cohen_kappa_score(severity_labels, severity_predictions, weights="quadratic")
        ),
        "macro_f1": float(
            f1_score(severity_labels, severity_predictions, average="macro", zero_division=0)
        ),
        "mean_absolute_error": float(
            mean_absolute_error(severity_labels, severity_predictions)
        ),
        "threshold_ge2": threshold_ge2,
        "threshold_ge3": threshold_ge3,
    }

    final_ge2 = classifier_pipeline()
    final_ge3 = classifier_pipeline()
    final_ge2.fit(severity_texts, (severity_labels >= 2).astype(int))
    final_ge3.fit(severity_texts, (severity_labels >= 3).astype(int))
    severity_model = OrdinalSeverityClassifier(
        final_ge2,
        final_ge3,
        threshold_ge2=threshold_ge2,
        threshold_ge3=threshold_ge3,
    )

    active = joblib.load(ACTIVE_ARTIFACT)
    gates = {
        "risk_recall_at_least_0_80": risk_metrics["risk_recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": risk_metrics["macro_f1"] >= 0.75,
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "qualified_for_integration": False,
        "qualification_note": "Cross-validation candidate; a new product-disjoint locked test is still required.",
        "development_gates": gates,
        "random_state": seed,
        "training_rows": len(records),
        "training_products": len(set(groups)),
        "risk_positive_rows": int(risk_labels.sum()),
        "risk_categories_used": False,
        "severity_uses_rating_token": True,
        "severity_weights": {"1": 0.33, "2": 0.67, "3": 1.0},
        "risk_prior": 0.15,
        "risk_prior_strength": 30,
        "sources": [
            {"path": str(TRAIN_FILE.relative_to(ROOT)), "sha256": sha256(TRAIN_FILE)},
            {"path": str(VALIDATION_FILE.relative_to(ROOT)), "sha256": sha256(VALIDATION_FILE)},
        ],
    }
    artifact = {
        "metadata": metadata,
        "sentiment_pipeline": active["sentiment_pipeline"],
        "sentiment_uncertainty_threshold": active.get("sentiment_uncertainty_threshold", 0.75),
        "risk_detection_pipeline": risk_model,
        "risk_detection_threshold": risk_threshold,
        "risk_severity_pipeline": severity_model,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    report = {
        "model_version": MODEL_VERSION,
        "evaluation_policy": "Five-fold stratified product-group out-of-fold evaluation on development data only.",
        "risk_detection": risk_metrics,
        "severity": severity_metrics,
        "development_gates": gates,
        "artifact": str(output.relative_to(ROOT)),
        "qualified_for_integration": False,
        "next_gate": "Evaluate once on a newly frozen product-disjoint locked test set.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.report, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
