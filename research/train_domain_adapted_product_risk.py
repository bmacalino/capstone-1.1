"""Train a FiReCS + Lazada domain-adapted binary product-risk candidate.

The 133-row Lazada test split is excluded. Development estimates use nested
grouped cross-validation: Lazada reviews stay grouped by product, while FiReCS
reviews are grouped by normalized duplicate text because product IDs are absent.
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

from defaketive_sentiment.manuscript import DEFAULT_HYBRID_ARTIFACT  # noqa: E402
from research.cross_validate_firecs_product_risk import (  # noqa: E402
    DEFAULT_DATASET as DEFAULT_FIRECS,
    DEFAULT_SOURCE_WORKBOOK,
    duplicate_group,
    inner_threshold,
    load_records as load_firecs_records,
    risk_metrics,
)
from research.evaluate_firecs_product_risk_locked_lazada import (  # noqa: E402
    product_metrics,
)
from research.train_firecs_product_risk import (  # noqa: E402
    grouped_oof_probabilities,
    normalized_text,
)
from research.train_risk_scoring_v1 import (  # noqa: E402
    positive_probability,
    select_risk_threshold,
)
from train_marketplace_v2 import classifier_pipeline, load_marketplace  # noqa: E402


MODEL_VERSION = "defaketive-domain-adapted-product-risk-0.1.0-candidate"
DEFAULT_LAZADA = (
    ROOT / "outputs/lazada_gold_test_v1/lazada_round1_all_adjudicated.csv"
)
DEFAULT_ARTIFACT = (
    SENTIMENT_ROOT / "models/domain_adapted_product_risk_candidate.joblib"
)
DEFAULT_REPORT = (
    SENTIMENT_ROOT
    / "evaluation/domain_adapted_product_risk_candidate_training.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_development_rows(
    firecs_path: Path, lazada_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    firecs = load_firecs_records(firecs_path)
    lazada_all = load_marketplace(lazada_path)
    lazada = [
        record
        for record in lazada_all
        if record["split"] in {"train", "validation"}
    ]
    excluded_test = [record for record in lazada_all if record["split"] == "test"]
    if len(lazada) != 774 or len(excluded_test) != 133:
        raise ValueError(
            "Expected 774 Lazada development rows and 133 excluded test rows."
        )
    development_products = {record["product_id"] for record in lazada}
    test_products = {record["product_id"] for record in excluded_test}
    if development_products & test_products:
        raise ValueError("Lazada development and excluded-test products overlap.")

    firecs_texts = {normalized_text(record["review_text"]) for record in firecs}
    cross_domain_overlap = [
        record["review_id"] for record in lazada if record["text"] in firecs_texts
    ]
    if cross_domain_overlap:
        raise ValueError(
            f"FiReCS and Lazada development text overlap: {cross_domain_overlap}"
        )
    return firecs, lazada


def evaluate_nested(
    texts: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    domains: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    probabilities = np.zeros(len(labels), dtype=float)
    predictions = np.full(len(labels), -1, dtype=int)
    folds = np.full(len(labels), -1, dtype=int)
    fold_reports: list[dict[str, Any]] = []

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(
        outer.split(indices, labels, groups), start=1
    ):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Group leakage in domain-adapted outer fold.")
        threshold, selection = inner_threshold(
            texts, labels, groups, train, seed + fold * 100
        )
        model = classifier_pipeline()
        model.fit(texts[train].tolist(), labels[train])
        fold_probability = positive_probability(model, texts[test].tolist())
        fold_prediction = (fold_probability >= threshold).astype(int)
        probabilities[test] = fold_probability
        predictions[test] = fold_prediction
        folds[test] = fold

        domain_results: dict[str, Any] = {}
        for domain in ("firecs", "lazada"):
            local = test[domains[test] == domain]
            domain_results[domain] = risk_metrics(
                labels[local], predictions[local], probabilities[local]
            )
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "selected_threshold": round(float(threshold), 4),
                "inner_threshold_selection": selection,
                "overall": risk_metrics(
                    labels[test], predictions[test], probabilities[test]
                ),
                "by_domain": domain_results,
            }
        )

    if np.any(predictions < 0) or np.any(folds < 0):
        raise RuntimeError("Nested cross-validation missed one or more rows.")
    by_domain = {}
    for domain in ("firecs", "lazada"):
        local = np.flatnonzero(domains == domain)
        by_domain[domain] = risk_metrics(
            labels[local], predictions[local], probabilities[local]
        )
    return (
        {
            "overall": risk_metrics(labels, predictions, probabilities),
            "by_domain": by_domain,
            "folds": fold_reports,
        },
        predictions,
        probabilities,
        folds,
    )


def train(
    firecs_path: Path,
    source_workbook: Path,
    lazada_path: Path,
    active_artifact_path: Path,
    output: Path,
    report_path: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    firecs, lazada = build_development_rows(firecs_path, lazada_path)
    texts = np.asarray(
        [normalized_text(record["review_text"]) for record in firecs]
        + [record["text"] for record in lazada],
        dtype=object,
    )
    labels = np.asarray(
        [record["risk_detected"] for record in firecs]
        + [int(record["risk_detected"]) for record in lazada],
        dtype=int,
    )
    groups = np.asarray(
        [f"firecs:{duplicate_group(record['review_text'])}" for record in firecs]
        + [f"lazada:{record['product_id']}" for record in lazada],
        dtype=object,
    )
    domains = np.asarray(
        ["firecs"] * len(firecs) + ["lazada"] * len(lazada), dtype=object
    )

    nested, nested_predictions, _nested_probabilities, nested_folds = evaluate_nested(
        texts, labels, groups, domains, seed=seed
    )
    threshold_probabilities, threshold_folds = grouped_oof_probabilities(
        texts, labels, groups, seed=seed
    )
    threshold, threshold_selection = select_risk_threshold(
        labels, threshold_probabilities
    )
    threshold_predictions = (threshold_probabilities >= threshold).astype(int)

    final_risk_model = classifier_pipeline()
    final_risk_model.fit(texts.tolist(), labels)
    active = joblib.load(active_artifact_path)
    if "sentiment_pipeline" not in active:
        raise ValueError("Active artifact has no sentiment pipeline to preserve.")

    firecs_metrics = nested["by_domain"]["firecs"]
    lazada_metrics = nested["by_domain"]["lazada"]
    gates = {
        "firecs_risk_recall_at_least_0_80": firecs_metrics["risk"]["recall"] >= 0.80,
        "firecs_macro_f1_at_least_0_75": firecs_metrics["macro_f1"] >= 0.75,
        "lazada_risk_recall_at_least_0_80": lazada_metrics["risk"]["recall"] >= 0.80,
        "lazada_macro_f1_at_least_0_75": lazada_metrics["macro_f1"] >= 0.75,
    }
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "deployment_status": "development_candidate",
        "qualified_for_integration": False,
        "qualified_for_new_locked_test": all(gates.values()),
        "qualification_note": (
            "Trained without the 133-row Lazada test split. A newly collected, "
            "product-disjoint locked test is required because the former test labels "
            "have now been inspected."
        ),
        "risk_policy": "binary product-only yes/no; no categories or severity",
        "score_formula": (
            "100 * (0.30 * negative_review_ratio + "
            "0.70 * detected_risk_review_ratio)"
        ),
        "random_state": seed,
        "risk_detection_threshold": float(threshold),
        "threshold_selection": (
            "five-fold stratified grouped out-of-fold predictions on combined development data"
        ),
        "training_rows": len(labels),
        "training_groups": len(set(groups)),
        "domain_rows": {"firecs": len(firecs), "lazada": len(lazada)},
        "lazada_products": len({record["product_id"] for record in lazada}),
        "excluded_lazada_test_rows": 133,
        "excluded_lazada_test_products": 7,
        "risk_distribution": {
            "no_risk": int((labels == 0).sum()),
            "risk": int(labels.sum()),
        },
        "sources": [
            {"path": str(firecs_path.relative_to(ROOT)), "sha256": sha256(firecs_path)},
            {
                "path": str(source_workbook.relative_to(ROOT)),
                "sha256": sha256(source_workbook),
            },
            {"path": str(lazada_path.relative_to(ROOT)), "sha256": sha256(lazada_path)},
            {
                "path": str(active_artifact_path.relative_to(ROOT)),
                "sha256": sha256(active_artifact_path),
                "use": "sentiment pipeline only",
            },
        ],
    }
    artifact = {
        "metadata": metadata,
        "sentiment_pipeline": active["sentiment_pipeline"],
        "risk_detection_pipeline": final_risk_model,
        "risk_detection_threshold": float(threshold),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    lazada_start = len(firecs)
    lazada_oof_predictions = nested_predictions[lazada_start:]
    report = {
        "metadata": metadata,
        "artifact": str(output.relative_to(ROOT)),
        "artifact_sha256": sha256(output),
        "nested_cross_validation": nested,
        "lazada_product_level_oof": product_metrics(
            lazada, lazada_oof_predictions
        ),
        "final_threshold_selection": {
            "threshold": round(float(threshold), 4),
            "selection_macro_f1": round(
                float(threshold_selection["macro_f1"]), 4
            ),
            "selection_risk_recall": round(
                float(threshold_selection["risk_recall"]), 4
            ),
            "oof_metrics": risk_metrics(
                labels, threshold_predictions, threshold_probabilities
            ),
            "fold_counts": dict(sorted(Counter(threshold_folds.tolist()).items())),
        },
        "integrity": {
            "nested_fold_counts": dict(sorted(Counter(nested_folds.tolist()).items())),
            "excluded_test_rows_used_for_training": 0,
            "cross_domain_normalized_text_overlap": 0,
        },
        "development_gates": gates,
        "passes_all_development_gates": all(gates.values()),
        "deployment_changed": False,
        "next_gate": "Collect and adjudicate new product-disjoint Lazada reviews.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firecs", type=Path, default=DEFAULT_FIRECS)
    parser.add_argument("--source-workbook", type=Path, default=DEFAULT_SOURCE_WORKBOOK)
    parser.add_argument("--lazada", type=Path, default=DEFAULT_LAZADA)
    parser.add_argument("--active-artifact", type=Path, default=DEFAULT_HYBRID_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train(
        args.firecs.resolve(),
        args.source_workbook.resolve(),
        args.lazada.resolve(),
        args.active_artifact.resolve(),
        args.output.resolve(),
        args.report.resolve(),
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
