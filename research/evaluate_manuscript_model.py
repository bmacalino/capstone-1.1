"""Diagnostically evaluate the hybrid-manuscript model on adjudicated reviews.

The current Lazada test labels have been inspected during development, so this
report is retrospective and must not be described as a fresh locked test.
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

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.manuscript import (
    HybridManuscriptRiskModel,
    ManuscriptRiskModel,
)
from research_utils import evaluate_labels


DEFAULT_DATASET = (
    ROOT
    / "outputs"
    / "lazada_gold_test_v1"
    / "lazada_round1_gold_test_v1.csv"
)
DEFAULT_REPORT = (
    SENTIMENT_ROOT / "evaluation" / "hybrid_manuscript_model_diagnostic.json"
)
DEFAULT_ERRORS = (
    SENTIMENT_ROOT / "evaluation" / "hybrid_manuscript_model_diagnostic_errors.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_sentiment(value: object) -> str:
    label = str(value or "").strip().casefold()
    return label if label in {"positive", "negative"} else "uncertain"


def expected_risk(value: object) -> str:
    return "risk" if str(value or "").strip().casefold() == "yes" else "no_risk"


def risk_metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    labels = ["no_risk", "risk"]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    return {
        "sample_count": len(actual),
        "labels": labels,
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(
            float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
            4,
        ),
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def evaluate(dataset: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Evaluation dataset contains no rows.")
    required = {
        "review_id",
        "product_id",
        "review_text",
        "adjudicated_label",
        "risk_detected",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Evaluation dataset is missing columns: {sorted(missing)}")

    model = HybridManuscriptRiskModel(allow_unqualified=True)
    analyses = [
        model.analyze(row["review_text"], rating=row.get("rating")) for row in rows
    ]
    sentiment_actual = [expected_sentiment(row["adjudicated_label"]) for row in rows]
    sentiment_predicted = [str(item["label"]) for item in analyses]
    risk_actual = [expected_risk(row["risk_detected"]) for row in rows]
    risk_predicted = [
        "risk" if (item.get("risk") or {}).get("detected") else "no_risk"
        for item in analyses
    ]

    sentiment = evaluate_labels(sentiment_actual, sentiment_predicted)
    risk = risk_metrics(risk_actual, risk_predicted)

    # Compare the deployed detector with the strict manuscript lexicon baseline
    # on exactly the same retrospective data.
    baseline = ManuscriptRiskModel()
    baseline_analyses = [
        baseline.analyze(row["review_text"], rating=row.get("rating")) for row in rows
    ]
    baseline_sentiment = evaluate_labels(
        sentiment_actual, [str(item["label"]) for item in baseline_analyses]
    )
    baseline_risk = risk_metrics(
        risk_actual,
        [
            "risk" if (item.get("risk") or {}).get("detected") else "no_risk"
            for item in baseline_analyses
        ],
    )
    errors: list[dict[str, Any]] = []
    for row, analysis, expected_s, predicted_s, expected_r, predicted_r in zip(
        rows,
        analyses,
        sentiment_actual,
        sentiment_predicted,
        risk_actual,
        risk_predicted,
    ):
        if expected_s == predicted_s and expected_r == predicted_r:
            continue
        evidence = [
            str(item.get("term") or "")
            for item in (analysis.get("risk") or {}).get("evidence") or []
            if item.get("term") and not item.get("negated")
        ]
        errors.append(
            {
                "review_id": row["review_id"],
                "product_id": row["product_id"],
                "expected_sentiment": expected_s,
                "predicted_sentiment": predicted_s,
                "expected_risk": expected_r,
                "predicted_risk": predicted_r,
                "active_risk_terms": "|".join(evidence),
                "review_text": row["review_text"],
            }
        )

    error_types = Counter()
    for error in errors:
        if error["expected_sentiment"] != error["predicted_sentiment"]:
            error_types[
                f"sentiment:{error['expected_sentiment']}->{error['predicted_sentiment']}"
            ] += 1
        if error["expected_risk"] != error["predicted_risk"]:
            error_types[f"risk:{error['expected_risk']}->{error['predicted_risk']}"] += 1

    gates = {
        "sentiment_macro_f1_at_least_0_75": sentiment["f1"] >= 0.75,
        "uncertain_f1_at_least_0_60": sentiment["per_class"]["uncertain"]["f1"] >= 0.60,
        "risk_recall_at_least_0_80": risk["per_class"]["risk"]["recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": risk["macro_f1"] >= 0.75,
    }
    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_status": "retrospective_diagnostic_not_fresh_locked_test",
            "limitation": (
                "These adjudicated labels were available during prior development and "
                "discussion. Use a new untouched product-disjoint set for final claims."
            ),
            "dataset": dataset.name,
            "dataset_sha256": sha256(dataset),
            "rows": len(rows),
            "products": len({row["product_id"] for row in rows}),
            "model_version": model.model_version,
            "sentiment_support": dict(Counter(sentiment_actual)),
            "risk_support": dict(Counter(risk_actual)),
        },
        "sentiment": sentiment,
        "risk_detection": risk,
        "lexicon_baseline": {
            "model_version": baseline.model_version,
            "sentiment": baseline_sentiment,
            "risk_detection": baseline_risk,
        },
        "comparison_to_lexicon_baseline": {
            "sentiment_macro_f1_delta": round(sentiment["f1"] - baseline_sentiment["f1"], 4),
            "risk_macro_f1_delta": round(risk["macro_f1"] - baseline_risk["macro_f1"], 4),
            "risk_recall_delta": round(
                risk["per_class"]["risk"]["recall"]
                - baseline_risk["per_class"]["risk"]["recall"],
                4,
            ),
        },
        "qualification_gates": gates,
        "passes_all_diagnostic_gates": all(gates.values()),
        "error_rows": len(errors),
        "error_type_counts": dict(error_types.most_common()),
    }
    return report, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    args = parser.parse_args()

    report, errors = evaluate(args.dataset.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.errors.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "review_id",
            "product_id",
            "expected_sentiment",
            "predicted_sentiment",
            "expected_risk",
            "predicted_risk",
            "active_risk_terms",
            "review_text",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
