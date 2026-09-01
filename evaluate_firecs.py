"""Evaluate the local DeFaketive sentiment model on the held-out FiReCS test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from defaketive_sentiment.model import MODEL_VERSION, DefaketiveSentimentModel
from research_utils import evaluate_labels, normalize_sentiment_label

DEFAULT_DATASET = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_test_set.csv"
DEFAULT_RESULT = SENTIMENT_ROOT / "evaluation" / "firecs_test_baseline.json"
DEFAULT_ERRORS = SENTIMENT_ROOT / "evaluation" / "firecs_test_errors.csv"


def load_firecs(path: Path) -> list[dict[str, str]]:
    """Load and validate a FiReCS CSV without changing its official split."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or [])
        if not {"review", "label"}.issubset(fields):
            raise ValueError("FiReCS CSV must contain review and label columns.")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            review = str(row.get("review") or "").strip()
            label = normalize_sentiment_label(row.get("label"))
            if not review:
                raise ValueError(f"FiReCS row {line_number} has an empty review.")
            if label not in {"negative", "neutral", "positive"}:
                raise ValueError(f"FiReCS row {line_number} has unsupported label {row.get('label')!r}.")
            rows.append({"review": review, "label": label})
    if not rows:
        raise ValueError("FiReCS CSV is empty.")
    return rows


def evaluate_firecs(
    path: Path, model: DefaketiveSentimentModel | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return benchmark metrics and misclassified records for one FiReCS split."""
    records = load_firecs(path)
    model = model or DefaketiveSentimentModel()
    predictions = []
    errors = []
    for record in records:
        analysis = model.analyze(record["review"])
        prediction = analysis["label"]
        predictions.append(prediction)
        if prediction != record["label"]:
            errors.append(
                {
                    "review": record["review"],
                    "actual_label": record["label"],
                    "predicted_label": prediction,
                    "compound": analysis["scores"]["compound"],
                    "sentiment_evidence": json.dumps(
                        analysis["sentiment_evidence"], ensure_ascii=False
                    ),
                    "risk_evidence": json.dumps(
                        analysis["risk"]["evidence"], ensure_ascii=False
                    ),
                }
            )
    metrics = evaluate_labels((record["label"] for record in records), predictions)
    metrics.update(
        {
            "dataset": "FiReCS",
            "split_file": path.name,
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_version": MODEL_VERSION,
            "license": "CC BY 4.0",
            "source": "https://huggingface.co/datasets/ccosme/FiReCS",
            "error_count": len(errors),
        }
    )
    return metrics, errors


def write_results(
    metrics: dict[str, Any],
    errors: list[dict[str, Any]],
    result_path: Path,
    errors_path: Path,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with errors_path.open("w", encoding="utf-8-sig", newline="") as target:
        fieldnames = [
            "review",
            "actual_label",
            "predicted_label",
            "compound",
            "sentiment_evidence",
            "risk_evidence",
        ]
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="FiReCS CSV split to evaluate"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RESULT, help="JSON metrics output"
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        default=DEFAULT_ERRORS,
        help="Misclassified-review CSV output",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics, errors = evaluate_firecs(args.dataset.resolve())
    write_results(metrics, errors, args.output.resolve(), args.errors_output.resolve())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
