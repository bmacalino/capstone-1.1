"""Evaluate local and Hugging Face sentiment models on marketplace annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for import_root in (ROOT, SENTIMENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from defaketive_sentiment.model import DefaketiveSentimentModel
from research_utils import evaluate_labels, normalize_sentiment_label


DEFAULT_DATASET = ROOT / "complete_manual_marketplace_reviews (1).csv"
SUPPORTED_LABELS = {"negative", "neutral", "positive"}
HF_MODELS = {
    "dost-asti/RoBERTa-tl-sentiment-analysis",
    "jjjardev/tagasenti_model",
}
HF_LABEL_ALIASES = {
    "dost-asti/RoBERTa-tl-sentiment-analysis": {
        # The model config exposes only LABEL_N. This mapping is inferred from
        # clear positive/negative/factual sanity checks and is recorded as a
        # limitation in the benchmark report.
        "label_0": "negative",
        "label_1": "positive",
        "label_2": "neutral",
    },
    "jjjardev/tagasenti_model": {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    },
}


def load_records(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"review_id", "platform", "product_id", "review_text", "adjudicated_label"}
    missing = required.difference(rows[0] if rows else {})
    if not rows:
        raise ValueError("Manual marketplace dataset is empty.")
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    clean: list[dict[str, str]] = []
    ids: set[str] = set()
    texts: set[str] = set()
    duplicate_ids = 0
    duplicate_texts = 0
    for row in rows:
        review_id = row["review_id"].strip()
        text = row["review_text"].strip()
        label = normalize_sentiment_label(row["adjudicated_label"])
        if not review_id or not text:
            continue
        duplicate_ids += review_id in ids
        duplicate_texts += text.casefold() in texts
        ids.add(review_id)
        texts.add(text.casefold())
        clean.append({**row, "review_text": text, "adjudicated_label": label})

    audit = {
        "source_rows": len(rows),
        "nonempty_rows": len(clean),
        "unique_review_ids": len(ids),
        "unique_casefolded_texts": len(texts),
        "duplicate_review_ids": duplicate_ids,
        "duplicate_casefolded_texts": duplicate_texts,
        "product_count": len({row["product_id"].strip() for row in clean}),
        "platform_counts": dict(Counter(row["platform"].strip().casefold() for row in clean)),
        "adjudicated_label_counts": dict(Counter(row["adjudicated_label"] for row in clean)),
        "blank_annotator_ids": sum(not row.get("annotator_id", "").strip() for row in clean),
    }
    return clean, audit


def local_predictions(
    records: list[dict[str, str]],
) -> tuple[list[str], list[float | None], list[dict[str, Any]]]:
    model = DefaketiveSentimentModel()
    labels: list[str] = []
    confidence: list[float | None] = []
    analyses: list[dict[str, Any]] = []
    for row in records:
        result = model.analyze(row["review_text"], row.get("rating"))
        analyses.append(result)
        labels.append(normalize_sentiment_label(result["label"]))
        classifier = result.get("classifier") or {}
        value = classifier.get("confidence")
        confidence.append(round(float(value), 6) if value is not None else None)
    return labels, confidence, analyses


def score_local_risk(
    records: list[dict[str, str]], analyses: list[dict[str, Any]]
) -> dict[str, Any]:
    valid_indices = [
        index
        for index, row in enumerate(records)
        if row.get("risk_detected", "").strip().casefold() in {"yes", "no"}
    ]
    actual_flags = [records[index]["risk_detected"].strip().casefold() for index in valid_indices]
    predicted_flags = [
        "yes" if analyses[index]["risk"]["detected"] else "no" for index in valid_indices
    ]

    category_rows = [
        index
        for index in valid_indices
        if records[index]["risk_detected"].strip().casefold() == "yes"
        and records[index].get("risk_category", "").strip()
    ]
    category_correct = sum(
        records[index]["risk_category"].strip().casefold()
        in {str(value).strip().casefold() for value in analyses[index]["risk"].get("categories", [])}
        for index in category_rows
    )

    severity_pairs: list[tuple[int, int]] = []
    for index in category_rows:
        raw_severity = records[index].get("risk_severity", "").strip()
        if not raw_severity:
            continue
        expected = int(float(raw_severity))
        evidence = analyses[index]["risk"].get("evidence", [])
        predicted = max((int(item.get("severity", 0)) for item in evidence), default=0)
        severity_pairs.append((expected, predicted))

    return {
        "detection": evaluate_labels(actual_flags, predicted_flags),
        "category": {
            "sample_count": len(category_rows),
            "exact_match_count": category_correct,
            "exact_match_accuracy": round(category_correct / len(category_rows), 4)
            if category_rows
            else None,
            "policy": "gold category matches any category predicted for the review",
        },
        "severity": {
            "sample_count": len(severity_pairs),
            "exact_match_accuracy": round(
                sum(expected == predicted for expected, predicted in severity_pairs)
                / len(severity_pairs),
                4,
            )
            if severity_pairs
            else None,
            "mean_absolute_error": round(
                sum(abs(expected - predicted) for expected, predicted in severity_pairs)
                / len(severity_pairs),
                4,
            )
            if severity_pairs
            else None,
            "policy": "prediction is the maximum matched-evidence severity; zero means no evidence",
        },
    }


def hf_predictions(
    model_id: str, records: list[dict[str, str]], batch_size: int
) -> tuple[list[str], list[float | None]]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Install transformers and torch to evaluate Hugging Face models.") from exc

    classifier = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=-1,
    )
    outputs = classifier(
        [row["review_text"] for row in records],
        batch_size=batch_size,
        truncation=True,
        max_length=128 if model_id == "jjjardev/tagasenti_model" else 256,
    )
    aliases = HF_LABEL_ALIASES[model_id]
    labels: list[str] = []
    confidence: list[float | None] = []
    for output in outputs:
        raw = str(output["label"]).strip().casefold()
        if raw not in aliases:
            raise ValueError(f"Unsupported label {output['label']!r} from {model_id}")
        labels.append(aliases[raw])
        score = output.get("score")
        confidence.append(round(float(score), 6) if score is not None else None)
    return labels, confidence


def score_model(
    name: str,
    records: list[dict[str, str]],
    predictions: list[str],
    confidence: list[float | None],
) -> dict[str, Any]:
    actual = [row["adjudicated_label"] for row in records]
    result: dict[str, Any] = {
        "model": name,
        "overall": evaluate_labels(actual, predictions),
        "by_platform": {},
    }
    available_confidence = [value for value in confidence if value is not None]
    result["mean_confidence"] = (
        round(sum(available_confidence) / len(available_confidence), 6)
        if available_confidence
        else None
    )
    for platform in sorted({row["platform"].strip().casefold() for row in records}):
        indices = [
            index
            for index, row in enumerate(records)
            if row["platform"].strip().casefold() == platform
        ]
        result["by_platform"][platform] = evaluate_labels(
            [actual[index] for index in indices],
            [predictions[index] for index in indices],
        )
    result["errors"] = [
        {
            "review_id": row["review_id"],
            "platform": row["platform"],
            "product_id": row["product_id"],
            "expected": row["adjudicated_label"],
            "predicted": prediction,
            "confidence": model_confidence,
            "review_text": row["review_text"],
        }
        for row, prediction, model_confidence in zip(records, predictions, confidence)
        if row["adjudicated_label"] != prediction
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["current", *sorted(HF_MODELS)],
        default=["current"],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    all_records, audit = load_records(args.dataset.resolve())
    compatible_indices = [
        index
        for index, row in enumerate(all_records)
        if row["adjudicated_label"] in SUPPORTED_LABELS
    ]
    records = [all_records[index] for index in compatible_indices]
    result: dict[str, Any] = {
        "dataset": args.dataset.name,
        "audit": audit,
        "evaluation_policy": {
            "included_labels": sorted(SUPPORTED_LABELS),
            "excluded_rows": len(all_records) - len(records),
            "excluded_label_counts": dict(
                Counter(
                    row["adjudicated_label"]
                    for row in all_records
                    if row["adjudicated_label"] not in SUPPORTED_LABELS
                )
            ),
            "primary_metric": "macro_f1",
            "dost_label_mapping": {
                "LABEL_0": "negative",
                "LABEL_1": "positive",
                "LABEL_2": "neutral",
                "status": "inferred; upstream config and model card do not document label names",
            },
        },
        "results": {},
    }
    for name in args.models:
        if name == "current":
            all_predictions, all_confidence, all_analyses = local_predictions(all_records)
            predictions = [all_predictions[index] for index in compatible_indices]
            confidence = [all_confidence[index] for index in compatible_indices]
        else:
            predictions, confidence = hf_predictions(name, records, args.batch_size)
        scored = score_model(name, records, predictions, confidence)
        if name == "current":
            scored["risk"] = score_local_risk(all_records, all_analyses)
        result["results"][name] = scored
        metrics = scored["overall"]
        print(
            f"{name}: n={metrics['sample_count']} accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['f1']:.4f}",
            flush=True,
        )

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
