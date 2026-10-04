"""Does the sentiment model hold up on real marketplace reviews?

Every accuracy figure so far was measured on FiReCS: short Taglish reviews like
the ones the models were trained on. The application reads Lazada and Temu
reviews, which include formal English, template text ("Compatibility: ...")
and very short ratings ("10 out of 10"). This script scores the same models on
the project's human-labelled marketplace set to measure that domain shift.

Test set: complete_manual_marketplace_reviews (1).csv -- 498 reviews labelled
by people (0% machine-labelled per outputs/legacy_label_provenance_audit).
"mixed" reviews are excluded because no three-class model can express them.
None of these reviews were used to train any model scored here.

Also measured: a proposed display rule that shows a review as "uncertain" when
its text reading contradicts its star rating, instead of a confident label.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from research_utils import (  # noqa: E402
    bootstrap_macro_f1_ci,
    evaluate_labels,
    holm_bonferroni,
    mcnemar_test,
    normalize_sentiment_label,
)

DEFAULT_DATASET = ROOT / "complete_manual_marketplace_reviews (1).csv"
DEFAULT_OUTPUT = ROOT / "research" / "marketplace_shift.json"
CHECKPOINTS = SENTIMENT_ROOT / "models" / "transformers"
TRANSFORMERS = {
    # Trained on normalized text, scored here on raw text (the mismatch the
    # raw-text retraining below corrects).
    "finetuned_xlm_roberta": CHECKPOINTS / "cardiffnlp__twitter-xlm-roberta-base-sentiment",
    "finetuned_dost_asti": CHECKPOINTS / "dost-asti__RoBERTa-tl-sentiment-analysis",
    # Retrained on raw text. FiReCS only: the comparison.
    "xlmr_firecs_raw": CHECKPOINTS / "xlmr_firecs_raw",
    # FiReCS + Lazada star labels: the model chosen in advance to ship. No
    # review of any product in this test set was in its training data.
    "xlmr_firecs_lazada_raw": CHECKPOINTS / "xlmr_firecs_lazada_raw",
    # Normalized text + star labels at half loss weight (rule fixed in advance
    # in research/PREREGISTRATION_xlmr_firecs_lazada_norm.md).
    "xlmr_firecs_lazada_norm": CHECKPOINTS / "xlmr_firecs_lazada_norm",
}
# Scored on the text form each was trained on, as the app serves it. The first
# XLM-R was trained normalized but has always been reported on raw text; that
# figure is kept as recorded.
NORMALIZED_INPUT = {"xlmr_firecs_lazada_norm"}
FIRECS_REPORTS = {
    "finetuned_xlm_roberta": "finetuned_cardiffnlp__twitter-xlm-roberta-base-sentiment.json",
    "finetuned_dost_asti": "finetuned_dost-asti__RoBERTa-tl-sentiment-analysis.json",
    "xlmr_firecs_raw": "finetuned_xlmr_firecs_raw.json",
    "xlmr_firecs_lazada_raw": "finetuned_xlmr_firecs_lazada_raw.json",
    "xlmr_firecs_lazada_norm": "finetuned_xlmr_firecs_lazada_norm.json",
}
REFERENCE = "deployed_tfidf"
LABELS = ("negative", "neutral", "positive")


def audit_key(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in folded).split())


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    records = []
    for row in rows:
        label = normalize_sentiment_label(row.get("adjudicated_label") or "")
        text = str(row.get("review_text") or "").strip()
        if label not in LABELS or not text:
            continue
        try:
            rating = int(float(row.get("rating") or 0)) or None
        except ValueError:
            rating = None
        records.append(
            {"text": text, "label": label, "rating": rating, "platform": str(row.get("platform") or "").strip()}
        )
    return records


def training_overlap(records: list[dict[str, Any]]) -> int:
    """Count test reviews whose text also appears in the FiReCS training split."""
    from evaluate_firecs import load_firecs

    train = {audit_key(item["review"]) for item in load_firecs(SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_train_set.csv")}
    return sum(1 for record in records if audit_key(record["text"]) in train)


def predict_tfidf(texts: list[str]) -> list[str]:
    import joblib
    from train_firecs import normalized_text

    pipeline = joblib.load(SENTIMENT_ROOT / "models" / "firecs_sentiment.joblib")["pipeline"]
    return [str(label) for label in pipeline.predict([normalized_text(text) for text in texts])]


def predict_transformer(checkpoint: Path, texts: list[str]) -> list[str]:
    import torch
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        device=0 if torch.cuda.is_available() else -1,
    )
    outputs = classifier(texts, batch_size=32, truncation=True, max_length=128)
    return [normalize_sentiment_label(output["label"]) for output in outputs]


def firecs_f1(name: str) -> float | None:
    """The model's FiReCS test macro F1, from its fine-tuning report."""
    path = ROOT / "research" / FIRECS_REPORTS[name]
    if not path.is_file():
        return None
    return json.loads(path.read_bytes().decode("utf-8"))["test"]["f1"]


def contradicts_rating(prediction: str, rating: int | None) -> bool:
    """Text reads negative on a 4-5 star review, or positive on a 1-2 star one."""
    if rating is None:
        return False
    return (prediction == "negative" and rating >= 4) or (prediction == "positive" and rating <= 2)


def rating_rule(records: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """What happens if contradicted labels are shown as uncertain instead."""
    flagged = [i for i, (r, p) in enumerate(zip(records, predictions)) if contradicts_rating(p, r["rating"])]
    wrong_flagged = sum(1 for i in flagged if predictions[i] != records[i]["label"])
    kept = [i for i in range(len(records)) if i not in set(flagged)]
    total_errors = sum(1 for r, p in zip(records, predictions) if r["label"] != p)
    kept_metrics = evaluate_labels([records[i]["label"] for i in kept], [predictions[i] for i in kept]) if kept else None
    return {
        "reviews_flagged_uncertain": len(flagged),
        "flagged_that_were_wrong": wrong_flagged,
        "flagged_that_were_right": len(flagged) - wrong_flagged,
        "total_errors_before": total_errors,
        "errors_caught_share": round(wrong_flagged / total_errors, 4) if total_errors else None,
        "coverage_after": round(len(kept) / len(records), 4),
        "accuracy_on_kept": kept_metrics["accuracy"] if kept_metrics else None,
        "macro_f1_on_kept": kept_metrics["f1"] if kept_metrics else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = load(args.dataset)
    texts = [record["text"] for record in records]
    truth = [record["label"] for record in records]

    predictions: dict[str, list[str]] = {}
    timings: dict[str, float] = {}
    started = time.perf_counter()
    predictions[REFERENCE] = predict_tfidf(texts)
    timings[REFERENCE] = time.perf_counter() - started
    for name, checkpoint in TRANSFORMERS.items():
        if not checkpoint.exists():
            print(f"skipping {name}: no checkpoint at {checkpoint}")
            continue
        started = time.perf_counter()
        if name in NORMALIZED_INPUT:
            from train_firecs import normalized_text

            predictions[name] = predict_transformer(checkpoint, [normalized_text(text) for text in texts])
        else:
            predictions[name] = predict_transformer(checkpoint, texts)
        timings[name] = time.perf_counter() - started

    results: dict[str, Any] = {}
    for name, predicted in predictions.items():
        metrics = evaluate_labels(truth, predicted)
        metrics["macro_f1_ci"] = bootstrap_macro_f1_ci(truth, predicted, iterations=args.bootstrap, seed=42)
        metrics["by_platform"] = {}
        for platform in sorted({r["platform"] for r in records}):
            platform_truth = [r["label"] for r in records if r["platform"] == platform]
            platform_predicted = [p for r, p in zip(records, predicted) if r["platform"] == platform]
            metrics["by_platform"][platform] = evaluate_labels(platform_truth, platform_predicted)
            metrics["by_platform"][platform]["macro_f1_ci"] = bootstrap_macro_f1_ci(
                platform_truth, platform_predicted, iterations=args.bootstrap, seed=42
            )
        # The failure the review screen showed: praise read as a complaint.
        metrics["positive_read_as_negative"] = sum(
            1 for t, p in zip(truth, predicted) if t == "positive" and p == "negative"
        )
        metrics["rating_rule"] = rating_rule(records, predicted)
        metrics["seconds"] = round(timings[name], 2)
        results[name] = metrics

    p_values = {}
    for name in predictions:
        if name == REFERENCE:
            continue
        test = mcnemar_test(truth, predictions[REFERENCE], predictions[name])
        results[name]["mcnemar_vs_deployed"] = test
        p_values[name] = test["p_value"]
    if p_values:
        for name, corrected in holm_bonferroni(p_values)["results"].items():
            results[name]["mcnemar_vs_deployed"].update(
                {"p_adjusted": corrected["p_adjusted"], "significant": corrected["significant"]}
            )

    firecs_reference = json.loads((ROOT / "research" / "sentiment_model_comparison.json").read_text(encoding="utf-8"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "file": args.dataset.name,
            "reviews_scored": len(records),
            "label_counts": dict(Counter(truth)),
            "platform_counts": dict(Counter(r["platform"] for r in records)),
            "excluded": "mixed or unlabelled reviews",
            "overlap_with_firecs_training": training_overlap(records),
            "label_provenance": "human (outputs/legacy_label_provenance_audit/summary.md: 0% machine-labelled)",
        },
        "firecs_macro_f1_for_comparison": {
            REFERENCE: firecs_reference["results"]["tfidf_logistic_regression_incumbent"]["f1"],
            **{name: firecs_f1(name) for name in predictions if name != REFERENCE},
        },
        "results": results,
        "qualified_for_integration": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report = run(args)
    args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
