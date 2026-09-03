"""Train and evaluate supervised FiReCS sentiment classifiers reproducibly.

Model selection uses only a stratified validation split from the official
training data. Exact duplicate training reviews and reviews duplicated in the
official test split are excluded before fitting. Held-out test labels never
participate in candidate selection; they are used for final reporting only
after the winner has been selected and retrained on the cleaned training set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from defaketive_sentiment.model import DefaketiveSentimentModel
from evaluate_firecs import load_firecs
from research_utils import evaluate_labels


MODEL_VERSION = "defaketive-firecs-tfidf-0.2.0"
AUXILIARY_MODEL_VERSION = "defaketive-multisource-tfidf-0.3.0-candidate"
LABELS = ("negative", "neutral", "positive")
DEFAULT_TRAIN = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_train_set.csv"
DEFAULT_TEST = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_test_set.csv"
DEFAULT_SENTITAGLISH = (
    SENTIMENT_ROOT
    / "data"
    / "sentitaglish"
    / "SentiTaglish_ProductsAndServices.csv"
)
DEFAULT_TAGASENTI = SENTIMENT_ROOT / "data" / "tagasenti" / "tagasenti_dataset.csv"
DEFAULT_MODEL = SENTIMENT_ROOT / "models" / "firecs_sentiment.joblib"
DEFAULT_RESULT = SENTIMENT_ROOT / "evaluation" / "firecs_trained_model.json"
DEFAULT_ERRORS = SENTIMENT_ROOT / "evaluation" / "firecs_trained_model_errors.csv"
DEFAULT_BASELINE = SENTIMENT_ROOT / "evaluation" / "firecs_test_baseline.json"
DEFAULT_INCUMBENT = SENTIMENT_ROOT / "evaluation" / "firecs_trained_model.json"


def normalized_text(value: Any) -> str:
    """Use the same stable text normalization as the runtime model."""
    return DefaketiveSentimentModel.normalize(value)


def audit_key(value: Any) -> str:
    """Create a punctuation-insensitive key used only for leakage checks."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    characters = (character if character.isalnum() else " " for character in normalized)
    return " ".join("".join(characters).split())


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sentitaglish(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load the human-annotated four-class corpus for three-class training.

    Mixed reviews are not silently forced into positive, neutral, or negative.
    They remain useful for a future four-class model but are excluded here.
    """
    label_map = {"1": "negative", "2": "neutral", "3": "positive"}
    records: list[dict[str, str]] = []
    mixed_rows = 0
    invalid_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"review", "sentiment"}.issubset(reader.fieldnames):
            raise ValueError("SentiTaglish CSV must contain review and sentiment columns")
        for row in reader:
            review = str(row.get("review") or "").strip()
            raw_label = str(row.get("sentiment") or "").strip()
            if raw_label == "4":
                mixed_rows += 1
                continue
            label = label_map.get(raw_label)
            if not review or label is None:
                invalid_rows += 1
                continue
            records.append(
                {"review": review, "label": label, "source": "sentitaglish_human"}
            )
    return records, {
        "file": path.name,
        "sha256": dataset_sha256(path),
        "usable_three_class_rows": len(records),
        "mixed_rows_excluded": mixed_rows,
        "invalid_rows_excluded": invalid_rows,
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/ccosme/SentiTaglishProductsAndServices",
    }


def load_tagasenti(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load TagaSenti while retaining its mixed-provenance warning."""
    label_map = {"0": "negative", "1": "neutral", "2": "positive"}
    records: list[dict[str, str]] = []
    invalid_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"sentence", "label"}.issubset(reader.fieldnames):
            raise ValueError("TagaSenti CSV must contain sentence and label columns")
        for row in reader:
            review = str(row.get("sentence") or "").strip()
            label = label_map.get(str(row.get("label") or "").strip())
            if not review or label is None:
                invalid_rows += 1
                continue
            records.append(
                {
                    "review": review,
                    "label": label,
                    "source": "tagasenti_mixed_provenance",
                }
            )
    return records, {
        "file": path.name,
        "sha256": dataset_sha256(path),
        "usable_rows": len(records),
        "invalid_rows_excluded": invalid_rows,
        "provenance_warning": (
            "The downloadable CSV has no row-level source field and combines human, "
            "synthetic, translated, and LLM-labeled data."
        ),
        "license": "CC BY-SA 4.0",
        "source": "https://huggingface.co/datasets/jjjardev/tagasenti",
    }


def audit_training_records(
    train_records: list[dict[str, str]], test_records: list[dict[str, str]]
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Deduplicate training data and remove exact normalized test overlaps."""
    test_texts = {audit_key(record["review"]) for record in test_records}
    seen: dict[str, str] = {}
    duplicates_removed = 0
    conflicting_duplicates = 0
    overlaps_removed = 0
    texts: list[str] = []
    labels: list[str] = []
    original_source_counts = Counter(
        record.get("source", "firecs_human") for record in train_records
    )
    clean_source_counts: Counter[str] = Counter()

    for record in train_records:
        text = normalized_text(record["review"])
        key = audit_key(record["review"])
        label = record["label"]
        if key in seen:
            duplicates_removed += 1
            conflicting_duplicates += int(seen[key] != label)
            continue
        seen[key] = label
        if key in test_texts:
            overlaps_removed += 1
            continue
        texts.append(text)
        labels.append(label)
        clean_source_counts[record.get("source", "firecs_human")] += 1

    audit = {
        "original_train_rows": len(train_records),
        "clean_train_rows": len(texts),
        "duplicate_train_rows_removed": duplicates_removed,
        "conflicting_duplicate_texts": conflicting_duplicates,
        "cross_split_train_rows_removed": overlaps_removed,
        "test_rows": len(test_records),
        "clean_train_label_counts": dict(sorted(Counter(labels).items())),
        "test_label_counts": dict(
            sorted(Counter(record["label"] for record in test_records).items())
        ),
        "original_source_counts": dict(sorted(original_source_counts.items())),
        "clean_source_counts": dict(sorted(clean_source_counts.items())),
    }
    return texts, labels, audit


def word_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=40_000,
        sublinear_tf=True,
        strip_accents=None,
        dtype=np.float32,
    )


def character_features() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def combined_features() -> FeatureUnion:
    return FeatureUnion(
        [("word", word_features()), ("character", character_features())]
    )


def candidate_factories() -> dict[str, Callable[[], Pipeline]]:
    return {
        "word_linear_svc": lambda: Pipeline(
            [
                ("features", word_features()),
                ("classifier", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "character_linear_svc": lambda: Pipeline(
            [
                ("features", character_features()),
                ("classifier", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "word_character_linear_svc": lambda: Pipeline(
            [
                ("features", combined_features()),
                ("classifier", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        "word_character_logistic_regression": lambda: Pipeline(
            [
                ("features", combined_features()),
                (
                    "classifier",
                    LogisticRegression(
                        C=4.0,
                        class_weight="balanced",
                        max_iter=1_000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
    }


def validation_experiment(
    texts: list[str], labels: list[str], random_state: int
) -> tuple[str, dict[str, Any]]:
    fit_texts, validation_texts, fit_labels, validation_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=random_state,
        stratify=labels,
    )
    candidates: dict[str, Any] = {}
    for name, factory in candidate_factories().items():
        started = time.perf_counter()
        pipeline = factory()
        pipeline.fit(fit_texts, fit_labels)
        predictions = pipeline.predict(validation_texts)
        metrics = evaluate_labels(validation_labels, predictions)
        metrics["fit_seconds"] = round(time.perf_counter() - started, 3)
        candidates[name] = metrics
        print(
            f"{name}: macro F1={metrics['f1']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"fit={metrics['fit_seconds']:.1f}s",
            flush=True,
        )

    selected = max(
        candidates,
        key=lambda name: (candidates[name]["f1"], candidates[name]["accuracy"]),
    )
    return selected, {
        "random_state": random_state,
        "strategy": "stratified 80/20 split of cleaned training data",
        "fit_rows": len(fit_texts),
        "validation_rows": len(validation_texts),
        "selection_metric": "macro_f1",
        "candidates": candidates,
        "selected_candidate": selected,
    }


def read_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_errors(
    path: Path,
    test_records: list[dict[str, str]],
    predictions: list[str],
) -> int:
    errors = [
        {
            "review": record["review"],
            "actual_label": record["label"],
            "predicted_label": prediction,
        }
        for record, prediction in zip(test_records, predictions)
        if record["label"] != prediction
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=["review", "actual_label", "predicted_label"]
        )
        writer.writeheader()
        writer.writerows(errors)
    return len(errors)


def train_and_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    train_records = load_firecs(args.train.resolve())
    test_records = load_firecs(args.test.resolve())
    for record in train_records:
        record["source"] = "firecs_human"

    auxiliary_audits: list[dict[str, Any]] = []
    if args.include_sentitaglish:
        records, source_audit = load_sentitaglish(args.sentitaglish.resolve())
        train_records.extend(records)
        auxiliary_audits.append(source_audit)
    if args.include_tagasenti:
        records, source_audit = load_tagasenti(args.tagasenti.resolve())
        train_records.extend(records)
        auxiliary_audits.append(source_audit)

    texts, labels, audit = audit_training_records(train_records, test_records)

    selected, validation = validation_experiment(texts, labels, args.random_state)
    print(f"Selected {selected}; retraining on {len(texts)} cleaned rows.", flush=True)
    final_pipeline = candidate_factories()[selected]()
    final_started = time.perf_counter()
    final_pipeline.fit(texts, labels)
    final_fit_seconds = round(time.perf_counter() - final_started, 3)

    # Predictions and metrics against the official test labels are computed
    # only after model selection and the final refit have completed.
    test_texts = [normalized_text(record["review"]) for record in test_records]
    test_predictions = final_pipeline.predict(test_texts).tolist()
    test_metrics = evaluate_labels(
        (record["label"] for record in test_records), test_predictions
    )
    error_count = write_errors(args.errors_output.resolve(), test_records, test_predictions)
    test_metrics["error_count"] = error_count

    baseline = read_baseline(args.baseline.resolve())
    baseline_f1 = float(baseline.get("f1") or 0.0)
    f1_delta = round(float(test_metrics["f1"]) - baseline_f1, 4)
    incumbent = read_baseline(args.incumbent.resolve())
    incumbent_metrics = incumbent.get("test_metrics") or {}
    incumbent_f1 = float(incumbent_metrics.get("f1") or incumbent.get("f1") or 0.0)
    incumbent_delta = round(float(test_metrics["f1"]) - incumbent_f1, 4)
    qualified = (
        f1_delta >= args.minimum_improvement
        and incumbent_delta >= args.minimum_incumbent_improvement
    )

    metadata = {
        "model_version": (
            AUXILIARY_MODEL_VERSION if auxiliary_audits else MODEL_VERSION
        ),
        "selected_candidate": selected,
        "labels": list(LABELS),
        "expects_normalized_text": True,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": args.random_state,
        "sklearn_version": sklearn.__version__,
        "training_audit": audit,
        "auxiliary_datasets": auxiliary_audits,
        "validation": validation,
        "final_fit_seconds": final_fit_seconds,
        "test_metrics": test_metrics,
        "baseline_macro_f1": baseline_f1,
        "macro_f1_delta": f1_delta,
        "minimum_required_improvement": args.minimum_improvement,
        "incumbent_macro_f1": incumbent_f1,
        "macro_f1_delta_vs_incumbent": incumbent_delta,
        "minimum_incumbent_improvement": args.minimum_incumbent_improvement,
        "qualified_for_integration": qualified,
        "train_file": args.train.name,
        "train_sha256": dataset_sha256(args.train.resolve()),
        "test_file": args.test.name,
        "test_sha256": dataset_sha256(args.test.resolve()),
        "license": (
            "CC BY-SA 4.0 combined auxiliary candidate"
            if args.include_tagasenti
            else "CC BY 4.0"
        ),
        "source": "https://huggingface.co/datasets/ccosme/FiReCS",
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_pipeline, "metadata": metadata}, args.model_output)
    artifact_hash = dataset_sha256(args.model_output.resolve())

    result = dict(metadata)
    result.update(
        {
            "artifact_file": args.model_output.name,
            "artifact_sha256": artifact_hash,
            "result_file": args.result_output.name,
            "errors_file": args.errors_output.name,
        }
    )
    args.result_output.parent.mkdir(parents=True, exist_ok=True)
    args.result_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--include-sentitaglish",
        action="store_true",
        help="Add non-mixed SentiTaglish rows after duplicate/leakage filtering",
    )
    parser.add_argument("--sentitaglish", type=Path, default=DEFAULT_SENTITAGLISH)
    parser.add_argument(
        "--include-tagasenti",
        action="store_true",
        help="Add TagaSenti rows after duplicate/leakage filtering",
    )
    parser.add_argument("--tagasenti", type=Path, default=DEFAULT_TAGASENTI)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--errors-output", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.02,
        help="Required test macro-F1 gain over the VADER baseline",
    )
    parser.add_argument(
        "--minimum-incumbent-improvement",
        type=float,
        default=0.0,
        help="Required macro-F1 gain over the currently trained classifier",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = train_and_evaluate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["qualified_for_integration"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
