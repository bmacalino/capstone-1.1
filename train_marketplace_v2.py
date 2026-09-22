"""Train the product-grouped DeFaketive marketplace sentiment and risk candidate."""

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

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parent
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from defaketive_sentiment.model import DefaketiveSentimentModel
from defaketive_sentiment.trained import feature_text
from evaluate_firecs import load_firecs
from research_utils import evaluate_labels, normalize_sentiment_label


MODEL_VERSION = "defaketive-marketplace-v2-0.1.0-candidate"
SENTIMENT_LABELS = {"negative", "neutral", "positive"}
BINARY_SENTIMENT_LABELS = {"negative", "positive"}
DEFAULT_MARKETPLACE = ROOT / "complete_manual_marketplace_reviews (1).csv"
DEFAULT_FIRECS_TRAIN = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_train_set.csv"
DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models" / "marketplace_v2_candidate.joblib"
DEFAULT_REPORT = SENTIMENT_ROOT / "evaluation" / "marketplace_v2_candidate.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(value: object) -> str:
    return feature_text(DefaketiveSentimentModel.normalize(value))


def load_marketplace(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "review_id",
        "platform",
        "product_id",
        "review_text",
        "adjudicated_label",
        "risk_detected",
        "risk_category",
        "risk_severity",
    }
    if not rows:
        raise ValueError("Marketplace annotation file is empty.")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing marketplace columns: {', '.join(sorted(missing))}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        review_id = row["review_id"].strip()
        text = row["review_text"].strip()
        product_id = row["product_id"].strip()
        if not review_id or not text or not product_id:
            raise ValueError(f"Row {line_number} is missing an ID, review, or product ID.")
        text_key = normalized_text(text)
        if review_id in seen_ids:
            raise ValueError(f"Duplicate review ID at row {line_number}: {review_id}")
        if text_key in seen_texts:
            raise ValueError(f"Duplicate normalized review text at row {line_number}.")
        seen_ids.add(review_id)
        seen_texts.add(text_key)

        risk_raw = row["risk_detected"].strip().casefold()
        risk_detected = {"yes": 1, "no": 0}.get(risk_raw)
        categories = tuple(
            sorted(
                {
                    value.strip().casefold()
                    for value in row["risk_category"].replace(",", "|").split("|")
                    if value.strip()
                }
            )
        )
        severity_raw = row["risk_severity"].strip()
        severity = int(float(severity_raw)) if severity_raw else None
        if severity is not None and severity not in {1, 2, 3}:
            raise ValueError(f"Unsupported severity at row {line_number}: {severity}")
        rating_raw = row.get("rating", "").strip()
        rating = int(float(rating_raw)) if rating_raw else None
        if rating is not None and rating not in {1, 2, 3, 4, 5}:
            raise ValueError(f"Unsupported rating at row {line_number}: {rating}")
        records.append(
            {
                "review_id": review_id,
                "platform": row["platform"].strip().casefold(),
                "product_id": product_id,
                "review": text,
                "text": text_key,
                "sentiment": normalize_sentiment_label(row["adjudicated_label"]),
                "risk_detected": risk_detected,
                "risk_categories": categories,
                "risk_severity": severity,
                "rating": rating,
                "annotator_id": row.get("annotator_id", "").strip(),
                "split": row.get("split", "").strip().casefold(),
            }
        )
    return records


def _sentiment_stratum(record: dict[str, Any]) -> str:
    label = record["sentiment"]
    return label if label in SENTIMENT_LABELS else "mixed"


def product_group_split(
    records: list[dict[str, Any]], random_state: int
) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic train/validation/test indices with disjoint products."""
    indices = np.arange(len(records))
    strata = np.asarray([_sentiment_stratum(record) for record in records])
    groups = np.asarray([record["product_id"] for record in records])
    overall = Counter(strata)

    def score(parts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
        desired = (0.70, 0.15, 0.15)
        value = 0.0
        for part, target in zip(parts, desired):
            counts = Counter(strata[part])
            value += 4 * abs(len(part) / len(records) - target)
            value += sum(
                abs(counts[label] / max(len(part), 1) - overall[label] / len(records))
                for label in overall
            )
            # Every partition should exercise the three production sentiment labels.
            value += 100 * sum(counts[label] == 0 for label in SENTIMENT_LABELS)
            risk_values = {records[index]["risk_detected"] for index in part}
            value += 100 * sum(label not in risk_values for label in (0, 1))
        return value

    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    # Select a split using labels/group sizes only, never model outcomes. Repeated
    # folds avoid letting one unusually large product consume the whole test set.
    for seed in range(random_state, random_state + 20):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for remaining, test in outer.split(indices, strata, groups):
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 101)
            for inner_train, inner_validation in inner.split(
                remaining, strata[remaining], groups[remaining]
            ):
                train = remaining[inner_train]
                validation = remaining[inner_validation]
                candidates.append((score((train, validation, test)), train, validation, test))
    _, train, validation, test = min(candidates, key=lambda candidate: candidate[0])

    product_sets = [set(groups[part]) for part in (train, validation, test)]
    if product_sets[0] & product_sets[1] or product_sets[0] & product_sets[2] or product_sets[1] & product_sets[2]:
        raise RuntimeError("Product leakage detected in marketplace split.")
    return train.tolist(), validation.tolist(), test.tolist()


def locked_product_split(
    records: list[dict[str, Any]],
) -> tuple[list[int], list[int], list[int]] | None:
    """Use a complete, product-disjoint split column when the dataset provides one."""
    values = [str(record.get("split", "")).strip().casefold() for record in records]
    if not any(values):
        return None
    if not all(values):
        raise ValueError("The marketplace split column must be populated for every row or left blank.")
    allowed = {"train", "validation", "test"}
    invalid = sorted(set(values).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported marketplace split values: {', '.join(invalid)}")

    parts = tuple(
        [index for index, value in enumerate(values) if value == name]
        for name in ("train", "validation", "test")
    )
    if any(not part for part in parts):
        raise ValueError("The marketplace split must contain train, validation, and test rows.")

    products: dict[str, str] = {}
    for record, split in zip(records, values):
        product_id = str(record["product_id"])
        previous = products.setdefault(product_id, split)
        if previous != split:
            raise ValueError(f"Product leakage in locked split for product_id {product_id}.")
    return parts


def feature_union(max_word_features: int, max_character_features: int) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.995,
                    max_features=max_word_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_character_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def classifier_pipeline(*, large: bool = False) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                feature_union(40_000 if large else 12_000, 60_000 if large else 24_000),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def multilabel_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", feature_union(12_000, 24_000)),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=4.0,
                        class_weight="balanced",
                        max_iter=2_000,
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )


def _select(records: list[dict[str, Any]], indices: list[int], predicate) -> list[int]:
    return [index for index in indices if predicate(records[index])]


def _counts(records: list[dict[str, Any]], indices: list[int], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(records[index][field]) for index in indices).items()))


def _risk_metrics(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "sample_count": len(actual),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro")), 4),
        "no_risk": {
            "precision": round(float(precision[0]), 4),
            "recall": round(float(recall[0]), 4),
            "f1": round(float(f1[0]), 4),
            "support": int(support[0]),
        },
        "risk": {
            "precision": round(float(precision[1]), 4),
            "recall": round(float(recall[1]), 4),
            "f1": round(float(f1[1]), 4),
            "support": int(support[1]),
        },
    }


def _multilabel_metrics(actual: np.ndarray, predicted: np.ndarray, classes: list[str]) -> dict[str, Any]:
    per_class = precision_recall_fscore_support(
        actual, predicted, average=None, zero_division=0
    )
    return {
        "sample_count": int(actual.shape[0]),
        "classes": classes,
        "exact_match_accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(actual, predicted, average="micro", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(actual, predicted)), 4),
        "per_class": {
            label: {
                "precision": round(float(per_class[0][index]), 4),
                "recall": round(float(per_class[1][index]), 4),
                "f1": round(float(per_class[2][index]), 4),
                "support": int(per_class[3][index]),
            }
            for index, label in enumerate(classes)
        },
    }


def category_predictions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold multilabel probabilities, falling back to the strongest category."""
    predicted = (probabilities >= threshold).astype(int)
    empty = np.flatnonzero(predicted.sum(axis=1) == 0)
    if len(empty):
        predicted[empty, np.argmax(probabilities[empty], axis=1)] = 1
    return predicted


def select_category_threshold(actual: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    return max(
        candidates,
        key=lambda threshold: (
            f1_score(
                actual,
                category_predictions(probabilities, threshold),
                average="macro",
                zero_division=0,
            ),
            threshold,
        ),
    )


def uncertainty_predictions(
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> list[str]:
    """Convert binary probabilities into positive, negative, or uncertain labels."""
    strongest = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    return [
        "uncertain" if confidence[index] < threshold else str(classes[class_index])
        for index, class_index in enumerate(strongest)
    ]


def select_uncertainty_threshold(
    actual: list[str],
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Tune abstention on validation data, never on the locked test partition."""
    expected = [
        label if label in BINARY_SENTIMENT_LABELS else "uncertain"
        for label in actual
    ]
    candidates = tuple(round(value / 100, 2) for value in range(50, 96, 5))
    return max(
        candidates,
        key=lambda threshold: (
            evaluate_labels(
                expected,
                uncertainty_predictions(probabilities, classes, threshold),
            )["f1"],
            -threshold,
        ),
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    records = load_marketplace(args.marketplace)
    sentiment_mode = getattr(args, "sentiment_mode", "three-class")
    locked_split = locked_product_split(records)
    if locked_split is None:
        train_indices, validation_indices, test_indices = product_group_split(
            records, args.random_state
        )
        split_policy = (
            "Repeated StratifiedGroupKFold by product_id, selected from label/size "
            "balance only; no product crosses partitions"
        )
    else:
        train_indices, validation_indices, test_indices = locked_split
        split_policy = "Locked split column supplied by the dataset; product leakage validated"
    firecs = load_firecs(args.firecs_train)

    sentiment_train_labels_allowed = (
        BINARY_SENTIMENT_LABELS if sentiment_mode == "binary-uncertain" else SENTIMENT_LABELS
    )
    local_sentiment_train = _select(
        records,
        train_indices,
        lambda row: row["sentiment"] in sentiment_train_labels_allowed,
    )
    firecs_sentiment_train = [
        row for row in firecs if row["label"] in sentiment_train_labels_allowed
    ]
    sentiment_train_texts = [
        normalized_text(row["review"]) for row in firecs_sentiment_train
    ]
    sentiment_train_labels = [row["label"] for row in firecs_sentiment_train]
    sentiment_train_texts.extend(records[index]["text"] for index in local_sentiment_train)
    sentiment_train_labels.extend(records[index]["sentiment"] for index in local_sentiment_train)

    evaluation_labels = (
        SENTIMENT_LABELS | {"mixed"}
        if sentiment_mode == "binary-uncertain"
        else SENTIMENT_LABELS
    )
    sentiment_validation = _select(
        records, validation_indices, lambda row: row["sentiment"] in evaluation_labels
    )
    sentiment_test = _select(
        records, test_indices, lambda row: row["sentiment"] in evaluation_labels
    )
    sentiment = classifier_pipeline(large=True)
    sentiment.fit(sentiment_train_texts, sentiment_train_labels)

    incumbent = DefaketiveSentimentModel()
    sentiment_uncertainty_threshold = None
    if sentiment_mode == "binary-uncertain":
        validation_actual = [
            records[index]["sentiment"]
            if records[index]["sentiment"] in BINARY_SENTIMENT_LABELS
            else "uncertain"
            for index in sentiment_validation
        ]
        test_actual = [
            records[index]["sentiment"]
            if records[index]["sentiment"] in BINARY_SENTIMENT_LABELS
            else "uncertain"
            for index in sentiment_test
        ]
        validation_probabilities = sentiment.predict_proba(
            [records[index]["text"] for index in sentiment_validation]
        )
        sentiment_uncertainty_threshold = select_uncertainty_threshold(
            [records[index]["sentiment"] for index in sentiment_validation],
            validation_probabilities,
            sentiment.classes_,
        )
        validation_predictions = uncertainty_predictions(
            validation_probabilities,
            sentiment.classes_,
            sentiment_uncertainty_threshold,
        )
        test_predictions = uncertainty_predictions(
            sentiment.predict_proba([records[index]["text"] for index in sentiment_test]),
            sentiment.classes_,
            sentiment_uncertainty_threshold,
        )
        incumbent_test_predictions = [
            (
                label
                if label in BINARY_SENTIMENT_LABELS
                else "uncertain"
            )
            for label in (
                incumbent.analyze(records[index]["review"], None)["label"]
                for index in sentiment_test
            )
        ]
        validation_sentiment = evaluate_labels(validation_actual, validation_predictions)
        test_sentiment = evaluate_labels(test_actual, test_predictions)
        incumbent_sentiment = evaluate_labels(test_actual, incumbent_test_predictions)
    else:
        validation_sentiment = evaluate_labels(
            [records[index]["sentiment"] for index in sentiment_validation],
            sentiment.predict([records[index]["text"] for index in sentiment_validation]),
        )
        test_sentiment = evaluate_labels(
            [records[index]["sentiment"] for index in sentiment_test],
            sentiment.predict([records[index]["text"] for index in sentiment_test]),
        )
        incumbent_test_predictions = [
            incumbent.analyze(records[index]["review"], None)["label"]
            for index in sentiment_test
        ]
        incumbent_sentiment = evaluate_labels(
            [records[index]["sentiment"] for index in sentiment_test],
            incumbent_test_predictions,
        )

    risk_train = _select(records, train_indices, lambda row: row["risk_detected"] is not None)
    risk_validation = _select(
        records, validation_indices, lambda row: row["risk_detected"] is not None
    )
    risk_test = _select(records, test_indices, lambda row: row["risk_detected"] is not None)
    risk_detection = classifier_pipeline()
    risk_detection.fit(
        [records[index]["text"] for index in risk_train],
        [records[index]["risk_detected"] for index in risk_train],
    )
    validation_risk = _risk_metrics(
        [records[index]["risk_detected"] for index in risk_validation],
        risk_detection.predict([records[index]["text"] for index in risk_validation]).tolist(),
    )
    test_risk = _risk_metrics(
        [records[index]["risk_detected"] for index in risk_test],
        risk_detection.predict([records[index]["text"] for index in risk_test]).tolist(),
    )

    incumbent_risk_predictions = [
        int(incumbent.analyze(records[index]["review"], None)["risk"]["detected"])
        for index in risk_test
    ]
    incumbent_risk = _risk_metrics(
        [records[index]["risk_detected"] for index in risk_test],
        incumbent_risk_predictions,
    )

    category_train = _select(
        records,
        train_indices,
        lambda row: row["risk_detected"] == 1 and bool(row["risk_categories"]),
    )
    category_validation = _select(
        records,
        validation_indices,
        lambda row: row["risk_detected"] == 1 and bool(row["risk_categories"]),
    )
    category_test = _select(
        records,
        test_indices,
        lambda row: row["risk_detected"] == 1 and bool(row["risk_categories"]),
    )
    category_binarizer = MultiLabelBinarizer()
    category_train_labels = category_binarizer.fit_transform(
        [records[index]["risk_categories"] for index in category_train]
    )
    risk_category = multilabel_pipeline()
    risk_category.fit(
        [records[index]["text"] for index in category_train], category_train_labels
    )

    validation_known = [
        index
        for index in category_validation
        if set(records[index]["risk_categories"]).issubset(category_binarizer.classes_)
    ]
    validation_category_actual = category_binarizer.transform(
        [records[index]["risk_categories"] for index in validation_known]
    )
    validation_category_probabilities = risk_category.predict_proba(
        [records[index]["text"] for index in validation_known]
    )
    category_threshold = select_category_threshold(
        validation_category_actual, validation_category_probabilities
    )

    def category_score(indices: list[int]) -> dict[str, Any]:
        known = [
            index
            for index in indices
            if set(records[index]["risk_categories"]).issubset(category_binarizer.classes_)
        ]
        actual = category_binarizer.transform(
            [records[index]["risk_categories"] for index in known]
        )
        probabilities = risk_category.predict_proba(
            [records[index]["text"] for index in known]
        )
        predicted = category_predictions(probabilities, category_threshold)
        result = _multilabel_metrics(actual, predicted, category_binarizer.classes_.tolist())
        result["unknown_category_rows_excluded"] = len(indices) - len(known)
        result["probability_threshold"] = category_threshold
        return result

    validation_category = category_score(category_validation)
    test_category = category_score(category_test)

    severity_train = _select(
        records,
        train_indices,
        lambda row: row["risk_detected"] == 1 and row["risk_severity"] is not None,
    )
    severity_validation = _select(
        records,
        validation_indices,
        lambda row: row["risk_detected"] == 1 and row["risk_severity"] is not None,
    )
    severity_test = _select(
        records,
        test_indices,
        lambda row: row["risk_detected"] == 1 and row["risk_severity"] is not None,
    )
    severity = classifier_pipeline()
    severity.fit(
        [records[index]["text"] for index in severity_train],
        [records[index]["risk_severity"] for index in severity_train],
    )

    def severity_score(indices: list[int]) -> dict[str, Any]:
        actual = [records[index]["risk_severity"] for index in indices]
        predicted = severity.predict([records[index]["text"] for index in indices]).tolist()
        return {
            "sample_count": len(actual),
            "accuracy": round(float(accuracy_score(actual, predicted)), 4),
            "macro_f1": round(float(f1_score(actual, predicted, average="macro", zero_division=0)), 4),
            "weighted_kappa": round(float(cohen_kappa_score(actual, predicted, weights="quadratic")), 4),
            "mean_absolute_error": round(float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))), 4),
        }

    validation_severity = severity_score(severity_validation)
    test_severity = severity_score(severity_test)

    uncertainty_label = "uncertain" if sentiment_mode == "binary-uncertain" else "neutral"
    gates = {
        "sentiment_macro_f1_at_least_0_75": test_sentiment["f1"] >= 0.75,
        "sentiment_beats_incumbent_by_0_02": test_sentiment["f1"] >= incumbent_sentiment["f1"] + 0.02,
        f"{uncertainty_label}_f1_at_least_0_60": test_sentiment["per_class"].get(uncertainty_label, {}).get("f1", 0) >= 0.60,
        "risk_recall_at_least_0_80": test_risk["risk"]["recall"] >= 0.80,
        "risk_macro_f1_at_least_0_75": test_risk["macro_f1"] >= 0.75,
        "category_macro_f1_at_least_0_65": test_category["macro_f1"] >= 0.65,
        "severity_weighted_kappa_at_least_0_65": test_severity["weighted_kappa"] >= 0.65,
    }
    qualified = all(gates.values())

    split_report = {}
    for name, indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        split_report[name] = {
            "rows": len(indices),
            "products": len({records[index]["product_id"] for index in indices}),
            "platforms": _counts(records, indices, "platform"),
            "sentiment": _counts(records, indices, "sentiment"),
            "product_ids": sorted({records[index]["product_id"] for index in indices}),
        }

    metadata = {
        "model_version": (
            f"{MODEL_VERSION}-binary-uncertain"
            if sentiment_mode == "binary-uncertain"
            else MODEL_VERSION
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "qualified_for_integration": qualified,
        "gates": gates,
        "random_state": args.random_state,
        "marketplace_file": args.marketplace.name,
        "marketplace_sha256": sha256(args.marketplace),
        "firecs_train_file": args.firecs_train.name,
        "firecs_train_sha256": sha256(args.firecs_train),
        "mixed_sentiment_policy": (
            "neutral and mixed map to uncertain for evaluation; both are excluded from "
            "binary sentiment fitting and retained for all risk tasks"
            if sentiment_mode == "binary-uncertain"
            else "mixed excluded from three-class sentiment only; retained for risk tasks"
        ),
        "sentiment_mode": sentiment_mode,
        "sentiment_uncertainty_threshold": sentiment_uncertainty_threshold,
        "category_policy": "pipe- or comma-delimited multilabel categories",
        "category_probability_threshold": category_threshold,
        "split_policy": split_policy,
    }
    artifact = {
        "metadata": metadata,
        "sentiment_pipeline": sentiment,
        "risk_detection_pipeline": risk_detection,
        "risk_category_pipeline": risk_category,
        "risk_category_binarizer": category_binarizer,
        "risk_category_threshold": category_threshold,
        "risk_severity_pipeline": severity,
        "sentiment_uncertainty_threshold": sentiment_uncertainty_threshold,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)

    report = {
        "metadata": metadata,
        "dataset": {
            "rows": len(records),
            "products": len({record["product_id"] for record in records}),
            "platforms": _counts(records, list(range(len(records))), "platform"),
            "sentiment": _counts(records, list(range(len(records))), "sentiment"),
            "blank_annotator_ids": sum(not record["annotator_id"] for record in records),
        },
        "splits": split_report,
        "training": {
            "sentiment_rows": len(sentiment_train_labels),
            "sentiment_local_rows": len(local_sentiment_train),
            "sentiment_firecs_rows": len(firecs_sentiment_train),
            "risk_detection_rows": len(risk_train),
            "risk_category_rows": len(category_train),
            "risk_severity_rows": len(severity_train),
        },
        "validation": {
            "sentiment": validation_sentiment,
            "risk_detection": validation_risk,
            "risk_category": validation_category,
            "risk_severity": validation_severity,
        },
        "test": {
            "sentiment": test_sentiment,
            "incumbent_sentiment": incumbent_sentiment,
            "risk_detection": test_risk,
            "incumbent_risk_detection": incumbent_risk,
            "risk_category": test_category,
            "risk_severity": test_severity,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marketplace", type=Path, default=DEFAULT_MARKETPLACE)
    parser.add_argument("--firecs-train", type=Path, default=DEFAULT_FIRECS_TRAIN)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sentiment-mode",
        choices=("three-class", "binary-uncertain"),
        default="three-class",
    )
    return parser.parse_args()


def main() -> None:
    report = train(parse_args())
    print(json.dumps({"qualified": report["metadata"]["qualified_for_integration"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
