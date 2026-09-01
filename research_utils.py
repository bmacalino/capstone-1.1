"""Research scoring and verified-dataset evaluation helpers."""

from __future__ import annotations

from typing import Iterable, Sequence


SENTIMENT_LABEL_ALIASES = {
    "0": "negative",
    "0.0": "negative",
    "1": "neutral",
    "1.0": "neutral",
    "2": "positive",
    "2.0": "positive",
}


def normalize_sentiment_label(value: object) -> str:
    """Normalize text labels and the public FiReCS numeric label encoding."""
    label = str(value).strip().casefold()
    return SENTIMENT_LABEL_ALIASES.get(label, label)


def score_sus(responses: Sequence[int | float]) -> float:
    """Calculate the standard 0..100 System Usability Scale score."""
    if len(responses) != 10:
        raise ValueError("SUS requires exactly 10 responses.")
    values = [float(value) for value in responses]
    if any(value < 1 or value > 5 for value in values):
        raise ValueError("SUS responses must be between 1 and 5.")
    adjusted = [value - 1 if index % 2 == 0 else 5 - value for index, value in enumerate(values)]
    return round(sum(adjusted) * 2.5, 2)


def score_umux(responses: Sequence[int | float]) -> float:
    """Calculate the four-item UMUX score on its standard 1..7 scale."""
    if len(responses) != 4:
        raise ValueError("UMUX requires exactly 4 responses.")
    values = [float(value) for value in responses]
    if any(value < 1 or value > 7 for value in values):
        raise ValueError("UMUX responses must be between 1 and 7.")
    adjusted = [values[0] - 1, 7 - values[1], values[2] - 1, 7 - values[3]]
    return round((sum(adjusted) / 24) * 100, 2)


def evaluate_labels(actual: Iterable[str], predicted: Iterable[str]) -> dict:
    """Return reproducible multiclass classification metrics and matrix."""
    actual_values = [normalize_sentiment_label(value) for value in actual]
    predicted_values = [normalize_sentiment_label(value) for value in predicted]
    if not actual_values or len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted labels must be non-empty and have equal length.")
    labels = sorted(set(actual_values) | set(predicted_values))
    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for verified model evaluation.") from exc
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_values, predicted_values, labels=labels, average="macro", zero_division=0
    )
    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        actual_values, predicted_values, labels=labels, average=None, zero_division=0
    )
    matrix = confusion_matrix(actual_values, predicted_values, labels=labels)
    return {
        "sample_count": len(actual_values),
        "labels": labels,
        "accuracy": round(float(accuracy_score(actual_values, predicted_values)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "per_class": {
            label: {
                "precision": round(float(class_precision[index]), 4),
                "recall": round(float(class_recall[index]), 4),
                "f1": round(float(class_f1[index]), 4),
                "support": int(class_support[index]),
            }
            for index, label in enumerate(labels)
        },
        "matrix": matrix.tolist(),
    }
