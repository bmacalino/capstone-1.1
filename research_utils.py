"""Research scoring and verified-dataset evaluation helpers."""

from __future__ import annotations

import math
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


def mcnemar_test(
    actual: Iterable[str],
    predicted_a: Iterable[str],
    predicted_b: Iterable[str],
) -> dict:
    """Test whether two models differ on the same test set (paired, two-sided).

    Compares only the rows where the two models disagree. ``only_a_correct`` is
    the count of rows model A got right and model B got wrong; ``only_b_correct``
    is the reverse. Rows both models got right, or both got wrong, carry no
    information about which is better and are excluded by design.

    Uses the exact binomial test when there are few disagreements and the
    continuity-corrected chi-square approximation otherwise, which is the
    conventional cutoff; ``method`` records which one produced the p-value.
    """
    actual_values = [normalize_sentiment_label(value) for value in actual]
    a_values = [normalize_sentiment_label(value) for value in predicted_a]
    b_values = [normalize_sentiment_label(value) for value in predicted_b]
    if not actual_values:
        raise ValueError("McNemar's test requires at least one row.")
    if not len(actual_values) == len(a_values) == len(b_values):
        raise ValueError("Actual and both prediction sequences must have equal length.")

    only_a = sum(
        1
        for truth, a_pred, b_pred in zip(actual_values, a_values, b_values)
        if a_pred == truth and b_pred != truth
    )
    only_b = sum(
        1
        for truth, a_pred, b_pred in zip(actual_values, a_values, b_values)
        if b_pred == truth and a_pred != truth
    )
    discordant = only_a + only_b

    if discordant == 0:
        p_value = 1.0
        method = "exact"
        statistic = 0.0
    elif discordant < 25:
        smaller = min(only_a, only_b)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
        method = "exact"
        statistic = float(smaller)
    else:
        statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
        # Survival function of chi-square with 1 degree of freedom.
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        method = "chi_square_continuity_corrected"

    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "discordant": discordant,
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 6),
        "method": method,
    }


def bootstrap_macro_f1_ci(
    actual: Iterable[str],
    predicted: Iterable[str],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Return a percentile bootstrap confidence interval for macro F1."""
    actual_values = [normalize_sentiment_label(value) for value in actual]
    predicted_values = [normalize_sentiment_label(value) for value in predicted]
    if not actual_values or len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted labels must be non-empty and have equal length.")
    if iterations < 1:
        raise ValueError("Bootstrap requires at least one iteration.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must be between 0 and 1.")

    try:
        import numpy as np
        from sklearn.metrics import f1_score
    except ImportError as exc:
        raise RuntimeError("numpy and scikit-learn are required for bootstrap intervals.") from exc

    labels = sorted(set(actual_values) | set(predicted_values))
    actual_array = np.asarray(actual_values)
    predicted_array = np.asarray(predicted_values)
    rng = np.random.default_rng(seed)
    size = len(actual_values)

    scores = []
    for _ in range(iterations):
        index = rng.integers(0, size, size=size)
        scores.append(
            f1_score(
                actual_array[index],
                predicted_array[index],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(scores, [alpha * 100.0, (1.0 - alpha) * 100.0])
    point = f1_score(
        actual_array, predicted_array, labels=labels, average="macro", zero_division=0
    )
    return {
        "macro_f1": round(float(point), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "confidence": confidence,
        "iterations": iterations,
        "seed": seed,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """Correct a family of p-values for multiple comparisons (Holm step-down).

    Running one test per challenger inflates the chance that at least one looks
    significant by luck. Holm controls that while staying less conservative than
    plain Bonferroni. Returns each comparison's adjusted p-value and whether it
    survives at ``alpha``.
    """
    if not p_values:
        raise ValueError("Holm-Bonferroni requires at least one p-value.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("Alpha must be between 0 and 1.")
    for name, value in p_values.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"p-value for {name!r} must be between 0 and 1.")

    ordered = sorted(p_values.items(), key=lambda item: float(item[1]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for index, (name, value) in enumerate(ordered):
        # Step-down: enforce monotonic non-decreasing adjusted p-values.
        candidate = min(1.0, (total - index) * float(value))
        running_max = max(running_max, candidate)
        adjusted[name] = running_max

    return {
        "alpha": alpha,
        "comparisons": total,
        "results": {
            name: {
                "p_value": round(float(p_values[name]), 6),
                "p_adjusted": round(adjusted[name], 6),
                "significant": adjusted[name] <= alpha,
            }
            for name in p_values
        },
    }
