"""Reusable risk-head helpers for marketplace model artifacts."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np


_DAMAGE_NEGATION_RE = re.compile(
    r"\b(?:no|not|without|walang|wala(?:ng)?|hindi)\s+(?:physical\s+)?"
    r"(?:damage(?:d)?|crack(?:ed)?|basag|dent(?:ed)?|yupi|scratch(?:ed|es)?)\b",
    re.IGNORECASE,
)
_PHYSICAL_DAMAGE_RE = re.compile(
    r"\b(?:basag|crack(?:ed|s)?|dent(?:ed|s)?|yupi|wet|basa|"
    r"scratch(?:ed|es)?|shatter(?:ed)?|physical(?:ly)?\s+damag(?:e|ed))\b",
    re.IGNORECASE,
)
_GENERAL_DAMAGE_RE = re.compile(r"\b(?:broken|damag(?:e|ed))\b", re.IGNORECASE)
_ARRIVAL_RE = re.compile(
    r"\b(?:arriv(?:e|ed|al)?|dumating|pagdating|receive(?:d)?|delivery|delivered|"
    r"shipping|courier|package|packaging|parcel|box)\b",
    re.IGNORECASE,
)


class HybridRiskCategoryClassifier:
    """Add a deterministic physical-damage signal to a learned category model.

    The base model remains responsible for categories present in training.  Damage
    is emitted separately because the first Lazada training split contains no
    examples annotated with that category.
    """

    def __init__(
        self,
        base_model: Any,
        base_classes: Iterable[str],
        output_classes: Iterable[str],
    ) -> None:
        self.base_model = base_model
        self.base_classes = np.asarray(list(base_classes), dtype=object)
        self.output_classes = np.asarray(list(output_classes), dtype=object)
        if len(set(self.output_classes.tolist())) != len(self.output_classes):
            raise ValueError("output_classes must be unique")
        if not set(self.base_classes).issubset(set(self.output_classes)):
            raise ValueError("base_classes must be a subset of output_classes")

    @staticmethod
    def damage_probability(text: object) -> float:
        value = str(text or "")
        if _DAMAGE_NEGATION_RE.search(value):
            return 0.02
        has_physical_damage = bool(_PHYSICAL_DAMAGE_RE.search(value))
        has_general_damage = bool(_GENERAL_DAMAGE_RE.search(value))
        has_arrival_context = bool(_ARRIVAL_RE.search(value))
        if (has_physical_damage or has_general_damage) and has_arrival_context:
            return 0.95
        if has_physical_damage:
            return 0.75
        if has_general_damage:
            return 0.15
        return 0.02

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        base_probabilities = np.asarray(
            self.base_model.predict_proba(values), dtype=float
        )
        if base_probabilities.ndim == 1:
            base_probabilities = base_probabilities.reshape(-1, 1)
        if base_probabilities.shape[1] != len(self.base_classes):
            raise ValueError("Base probability columns do not match base_classes")

        probabilities = np.zeros((len(values), len(self.output_classes)), dtype=float)
        output_indices = {
            category: index for index, category in enumerate(self.output_classes)
        }
        for base_index, category in enumerate(self.base_classes):
            probabilities[:, output_indices[category]] = base_probabilities[:, base_index]
        if "damage" in output_indices:
            probabilities[:, output_indices["damage"]] = [
                self.damage_probability(value) for value in values
            ]
        return probabilities


def rating_feature_text(text: object, rating: object = None) -> str:
    """Append a stable rating token while preserving support for missing ratings."""
    normalized = str(text or "").strip()
    try:
        value = int(float(str(rating).strip()))
    except (TypeError, ValueError):
        value = 0
    token = f"__rating_{value}" if value in {1, 2, 3, 4, 5} else "__rating_unknown"
    return f"{normalized} {token}".strip()


class OrdinalSeverityClassifier:
    """Predict severity 1/2/3 using cumulative binary classifiers."""

    classes_ = np.asarray([1, 2, 3])

    def __init__(
        self,
        model_ge2: Any,
        model_ge3: Any,
        *,
        threshold_ge2: float,
        threshold_ge3: float,
    ) -> None:
        self.model_ge2 = model_ge2
        self.model_ge3 = model_ge3
        self.threshold_ge2 = float(threshold_ge2)
        self.threshold_ge3 = float(threshold_ge3)

    @staticmethod
    def _positive_probability(model: Any, texts: Iterable[str]) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(list(texts)), dtype=float)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]

    def cumulative_probabilities(self, texts: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        values = list(texts)
        probability_ge2 = self._positive_probability(self.model_ge2, values)
        probability_ge3 = self._positive_probability(self.model_ge3, values)
        probability_ge3 = np.minimum(probability_ge2, probability_ge3)
        return probability_ge2, probability_ge3

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        probability_ge2, probability_ge3 = self.cumulative_probabilities(texts)
        return np.column_stack(
            (
                1.0 - probability_ge2,
                probability_ge2 - probability_ge3,
                probability_ge3,
            )
        )

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        probability_ge2, probability_ge3 = self.cumulative_probabilities(texts)
        return np.where(
            probability_ge3 >= self.threshold_ge3,
            3,
            np.where(probability_ge2 >= self.threshold_ge2, 2, 1),
        )


class CalibratedBinaryClassifier:
    """Apply a product-group out-of-fold probability calibrator to a classifier."""

    classes_ = np.asarray([0, 1])

    def __init__(self, base_model: Any, calibrator: Any, *, threshold: float = 0.5) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.threshold = float(threshold)

    @staticmethod
    def _positive_probability(model: Any, texts: Iterable[str]) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(list(texts)), dtype=float)
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)]

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        raw = self._positive_probability(self.base_model, texts)
        calibrated = self._positive_probability(self.calibrator, self._logit(raw))
        return np.column_stack((1 - calibrated, calibrated))

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] >= self.threshold).astype(int)
