"""Load and explain the locally trained FiReCS sentiment classifier."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "firecs_sentiment.joblib"

FEATURE_MARKERS = {
    "contrast": re.compile(r"(?<!\w)(?:but|pero|kaso|subalit|ngunit|however)(?!\w)"),
    "functionality": re.compile(
        r"(?<!\w)(?:work|works|working|function|functional|charge|charging|power|gumana|gumagana|sira|sirang)(?!\w)"
    ),
    "quality": re.compile(
        r"(?<!\w)(?:quality|durable|durability|matibay|tibay|defect|defective|sira|basag)(?!\w)"
    ),
    "delivery": re.compile(
        r"(?<!\w)(?:delivery|deliver|shipping|ship|dumating|nadeliver|courier)(?!\w)"
    ),
    "packaging": re.compile(
        r"(?<!\w)(?:package|packaging|box|balot|parcel)(?!\w)"
    ),
    "order_accuracy": re.compile(
        r"(?<!\w)(?:color|colour|size|kulay|mali|wrong|kulang|missing|complete)(?!\w)"
    ),
}


def feature_text(normalized_text: str) -> str:
    """Append sparse structural and aspect markers to the normalized review."""
    markers = [
        f"feature_{name}"
        for name, pattern in FEATURE_MARKERS.items()
        if pattern.search(normalized_text)
    ]
    return f"{normalized_text} {' '.join(markers)}".strip()


class TrainedModelError(RuntimeError):
    """Raised when a local classifier artifact is missing or incompatible."""


class TrainedSentimentClassifier:
    """Small runtime wrapper around the trusted local scikit-learn pipeline."""

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT):
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.is_file():
            raise TrainedModelError(f"Trained sentiment artifact not found: {self.artifact_path}")
        try:
            artifact = joblib.load(self.artifact_path)
        except Exception as exc:  # pragma: no cover - depends on artifact corruption
            raise TrainedModelError(f"Could not load trained sentiment artifact: {exc}") from exc

        if not isinstance(artifact, dict):
            raise TrainedModelError("Trained sentiment artifact must contain a dictionary.")
        pipeline = artifact.get("pipeline")
        metadata = artifact.get("metadata")
        if not hasattr(pipeline, "named_steps") or not isinstance(metadata, dict):
            raise TrainedModelError("Trained sentiment artifact has an unsupported structure.")
        if not metadata.get("qualified_for_integration"):
            raise TrainedModelError("Trained sentiment artifact did not pass the integration gate.")

        features = pipeline.named_steps.get("features")
        classifier = pipeline.named_steps.get("classifier")
        if features is None or classifier is None or not hasattr(classifier, "classes_"):
            raise TrainedModelError("Trained sentiment pipeline is missing required stages.")

        self.pipeline = pipeline
        self.features = features
        self.classifier = classifier
        self.metadata = metadata
        self.model_version = str(metadata.get("model_version") or "firecs-trained-unknown")
        self._feature_names: np.ndarray | None = None

    def _explain(
        self, matrix: Any, predicted_label: str, limit: int = 6
    ) -> list[dict[str, Any]]:
        """Return the strongest active linear features supporting the prediction."""
        if not hasattr(self.classifier, "coef_"):
            return []
        if self._feature_names is None:
            self._feature_names = np.asarray(self.features.get_feature_names_out())

        classes = list(self.classifier.classes_)
        try:
            class_index = classes.index(predicted_label)
        except ValueError:
            return []
        row = matrix.getrow(0)
        if not row.nnz:
            return []
        contributions = row.data * self.classifier.coef_[class_index, row.indices]
        ranked = np.argsort(contributions)[::-1]
        evidence = []
        for position in ranked:
            contribution = float(contributions[position])
            if contribution <= 0:
                break
            raw_name = str(self._feature_names[row.indices[position]])
            feature_type, _, feature = raw_name.partition("__")
            evidence.append(
                {
                    "feature": feature or raw_name,
                    "feature_type": feature_type if feature else "text",
                    "contribution": round(contribution, 6),
                }
            )
            if len(evidence) >= limit:
                break
        return evidence

    def predict(self, normalized_text: str) -> dict[str, Any]:
        matrix = self.features.transform([feature_text(normalized_text)])
        label = str(self.classifier.predict(matrix)[0])
        probabilities: dict[str, float] = {}
        confidence = None
        if hasattr(self.classifier, "predict_proba"):
            values = self.classifier.predict_proba(matrix)[0]
            probabilities = {
                str(name): round(float(value), 6)
                for name, value in zip(self.classifier.classes_, values)
            }
            confidence = round(float(max(values)), 6)
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "evidence": self._explain(matrix, label),
        }
