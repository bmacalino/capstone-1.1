"""Standalone DeFaketive sentiment and product-risk analyzer."""

from .model import (
    MODEL_VERSION,
    TRAINED_MODEL_VERSION,
    VADER_MODEL_VERSION,
    DefaketiveSentimentModel,
    ModelSetupError,
)
from .trained import TrainedModelError, TrainedSentimentClassifier

__all__ = [
    "MODEL_VERSION",
    "TRAINED_MODEL_VERSION",
    "VADER_MODEL_VERSION",
    "DefaketiveSentimentModel",
    "ModelSetupError",
    "TrainedModelError",
    "TrainedSentimentClassifier",
]
