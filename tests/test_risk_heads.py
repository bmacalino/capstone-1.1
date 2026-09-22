from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

SENTIMENT_ROOT = Path(__file__).resolve().parents[1] / "sentiment-analysis"
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from defaketive_sentiment.risk_heads import (
    CalibratedBinaryClassifier,
    HybridRiskCategoryClassifier,
    OrdinalSeverityClassifier,
    rating_feature_text,
)


class _ProbabilityModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, positives):
        self.positives = np.asarray(positives, dtype=float)

    def predict_proba(self, texts):
        count = len(list(texts))
        positives = self.positives[:count]
        return np.column_stack((1 - positives, positives))


class _CategoryProbabilityModel:
    def predict_proba(self, texts):
        return np.asarray([[0.7, 0.2] for _ in texts], dtype=float)


class RiskHeadTests(unittest.TestCase):
    def test_calibrated_binary_classifier_uses_its_threshold(self):
        model = CalibratedBinaryClassifier(
            _ProbabilityModel([0.6]),
            _ProbabilityModel([0.7]),
            threshold=0.65,
        )
        self.assertEqual(model.predict(["review"]).tolist(), [1])
        self.assertAlmostEqual(model.predict_proba(["review"])[0, 1], 0.7)

    def test_rating_feature_text_handles_known_and_missing_ratings(self):
        self.assertEqual(rating_feature_text("mahina", 2), "mahina __rating_2")
        self.assertEqual(rating_feature_text("mahina", None), "mahina __rating_unknown")

    def test_ordinal_probabilities_are_valid_and_monotonic(self):
        model = OrdinalSeverityClassifier(
            _ProbabilityModel([0.8, 0.4]),
            _ProbabilityModel([0.9, 0.2]),
            threshold_ge2=0.5,
            threshold_ge3=0.7,
        )
        probabilities = model.predict_proba(["a", "b"])
        np.testing.assert_allclose(probabilities.sum(axis=1), [1.0, 1.0])
        np.testing.assert_array_equal(model.predict(["a", "b"]), [3, 1])

    def test_hybrid_category_columns_align_with_output_classes(self):
        model = HybridRiskCategoryClassifier(
            _CategoryProbabilityModel(),
            ["defect", "performance"],
            ["damage", "defect", "performance"],
        )
        probabilities = model.predict_proba(["arrived cracked"])
        np.testing.assert_allclose(probabilities[0], [0.95, 0.7, 0.2])

    def test_hybrid_damage_rule_handles_physical_damage_and_negation(self):
        self.assertEqual(
            HybridRiskCategoryClassifier.damage_probability("May basag sa gilid"),
            0.75,
        )
        self.assertEqual(
            HybridRiskCategoryClassifier.damage_probability("No damage on arrival"),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
