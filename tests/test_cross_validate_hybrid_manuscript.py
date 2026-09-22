import unittest

import numpy as np

from research.cross_validate_hybrid_manuscript import (
    evaluation_sentiment,
    product_risk_score,
    risk_level,
    three_class_predictions,
)


class HybridManuscriptCrossValidationTests(unittest.TestCase):
    def test_non_binary_sentiments_map_to_uncertain(self):
        self.assertEqual(evaluation_sentiment("mixed"), "uncertain")
        self.assertEqual(evaluation_sentiment("neutral"), "uncertain")
        self.assertEqual(evaluation_sentiment("uncertain"), "uncertain")
        self.assertEqual(evaluation_sentiment("positive"), "positive")

    def test_product_score_uses_thirty_seventy_formula(self):
        score = product_risk_score(
            ["positive", "negative", "uncertain", "negative"],
            [0, 0, 1, 0],
        )
        self.assertAlmostEqual(score, 32.5)

    def test_risk_level_boundaries_use_percent_scale(self):
        self.assertEqual(risk_level(30), "low")
        self.assertEqual(risk_level(30.01), "moderate")
        self.assertEqual(risk_level(60), "moderate")
        self.assertEqual(risk_level(60.01), "high")

    def test_three_class_threshold_can_prefer_uncertain(self):
        predicted = three_class_predictions(
            np.asarray(
                [
                    [0.42, 0.20, 0.38],
                    [0.10, 0.75, 0.15],
                ]
            ),
            0.35,
        )
        self.assertEqual(predicted, ["uncertain", "positive"])


if __name__ == "__main__":
    unittest.main()
