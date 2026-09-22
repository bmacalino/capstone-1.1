import unittest

import numpy as np

from train_marketplace_v2 import (
    category_predictions,
    locked_product_split,
    select_category_threshold,
    select_uncertainty_threshold,
    uncertainty_predictions,
)


class MarketplaceV2TrainingTests(unittest.TestCase):
    def test_category_predictions_always_selects_at_least_one_category(self):
        probabilities = np.asarray([[0.10, 0.20, 0.30], [0.70, 0.10, 0.05]])
        predicted = category_predictions(probabilities, 0.50)
        self.assertEqual(predicted.tolist(), [[0, 0, 1], [1, 0, 0]])

    def test_category_threshold_is_selected_from_validation_candidates(self):
        actual = np.asarray([[1, 0], [0, 1], [1, 0]])
        probabilities = np.asarray([[0.45, 0.10], [0.20, 0.40], [0.42, 0.08]])
        # Equal-F1 ties choose the more conservative threshold.
        self.assertEqual(select_category_threshold(actual, probabilities), 0.50)

    def test_locked_product_split_uses_supplied_partitions(self):
        records = [
            {"product_id": "p1", "split": "train"},
            {"product_id": "p1", "split": "train"},
            {"product_id": "p2", "split": "validation"},
            {"product_id": "p3", "split": "test"},
        ]
        self.assertEqual(locked_product_split(records), ([0, 1], [2], [3]))

    def test_locked_product_split_rejects_product_leakage(self):
        records = [
            {"product_id": "p1", "split": "train"},
            {"product_id": "p2", "split": "validation"},
            {"product_id": "p1", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "Product leakage"):
            locked_product_split(records)

    def test_locked_product_split_falls_back_when_column_is_blank(self):
        records = [{"product_id": "p1", "split": ""}]
        self.assertIsNone(locked_product_split(records))

    def test_uncertainty_predictions_abstain_below_threshold(self):
        probabilities = np.asarray([[0.80, 0.20], [0.52, 0.48], [0.10, 0.90]])
        classes = np.asarray(["negative", "positive"])
        self.assertEqual(
            uncertainty_predictions(probabilities, classes, 0.60),
            ["negative", "uncertain", "positive"],
        )

    def test_uncertainty_threshold_is_selected_from_validation_labels(self):
        actual = ["negative", "mixed", "neutral", "positive"]
        probabilities = np.asarray(
            [[0.90, 0.10], [0.52, 0.48], [0.45, 0.55], [0.10, 0.90]]
        )
        classes = np.asarray(["negative", "positive"])
        self.assertEqual(select_uncertainty_threshold(actual, probabilities, classes), 0.60)


if __name__ == "__main__":
    unittest.main()
