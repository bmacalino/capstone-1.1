from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.train_firecs_product_risk import grouped_oof_probabilities, normalized_text


class FirecsProductRiskTrainingTests(unittest.TestCase):
    def test_normalized_text_matches_runtime_shape(self):
        self.assertEqual(
            normalized_text("  NOT Working!!!  "),
            "not working!!! feature_functionality",
        )

    def test_grouped_oof_returns_one_probability_and_fold_per_row(self):
        negative = [f"safe product example {index}" for index in range(10)]
        positive = [f"broken product example {index}" for index in range(10)]
        texts = np.asarray(negative + positive, dtype=object)
        labels = np.asarray([0] * 10 + [1] * 10, dtype=int)
        groups = np.asarray([f"group-{index}" for index in range(20)], dtype=object)
        probabilities, folds = grouped_oof_probabilities(
            texts, labels, groups, seed=42
        )
        self.assertEqual(len(probabilities), 20)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        self.assertEqual(set(folds.tolist()), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
