from __future__ import annotations

import sys
import unittest
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.train_risk_scoring_v1 import (
    DEFAULT_ARTIFACT,
    expected_calibration_error,
    select_risk_threshold,
)


class RiskScoringV1TrainingTests(unittest.TestCase):
    def test_threshold_selection_meets_recall_gate_when_possible(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.1, 0.4, 0.45, 0.9])
        threshold, metrics = select_risk_threshold(labels, probabilities)
        self.assertGreaterEqual(metrics["risk_recall"], 0.8)
        self.assertGreaterEqual(threshold, 0.2)

    def test_calibration_error_is_zero_for_matching_bins(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertEqual(expected_calibration_error(labels, probabilities), 0.0)

    def test_candidate_artifact_has_no_risk_category_head(self):
        artifact = joblib.load(DEFAULT_ARTIFACT)
        self.assertNotIn("risk_category_pipeline", artifact)
        self.assertNotIn("risk_category_binarizer", artifact)
        self.assertIn("risk_detection_pipeline", artifact)
        self.assertIn("risk_severity_pipeline", artifact)


if __name__ == "__main__":
    unittest.main()
