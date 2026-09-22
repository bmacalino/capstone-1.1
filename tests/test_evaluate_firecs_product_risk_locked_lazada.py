from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.evaluate_firecs_product_risk_locked_lazada import product_metrics


class FirecsProductRiskLockedLazadaTests(unittest.TestCase):
    def test_product_metrics_report_seventy_point_contribution(self):
        records = [
            {"product_id": "a", "risk_detected": 1},
            {"product_id": "a", "risk_detected": 0},
            {"product_id": "b", "risk_detected": 0},
            {"product_id": "b", "risk_detected": 0},
        ]
        result = product_metrics(records, np.asarray([1, 1, 0, 0], dtype=int))
        self.assertEqual(result["product_count"], 2)
        self.assertEqual(
            result["unweighted_mean_absolute_risk_contribution_error_points"],
            17.5,
        )


if __name__ == "__main__":
    unittest.main()
