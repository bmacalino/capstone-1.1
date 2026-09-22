import unittest

from research.sample_size_study import evaluate, risk_level, risk_score


class TestSampleSizeStudy(unittest.TestCase):
    def test_score_matches_application_formula(self):
        rows = [
            {"negative": True, "risk_detected": True},
            {"negative": False, "risk_detected": False},
        ]
        self.assertEqual(risk_score(rows), 50.0)
        self.assertEqual(risk_level(30), "Low")
        self.assertEqual(risk_level(31), "Moderate")
        self.assertEqual(risk_level(61), "High")

    def test_small_coverage_can_only_produce_pilot_candidate(self):
        products = [{
            "product_id": "one",
            "platform": "temu",
            "reviews": [{"negative": False, "risk_detected": False}] * 30,
        }]
        result = evaluate(products, sizes=[10, 30], trials=20, seed=1)
        self.assertEqual(result["status"], "pilot_only")
        self.assertIsNone(result["recommendation"])
        self.assertEqual(result["pilot_candidate"], 10)


if __name__ == "__main__":
    unittest.main()
