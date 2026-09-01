import unittest

from research_utils import evaluate_labels, normalize_sentiment_label, score_sus, score_umux


class TestResearchUtils(unittest.TestCase):
    def test_sus_extremes_and_midpoint(self):
        self.assertEqual(score_sus([5, 1] * 5), 100.0)
        self.assertEqual(score_sus([1, 5] * 5), 0.0)
        self.assertEqual(score_sus([3] * 10), 50.0)

    def test_umux_extremes_and_midpoint(self):
        self.assertEqual(score_umux([7, 1, 7, 1]), 100.0)
        self.assertEqual(score_umux([1, 7, 1, 7]), 0.0)
        self.assertEqual(score_umux([4, 4, 4, 4]), 50.0)

    def test_invalid_response_counts(self):
        with self.assertRaises(ValueError):
            score_sus([3] * 9)
        with self.assertRaises(ValueError):
            score_umux([4] * 3)

    def test_firecs_numeric_labels_are_normalized(self):
        self.assertEqual(normalize_sentiment_label(0.0), "negative")
        self.assertEqual(normalize_sentiment_label("1.0"), "neutral")
        self.assertEqual(normalize_sentiment_label(2), "positive")

    def test_evaluation_includes_macro_and_per_class_metrics(self):
        result = evaluate_labels(
            [0.0, 1.0, 2.0, 2.0],
            ["negative", "positive", "positive", "neutral"],
        )

        self.assertEqual(result["labels"], ["negative", "neutral", "positive"])
        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["per_class"]["positive"]["support"], 2)
        self.assertEqual(len(result["matrix"]), 3)


if __name__ == "__main__":
    unittest.main()
