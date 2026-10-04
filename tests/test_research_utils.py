import unittest

from research_utils import (
    bootstrap_macro_f1_ci,
    evaluate_labels,
    holm_bonferroni,
    mcnemar_test,
    normalize_sentiment_label,
    score_sus,
    score_umux,
)


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


class TestMcNemarTest(unittest.TestCase):
    def test_counts_only_discordant_rows(self):
        # Rows 1-2: both correct. Row 3: only A correct. Row 4: only B correct.
        result = mcnemar_test(
            ["positive", "negative", "positive", "negative"],
            ["positive", "negative", "positive", "positive"],
            ["positive", "negative", "negative", "negative"],
        )

        self.assertEqual(result["only_a_correct"], 1)
        self.assertEqual(result["only_b_correct"], 1)
        self.assertEqual(result["discordant"], 2)

    def test_exact_binomial_p_value(self):
        # A correct on 3 rows B misses, B never wins: p = 2 * C(3,0) / 2^3 = 0.25.
        actual = ["positive"] * 3
        result = mcnemar_test(actual, ["positive"] * 3, ["negative"] * 3)

        self.assertEqual(result["method"], "exact")
        self.assertEqual(result["discordant"], 3)
        self.assertEqual(result["p_value"], 0.25)

    def test_identical_models_are_not_significant(self):
        predictions = ["positive", "negative", "neutral"]
        result = mcnemar_test(predictions, predictions, predictions)

        self.assertEqual(result["discordant"], 0)
        self.assertEqual(result["p_value"], 1.0)

    def test_two_sided_p_value_is_capped_at_one(self):
        # Perfectly balanced disagreement must not produce p > 1.
        actual = ["positive"] * 10
        a_predictions = ["positive"] * 5 + ["negative"] * 5
        b_predictions = ["negative"] * 5 + ["positive"] * 5
        result = mcnemar_test(actual, a_predictions, b_predictions)

        self.assertEqual(result["only_a_correct"], 5)
        self.assertEqual(result["only_b_correct"], 5)
        self.assertEqual(result["p_value"], 1.0)

    def test_large_samples_use_chi_square_approximation(self):
        # 30 wins for A, 10 for B: chi-square = (|30-10|-1)^2 / 40 = 9.025.
        actual = ["positive"] * 40
        a_predictions = ["positive"] * 30 + ["negative"] * 10
        b_predictions = ["negative"] * 30 + ["positive"] * 10
        result = mcnemar_test(actual, a_predictions, b_predictions)

        self.assertEqual(result["method"], "chi_square_continuity_corrected")
        self.assertEqual(result["statistic"], 9.025)
        self.assertLess(result["p_value"], 0.01)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            mcnemar_test(["positive", "negative"], ["positive"], ["negative", "negative"])


class TestBootstrapMacroF1(unittest.TestCase):
    def test_perfect_predictions_give_a_degenerate_interval(self):
        labels = ["positive", "negative", "neutral"] * 10
        result = bootstrap_macro_f1_ci(labels, labels, iterations=50)

        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["ci_lower"], 1.0)
        self.assertEqual(result["ci_upper"], 1.0)

    def test_interval_brackets_the_point_estimate(self):
        actual = ["positive", "negative", "neutral"] * 20
        predicted = list(actual)
        predicted[0] = "negative"
        predicted[5] = "positive"
        result = bootstrap_macro_f1_ci(actual, predicted, iterations=200)

        self.assertLessEqual(result["ci_lower"], result["macro_f1"])
        self.assertLessEqual(result["macro_f1"], result["ci_upper"])

    def test_same_seed_reproduces_the_interval(self):
        actual = ["positive", "negative", "neutral"] * 20
        predicted = list(actual)
        predicted[3] = "positive"
        first = bootstrap_macro_f1_ci(actual, predicted, iterations=100, seed=7)
        second = bootstrap_macro_f1_ci(actual, predicted, iterations=100, seed=7)

        self.assertEqual(first["ci_lower"], second["ci_lower"])
        self.assertEqual(first["ci_upper"], second["ci_upper"])


class TestHolmBonferroni(unittest.TestCase):
    def test_step_down_adjustment_and_monotonicity(self):
        # Sorted: a=0.01 -> 3*0.01=0.03, c=0.03 -> 2*0.03=0.06,
        # b=0.04 -> 1*0.04=0.04 but is floored to the running max 0.06.
        result = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
        results = result["results"]

        self.assertEqual(result["comparisons"], 3)
        self.assertEqual(results["a"]["p_adjusted"], 0.03)
        self.assertEqual(results["c"]["p_adjusted"], 0.06)
        self.assertEqual(results["b"]["p_adjusted"], 0.06)

    def test_only_surviving_comparisons_are_significant(self):
        results = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})["results"]

        self.assertTrue(results["a"]["significant"])
        self.assertFalse(results["b"]["significant"])
        self.assertFalse(results["c"]["significant"])

    def test_single_comparison_is_unchanged(self):
        results = holm_bonferroni({"only": 0.02})["results"]

        self.assertEqual(results["only"]["p_adjusted"], 0.02)
        self.assertTrue(results["only"]["significant"])

    def test_invalid_p_values_are_rejected(self):
        with self.assertRaises(ValueError):
            holm_bonferroni({"a": 1.5})
        with self.assertRaises(ValueError):
            holm_bonferroni({})


if __name__ == "__main__":
    unittest.main()
