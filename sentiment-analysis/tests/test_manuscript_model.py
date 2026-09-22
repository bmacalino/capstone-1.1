import unittest

from defaketive_sentiment.manuscript import (
    HYBRID_RISK_SCORE_METHOD,
    HybridManuscriptRiskModel,
    ManuscriptModelError,
    ManuscriptRiskModel,
    RISK_SCORE_METHOD,
)


class TestManuscriptRiskModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ManuscriptRiskModel()

    def test_uses_vader_and_binary_defect_indicator_without_severity(self):
        result = self.model.analyze("Maganda pero mabilis masira.", rating=5)

        self.assertEqual(result["label_source"], "vader_rules")
        self.assertTrue(result["risk"]["detected"])
        self.assertEqual(result["risk"]["score"], 1.0)
        self.assertEqual(result["risk"]["score_basis"], "binary_defect_indicator")
        self.assertTrue(result["risk"]["evidence"])
        self.assertNotIn("severity", result["risk"]["evidence"][0])

    def test_star_rating_does_not_override_vader_text_label(self):
        result = self.model.analyze("ordinary item", rating=1)

        self.assertEqual(result["label"], "uncertain")
        self.assertEqual(result["vader_label"], "neutral")
        self.assertIsNone(result["label_adjustment"])

    def test_product_score_is_thirty_percent_negative_plus_seventy_percent_defect(self):
        records = [
            self.model.analyze("Maganda at sulit."),
            self.model.analyze("Hindi maganda."),
            self.model.analyze("Sira at hindi gumagana."),
            self.model.analyze("Ordinary item."),
        ]
        summary = self.model.summarize(records)

        expected = 100 * (0.30 * 0.50 + 0.70 * 0.25)
        self.assertAlmostEqual(summary["risk_score"], expected)
        self.assertEqual(summary["risk_review_count"], 1)
        self.assertEqual(summary["risk_score_method"], RISK_SCORE_METHOD)
        self.assertAlmostEqual(summary["negative_contribution_points"], 15.0)
        self.assertAlmostEqual(summary["defect_contribution_points"], 17.5)

    def test_sampling_weights_are_used_for_population_ratios(self):
        positive = self.model.analyze("Maganda at sulit.")
        positive["risk"]["sampling_weight"] = 3.0
        negative = self.model.analyze("Sira agad.")
        negative["risk"]["sampling_weight"] = 1.0

        summary = self.model.summarize([positive, negative])

        self.assertAlmostEqual(summary["sentiment_ratios"]["negative"], 0.25)
        self.assertAlmostEqual(summary["keyword_failure_rate"], 0.25)
        self.assertAlmostEqual(summary["risk_score"], 25.0)

    def test_negated_defect_does_not_enter_failure_rate(self):
        summary = self.model.summarize(
            [self.model.analyze("Walang sira at gumagana nang maayos.")]
        )

        self.assertEqual(summary["risk_review_count"], 0)
        self.assertEqual(summary["keyword_failure_rate"], 0.0)


class TestHybridArtifactGate(unittest.TestCase):
    """The shipped artifact does not pass its own evaluation gates.

    `trained.py` and `marketplace_v2.py` have always refused an artifact marked
    `qualified_for_integration: false`. This class did not check, so the
    application served a model its own gate would have rejected, and every
    number reported from that path inherits the problem. These tests pin the
    check so the hole cannot reopen.
    """

    def test_default_load_is_refused(self):
        with self.assertRaises(ManuscriptModelError):
            HybridManuscriptRiskModel()

    def test_research_baseline_can_opt_in_explicitly(self):
        model = HybridManuscriptRiskModel(allow_unqualified=True)
        self.assertFalse(model.qualified_for_integration)


class TestHybridManuscriptRiskModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loaded deliberately as a research baseline. The badge model is the
        # production path; this one is retained only for the comparison the
        # validation plan requires.
        cls.model = HybridManuscriptRiskModel(allow_unqualified=True)

    def test_artifact_contains_only_required_prediction_heads(self):
        self.assertIn("sentiment_pipeline", self.model.artifact)
        self.assertIn("risk_detection_pipeline", self.model.artifact)
        self.assertNotIn("risk_severity_pipeline", self.model.artifact)
        self.assertNotIn("risk_category_pipeline", self.model.artifact)
        self.assertEqual(
            set(self.model.artifact["sentiment_pipeline"].classes_),
            {"negative", "positive", "uncertain"},
        )

    def test_trained_heads_drive_decisions_but_severity_is_removed(self):
        result = self.model.analyze("Hindi na gumana after two days.", rating=1)

        self.assertEqual(result["label_source"], "marketplace_direct_three_class")
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertEqual(result["risk"]["score"], 1.0)
        self.assertEqual(
            result["risk"]["score_basis"], "binary_trained_risk_indicator"
        )
        self.assertNotIn("severity", result["risk"])
        self.assertNotIn("expected_severity", result["risk"])

    def test_batch_analysis_is_identical_to_one_at_a_time(self):
        # Batching exists purely for speed once the review cap was removed. If
        # it ever diverges from the per-review path it is silently changing
        # results, so equivalence is asserted rather than assumed.
        reviews = [
            "Hindi na gumana after two days.",
            "Maganda at sulit sa presyo.",
            "Sobrang init tapos namaga ang battery.",
            "peke ang item, hindi original",
            "walang sira at gumagana nang maayos",
            "",
        ]
        ratings = [1, 5, 1, 2, 5, None]
        one_at_a_time = [
            self.model.analyze(review, rating=rating)
            for review, rating in zip(reviews, ratings)
        ]
        self.assertEqual(self.model.analyze_batch(reviews, ratings), one_at_a_time)

    def test_batch_analysis_handles_an_empty_list(self):
        self.assertEqual(self.model.analyze_batch([]), [])

    def test_batch_analysis_defaults_ratings_to_none(self):
        self.assertEqual(
            self.model.analyze_batch(["Hindi na gumana."]),
            [self.model.analyze("Hindi na gumana.", rating=None)],
        )

    def test_hybrid_summary_uses_binary_risk_and_manuscript_weights(self):
        records = [
            {"label": "positive", "scores": {}, "risk": {"detected": False}},
            {"label": "negative", "scores": {}, "risk": {"detected": True}},
            {"label": "uncertain", "scores": {}, "risk": {"detected": True}},
            {"label": "positive", "scores": {}, "risk": {"detected": False}},
        ]

        summary = self.model.summarize(records)

        self.assertEqual(summary["risk_score_method"], HYBRID_RISK_SCORE_METHOD)
        self.assertEqual(summary["risk_review_count"], 2)
        self.assertAlmostEqual(summary["negative_contribution_points"], 7.5)
        self.assertAlmostEqual(summary["defect_contribution_points"], 35.0)
        self.assertAlmostEqual(summary["risk_score"], 42.5)
        self.assertNotIn("average_risk_severity", summary)


if __name__ == "__main__":
    unittest.main()
