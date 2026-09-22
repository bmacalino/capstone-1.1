import unittest

from defaketive_sentiment.analyze_reviews import analyze_product_json
from defaketive_sentiment.model import DefaketiveSentimentModel


class TestDefaketiveSentimentModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DefaketiveSentimentModel()
        cls.rule_model = DefaketiveSentimentModel(use_trained=False)

    def test_positive_tagalog_review(self):
        result = self.model.analyze("Maganda, matibay, at sulit sa presyo!")
        self.assertEqual(result["label"], "positive")
        self.assertFalse(result["risk"]["detected"])

    def test_tagalog_negation(self):
        result = self.rule_model.analyze("Hindi maganda ang item.")
        self.assertEqual(result["label"], "negative")

    def test_contrast_and_durability_failure(self):
        result = self.rule_model.analyze("Maganda pero mabilis masira.")
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("durability", result["risk"]["categories"])

    def test_negated_defect_is_not_active_risk(self):
        result = self.rule_model.analyze("Walang sira at gumagana nang maayos.")
        self.assertEqual(result["label"], "positive")
        self.assertFalse(result["risk"]["detected"])
        self.assertTrue(any(item["negated"] for item in result["risk"]["evidence"]))

    def test_counterfeit_phrase(self):
        result = self.model.analyze("Mukhang peke at hindi original.")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("counterfeit", result["risk"]["categories"])
        self.assertEqual(result["risk"]["score"], 1.0)

    def test_narrative_failure_is_negative_and_risky(self):
        result = self.model.analyze("Nasira agad after two days.")
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("durability", result["risk"]["categories"])

    def test_observed_taglish_failure_phrase(self):
        result = self.model.analyze("bilis masira 1 week palang")
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("durability", result["risk"]["categories"])

    def test_dont_buy_warning_is_negative(self):
        self.assertEqual(self.model.analyze("DONT BUY")["label"], "negative")
        self.assertEqual(self.model.analyze("Don't buy this item")["label"], "negative")

    def test_glitch_outweighs_packaging_praise_and_records_risk(self):
        result = self.rule_model.analyze(
            "The packaging is good but there is a glitch while using the phone."
        )
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("performance", result["risk"]["categories"])

    def test_observed_misspelling_and_thumbs_down_are_negative(self):
        result = self.model.analyze("sosovery dizapoytedd!!!!! 👎")
        self.assertEqual(result["label"], "negative")
        self.assertTrue(any(item["term"] == "👎" for item in result["sentiment_evidence"]))

    def test_common_positive_and_negative_emojis(self):
        self.assertEqual(self.rule_model.analyze("😊❤️")["label"], "positive")
        self.assertEqual(self.rule_model.analyze("👎")["label"], "negative")

    def test_negated_counterfeit_claim(self):
        result = self.rule_model.analyze("Not fake. Authentic and works well.")
        self.assertFalse(result["risk"]["detected"])
        self.assertEqual(result["label"], "positive")

    def test_empty_review(self):
        result = self.model.analyze("")
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["scores"]["neu"], 1.0)

    def test_safety_complaint_is_not_misclassified_by_incidental_praise(self):
        result = self.rule_model.analyze(
            "The case was on fire. Very lucky it did not burn the house down."
        )
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("safety", result["risk"]["categories"])

    def test_low_rating_resolves_positive_vader_words_in_negative_review(self):
        fallback = DefaketiveSentimentModel(use_trained=False)
        result = fallback.analyze(
            "I am disappointed. The wires are exposed, but I would appreciate a refund.",
            rating=1,
        )
        self.assertEqual(result["label"], "negative")
        self.assertEqual(
            result["label_adjustment"]["reason"],
            "low_rating_with_negative_evidence",
        )

    def test_trained_classifier_is_loaded_with_probabilities_and_evidence(self):
        result = self.model.analyze(
            "okay naman ang item pero medyo manipis ang material"
        )

        self.assertEqual(result["model_version"], "defaketive-firecs-tfidf-0.2.0")
        self.assertIsNotNone(result["classifier"])
        self.assertIn(result["classifier"]["label"], {"negative", "neutral", "positive"})
        self.assertAlmostEqual(
            sum(result["classifier"]["probabilities"].values()), 1.0, places=4
        )
        self.assertTrue(result["classifier"]["evidence"])

    def test_mixed_reviews_remain_neutral_with_supervised_classifier(self):
        reviews = [
            "Okay naman pero wrong color ang dumating.",
            "Maganda naman siya kaso may leak, pero overall okay.",
            "Good packaging, but the item does not work.",
            "Okay lang naman, may minor defect pero nagagamit pa.",
        ]

        for review in reviews:
            with self.subTest(review=review):
                result = self.model.analyze(review)
                self.assertEqual(result["label"], "neutral")
                self.assertEqual(
                    result["label_source"], "firecs_supervised_classifier"
                )

    def test_mixed_review_keeps_risk_evidence_separate_from_sentiment(self):
        result = self.model.analyze(
            "Okay lang naman, may minor defect pero nagagamit pa."
        )

        self.assertEqual(result["label"], "neutral")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("defect", result["risk"]["categories"])

    def test_missing_trained_artifact_falls_back_to_vader(self):
        model = DefaketiveSentimentModel(
            trained_model_path="missing-firecs-model.joblib"
        )
        result = model.analyze("Maganda at sulit")

        self.assertEqual(result["model_version"], "defaketive-vader-seed-0.1.2")
        self.assertEqual(result["label_source"], "vader_rules")
        self.assertIsNone(result["classifier"])
        self.assertIn("not found", model.trained_model_error)

    def test_low_rating_resolves_unrecognized_terse_complaint(self):
        result = self.model.analyze("The volume automatically goes down.", rating=1)
        self.assertEqual(result["label"], "negative")
        self.assertEqual(
            result["label_adjustment"]["reason"], "low_rating_neutral_text"
        )

    def test_battery_drain_outweighs_free_item_wording(self):
        result = self.rule_model.analyze(
            "Claimed from a free items deal, but the battery is draining 40% in two hours."
        )
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("performance", result["risk"]["categories"])

    def test_low_rating_does_not_override_clearly_positive_text(self):
        result = self.rule_model.analyze("Great, the battery lasts a long time", rating=1)
        self.assertEqual(result["label"], "positive")
        self.assertIsNone(result["label_adjustment"])

    def test_hyphenated_poor_quality_is_negative(self):
        result = self.model.analyze("Poor-quality item")
        self.assertEqual(result["label"], "negative")

    def test_product_json_passes_rating_context_to_model(self):
        data = [
            {
                "name": "Test gadget",
                "comments": [
                    {"content": "The volume automatically goes down.", "rating": 1}
                ],
            }
        ]
        analyzed = analyze_product_json(data, self.model)
        self.assertEqual(
            analyzed[0]["comments"][0]["sentiment_analysis"]["label"],
            "negative",
        )

    def test_product_json_marks_duplicates_and_aggregates_unique_reviews(self):
        data = [
            {
                "name": "Test gadget",
                "comments": [
                    {"content": "Maganda at sulit"},
                    {"content": "  MAGANDA at sulit!!!  "},
                    {"content": "Sira agad"},
                ],
            }
        ]
        analyzed = analyze_product_json(data, self.model)
        comments = analyzed[0]["comments"]
        self.assertFalse(comments[0]["is_duplicate"])
        self.assertTrue(comments[1]["is_duplicate"])
        self.assertEqual(analyzed[0]["sentiment_summary"]["review_count"], 2)
        self.assertEqual(analyzed[0]["sentiment_summary"]["duplicate_review_count"], 1)

    def test_emoji_only_reviews_are_fingerprinted_and_summarized(self):
        data = [
            {
                "name": "Emoji reviews",
                "comments": [
                    {"content": "😊❤️"},
                    {"content": " 😊❤️ "},
                    {"content": "👎"},
                ],
            }
        ]

        analyzed = analyze_product_json(data, self.rule_model)
        summary = analyzed[0]["sentiment_summary"]
        self.assertEqual(summary["review_count"], 2)
        self.assertEqual(summary["duplicate_review_count"], 1)
        self.assertEqual(summary["sentiment_counts"]["positive"], 1)
        self.assertEqual(summary["sentiment_counts"]["negative"], 1)

    def test_summary_formula(self):
        analyses = [
            self.model.analyze("Maganda at sulit"),
            self.model.analyze("Sira at hindi gumagana"),
        ]
        summary = self.model.summarize(analyses)
        self.assertEqual(summary["review_count"], 2)
        self.assertEqual(summary["risk_review_count"], 1)
        self.assertEqual(summary["risk_score"], 50.0)
        self.assertEqual(summary["risk_score_scale"], "percent")

    def test_marketplace_summary_uses_risk_confidence_severity_and_sample_size(self):
        from defaketive_sentiment.marketplace_v2 import MarketplaceV2Model

        analyses = [
            {
                "label": "negative",
                "scores": {"compound": -0.8},
                "risk": {
                    "detected": True,
                    "confidence": 0.9,
                    "score": 0.8,
                    "categories": ["defect"],
                    "severity": 3,
                },
            },
            {
                "label": "uncertain",
                "scores": {"compound": 0.0},
                "risk": {
                    "detected": False,
                    "confidence": 0.2,
                    "score": 0.1,
                    "categories": [],
                    "severity": None,
                },
            },
        ]
        summary = MarketplaceV2Model.summarize(analyses)
        self.assertEqual(summary["sentiment_counts"]["uncertain"], 1)
        self.assertEqual(summary["risk_review_count"], 1)
        self.assertEqual(summary["raw_risk_score"], 45.0)
        self.assertEqual(summary["evidence_confidence"], 0.0625)
        self.assertEqual(summary["risk_score"], 16.88)
        self.assertEqual(
            summary["risk_score_method"],
            "calibrated_probability_expected_severity",
        )
        self.assertEqual(summary["risk_categories"], {})
        self.assertEqual(len(summary["risk_score_ci_95"]), 2)

    @staticmethod
    def _mixed_star_product():
        comments = []
        for rating, count in ((1, 10), (2, 20), (3, 30), (4, 40), (5, 100)):
            comments.extend(
                {
                    "content": f"unique review {rating}-{index}",
                    "rating": rating,
                }
                for index in range(count)
            )
        return [{"comments": comments, "reported_review_count": 200}]

    def test_legacy_review_limit_caps_large_review_sets_proportionally(self):
        # The capped, star-proportional behaviour the manuscript results were
        # produced under. Now opt-in, so the baseline stays reproducible.
        analyzed = analyze_product_json(
            self._mixed_star_product(), self.model, review_limit=100
        )
        included = [
            review
            for review in analyzed[0]["comments"]
            if review.get("included_in_analysis")
        ]
        self.assertEqual(len(included), 100)
        self.assertEqual(
            {
                rating: sum(review["rating"] == rating for review in included)
                for rating in range(1, 6)
            },
            {1: 5, 2: 10, 3: 15, 4: 20, 5: 50},
        )
        self.assertEqual(analyzed[0]["sentiment_summary"]["analysis_limit"], 100)

    def test_default_analyzes_every_usable_review(self):
        # Hazard reports run at roughly half a percent of reviews, so a 100 of
        # 200 sample would be expected to miss them. No cap by default.
        analyzed = analyze_product_json(self._mixed_star_product(), self.model)
        included = [
            review
            for review in analyzed[0]["comments"]
            if review.get("included_in_analysis")
        ]
        self.assertEqual(len(included), 200)
        summary = analyzed[0]["sentiment_summary"]
        self.assertNotIn("analysis_limit", summary)
        self.assertEqual(summary["coverage"]["reviews_analyzed"], 200)
        self.assertEqual(summary["coverage"]["sampling"], "census_of_scraped_reviews")

    def test_every_analyzed_review_carries_unit_weight_without_sampling(self):
        # Weight 1 is the identity for the legacy weighted summarize, which is
        # what keeps the baseline comparison valid after the cap is removed.
        analyzed = analyze_product_json(self._mixed_star_product(), self.model)
        weights = {
            review["sentiment_analysis"]["risk"]["sampling_weight"]
            for review in analyzed[0]["comments"]
            if review.get("included_in_analysis")
        }
        self.assertEqual(weights, {1.0})

    def test_coverage_reports_completeness_against_the_listed_star_counts(self):
        data = self._mixed_star_product()
        data[0]["review_star_distribution"] = {"1": 10, "2": 20, "3": 30, "4": 40, "5": 100}
        summary = analyze_product_json(data, self.model)[0]["sentiment_summary"]
        self.assertEqual(summary["coverage"]["scrape_completeness"], "complete")
        self.assertEqual(summary["coverage"]["collected_by_star"]["5"], 100)

    def test_coverage_flags_a_short_star_bucket_as_partial(self):
        # A missing slice of low-star reviews is exactly where hazards live.
        data = self._mixed_star_product()
        data[0]["review_star_distribution"] = {"1": 200, "2": 20, "3": 30, "4": 40, "5": 100}
        summary = analyze_product_json(data, self.model)[0]["sentiment_summary"]
        self.assertEqual(summary["coverage"]["scrape_completeness"], "partial")
        self.assertIn("1-star", summary["coverage"]["completeness_reason"])

    def test_coverage_flags_a_star_filtered_scrape_as_partial(self):
        data = self._mixed_star_product()
        data[0]["applied_rating_filters"] = [1, 2, 3]
        summary = analyze_product_json(data, self.model)[0]["sentiment_summary"]
        self.assertEqual(summary["coverage"]["scrape_completeness"], "partial")

    def test_coverage_reports_a_review_date_range(self):
        data = [
            {
                "comments": [
                    {"content": "maganda naman", "rating": 5, "reviewed_at": "2025-03-04"},
                    {"content": "sira agad ang unit", "rating": 1, "reviewed_at": "2026-08-19"},
                ],
                "reported_review_count": 2,
            }
        ]
        coverage = analyze_product_json(data, self.model)[0]["sentiment_summary"]["coverage"]
        self.assertEqual(coverage["review_date_range"], {"first": "2025-03-04", "last": "2026-08-19"})


if __name__ == "__main__":
    unittest.main()
