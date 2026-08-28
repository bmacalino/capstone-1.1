import unittest

from defaketive_sentiment.analyze_reviews import analyze_product_json
from defaketive_sentiment.model import DefaketiveSentimentModel


class TestDefaketiveSentimentModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DefaketiveSentimentModel()

    def test_positive_tagalog_review(self):
        result = self.model.analyze("Maganda, matibay, at sulit sa presyo!")
        self.assertEqual(result["label"], "positive")
        self.assertFalse(result["risk"]["detected"])

    def test_tagalog_negation(self):
        result = self.model.analyze("Hindi maganda ang item.")
        self.assertEqual(result["label"], "negative")

    def test_contrast_and_durability_failure(self):
        result = self.model.analyze("Maganda pero mabilis masira.")
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("durability", result["risk"]["categories"])

    def test_negated_defect_is_not_active_risk(self):
        result = self.model.analyze("Walang sira at gumagana nang maayos.")
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
        result = self.model.analyze(
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
        self.assertEqual(self.model.analyze("😊❤️")["label"], "positive")
        self.assertEqual(self.model.analyze("👎")["label"], "negative")

    def test_negated_counterfeit_claim(self):
        result = self.model.analyze("Not fake. Authentic and works well.")
        self.assertFalse(result["risk"]["detected"])
        self.assertEqual(result["label"], "positive")

    def test_empty_review(self):
        result = self.model.analyze("")
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["scores"]["neu"], 1.0)

    def test_safety_complaint_is_not_misclassified_by_incidental_praise(self):
        result = self.model.analyze(
            "The case was on fire. Very lucky it did not burn the house down."
        )
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("safety", result["risk"]["categories"])

    def test_low_rating_resolves_positive_vader_words_in_negative_review(self):
        result = self.model.analyze(
            "I am disappointed. The wires are exposed, but I would appreciate a refund.",
            rating=1,
        )
        self.assertEqual(result["label"], "negative")
        self.assertEqual(
            result["label_adjustment"]["reason"],
            "low_rating_with_negative_evidence",
        )

    def test_low_rating_resolves_unrecognized_terse_complaint(self):
        result = self.model.analyze("The volume automatically goes down.", rating=1)
        self.assertEqual(result["label"], "negative")
        self.assertEqual(
            result["label_adjustment"]["reason"], "low_rating_neutral_text"
        )

    def test_battery_drain_outweighs_free_item_wording(self):
        result = self.model.analyze(
            "Claimed from a free items deal, but the battery is draining 40% in two hours."
        )
        self.assertEqual(result["label"], "negative")
        self.assertTrue(result["risk"]["detected"])
        self.assertIn("performance", result["risk"]["categories"])

    def test_low_rating_does_not_override_clearly_positive_text(self):
        result = self.model.analyze("Great, the battery lasts a long time", rating=1)
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

        analyzed = analyze_product_json(data, self.model)
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


if __name__ == "__main__":
    unittest.main()
