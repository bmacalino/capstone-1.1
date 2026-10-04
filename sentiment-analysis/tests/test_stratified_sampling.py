import unittest

from defaketive_sentiment.analyze_reviews import (
    STRATIFIED_BY_STAR,
    _completeness,
    _stratified_completeness,
    _stratum_weights,
    analyze_product_json,
)
from defaketive_sentiment.hazard_model import HazardBadgeModel
from defaketive_sentiment.product_badges import ProductBadgeAggregator

# The Anker charger listing, measured 27 Sept 2026.
LISTED = {"1": 35, "2": 9, "3": 42, "4": 102, "5": 13704}


def complete_collection(**overrides):
    collection = {star: {"complete": True, "stop_reason": "target_reached"} for star in LISTED}
    collection.update(overrides)
    return collection


def stratified_product(**extra):
    return {
        "sampling_method": STRATIFIED_BY_STAR,
        "review_star_distribution": dict(LISTED),
        "star_collection": complete_collection(),
        "applied_rating_filters": [3, 2, 4, 1, 5],
        **extra,
    }


class TestStratifiedCompleteness(unittest.TestCase):
    def test_every_level_read_to_quota_or_end_is_sufficient(self):
        self.assertEqual(_stratified_completeness(stratified_product()), ("stratified", None))
        # Star filters no longer mean "partial" when the sample was stratified.
        self.assertEqual(_completeness(stratified_product(), {}, 300, 13892, 0.05)[0], "stratified")

    def test_a_level_cut_short_is_partial(self):
        product = stratified_product(
            star_collection=complete_collection(**{"1": {"complete": False, "stop_reason": "page_signature_timeout_attempt_3"}})
        )
        status, reason = _stratified_completeness(product)
        self.assertEqual(status, "partial")
        self.assertIn("1-star", reason)

    def test_unknown_per_star_counts_cannot_be_weighted(self):
        product = stratified_product()
        del product["review_star_distribution"]
        self.assertEqual(_stratified_completeness(product)[0], "partial")

    def test_a_level_with_no_ratings_needs_no_reviews(self):
        product = stratified_product(review_star_distribution={**LISTED, "2": 0})
        product["star_collection"].pop("2")
        self.assertEqual(_stratified_completeness(product)[0], "stratified")

    def test_a_filter_warning_makes_it_partial(self):
        product = stratified_product(review_filter_warning="Could not apply the 3-star filter")
        self.assertEqual(_stratified_completeness(product)[0], "partial")

    def test_unstratified_products_keep_the_old_rule(self):
        self.assertEqual(
            _completeness({"applied_rating_filters": [1]}, {}, 10, 100, 0.05)[0], "partial"
        )


class TestStratumWeights(unittest.TestCase):
    def test_each_review_stands_for_its_share_of_the_listing(self):
        sampled = [{"source_rating_filter": 5}] * 60 + [{"source_rating_filter": 1}] * 20
        weights = _stratum_weights(LISTED, sampled)
        self.assertAlmostEqual(weights[5], 13704 / 60)
        self.assertAlmostEqual(weights[1], 35 / 20)
        # Levels with no analysed review get no weight.
        self.assertNotIn(3, weights)

    def test_the_rating_is_used_when_no_filter_was_recorded(self):
        self.assertAlmostEqual(_stratum_weights(LISTED, [{"rating": 4}] * 2)[4], 51.0)


def record(label, weight, *tags):
    sentences = [{"index": 0, "text": "x", "tags": list(tags), "high_confidence": {}}] if tags else []
    return {"label": label, "sentiment": {"label": label}, "sentences": sentences, "sampling_weight": weight}


class TestWeightedRates(unittest.TestCase):
    def setUp(self):
        self.aggregator = ProductBadgeAggregator(baselines={})
        five, one = 13704 / 60, 35 / 20
        self.records = (
            [record("positive", five) for _ in range(60)]
            + [record("negative", one, "defect") for _ in range(10)]
            + [record("negative", one) for _ in range(10)]
        )

    def test_rates_describe_the_listing_not_the_sample(self):
        summary = self.aggregator.summarize(self.records)
        defect = summary["flags"]["defect"]
        # Unweighted this would be 10 of 80 = 12.5%.
        self.assertAlmostEqual(defect["rate"], 10 * 35 / 20 / (13704 + 35), places=6)
        self.assertEqual(defect["flagged_reviews"], 10)
        self.assertTrue(summary["weighted"])

    def test_the_interval_uses_the_effective_sample_size(self):
        weights = [item["sampling_weight"] for item in self.records]
        n_eff = ProductBadgeAggregator.effective_sample_size(weights)
        self.assertAlmostEqual(n_eff, 13739 ** 2 / (60 * (13704 / 60) ** 2 + 20 * (35 / 20) ** 2))
        self.assertLess(n_eff, len(self.records))
        self.assertEqual(self.aggregator.summarize(self.records)["effective_sample_size"], round(n_eff, 1))

    def test_sentiment_shares_are_weighted_but_counts_are_not(self):
        summary = self.aggregator.summarize(self.records)
        self.assertEqual(summary["sentiment_counts"], {"positive": 60, "uncertain": 0, "negative": 20})
        self.assertAlmostEqual(summary["sentiment_shares"]["positive"], 13704 / 13739, places=5)
        self.assertGreater(summary["positive_ratio_lower_bound"], 0.9)

    def test_equal_weights_match_the_unweighted_rate(self):
        weighted = [dict(item, sampling_weight=1.0) for item in self.records]
        plain = [{key: value for key, value in item.items() if key != "sampling_weight"} for item in self.records]
        first = self.aggregator.summarize(weighted)["flags"]["defect"]
        second = self.aggregator.summarize(plain)["flags"]["defect"]
        self.assertAlmostEqual(first["rate"], second["rate"])
        self.assertAlmostEqual(first["rate_lower_bound"], second["rate_lower_bound"])

    def test_a_census_summary_has_no_weighting_fields(self):
        plain = [{key: value for key, value in item.items() if key != "sampling_weight"} for item in self.records]
        summary = self.aggregator.summarize(plain)
        self.assertNotIn("weighted", summary)
        self.assertNotIn("sentiment_shares", summary)


class TestStratifiedProductEndToEnd(unittest.TestCase):
    def test_a_stratified_product_is_weighted_and_not_blocked_by_coverage(self):
        comments = []
        for star, texts in {
            5: ["Maganda naman, sulit", "Ang bilis ng charge, recommended", "Legit Anker, salamat seller"],
            4: ["Ok naman medyo mainit lang", "Good quality pero matagal delivery"],
            3: ["Pwede na", "Medyo mahina pero gumagana"],
            1: ["Sira agad hindi na gumagana", "Hindi original, peke to"],
        }.items():
            for index in range(10):
                for text in texts:
                    comments.append({
                        "content": f"{text} ({index})", "rating": star, "source_rating_filter": star,
                    })
        product = stratified_product(
            link="https://www.lazada.com.ph/products/pdp-i4644812363-s1.html",
            total_rating=13892,
            review_star_distribution={**LISTED, "2": 0},
            comments=comments,
            star_quota=60,
        )
        product["star_collection"].pop("2")
        analyze_product_json([product], HazardBadgeModel(allow_unqualified=True))
        summary = product["sentiment_summary"]
        coverage = summary["coverage"]
        self.assertEqual(coverage["scrape_completeness"], "stratified")
        self.assertEqual(coverage["sampling"], STRATIFIED_BY_STAR)
        self.assertTrue(coverage["weighted"])
        self.assertAlmostEqual(coverage["weights_by_star"]["5"], round(13704 / 30, 4))
        self.assertNotEqual(summary["verdict"]["state"], "insufficient_evidence")
        self.assertTrue(summary["weighted"])
        weights = {comment["sentiment_analysis"]["sampling_weight"] for comment in comments if comment["rating"] == 1}
        self.assertEqual(weights, {round(35 / 20, 6)})


if __name__ == "__main__":
    unittest.main()
