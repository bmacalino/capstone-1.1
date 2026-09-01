import json
import unittest

from dashboard_utils import (
    analysis_is_current,
    cached_product_matches,
    csv_bytes,
    detect_marketplace,
    load_products,
    has_usable_products,
    merge_product_catalogs,
    normalized_risk_score,
    price_value,
    platform_name,
    product_image_url,
    product_reliability,
    product_risk_level,
    product_risk_percent,
    rank_recommendations,
    recommendation_search_query,
    reliability_score,
    review_signal_counts,
    review_rows,
    risk_keyword_counts,
    risk_level,
    risk_score_percent,
    search_slug,
    sentiment_model_version,
    weighted_risk_breakdown,
)


class TestDashboardUtils(unittest.TestCase):
    def test_analysis_version_check_rejects_stale_saved_results(self):
        product = {
            "comments": [
                {
                    "content": "Sira agad",
                    "sentiment_analysis": {"model_version": "old-model"},
                }
            ],
            "sentiment_summary": {"risk_score_scale": "percent"},
        }
        self.assertFalse(analysis_is_current([product]))
        product["comments"][0]["sentiment_analysis"]["model_version"] = (
            sentiment_model_version()
        )
        self.assertTrue(analysis_is_current([product]))

    def test_cached_product_matching_uses_names_and_normalized_links(self):
        products = [
            {"name": "Raspberry Pi Pico 2 Board", "link": "https://shopee.ph/sample-i.1.2?tracking=1", "sentiment_summary": {"risk_score": 20}},
            {"name": "Wireless Earbuds", "link": "https://www.lazada.com.ph/products/earbuds-i1-s2.html", "sentiment_summary": {"risk_score": 10}},
        ]
        self.assertEqual(cached_product_matches(products, "raspberry pico")[0]["name"], "Raspberry Pi Pico 2 Board")
        self.assertEqual(
            cached_product_matches(products, "https://shopee.ph/sample-i.1.2?from=share")[0]["name"],
            "Raspberry Pi Pico 2 Board",
        )
        self.assertEqual(cached_product_matches(products, "not in database"), [])

    def test_recommendations_are_similar_low_risk_and_well_reviewed(self):
        target = {
            "link": "target",
            "name": "Anker 25000mAh Laptop Power Bank",
            "brand": "Anker",
            "category": "Power Banks",
            "sentiment_summary": {"risk_score": 1, "risk_score_scale": "percent"},
        }
        good_match = {
            "link": "good",
            "name": "UGREEN 20000mAh USB-C Power Bank",
            "brand": "UGREEN",
            "category": "Power Banks",
            "rating": 4.8,
            "sentiment_summary": {
                "risk_score": 4,
                "risk_score_scale": "percent",
                "review_count": 30,
                "sentiment_ratios": {"positive": 0.9},
            },
        }
        risky_match = {
            "link": "risky",
            "name": "Anker Power Bank",
            "brand": "Anker",
            "category": "Power Banks",
            "rating": 4.9,
            "sentiment_summary": {
                "risk_score": 70,
                "risk_score_scale": "percent",
                "review_count": 30,
                "sentiment_ratios": {"positive": 0.9},
            },
        }
        unrelated = {
            "link": "unrelated",
            "name": "Low-Power USB-C Raspberry Pi Development Board",
            "rating": 5,
            "sentiment_summary": {
                "risk_score": 0,
                "risk_score_scale": "percent",
                "review_count": 30,
                "sentiment_ratios": {"positive": 1},
            },
        }

        recommendations = rank_recommendations(
            target, [target, risky_match, unrelated, good_match], limit=3
        )

        self.assertEqual([item["link"] for item in recommendations], ["good"])
        self.assertEqual(recommendation_search_query(target), "Anker power bank")

    def test_merge_product_catalogs_normalizes_tracking_links(self):
        current = {"name": "Current", "link": "https://www.lazada.com.ph/products/item-i1.html?tracking=1"}
        stale = {"name": "Stale", "link": "https://lazada.com.ph/products/item-i1.html?tracking=2"}
        other = {"name": "Other", "link": "https://shopee.ph/other-i.1.2"}
        self.assertEqual(merge_product_catalogs([current], [stale, other]), [current, other])

    def test_usable_product_detection_rejects_blank_failed_scrapes(self):
        self.assertFalse(has_usable_products([{"link": "https://shopee.ph/item-i.1.2", "comments": []}]))
        self.assertTrue(has_usable_products([{"name": "Real product", "comments": []}]))
        self.assertTrue(has_usable_products([{"review_status": "no_reviews", "comments": []}]))

    def test_search_slug(self):
        self.assertEqual(search_slug("Laptop Stand & Bag"), "laptop_stand_bag")
        self.assertEqual(search_slug("!!!"), "search")

    def test_marketplace_detection_and_display(self):
        self.assertEqual(detect_marketplace("https://shopee.ph/item-i.1.2"), "shopee")
        self.assertEqual(detect_marketplace("https://www.lazada.com.ph/products/item.html"), "lazada")
        self.assertEqual(detect_marketplace("https://www.temu.com/ph-en/goods.html"), "temu")
        self.assertEqual(detect_marketplace("wireless earbuds", "lazada"), "lazada")
        self.assertEqual(platform_name({"platform": "temu"}), "Temu PH")
        with self.assertRaises(ValueError):
            detect_marketplace("https://example.com/item")

    def test_product_image_url_normalizes_common_nested_fields(self):
        self.assertEqual(
            product_image_url({"img": "//img.lazcdn.com/product.jpg"}),
            "https://img.lazcdn.com/product.jpg",
        )
        self.assertEqual(
            product_image_url(
                {
                    "image": {"src": "", "original": [None, {"url": "https://cdn.example/item.png"}]},
                    "thumbnail": "https://cdn.example/fallback.png",
                }
            ),
            "https://cdn.example/item.png",
        )

    def test_product_image_url_rejects_non_web_and_invalid_values(self):
        self.assertEqual(product_image_url({"img": "C:\\downloads\\product.jpg"}), "")
        self.assertEqual(product_image_url({"image": "javascript:alert(1)"}), "")
        self.assertEqual(product_image_url({"images": [None, "", "/relative/image.jpg"]}), "")
        self.assertEqual(product_image_url({"img": "", "thumbnail_url": "https://cdn.example/valid.jpg"}), "https://cdn.example/valid.jpg")

    def test_price_value_uses_range_midpoint(self):
        self.assertEqual(price_value("₱1,000 - ₱1,500"), 1250.0)
        self.assertEqual(price_value("â‚±299"), 299.0)
        self.assertIsNone(price_value(""))

    def test_load_and_flatten_reviews(self):
        payload = [{"name": "Item", "comments": [{"rating": 5, "content": "Good"}]}]
        products = load_products(json.dumps(payload).encode())
        rows = review_rows(products)
        self.assertEqual(rows[0]["Product"], "Item")
        self.assertEqual(rows[0]["Sentiment"], "unanalyzed")
        self.assertIn(b"Review", csv_bytes(rows))

    def test_risk_display_helpers(self):
        self.assertEqual(normalized_risk_score(61), 0.61)
        self.assertEqual(normalized_risk_score(0.4), 0.4)
        self.assertEqual(normalized_risk_score(1.0, "percent"), 0.01)
        self.assertEqual(normalized_risk_score(1.0, "ratio"), 1.0)
        self.assertEqual(risk_score_percent(1.0, "percent"), 1.0)
        self.assertEqual(risk_level(30), "Low")
        self.assertEqual(risk_level(31), "Moderate")
        self.assertEqual(risk_level(61), "High")
        self.assertEqual(risk_level(1.0, "percent"), "Low")
        self.assertEqual(risk_level(1.0, "ratio"), "High")
        self.assertEqual(reliability_score(61), 39.0)
        self.assertEqual(reliability_score(20, 0.75), 55.0)
        self.assertEqual(reliability_score(1.0, 0.6667, "percent"), 65.7)

    def test_product_risk_helpers_respect_explicit_percent_scale(self):
        product = {
            "sentiment_summary": {
                "risk_score": 1.0,
                "risk_score_scale": "percent",
                "sentiment_ratios": {"positive": 0.6667},
            }
        }
        self.assertEqual(product_risk_percent(product), 1.0)
        self.assertEqual(product_risk_level(product), "Low")
        self.assertEqual(product_reliability(product), 65.7)

    def test_product_reliability(self):
        same_category = {"link": "same", "category": "Audio", "sentiment_summary": {"risk_score": 10, "sentiment_ratios": {"positive": .7}, "review_count": 5}}
        self.assertEqual(product_reliability(same_category), 60.0)

    def test_keyword_counts_ignore_negated_evidence(self):
        product = {"comments": [
            {"sentiment_analysis": {"risk": {"evidence": [
                {"term": "Sira", "negated": False},
                {"term": "peke", "negated": True},
                {"term": "sira", "negated": False},
            ]}}},
            {"is_duplicate": True, "sentiment_analysis": {"risk": {"evidence": [
                {"term": "scam", "negated": False},
            ]}}},
        ]}
        self.assertEqual(risk_keyword_counts(product), {"sira": 1})

    def test_review_signal_counts_once_per_unique_review(self):
        product = {"comments": [
            {"sentiment_analysis": {"sentiment_evidence": [
                {"term": "Good", "negated": False},
                {"term": "good", "negated": False},
                {"term": "nice", "negated": True},
            ]}},
            {"sentiment_analysis": {"sentiment_evidence": [
                {"term": "good", "negated": False},
                {"term": "sulit", "negated": False},
            ]}},
            {"is_duplicate": True, "sentiment_analysis": {"sentiment_evidence": [
                {"term": "excellent", "negated": False},
            ]}},
        ]}
        self.assertEqual(review_signal_counts(product), {"good": 2, "sulit": 1})

    def test_weighted_risk_breakdown_uses_point_contributions(self):
        product = {"sentiment_summary": {
            "review_count": 10,
            "sentiment_counts": {"positive": 7, "neutral": 1, "negative": 2},
            "sentiment_ratios": {"positive": .7, "neutral": .1, "negative": .2},
            "risk_review_count": 1,
            "keyword_failure_rate": .1,
        }}
        breakdown = weighted_risk_breakdown(product)
        self.assertEqual(breakdown["negative_reviews"], 2)
        self.assertEqual(breakdown["risk_reviews"], 1)
        self.assertAlmostEqual(breakdown["negative_points"], 6.0)
        self.assertAlmostEqual(breakdown["defect_points"], 7.0)
        self.assertAlmostEqual(breakdown["total_points"], 13.0)

    def test_weighted_risk_breakdown_handles_no_reviews(self):
        breakdown = weighted_risk_breakdown({})
        self.assertEqual(breakdown["review_count"], 0)
        self.assertEqual(breakdown["total_points"], 0.0)


if __name__ == "__main__":
    unittest.main()
