import json
import unittest

from dashboard_utils import (
    analysis_is_current,
    badge_view,
    cached_product_matches,
    csv_bytes,
    detect_marketplace,
    load_products,
    has_usable_products,
    merge_product_catalogs,
    normalized_risk_score,
    price_value,
    platform_name,
    product_rows,
    product_reliability,
    rank_recommendations,
    reliability_score,
    review_rows,
    search_slug,
    sentiment_model_version,
)


def quote(text, span=None, subtype=None, confident=True, score=5.0, negated=False, rating=None):
    """One evidence sentence as the badge aggregator records it."""
    terms = [{"span": list(span), "negated": negated}] if span else []
    return {
        "text": text,
        "score": score,
        "high_confidence": confident,
        "hazard_subtype": subtype,
        "terms": terms,
        "rating": rating,
    }


def badge_summary(
    state="no_warning_signs",
    *,
    usable=40,
    safety=None,
    counterfeit=0,
    defect=0,
    detail="",
    positive_lb=0.8,
):
    """A minimal schema-2.0 summary carrying only the fields the view reads."""
    return {
        "schema_version": "defaketive-summary-2.0",
        "verdict": {"state": state, "detail": detail},
        "usable_review_count": usable,
        "positive_ratio_lower_bound": positive_lb,
        "safety": safety or {"flagged_reviews": 0, "hazard_subtypes": {}, "evidence": []},
        "flags": {
            "counterfeit": {"flagged_reviews": counterfeit, "evidence": []},
            "defect": {"flagged_reviews": defect, "evidence": []},
        },
    }


def fire_reports(count):
    return {
        "flagged_reviews": count,
        "hazard_subtypes": {"fire_explosion_swelling": {"reviews": count, "severity": 4}},
        "evidence": [
            quote(f"nasunog yung charger {index}", span=(0, 7), subtype="fire_explosion_swelling")
            for index in range(count)
        ],
    }


class TestBadgeView(unittest.TestCase):
    def test_safety_warning_says_what_reviewers_report(self):
        view = badge_view(
            {"sentiment_summary": badge_summary("safety_warning", safety=fire_reports(3))}
        )
        self.assertEqual(view["tone"], "high")
        self.assertEqual(view["headline"], "Reviewers report safety problems")
        self.assertEqual(
            view["detail"],
            "Of 40 reviews, 3 mention fire, explosion or a swollen battery. "
            "Read them below and judge for yourself.",
        )
        self.assertEqual(view["counts"], {"safety": 3, "counterfeit": 0, "defect": 0})

    def test_counts_agree_with_their_verbs(self):
        view = badge_view(
            {"sentiment_summary": badge_summary(safety=fire_reports(1), counterfeit=1, defect=2)}
        )
        summaries = {badge["category"]: badge["summary"] for badge in view["badges"]}
        self.assertEqual(summaries["safety"], "1 review mentions fire, explosion or a swollen battery")
        self.assertEqual(summaries["counterfeit"], "1 review says it may not be genuine")
        self.assertEqual(summaries["defect"], "2 reviews say it broke or did not work")

    def test_too_few_reviews_still_surfaces_a_report(self):
        # "Not enough reviews to judge" must not hide a fire report among them.
        view = badge_view(
            {
                "sentiment_summary": badge_summary(
                    "insufficient_evidence",
                    usable=4,
                    safety=fire_reports(1),
                    detail="Only 4 usable reviews; at least 30 are needed",
                )
            }
        )
        self.assertEqual(view["headline"], "Not enough reviews to judge")
        self.assertEqual(
            view["detail"],
            "Only 4 usable reviews; at least 30 are needed. Even so, 1 review mentions "
            "fire, explosion or a swollen battery. Read it below.",
        )

    def test_a_clean_result_is_never_called_safe(self):
        view = badge_view(
            {
                "sentiment_summary": badge_summary(
                    detail="No warning signs found in 40 reviews. Absence of complaints "
                    "is not proof that a product is safe."
                )
            }
        )
        self.assertEqual(view["headline"], "No warning signs found")
        self.assertIn("not proof", view["detail"])
        for text in [view["headline"]] + [badge["summary"] for badge in view["badges"]]:
            self.assertNotIn("safe", text.casefold().split())

    def test_an_isolated_report_is_not_described_as_no_complaints(self):
        view = badge_view(
            {
                "sentiment_summary": badge_summary(
                    counterfeit=1,
                    detail="No warning signs found in 40 reviews. Absence of complaints "
                    "is not proof that a product is safe.",
                )
            }
        )
        self.assertEqual(view["headline"], "No warning signs found")
        self.assertNotIn("Absence of complaints", view["detail"])
        self.assertEqual(
            view["detail"],
            "No problem reaches a warning level in 40 reviews, but 1 review says it may not "
            "be genuine. Read it below. This is not proof that a product is safe.",
        )

    def test_the_most_serious_hazard_is_named_and_others_noted(self):
        safety = {
            "flagged_reviews": 3,
            "hazard_subtypes": {
                "smoke_smell": {"reviews": 2, "severity": 3},
                "shock_spark_short": {"reviews": 1, "severity": 4},
            },
            "evidence": [],
        }
        badge = badge_view({"sentiment_summary": badge_summary(safety=safety)})["badges"][0]
        self.assertEqual(
            badge["summary"],
            "3 reviews mention electric shocks, sparks or short circuits, and other hazards",
        )

    def test_evidence_is_distinct_capped_and_ranked_by_confidence(self):
        evidence = (
            [quote("weak one", confident=False, score=9.0)]
            + [quote(f"strong {index}", score=float(index)) for index in range(6)]
            + [quote("Strong 5")]
        )
        safety = {"flagged_reviews": 8, "hazard_subtypes": {}, "evidence": evidence}
        badge = badge_view({"sentiment_summary": badge_summary(safety=safety)})["badges"][0]
        texts = [item["text"] for item in badge["evidence"]]
        self.assertEqual(len(texts), 5)
        self.assertEqual(texts[0], "strong 5")
        self.assertEqual(texts.count("Strong 5") + texts.count("strong 5"), 1)
        self.assertNotIn("weak one", texts)  # confidence outranks a higher score
        # Seven distinct quotes ("Strong 5" repeats "strong 5"), five shown.
        self.assertEqual(badge["more_evidence"], 2)

    def test_only_active_terms_are_highlighted(self):
        safety = {
            "flagged_reviews": 1,
            "hazard_subtypes": {},
            "evidence": [quote("hindi nasunog", span=(6, 13), negated=True)],
        }
        badge = badge_view({"sentiment_summary": badge_summary(safety=safety)})["badges"][0]
        self.assertEqual(badge["evidence"][0]["highlights"], [])

    def test_a_legacy_summary_is_reported_as_not_analyzed(self):
        view = badge_view({"sentiment_summary": {"risk_score": 12, "risk_score_scale": "percent"}})
        self.assertFalse(view["available"])
        self.assertEqual(view["headline"], "Not analyzed yet")

    def test_export_rows_carry_badges_not_scores(self):
        row = product_rows(
            [
                {
                    "name": "X",
                    "comments": [],
                    "sentiment_summary": badge_summary(
                        "safety_warning", safety=fire_reports(2), defect=1
                    ),
                }
            ]
        )[0]
        self.assertEqual(row["Verdict"], "Reviewers report safety problems")
        self.assertEqual((row["Safety reports"], row["Defect reports"]), (2, 1))
        self.assertNotIn("Risk score", row)
        self.assertNotIn("Reliability score", row)

    def test_a_current_badge_analysis_is_not_reanalysed(self):
        product = {
            "comments": [{"sentiment_analysis": {"model_version": sentiment_model_version()}}],
            "sentiment_summary": badge_summary(),
        }
        self.assertTrue(analysis_is_current([product]))


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

    def test_recommendations_are_similar_and_show_no_warning_signs(self):
        def listing(link, name, summary, category="Power Banks", rating=4.8, brand=""):
            return {
                "link": link,
                "name": name,
                "brand": brand,
                "category": category,
                "rating": rating,
                "sentiment_summary": summary,
            }

        target = listing(
            "target",
            "Anker 25000mAh Laptop Power Bank",
            badge_summary("safety_warning", safety=fire_reports(3)),
            brand="Anker",
        )
        good = listing("good", "UGREEN 20000mAh USB-C Power Bank", badge_summary())
        risky = listing(
            "risky",
            "Anker Power Bank",
            badge_summary("safety_warning", safety=fire_reports(2)),
            brand="Anker",
        )
        # "No warning signs" overall, yet one review reports a fire: never
        # offer that as the safer choice.
        one_fire = listing(
            "one-fire", "Baseus 20000mAh Power Bank", badge_summary(safety=fire_reports(1))
        )
        thin_praise = listing("thin", "Romoss 20000mAh Power Bank", badge_summary(positive_lb=0.3))
        # A legacy score cannot vouch for a product under the current checks.
        legacy = listing(
            "legacy",
            "Xiaomi 20000mAh Power Bank",
            {"risk_score": 1, "risk_score_scale": "percent", "review_count": 50},
        )
        unrelated = listing(
            "unrelated",
            "Raspberry Pi Development Board",
            badge_summary(),
            category="Microcontrollers",
            rating=5,
        )

        recommendations = rank_recommendations(
            target, [target, risky, one_fire, thin_praise, legacy, unrelated, good], limit=3
        )

        self.assertEqual([item["link"] for item in recommendations], ["good"])

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
        self.assertEqual(reliability_score(61), 39.0)
        self.assertEqual(reliability_score(20, 0.75), 55.0)
        self.assertEqual(reliability_score(1.0, 0.6667, "percent"), 65.7)

    def test_product_reliability_respects_explicit_percent_scale(self):
        product = {
            "sentiment_summary": {
                "risk_score": 1.0,
                "risk_score_scale": "percent",
                "sentiment_ratios": {"positive": 0.6667},
            }
        }
        self.assertEqual(product_reliability(product), 65.7)

    def test_product_reliability(self):
        same_category = {"link": "same", "category": "Audio", "sentiment_summary": {"risk_score": 10, "sentiment_ratios": {"positive": .7}, "review_count": 5}}
        self.assertEqual(product_reliability(same_category), 60.0)


if __name__ == "__main__":
    unittest.main()
