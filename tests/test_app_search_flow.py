import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard_utils import sentiment_model_version


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def sample_products(count=6):
    products = []
    for index in range(count):
        products.append(
            {
                "link": f"https://shopee.ph/sample-{index}-i.1.{index + 1}",
                "name": f"Sample gadget {index + 1}",
                "price": f"₱{499 + index * 100}",
                "rating": 4.5,
                "platform": "shopee",
                "comments": [
                    {
                        "author": "Shopper",
                        "rating": 5,
                        "content": "Good product",
                        "sentiment_analysis": {
                            "model_version": sentiment_model_version(),
                            "label": "positive",
                            "risk": {"detected": False, "evidence": []},
                        },
                    }
                ],
                "sentiment_summary": {
                    "review_count": 1,
                    "risk_score": 10,
                    "risk_score_scale": "percent",
                    "keyword_failure_rate": 0,
                    "sentiment_counts": {"positive": 1, "neutral": 0, "negative": 0},
                    "sentiment_ratios": {"positive": 1, "neutral": 0, "negative": 0},
                },
            }
        )
    return products


class TestAppSearchFlow(unittest.TestCase):
    def assert_landing_destination_returns_home(self, button_label, expected_page):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.session_state["page"] = "Search"
        app.session_state["uploaded_products"] = sample_products(1)
        app.run()

        next(button for button in app.button if button.label == button_label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["page"], expected_page)
        self.assertEqual(app.session_state["navigation"], expected_page)

        next(button for button in app.button if button.label == "← Back to home").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["page"], "Search")
        self.assertEqual(app.session_state["navigation"], "Search")
        self.assertIn("Bago mo bilhin", " ".join(markdown.value for markdown in app.markdown))

    def test_reports_navigation_can_return_home(self):
        self.assert_landing_destination_returns_home("Reports", "Search results")

    def test_dashboard_navigation_can_return_home(self):
        self.assert_landing_destination_returns_home("Dashboard", "Admin overview")

    def test_landing_sidebar_hide_rule_is_scoped_to_home(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.session_state["page"] = "Search"
        app.session_state["uploaded_products"] = sample_products(1)
        app.run()
        styles = " ".join(markdown.value for markdown in app.markdown)

        self.assertFalse(app.exception)
        self.assertIn(
            'body:has(.st-key-landing_hero) section[data-testid="stSidebar"]',
            styles,
        )

    def test_dashboard_and_report_styles_follow_streamlit_theme(self):
        for page in ("Admin overview", "Feedback survey"):
            with self.subTest(page=page):
                app = AppTest.from_file(APP_PATH, default_timeout=90)
                app.session_state["page"] = page
                app.session_state["uploaded_products"] = sample_products(1)
                app.run()
                styles = " ".join(markdown.value for markdown in app.markdown)

                self.assertFalse(app.exception)
                self.assertIn("--def-bg:var(--background-color", styles)
                self.assertIn("--def-surface:var(--secondary-background-color", styles)
                self.assertIn("--def-text:var(--text-color", styles)
                self.assertIn("background:var(--def-surface)", styles)
                self.assertIn("color:var(--def-text)", styles)

    def test_shopper_can_choose_live_marketplace_scraping(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.session_state["page"] = "Landing"
        app.session_state["uploaded_products"] = sample_products()

        app.run()

        self.assertFalse(app.exception)
        self.assertTrue(any(widget.label == "Live marketplace scan" for widget in app.checkbox))
        self.assertTrue(any(widget.label == "Marketplace" for widget in app.selectbox))

    def test_shopper_can_use_saved_search_without_starting_scraper(self):
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.session_state["page"] = "Landing"
        app.session_state["uploaded_products"] = sample_products()
        app.run()

        next(widget for widget in app.text_input if widget.label == "Product link or search").input(
            "sample gadget"
        )
        next(widget for widget in app.checkbox if widget.label == "Live marketplace scan").uncheck()
        next(button for button in app.button if button.label == "Check now").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["page"], "Search")
        self.assertEqual(app.session_state["search_mode"], "name")

    def app_with_products(self, mode, products=None):
        products = products or sample_products()
        app = AppTest.from_file(APP_PATH, default_timeout=90)
        app.session_state["page"] = "Search"
        app.session_state["uploaded_products"] = products
        app.session_state["search_mode"] = mode
        app.session_state["search_result_query"] = "sample gadget"
        app.session_state["search_result_page"] = 0
        if mode == "link":
            app.session_state["inline_result_link"] = products[0]["link"]
        return app.run()

    def test_name_search_grid_opens_inline_report_and_returns(self):
        app = self.app_with_products("name")
        self.assertFalse(app.exception)
        self.assertIn("Results for", " ".join(markdown.value for markdown in app.markdown))
        view_buttons = [button for button in app.button if button.label == "Analyze reviews →"]
        self.assertEqual(len(view_buttons), 6)

        view_buttons[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Analysis for", " ".join(markdown.value for markdown in app.markdown))
        back = next(button for button in app.button if button.label == "← Back to product results")
        back.click().run()
        self.assertIn("Results for", " ".join(markdown.value for markdown in app.markdown))

    def test_link_search_opens_only_inline_report(self):
        app = self.app_with_products("link")
        text = " ".join(markdown.value for markdown in app.markdown)
        self.assertFalse(app.exception)
        self.assertIn("Analysis for", text)
        self.assertNotIn("Results for", text)
        self.assertFalse(any(button.label == "← Back to product results" for button in app.button))

    def test_analysis_header_shows_listing_image_and_rating_star(self):
        products = sample_products(1)
        products[0]["img"] = "//cdn.example.com/product.jpg"

        app = self.app_with_products("link", products)
        header = next(
            markdown.value
            for markdown in app.markdown
            if '<div class="analysis-product">' in markdown.value
        )

        self.assertFalse(app.exception)
        self.assertIn('src="https://cdn.example.com/product.jpg"', header)
        self.assertIn('class="rating-star"', header)
        self.assertIn("4.5 star rating", header)

    def test_analysis_header_uses_designed_image_placeholder(self):
        app = self.app_with_products("link", sample_products(1))
        header = next(
            markdown.value
            for markdown in app.markdown
            if '<div class="analysis-product">' in markdown.value
        )

        self.assertFalse(app.exception)
        self.assertIn("analysis-product-placeholder", header)
        self.assertIn("Listing image unavailable", header)

    def test_recommendation_action_has_readable_primary_styling(self):
        app = self.app_with_products("link", sample_products(1))
        styles = " ".join(markdown.value for markdown in app.markdown)

        self.assertFalse(app.exception)
        self.assertTrue(any(button.label.startswith("Find ") for button in app.button))
        self.assertIn('class*="recommendation_action"', styles)
        self.assertIn("background:#3E4784 !important", styles)
        self.assertIn("color:#FFF !important", styles)

    def test_one_percent_risk_is_rendered_as_low_not_high(self):
        products = sample_products(1)
        products[0]["sentiment_summary"].update(
            {
                "risk_score": 1.0,
                "risk_score_scale": "percent",
                "sentiment_ratios": {
                    "positive": 0.6667,
                    "neutral": 0.3,
                    "negative": 0.0333,
                },
            }
        )

        app = self.app_with_products("link", products)
        text = " ".join(markdown.value for markdown in app.markdown)

        self.assertFalse(app.exception)
        self.assertIn("Lower risk", text)
        self.assertNotIn("Not recommended", text)

    def test_empty_risk_terms_show_review_signals_and_clear_weighted_math(self):
        products = sample_products(1)
        products[0]["comments"][0]["sentiment_analysis"]["sentiment_evidence"] = [
            {"term": "good", "negated": False},
            {"term": "sulit", "negated": False},
        ]
        products[0]["sentiment_summary"].update(
            {"risk_score": 0, "risk_score_scale": "percent", "risk_review_count": 0}
        )

        app = self.app_with_products("link", products)
        text = " ".join(markdown.value for markdown in app.markdown)

        self.assertFalse(app.exception)
        self.assertIn("Weighted score breakdown", text)
        self.assertIn("0.0 + 0.0 contribution points", text)
        self.assertIn("Common review language", text)
        self.assertIn("No active risk terms in this sample", text)
        self.assertIn("not included in the risk score", text)

    def test_active_risk_terms_replace_context_fallback(self):
        products = sample_products(1)
        products[0]["comments"][0]["sentiment_analysis"].update(
            {
                "label": "negative",
                "risk": {
                    "detected": True,
                    "evidence": [{"term": "sira", "negated": False}],
                },
            }
        )
        products[0]["sentiment_summary"].update(
            {
                "risk_score": 100,
                "risk_score_scale": "percent",
                "risk_review_count": 1,
                "keyword_failure_rate": 1,
                "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 1},
                "sentiment_ratios": {"positive": 0, "neutral": 0, "negative": 1},
            }
        )

        app = self.app_with_products("link", products)
        text = " ".join(markdown.value for markdown in app.markdown)

        self.assertFalse(app.exception)
        self.assertIn("Detected defect terms", text)
        self.assertIn("RISK EVIDENCE", text)
        self.assertIn("sira", text)
        self.assertNotIn("Common review language", text)

    def test_analysis_shows_good_similar_product_recommendation(self):
        products = sample_products(2)
        products[0].update({"name": "Anker 25000mAh Power Bank", "brand": "Anker", "category": "Power Banks"})
        products[1].update({"name": "UGREEN 20000mAh Power Bank", "brand": "UGREEN", "category": "Power Banks", "rating": 4.8})
        products[1]["sentiment_summary"].update(
            {
                "review_count": 30,
                "risk_score": 5,
                "risk_score_scale": "percent",
                "sentiment_ratios": {"positive": 0.9, "neutral": 0.08, "negative": 0.02},
            }
        )

        app = self.app_with_products("link", products)
        text = " ".join(markdown.value for markdown in app.markdown)

        self.assertFalse(app.exception)
        self.assertIn("Recommended similar products", text)
        self.assertIn("UGREEN 20000mAh Power Bank", text)

    def test_name_grid_tolerates_duplicate_links_and_invalid_review_counts(self):
        products = sample_products(2)
        products[1]["link"] = products[0]["link"]
        products[1]["sentiment_summary"]["review_count"] = "unknown"

        app = self.app_with_products("name", products)

        self.assertFalse(app.exception)
        self.assertEqual(len([button for button in app.button if button.label == "Analyze reviews →"]), 2)


if __name__ == "__main__":
    unittest.main()
