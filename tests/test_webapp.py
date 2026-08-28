import unittest
from unittest.mock import patch

import webapp


SAMPLE_PRODUCT = {
    "link": "https://shopee.ph/sample-i.1.2",
    "name": "Sample gadget",
    "price": "₱499",
    "rating": 4.5,
    "platform": "shopee",
    "comments": [{"author": "Shopper", "rating": 5, "content": "Good product"}],
    "sentiment_summary": {
        "review_count": 1,
        "risk_score": 10,
        "risk_score_scale": "percent",
        "risk_review_count": 0,
        "keyword_failure_rate": 0,
        "sentiment_counts": {"positive": 1, "neutral": 0, "negative": 0},
        "sentiment_ratios": {"positive": 1, "neutral": 0, "negative": 0},
    },
}


class TestWebApp(unittest.TestCase):
    def setUp(self):
        webapp._catalogs.clear()
        webapp._jobs.clear()
        self.app = webapp.create_app({"TESTING": True, "SECRET_KEY": "test"})
        self.client = self.app.test_client()
        with self.client.session_transaction() as state:
            state["catalog_key"] = "webapp-test"
        webapp._catalogs["webapp-test"] = [SAMPLE_PRODUCT]

    def test_serves_html_shell_and_current_catalog(self):
        page = self.client.get("/")
        catalog = self.client.get("/api/products")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"DeFaketive", page.data)
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json["count"], 1)

    def test_saved_search_and_product_detail_are_available(self):
        response = self.client.post("/api/search", json={"query": "sample gadget", "live": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["count"], 1)
        detail = self.client.get("/api/products/detail", query_string={"link": SAMPLE_PRODUCT["link"]})
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json["product"]["name"], "Sample gadget")

    def test_survey_uses_existing_research_scoring(self):
        with patch("webapp.save_survey_response") as save:
            response = self.client.post("/api/survey", json={"frequency": "Weekly", "sus": [3] * 10, "umux": [4] * 4})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sus_score"], 50.0)
        self.assertEqual(response.json["umux_score"], 50.0)
        save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
