import subprocess
import unittest
from io import BytesIO
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
        self.assertIn(b"Analyzed products", page.data)
        self.assertNotIn(b"Check another product", page.data)
        self.assertIn(b"Back to analyzed products", page.data)
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json["count"], 1)

    def test_search_interface_has_smart_actions_feedback_and_progress(self):
        page = self.client.get("/")
        script = self.client.get("/static/js/app.js")
        script_data = script.get_data()
        script.close()

        self.assertIn(b"What product do you want to check?", page.data)
        self.assertNotIn(b'name="search-mode"', page.data)
        self.assertIn(b'name="marketplace" value="shopee"', page.data)
        self.assertIn(b'name="marketplace" value="lazada"', page.data)
        self.assertIn(b'name="marketplace" value="temu"', page.data)
        self.assertIn(b'id="advanced-options"', page.data)
        self.assertIn(b'id="saved-match"', page.data)
        self.assertIn(b'data-search-intent="saved"', page.data)
        self.assertIn(b'data-search-intent="live"', page.data)
        self.assertIn(b'id="marketplace-detection"', page.data)
        self.assertIn(b'id="search-feedback"', page.data)
        self.assertIn(b'data-job-stage="1"', page.data)
        self.assertIn(b'id="job-elapsed"', page.data)
        self.assertLess(page.data.index(b'id="job-panel"'), page.data.index(b'</form>'))
        self.assertIn(b"function queryContext", script_data)
        self.assertIn(b"function savedProductForQuery", script_data)
        self.assertIn(b"event.submitter", script_data)
        self.assertIn(b"Refresh with new reviews", script_data)

    def test_dashboard_is_shopper_focused(self):
        page = self.client.get("/")
        overview = self.client.get("/api/admin/overview")

        self.assertIn(b"Shopping insights", page.data)
        self.assertIn(b"simple overview", page.data)
        self.assertIn(b"All analyzed products", page.data)
        self.assertIn(b"Update insights", page.data)
        self.assertNotIn(b"Scraper manager", page.data)
        self.assertNotIn(b"Lexicon dictionary", page.data)
        self.assertEqual(
            overview.json["summary"],
            {
                "products": 1,
                "reviews": 1,
                "average_reliability": 90.0,
                "high_risk": 0,
                "marketplaces": 1,
            },
        )

    def test_dashboard_reports_a_products_main_risk_issue(self):
        product = {
            **SAMPLE_PRODUCT,
            "sentiment_summary": {
                **SAMPLE_PRODUCT["sentiment_summary"],
                "risk_score": 80,
            },
            "comments": [
                {
                    "content": "Sira agad",
                    "sentiment_analysis": {
                        "risk": {
                            "detected": True,
                            "categories": ["defect"],
                            "evidence": [
                                {"category": "defect", "term": "sira", "negated": False}
                            ],
                        }
                    },
                }
            ],
        }
        webapp._catalogs["webapp-test"] = [product]

        overview = self.client.get("/api/admin/overview")

        self.assertEqual(overview.json["summary"]["high_risk"], 1)
        self.assertEqual(overview.json["flagged"][0]["Main issue"], "Defect")

    def test_analysis_ui_explains_scores_and_filters_review_evidence(self):
        response = self.client.get("/static/js/app.js")
        script = response.get_data()
        response.close()

        self.assertIn(b"Lower is better", script)
        self.assertIn(b"Higher is better", script)
        self.assertIn(b"Reviews containing risk signals", script)
        self.assertIn(b"Top reported issues", script)
        self.assertIn(b'data-review-filter="risk"', script)
        self.assertIn(b'id="review-search"', script)
        self.assertIn(b"Duplicates excluded", script)
        self.assertNotIn(b"Active defect/fraud terms", script)
        self.assertNotIn(b'id="sentiment-chart"', script)

    def test_top_navigation_is_shopper_focused_and_accessible(self):
        page = self.client.get("/")
        script_response = self.client.get("/static/js/app.js")
        script = script_response.get_data()
        script_response.close()

        self.assertIn(b'>Home</a>', page.data)
        self.assertIn(b'>Analyzed products</a>', page.data)
        self.assertIn(b'>Insights</a>', page.data)
        self.assertIn(b'data-focus-search>Check a product</a>', page.data)
        self.assertIn(b'aria-label="Open navigation"', page.data)
        self.assertIn(b'class="nav-toggle-line"', page.data)
        self.assertNotIn(b'>Reports</a>', page.data)
        self.assertNotIn(b'>Dashboard</a>', page.data)
        self.assertIn(b"function setActiveNavigation", script)
        self.assertIn(b'event.key === "Escape"', script)
        self.assertIn(b"focusProductSearch", script)

    def test_product_link_marketplace_overrides_selected_marketplace(self):
        job = {
            "id": "temu-job",
            "state": "running",
            "stage": 1,
            "platform": "temu",
        }
        with patch("webapp._start_job", return_value=job) as start_job:
            response = self.client.post(
                "/api/search",
                json={
                    "query": "https://www.temu.com/ph-en/goods.html",
                    "platform": "lazada",
                    "live": True,
                    "product_count": 1,
                    "review_count": 30,
                },
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json["job"]["platform"], "temu")
        start_job.assert_called_once_with(
            "webapp-test",
            "https://www.temu.com/ph-en/goods.html",
            "temu",
            1,
            30,
        )

    def test_job_exposes_limits_and_progress_metadata(self):
        with patch("webapp.threading.Thread") as thread:
            job = webapp._start_job("catalog", "earphones", "lazada", 4, 25)

        self.assertEqual(job["platform"], "lazada")
        self.assertEqual(job["product_limit"], 4)
        self.assertEqual(job["review_limit"], 25)
        self.assertEqual(job["stage_label"], "Collect reviews")
        self.assertEqual(job["progress"], 18)
        thread.return_value.start.assert_called_once()

    def test_saved_search_and_product_detail_are_available(self):
        response = self.client.post("/api/search", json={"query": "sample gadget", "live": False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["count"], 1)
        detail = self.client.get("/api/products/detail", query_string={"link": SAMPLE_PRODUCT["link"]})
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json["product"]["name"], "Sample gadget")

    def test_lazada_live_search_uses_clean_browser_session(self):
        command, _target = webapp._command_for_scrape("earphones", "lazada", 1, 1)

        self.assertIn("earphones", command)
        self.assertNotIn("--chrome-user-data-dir", command)

    def test_network_access_denied_has_actionable_launcher_message(self):
        result = subprocess.CompletedProcess(
            args=["python", "lazada_scraper.py"],
            returncode=1,
            stdout="unknown error: net::ERR_NETWORK_ACCESS_DENIED",
            stderr="",
        )

        message = webapp._scraper_failure_message(result)

        self.assertIn("ERR_NETWORK_ACCESS_DENIED", message)
        self.assertIn("run_capstone.cmd", message)
        self.assertNotIn("unknown error", message)

    def test_other_scraper_errors_keep_the_original_detail(self):
        result = subprocess.CompletedProcess(
            args=["python", "scraper.py"],
            returncode=1,
            stdout="No real products were extracted",
            stderr="",
        )

        self.assertEqual(
            webapp._scraper_failure_message(result),
            "No real products were extracted",
        )

    def test_background_job_exception_also_normalizes_network_denial(self):
        self.assertEqual(
            webapp._user_facing_failure(
                "RuntimeError: navigation failed: net::ERR_NETWORK_ACCESS_DENIED"
            ),
            webapp.NETWORK_ACCESS_DENIED_MESSAGE,
        )

    def test_survey_uses_existing_research_scoring(self):
        with patch("webapp.save_survey_response") as save:
            response = self.client.post("/api/survey", json={"frequency": "Weekly", "sus": [3] * 10, "umux": [4] * 4})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sus_score"], 50.0)
        self.assertEqual(response.json["umux_score"], 50.0)
        save.assert_called_once()

    def test_evaluation_accepts_firecs_columns_and_numeric_labels(self):
        dataset = b"review,label\nsira at hindi gumagana,0.0\nmaganda at sulit,2.0\n"
        with patch("webapp.save_evaluation_run") as save:
            response = self.client.post(
                "/api/evaluation",
                data={"file": (BytesIO(dataset), "FiReCS_sample.csv")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["sample_count"], 2)
        self.assertEqual(response.json["labels"], ["negative", "positive"])
        save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
