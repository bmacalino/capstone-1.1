import json
import subprocess
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
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
        "schema_version": "defaketive-summary-2.0",
        "verdict": {
            "state": "insufficient_evidence",
            "detail": "Only 1 usable reviews; at least 30 are needed",
        },
        "usable_review_count": 1,
        "safety": {"flagged_reviews": 0, "hazard_subtypes": {}, "evidence": []},
        "flags": {
            "counterfeit": {"flagged_reviews": 0, "evidence": []},
            "defect": {"flagged_reviews": 0, "evidence": []},
        },
        "sentiment_counts": {"positive": 1, "neutral": 0, "negative": 0},
    },
}


def fire_product(link="https://shopee.ph/fire-i.1.3"):
    """A badge summary whose reviews report fires, and one defect."""
    fire = {
        "text": "nasunog yung charger pagka saksak",
        "score": 3.0,
        "high_confidence": True,
        "hazard_subtype": "fire_explosion_swelling",
        "terms": [{"span": [0, 7], "negated": False}],
    }
    return {
        **SAMPLE_PRODUCT,
        "link": link,
        "name": "VoltMax power bank",
        "sentiment_summary": {
            **SAMPLE_PRODUCT["sentiment_summary"],
            "verdict": {"state": "safety_warning", "detail": "Two independent fire reports"},
            "usable_review_count": 40,
            "safety": {
                "flagged_reviews": 3,
                "hazard_subtypes": {"fire_explosion_swelling": {"reviews": 3, "severity": 4}},
                "evidence": [fire],
            },
            "flags": {
                "counterfeit": {"flagged_reviews": 0, "evidence": []},
                "defect": {"flagged_reviews": 1, "evidence": []},
            },
        },
    }


class TestWebApp(unittest.TestCase):
    def test_verification_block_has_clear_message_without_traceback(self):
        message = webapp._user_facing_failure("Traceback: SHOPEE_VERIFICATION_BLOCKED: failure")
        self.assertIn("Wait before retrying", message)
        self.assertNotIn("Traceback", message)

    def setUp(self):
        webapp._catalogs.clear()
        webapp._jobs.clear()
        # A fresh accounts database per test, so guest scans never carry over.
        accounts_folder = tempfile.TemporaryDirectory()
        self.addCleanup(accounts_folder.cleanup)
        self.app = webapp.create_app({"TESTING": True, "SECRET_KEY": "test", "ACCOUNTS_DB": str(Path(accounts_folder.name) / "accounts.db")})
        self.client = self.app.test_client()
        with self.client.session_transaction() as state:
            state["catalog_key"] = "webapp-test"
        webapp._catalogs["webapp-test"] = [SAMPLE_PRODUCT]

    def test_serves_html_shell_and_current_catalog(self):
        page = self.client.get("/")
        catalog = self.client.get("/api/products")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"DeFaketive", page.data)
        self.assertNotIn(b"Check another product", page.data)
        self.assertIn(b"Analyze another product", page.data)
        self.assertEqual(catalog.status_code, 200)
        self.assertEqual(catalog.json["count"], 1)

    def test_search_interface_has_smart_actions_feedback_and_progress(self):
        page = self.client.get("/")
        script = self.client.get("/static/js/app.js")
        script_data = script.get_data()
        script.close()

        self.assertIn(b"SCAN A PRODUCT", page.data)
        self.assertIn(b"Paste a product link", page.data)
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
        search_form = page.data.index(b'id="search-form"')
        self.assertLess(page.data.index(b'id="job-panel"', search_form), page.data.index(b'</form>', search_form))
        self.assertIn(b"function queryContext", script_data)
        self.assertIn(b"function savedProductForQuery", script_data)
        self.assertIn(b"event.submitter", script_data)
        self.assertIn(b"Refresh with new reviews", script_data)

    def test_history_replaces_dashboard_and_overview_api_remains_available(self):
        page = self.client.get("/")
        overview = self.client.get("/api/admin/overview")

        self.assertIn(b"Analysis history", page.data)
        self.assertIn(b'data-view="analyze"', page.data)
        self.assertNotIn(b'data-view="admin"', page.data)
        self.assertNotIn(b"Shopping insights", page.data)
        self.assertNotIn(b"Scraper manager", page.data)
        self.assertNotIn(b"Lexicon dictionary", page.data)
        self.assertEqual(
            overview.json["summary"],
            {
                "products": 1,
                "reviews": 1,
                "safety_warnings": 0,
                "with_safety_reports": 0,
                "not_analyzed": 0,
                "marketplaces": 1,
            },
        )

    def test_landing_page_routes_sign_in_to_figma_home(self):
        page = self.client.get("/")
        script_response = self.client.get("/static/js/app.js")
        script = script_response.get_data()
        script_response.close()

        self.assertIn(b"SPOT FAKES &amp;<br>DEFECTS<br>BEFORE YOU BUY", page.data)
        self.assertIn(b'id="landing" class="view active pub-landing"', page.data)
        self.assertIn(b"css/public.css", page.data)
        self.assertIn(b'id="workspace-shell"', page.data)
        self.assertIn(b'data-view="home"', page.data)
        self.assertIn(b"data-signin data-auth-guest>Sign In", page.data)
        self.assertIn(b'id="search-form"', page.data)
        self.assertIn(b'id="platforms"', page.data)
        self.assertIn(b"THE MINDS<br>BEHIND A SAFER<br>SHOPPING EXPERIENCE.", page.data)
        self.assertIn(b"assets/landing/hero.jpg", page.data)
        self.assertNotIn(b"data-explore-tab", page.data)
        self.assertNotIn(b"function activateExploreTab", script)
        self.assertIn(b"Recently analyzed", page.data)
        # The app frame: a top bar and a floating icon rail replace the old sidebar.
        self.assertIn(b'class="app-topbar"', page.data)
        self.assertIn(b'class="app-rail" aria-label="Workspace navigation"', page.data)
        self.assertNotIn(b"workspace-sidebar", page.data)
        self.assertIn(b'id="home-overview"', page.data)
        self.assertIn(b'id="home-stores"', page.data)
        # Signed-in history and wishlist are kept in the account, not the browser.
        self.assertIn(b"function loadLibrary", script)
        self.assertIn(b'if (state.user) return;', script)
        self.assertIn(b"Too few reviews", script)
        self.assertIn(b'data-view="history"', page.data)
        self.assertIn(b'data-view="wishlist"', page.data)
        self.assertIn(b'data-view="settings"', page.data)
        self.assertNotIn(b'data-home-report=', page.data)
        self.assertNotIn(b'data-workspace-placeholder=', page.data)
        self.assertIn(b'class="app-brand-mark"', page.data)
        self.assertNotIn(b'id="dashboard"', page.data)
        self.assertNotIn(b'data-view="dashboard"', page.data)
        self.assertNotIn(b"Your safer-shopping dashboard", page.data)
        self.assertNotIn(b"function renderHomeDashboard", script)
        self.assertIn(b"function openSignedInPage", script)
        # Sign In opens the sign-in pop-up, or Home when already signed in.
        self.assertIn(b'async function openSignedInPage() {\n  if (state.user) go("home");\n  else openAuth("signin");', script)
        for marker in (b'data-view="login"', b'id="signin-dialog"', b'id="signup-dialog"', b'id="terms-dialog"',
                       b'autocomplete="current-password"', b'autocomplete="new-password"', b'data-auth-open="signup"'):
            self.assertIn(marker, page.data)
        # "Forgot password?" stays hidden until the app can send email.
        self.assertNotIn(b"Forgot", page.data)
        self.assertIn(b'if (view === "dashboard") view = "home"', script)
        self.assertIn(b"context.isUrl && state.products.length === 1", script)

    def test_dashboard_reports_a_products_main_risk_issue(self):
        webapp._catalogs["webapp-test"] = [SAMPLE_PRODUCT, fire_product()]

        overview = self.client.get("/api/admin/overview")

        self.assertEqual(overview.json["summary"]["safety_warnings"], 1)
        self.assertEqual(overview.json["summary"]["with_safety_reports"], 1)
        flagged = overview.json["flagged"]
        self.assertEqual(len(flagged), 1)
        # Three safety reports outnumber the one defect report.
        self.assertEqual(flagged[0]["Main issue"], "Safety")
        self.assertEqual(flagged[0]["Verdict"], "Reviewers report safety problems")

    def test_a_product_with_fire_reports_is_served_as_a_safety_warning(self):
        # Regression: the screens read a risk_score the badge model does not
        # produce, defaulted it to 0, and showed fire reports as "Low risk".
        product = fire_product()
        webapp._catalogs["webapp-test"] = [product]

        listed = self.client.get("/api/products").json["products"][0]["badge_view"]
        detail = self.client.get(
            "/api/products/detail", query_string={"link": product["link"]}
        ).json["product"]["badge_view"]

        for view in (listed, detail):
            self.assertEqual(view["tone"], "high")
            self.assertEqual(view["headline"], "Reviewers report safety problems")
            self.assertEqual(view["counts"]["safety"], 3)
        quote = detail["badges"][0]["evidence"][0]
        self.assertEqual(quote["text"], "nasunog yung charger pagka saksak")
        self.assertEqual(quote["highlights"], [[0, 7]])

    def test_exported_products_carry_badges_not_scores(self):
        webapp._catalogs["webapp-test"] = [fire_product()]
        response = self.client.get("/api/export/products.csv")
        header = response.get_data().decode("utf-8-sig").splitlines()[0]
        response.close()
        self.assertIn("Safety reports", header)
        self.assertNotIn("Risk score", header)
        self.assertNotIn("Reliability", header)

    def test_analysis_ui_explains_scores_and_filters_review_evidence(self):
        response = self.client.get("/static/js/app.js")
        script = response.get_data()
        response.close()

        # The report quotes what reviewers said and explains why there is no score.
        self.assertIn(b"Three checks, in their own words", script)
        self.assertIn(b"Why you see reviews, not a score", script)
        self.assertIn(b"no flags is not proof of safety", script)
        self.assertIn(b"function highlightSpans", script)
        self.assertIn(b'data-review-filter="reported"', script)
        self.assertIn(b'data-review-filter="uncertain"', script)
        self.assertIn(b'id="review-search"', script)
        self.assertIn(b"duplicates excluded", script)
        # Nothing of the retired 0-100 score survives, including the 30/70
        # formula it was built from and the "safe to buy" clearance.
        for retired in (
            b"risk_score",
            b"Lower is better",
            b"Thirty percent comes from the negative-review ratio",
            b"weighted risk score",
            b"Generally safe to buy",
            b'data-review-filter="risk"',
        ):
            self.assertNotIn(retired, script)
        self.assertNotIn(b'id="sentiment-chart"', script)
        # The redesigned report keeps the Figma layout but none of its score-era
        # content: no final score, no score table, no invented aspect scores,
        # no "safe to buy" or "avoid", and no reviewer name the data never had.
        for invented in (b"Final risk score", b"Score Components", b"Safe to buy", b"Aspect Scores",
                         b"Verified shopper", b"/ 100", b"Recommended Alternatives", b"Confidence", b"AiAssistant"):
            self.assertNotIn(invented, script)
        for honest in (b"How DeFaketive Labels a Product", b"Warning words found", b"What the check looks for",
                       b"Duplicates excluded", b"No warning signs", b"function warningWords", b"term.negated"):
            self.assertIn(honest, script)

    def test_guest_report_has_the_out_of_scans_notice(self):
        page = self.client.get("/")
        script_response = self.client.get("/static/js/app.js")
        script = script_response.get_data()
        script_response.close()
        self.assertIn(b'id="guest-limit-notice"', page.data)
        self.assertIn("You’ve used today’s free scans".encode("utf-8"), page.data)
        self.assertNotIn(b"credit limit", page.data)
        # Shown only to a guest with no scans left, on a report.
        self.assertIn(b'!state.user && Number(state.guest?.left) === 0 && currentView() === "analysis"', script)

    def test_top_navigation_is_shopper_focused_and_accessible(self):
        page = self.client.get("/")
        script_response = self.client.get("/static/js/app.js")
        script = script_response.get_data()
        script_response.close()

        # The logo is the way home; Products opens the list of checked products.
        self.assertIn(b'data-route="results">Products</a>', page.data)
        self.assertIn(b'>About</a>', page.data)
        self.assertIn(b'>FAQ</a>', page.data)
        self.assertIn(b'aria-label="Open navigation"', page.data)
        self.assertIn(b'class="nav-toggle-line"', page.data)
        self.assertNotIn(b'>Reports</a>', page.data)
        self.assertNotIn(b'>Dashboard</a>', page.data)
        self.assertNotIn(b'>Insights</a>', page.data)
        self.assertNotIn(b'>Check a product</a>', page.data)
        self.assertIn(b"function setActiveNavigation", script)
        self.assertIn(b'event.key === "Escape"', script)
        self.assertIn(b"focusProductSearch", script)

    def test_pages_never_describe_the_retired_score_or_invented_numbers(self):
        # The Figma screens still carried the old 0-1 score, its 30/70 formula,
        # marketplaces the app does not support, and made-up statistics.
        page = self.client.get("/").get_data(as_text=True)
        for claim in (
            "Risk Score", "KFR", "NLTK", "lexicon-based", "Amazon", "TikTok", "verified",
            "Reliability Score", "4,821", "28,493", "96.2%", "Flagged as Avoid",
            "Safe, Caution, Avoid", "90 days", "Free Plan", "community favorites", "Watch Demo",
        ):
            self.assertNotIn(claim, page)

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

    def test_lazada_scans_read_sixty_reviews_from_each_star_level(self):
        command, _target = webapp._command_for_scrape(
            "https://www.lazada.com.ph/products/pdp-i4644812363-s1.html", "lazada", 1, 100
        )
        self.assertIn("--all-star-types", command)
        self.assertEqual(command[command.index("--star-limit-per-type") + 1], "60")

    def test_other_marketplaces_keep_the_review_count(self):
        command, _target = webapp._command_for_scrape("earphones", "temu", 1, 40)
        self.assertNotIn("--all-star-types", command)
        self.assertEqual(command[command.index("-r") + 1], "40")

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



class TestModelUnavailable(unittest.TestCase):
    """A missing model must be explained, never shown as an empty catalog."""

    def setUp(self):
        webapp._catalogs.clear()
        self.addCleanup(webapp._catalogs.clear)
        self.client = webapp.create_app({"TESTING": True, "SECRET_KEY": "test"}).test_client()
        self.missing = webapp.ModelUnavailableError(
            "The sentiment model is not installed. Build it with research/slim_sentiment_model.py"
        )

    def results_file(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        target = Path(folder.name) / "results.json"
        target.write_bytes(json.dumps([SAMPLE_PRODUCT]).encode("utf-8"))
        return target

    def test_the_catalog_reports_the_missing_model_instead_of_an_empty_list(self):
        webapp._analysed_files.clear()
        with patch.object(webapp, "result_files", return_value=[self.results_file()]),                 patch.object(webapp, "load_saved_products", side_effect=AssertionError("must not fall through")),                 patch.object(webapp, "sentiment_model_version", return_value="unavailable"),                 patch.object(webapp, "analysis_is_current", return_value=False),                 patch.object(webapp, "analyze_products", side_effect=self.missing):
            response = self.client.get("/api/products")
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json["model_unavailable"])
        self.assertIn("Nothing was analyzed", response.json["error"])
        self.assertIn("slim_sentiment_model.py", response.json["detail"])

    def test_a_live_scan_is_refused_before_any_scraping(self):
        with patch.object(webapp, "require_model", side_effect=self.missing),                 patch.object(webapp, "_start_job") as start:
            response = self.client.post(
                "/api/search",
                json={"query": "https://www.lazada.com.ph/products/pdp-i1234567.html", "platform": "lazada", "live": True},
            )
        self.assertEqual(response.status_code, 503)
        start.assert_not_called()

    def test_an_upload_reports_the_missing_model(self):
        with patch.object(webapp, "_set_products", side_effect=self.missing):
            response = self.client.post(
                "/api/upload",
                data={"file": (BytesIO(json.dumps([SAMPLE_PRODUCT]).encode("utf-8")), "results.json")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json["model_unavailable"])

    def test_a_results_file_is_analysed_once_for_all_sessions(self):
        # The transformer takes ~19 s per 100 reviews; each new browser session
        # used to re-analyse the same file.
        webapp._analysed_files.clear()
        self.addCleanup(webapp._analysed_files.clear)
        with patch.object(webapp, "result_files", return_value=[self.results_file()]),                 patch.object(webapp, "sentiment_model_version", return_value="v-test"),                 patch.object(webapp, "analysis_is_current", return_value=False),                 patch.object(webapp, "analyze_products", side_effect=lambda products: products) as analyze:
            for key in ("session-a", "session-b"):
                with self.client.session_transaction() as state:
                    state["catalog_key"] = key
                self.assertEqual(self.client.get("/api/products").json["count"], 1)
        self.assertEqual(analyze.call_count, 1)

    def test_the_home_page_shows_the_reason(self):
        script = self.client.get("/static/js/app.js")
        data = script.get_data()
        script.close()
        self.assertIn(b"Review analysis is unavailable", data)
        self.assertIn(b"model_unavailable", data)


if __name__ == "__main__":
    unittest.main()
