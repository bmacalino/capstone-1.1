import sys
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from selenium.common.exceptions import TimeoutException

from src.retriv import ShopeeScraper, _position_scraper_window, _progress, build_parser


class TestShopeeScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = ShopeeScraper(
            search_term="test",
            max_products=5,
            index_only=False,
            review_limit=10,
            all_star_types=False,
            star_limit_per_type=5,
        )

    def test_disables_progress_output_when_stderr_is_not_interactive(self):
        captured_stderr = Mock()
        captured_stderr.isatty.return_value = False
        products = [{"link": "https://shopee.ph/sample-i.1.2"}]
        with patch.object(sys, "stderr", captured_stderr), patch("src.retriv.tqdm") as tqdm:
            _progress(products, desc="Processing products")

        tqdm.assert_called_once_with(products, desc="Processing products", disable=True)

    def test_initialization(self):
        self.assertEqual(self.scraper.search_term, "test")
        self.assertEqual(self.scraper.max_products, 5)
        self.assertFalse(self.scraper.index_only)
        self.assertEqual(self.scraper.review_limit, 10)
        self.assertEqual(self.scraper.out_file, "shopee_ph_test.json")
        self.assertIn("--window-size=900,700", self.scraper.options.arguments)
        self.assertNotIn("--start-maximized", self.scraper.options.arguments)
        self.assertNotIn("--disable-blink-features=AutomationControlled", self.scraper.options.arguments)
        self.assertNotIn("--no-sandbox", self.scraper.options.arguments)

    @patch("src.retriv.webdriver.Chrome")
    @patch.object(ShopeeScraper, "_cached_driver", return_value="matching-driver.exe")
    @patch.object(ShopeeScraper, "_detect_chrome_major", return_value=152)
    @patch("src.retriv._position_scraper_window")
    def test_standard_browser_uses_cache_and_closes_after_empty_search(self, position, major, cached, chrome):
        self.scraper.output_data = {}
        self.scraper._scrape_page = Mock(return_value=[])
        with self.assertRaisesRegex(RuntimeError, "SHOPEE_COLLECTION_FAILED"):
            self.scraper.execute()
        kwargs = chrome.call_args.kwargs
        self.assertEqual(kwargs["service"].path, "matching-driver.exe")
        self.assertIs(kwargs["options"], self.scraper.options)
        chrome.return_value.set_page_load_timeout.assert_called_once_with(45)
        chrome.return_value.quit.assert_called_once()

    def test_closed_browser_does_not_become_empty_success(self):
        from selenium.common.exceptions import InvalidSessionIdException
        self.scraper._safe_get = Mock(side_effect=InvalidSessionIdException("disconnected"))
        product = {"link": "https://shopee.ph/product/1/2"}
        with self.assertRaisesRegex(RuntimeError, "SHOPEE_BROWSER_CLOSED"):
            self.scraper._scrape_details(product)
        self.assertNotIn("comments", product)

    @patch("src.retriv.webdriver.Chrome")
    @patch.object(ShopeeScraper, "_cached_driver", return_value="matching-driver.exe")
    @patch.object(ShopeeScraper, "_detect_chrome_major", return_value=152)
    @patch("src.retriv._position_scraper_window")
    def test_failed_details_are_not_saved_and_cleanup_preserves_error(self, position, major, cached, chrome):
        self.scraper.output_data = {}
        self.scraper.index_only = False
        self.scraper._scrape_page = Mock(return_value=[{"link": "https://shopee.ph/product/1/2"}])
        self.scraper._scrape_details = Mock(side_effect=RuntimeError("SHOPEE_BROWSER_CLOSED"))
        self.scraper._periodic_save = Mock()
        chrome.return_value.quit.side_effect = RuntimeError("already disconnected")
        with self.assertRaisesRegex(RuntimeError, "SHOPEE_BROWSER_CLOSED"):
            self.scraper.execute()
        self.scraper._periodic_save.assert_not_called()
        self.assertEqual(self.scraper.output_data, {})

    @patch("src.retriv.os.path.isfile", return_value=True)
    @patch("src.retriv.subprocess.run")
    def test_cached_driver_requires_matching_major(self, run, _isfile):
        run.return_value = Mock(returncode=0, stdout="ChromeDriver 152.0.7977.82")
        self.assertIsNotNone(ShopeeScraper._cached_driver(152))
        self.assertIsNone(ShopeeScraper._cached_driver(151))
        run.return_value = Mock(returncode=1, stdout="ChromeDriver 152.0.7977.82")
        self.assertIsNone(ShopeeScraper._cached_driver(152))

    @patch("src.retriv.os.path.isfile", return_value=False)
    @patch("src.retriv.subprocess.run")
    def test_missing_cached_driver_uses_normal_setup(self, run, _isfile):
        self.assertIsNone(ShopeeScraper._cached_driver(152))
        run.assert_not_called()

    def test_scraper_window_leaves_dashboard_visible(self):
        driver = Mock()
        driver.execute_script.return_value = {"width": 1920, "height": 1080}

        _position_scraper_window(driver)

        driver.set_window_size.assert_called_once_with(1050, 820)
        driver.set_window_position.assert_called_once_with(854, 40)

    def test_captcha_check_accepts_byte_url(self):
        class FakeDriver:
            current_url = b"https://shopee.ph/search?keyword=test"

        self.scraper.driver = FakeDriver()
        self.assertFalse(self.scraper._check_captcha())

    @patch("src.retriv.time.sleep")
    def test_terminal_verification_failure_allows_manual_completion(self, sleep):
        class Body:
            text = "Please Try Again Later. Verification can't be completed."

        class FakeDriver:
            current_url = "https://shopee.ph/verify/captcha?anti_bot_tracking_id=test"

            @staticmethod
            def find_element(_by, _value):
                return Body()

            @staticmethod
            def execute_script(script):
                if "document.body.innerText" in script:
                    return Body.text
                return {}

        self.scraper.driver = FakeDriver()
        def complete(_message):
            self.scraper.driver.current_url = "https://shopee.ph/product/1/2"
        with patch("builtins.input", side_effect=complete) as prompt:
            self.assertTrue(self.scraper._check_captcha())
        prompt.assert_called_once()

    @patch("src.retriv.time.sleep")
    def test_dashboard_waits_through_failed_captcha_until_user_completes(self, sleep):
        from unittest.mock import PropertyMock
        self.scraper.no_prompt = True
        self.scraper.driver = Mock()
        type(self.scraper.driver).current_url = PropertyMock(side_effect=[
            "https://shopee.ph/verify/captcha",
            "https://shopee.ph/verify/captcha",
            "https://shopee.ph/product/1/2",
        ])
        self.scraper._capture_login_diagnostics = Mock()
        self.scraper._verification_page_text = Mock(return_value="verification can't be completed")
        self.assertTrue(self.scraper._check_captcha())
        self.scraper.driver.get.assert_not_called()

    def test_manual_verification_still_has_a_timeout(self):
        self.scraper.no_prompt = True
        self.scraper.driver = Mock(current_url="https://shopee.ph/verify/captcha")
        self.scraper._capture_login_diagnostics = Mock()
        self.scraper._verification_page_text = Mock(return_value="")
        with patch("src.retriv.time.monotonic", side_effect=[0, 0, 10000]):
            with self.assertRaisesRegex(TimeoutError, "verification was not completed"):
                self.scraper._check_captcha()

    def test_verification_text_avoids_webdriver_full_page_text_extraction(self):
        self.scraper.driver = Mock()
        self.scraper.driver.execute_script.return_value = "Please Try Again Later"
        self.assertEqual(self.scraper._verification_page_text(), "please try again later")
        self.scraper.driver.find_element.assert_not_called()

    def test_detects_installed_chrome_major(self):
        major = self.scraper._detect_chrome_major()
        if major is not None:
            self.assertGreaterEqual(major, 1)

    def test_parse_star_text(self):
        self.assertEqual(self.scraper._parse_star_text("1,2k"), 1200)
        self.assertEqual(self.scraper._parse_star_text("1.2K"), 1200)
        self.assertEqual(self.scraper._parse_star_text("15k"), 15000)
        self.assertEqual(self.scraper._parse_star_text("1,234"), 1234)
        self.assertEqual(self.scraper._parse_star_text("2M"), 2000000)
        self.assertEqual(self.scraper._parse_star_text("100"), 100)
        self.assertEqual(self.scraper._parse_star_text("invalid"), 0)

    def test_parse_ph_rating_filters(self):
        self.assertEqual(self.scraper._parse_rating_filter("5 Star (1.2K)"), ("5_star", 1200))
        self.assertEqual(self.scraper._parse_rating_filter("4 Stars (99)"), ("4_star", 99))
        self.assertEqual(self.scraper._parse_rating_filter("All (1,234)"), ("all", 1234))
        self.assertEqual(self.scraper._parse_rating_filter("1 Star\n82"), ("1_star", 82))
        self.assertEqual(self.scraper._parse_rating_filter("All\n5.6K"), ("all", 5600))
        self.assertEqual(self.scraper._parse_rating_filter("With Comments (25)"), ("commented", 25))
        self.assertEqual(self.scraper._parse_rating_filter("With Media (10)"), ("media", 10))

    def test_zero_all_count_falls_back_to_sum_of_star_counts(self):
        detailed_rating = {
            "all": 0,
            "5_star": 5000,
            "4_star": 347,
            "3_star": 139,
            "2_star": 32,
            "1_star": 82,
        }
        self.assertEqual(self.scraper._total_rating_from_breakdown(detailed_rating), 5600)

    def test_collects_lowest_ratings_first_until_limit(self):
        class FakeDriver:
            def __init__(self):
                self.selected = None
                self.clicks = []
                self.filters = []

            def find_elements(self, *_args):
                return self.filters

        class FakeFilter:
            def __init__(self, driver, star, count):
                self.driver = driver
                self.star = star
                self.text = f"{star} Star ({count})"

            def click(self):
                self.driver.selected = self.star
                self.driver.clicks.append(self.star)

        driver = FakeDriver()
        driver.filters = [
            FakeFilter(driver, 5, 100),
            FakeFilter(driver, 2, 100),
            FakeFilter(driver, 1, 2),
        ]
        self.scraper.driver = driver
        self.scraper._collect_reviews = lambda limit: [
            {"rating": driver.selected} for _ in range(limit)
        ]

        reviews = self.scraper._collect_lowest_rating_reviews(5)

        self.assertEqual([review["rating"] for review in reviews], [1, 1, 2, 2, 2])
        self.assertEqual(driver.clicks, [1, 2])

    def test_builds_shopee_ph_search_url(self):
        scraper = ShopeeScraper("laptop stand & bag", 1, True, 0)
        parts = urlsplit(scraper._build_search_url())
        query = parse_qs(parts.query)

        self.assertEqual(parts.scheme, "https")
        self.assertEqual(parts.netloc, "shopee.ph")
        self.assertEqual(parts.path, "/search")
        self.assertEqual(query["keyword"], ["laptop stand & bag"])
        self.assertEqual(query["page"], ["0"])
        self.assertEqual(query["sortBy"], ["sales"])

    def test_accepts_and_canonicalizes_shopee_ph_product_url(self):
        url = (
            "https://www.shopee.ph/Baseus-Bowie-MC2-i.131196305.55762651089"
            "?extraParams=tracking"
        )
        self.assertEqual(
            self.scraper._normalize_product_url(url),
            "https://shopee.ph/Baseus-Bowie-MC2-i.131196305.55762651089",
        )

    def test_rejects_non_ph_and_non_product_urls(self):
        with self.assertRaisesRegex(ValueError, "Only shopee.ph"):
            self.scraper._normalize_product_url("https://shopee.vn/Example-i.1.2")
        with self.assertRaisesRegex(ValueError, "not a recognized"):
            self.scraper._normalize_product_url("https://shopee.ph/search?keyword=test")

    def test_direct_product_url_mode(self):
        url = "https://shopee.ph/sample-i.1325174344.27161131426?tracking=1"
        scraper = ShopeeScraper(
            "",
            1,
            False,
            10,
            product_urls=[url],
            output="direct_test.json",
        )
        product = scraper._product_from_url(url)
        self.assertEqual(
            product["link"],
            "https://shopee.ph/sample-i.1325174344.27161131426",
        )
        self.assertEqual(product["market"], "PH")
        self.assertEqual(scraper.out_file, "direct_test.json")

    def test_parser_supports_dashboard_options(self):
        args = build_parser().parse_args(
            [
                "--product-url",
                "https://shopee.ph/sample-i.1.2",
                "--output",
                "result.json",
                "--no-prompt",
                "--verification-timeout",
                "600",
            ]
        )
        self.assertEqual(args.product_url, ["https://shopee.ph/sample-i.1.2"])
        self.assertEqual(args.output, "result.json")
        self.assertTrue(args.no_prompt)
        self.assertEqual(args.verification_timeout, 600)

    def test_normalizes_atomic_review_snapshot(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script):
                return [{
                    "comment_id": "123",
                    "author": "buyer_one",
                    "rating": 5,
                    "time": "2026-06-19 09:14 | Variation: Silver",
                    "content": "Stability: sturdy\nVery good item",
                    "seller_respond": "",
                    "like_count_text": "1.2K",
                }]

        self.scraper.driver = FakeDriver()
        reviews = self.scraper._reviews_from_current_page()
        self.assertEqual(reviews[0]["comment_id"], "123")
        self.assertEqual(reviews[0]["author"], "buyer_one")
        self.assertEqual(reviews[0]["rating"], 5)
        self.assertEqual(reviews[0]["like_count"], 1200)

    def test_safe_get(self):
        self.assertTrue(hasattr(self.scraper, "_safe_get"))

    def test_verification_detection_ignores_product_names_and_query_strings(self):
        for value in (None, "", "https://shopee.ph/security-camera-auth-check-i.1.2", "https://shopee.ph/search?keyword=login+error", "https://example.com/verify"):
            with self.subTest(value=value):
                self.assertFalse(self.scraper._is_verification_url(value))
        for value in ("https://shopee.ph/buyer/login?next=/product/1/2", b"https://shopee.ph/verify/captcha"):
            self.assertTrue(self.scraper._is_verification_url(value))

    @patch("src.retriv.time.sleep")
    def test_verification_redirect_does_not_reload_product(self, sleep):
        url = "https://shopee.ph/product/1/2"
        self.scraper.driver = Mock(current_url=url)
        self.scraper._check_captcha = Mock(return_value=True)
        self.scraper._safe_get(url)
        self.scraper.driver.get.assert_called_once_with(url)

    @patch("src.retriv.time.sleep")
    def test_verification_return_navigation_is_bounded(self, sleep):
        self.scraper.driver = Mock(current_url="https://shopee.ph/")
        self.scraper._check_captcha = Mock(return_value=True)
        with self.assertRaisesRegex(RuntimeError, "SHOPEE_COLLECTION_FAILED"):
            self.scraper._safe_get("https://shopee.ph/product/1/2")
        self.assertEqual(self.scraper.driver.get.call_count, 2)

    def test_retrieve_products(self):
        self.assertTrue(hasattr(self.scraper, "_retrieve_products"))


if __name__ == "__main__":
    unittest.main()
