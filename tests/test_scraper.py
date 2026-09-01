import sys
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from selenium.common.exceptions import TimeoutException

from src.retriv import ShopeeScraper, _progress, build_parser


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

    def test_captcha_check_accepts_byte_url(self):
        class FakeDriver:
            current_url = b"https://shopee.ph/search?keyword=test"

        self.scraper.driver = FakeDriver()
        self.assertFalse(self.scraper._check_captcha())

    def test_terminal_verification_failure_stops_without_prompting(self):
        class Body:
            text = "Please Try Again Later. Verification can't be completed."

        class FakeDriver:
            current_url = "https://shopee.ph/verify/captcha?anti_bot_tracking_id=test"

            @staticmethod
            def find_element(_by, _value):
                return Body()

        self.scraper.driver = FakeDriver()
        with self.assertRaisesRegex(TimeoutException, "SHOPEE_VERIFICATION_BLOCKED"):
            self.scraper._check_captcha()

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

    def test_retrieve_products(self):
        self.assertTrue(hasattr(self.scraper, "_retrieve_products"))


if __name__ == "__main__":
    unittest.main()
