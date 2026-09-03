import unittest
import pickle
from unittest.mock import patch
from selenium.common.exceptions import StaleElementReferenceException
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

from src.lazada_scraper import LazadaScraper, build_parser


class TestLazadaScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = LazadaScraper(
            keyword="laptop stand",
            max_products=5,
            review_limit=20,
            pages=2,
            pause_for_verification=False,
        )

    def test_initialization(self):
        self.assertEqual(self.scraper.keyword, "laptop stand")
        self.assertEqual(self.scraper.max_products, 5)
        self.assertEqual(self.scraper.review_limit, 20)
        self.assertEqual(self.scraper.pages, 2)
        self.assertEqual(self.scraper.min_rating, 0.0)
        self.assertEqual(self.scraper.min_reviews, 0)
        self.assertEqual(self.scraper.out_file, "lazada_ph_laptop_stand.json")
        self.assertEqual(
            self.scraper.discovery_checkpoint_file,
            "lazada_ph_laptop_stand_discovery.json",
        )
        self.assertIn("--no-first-run", self.scraper.options.arguments)
        self.assertIn("--disable-quic", self.scraper.options.arguments)
        self.assertIn("--window-size=900,700", self.scraper.options.arguments)
        self.assertNotIn("--start-maximized", self.scraper.options.arguments)
        self.assertIn("--disable-search-engine-choice-screen", self.scraper.options.arguments)

    def test_safe_get_retries_chrome_network_error_page(self):
        error_page = '<div id="main-frame-error">ERR_QUIC_PROTOCOL_ERROR</div>'

        class Driver:
            def __init__(self):
                self.calls = 0
                self.page_source = ""

            def get(self, _url):
                self.calls += 1
                self.page_source = error_page if self.calls == 1 else "<main>Products</main>"

            def execute_script(self, _script):
                return "complete"

        driver = Driver()
        self.scraper.driver = driver
        checks = []
        self.scraper._check_verification = lambda: checks.append(True)

        with patch("src.lazada_scraper.time.sleep"):
            self.scraper._safe_get("https://www.lazada.com.ph/catalog/?q=earphones")

        self.assertEqual(driver.calls, 2)
        self.assertEqual(checks, [True])

    def test_safe_get_reports_persistent_chrome_network_error(self):
        class Driver:
            page_source = '<div id="sub-frame-error-details">ERR_QUIC_PROTOCOL_ERROR</div>'

            def __init__(self):
                self.calls = 0

            def get(self, _url):
                self.calls += 1

            def execute_script(self, _script):
                return "complete"

        driver = Driver()
        self.scraper.driver = driver
        saved = []
        self.scraper._save_debug_artifacts = saved.append

        with patch("src.lazada_scraper.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "ERR_QUIC_PROTOCOL_ERROR"):
                self.scraper._safe_get("https://www.lazada.com.ph/catalog/?q=earphones")

        self.assertEqual(driver.calls, 3)
        self.assertEqual(saved, ["network_error"])

    def test_product_image_prefers_current_v2_gallery(self):
        class Element:
            def __init__(self, **attributes):
                self.attributes = attributes

            def get_attribute(self, name):
                return self.attributes.get(name, "")

        class Root:
            def find_elements(self, _by, selector):
                elements = {
                    '.gallery-preview-panel-v2__content img': [
                        Element(src="https://img.lazcdn.com/v2-product.jpg")
                    ],
                    'meta[property="og:image"]': [
                        Element(content="https://img.lazcdn.com/metadata.jpg")
                    ],
                }
                return elements.get(selector, [])

        self.assertEqual(
            self.scraper._product_image_url(Root()),
            "https://img.lazcdn.com/v2-product.jpg",
        )

    def test_product_image_falls_back_to_social_metadata(self):
        class Element:
            def get_attribute(self, name):
                return "https://img.lazcdn.com/metadata.jpg" if name == "content" else ""

        class Root:
            def find_elements(self, _by, selector):
                if selector == 'meta[property="og:image"]':
                    return [Element()]
                return []

        self.assertEqual(
            self.scraper._product_image_url(Root()),
            "https://img.lazcdn.com/metadata.jpg",
        )

    def test_requires_keyword_or_product_url(self):
        with self.assertRaises(ValueError):
            LazadaScraper()

    def test_category_url_is_a_valid_discovery_source(self):
        url = "https://www.lazada.com.ph/shop-mobiles-tablets/?sort=popularity#items"
        scraper = LazadaScraper(category_urls=[url], pause_for_verification=False)

        self.assertEqual(
            scraper.category_urls,
            ["https://www.lazada.com.ph/shop-mobiles-tablets/?sort=popularity"],
        )
        self.assertEqual(scraper.out_file, "lazada_ph_shop_mobiles_tablets.json")

    def test_normalizes_and_validates_category_urls(self):
        self.assertEqual(
            self.scraper._normalize_category_url(
                "//lazada.com.ph/shop-electronic-accessories/?sort=popularity#catalog"
            ),
            "https://www.lazada.com.ph/shop-electronic-accessories/?sort=popularity",
        )
        self.assertEqual(
            self.scraper._normalize_category_url(
                "https://www.lazada.co.id/shop-electronic-accessories/"
            ),
            "",
        )
        self.assertEqual(
            self.scraper._normalize_category_url(
                "https://www.lazada.com.ph/products/sample-i1-s2.html"
            ),
            "",
        )
        self.assertEqual(
            self.scraper._normalize_category_url("https://www.lazada.com.ph/user/login"),
            "",
        )

    def test_category_pagination_preserves_filters_and_replaces_page(self):
        url = (
            "https://www.lazada.com.ph/shop-electronic-accessories/"
            "?sort=popularity&page=9&price=100-500"
        )
        parts = urlsplit(self.scraper._build_category_page_url(url, 3))
        query = parse_qs(parts.query)

        self.assertEqual(parts.path, "/shop-electronic-accessories/")
        self.assertEqual(query["sort"], ["popularity"])
        self.assertEqual(query["price"], ["100-500"])
        self.assertEqual(query["page"], ["3"])

    def test_cli_accepts_repeated_category_urls(self):
        args = build_parser().parse_args(
            [
                "--category-url",
                "https://www.lazada.com.ph/shop-mobiles-tablets/",
                "--category-url",
                "https://www.lazada.com.ph/shop-electronic-accessories/",
                "--discovery-checkpoint",
                "queue.json",
            ]
        )

        self.assertEqual(len(args.category_url), 2)
        self.assertEqual(args.discovery_checkpoint, "queue.json")

    def test_discovery_checkpoint_round_trip(self):
        category_url = "https://www.lazada.com.ph/shop-computer-accessories/"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            checkpoint = Path(directory) / "queue.json"
            scraper = LazadaScraper(
                category_urls=[category_url],
                output=str(output),
                discovery_checkpoint=str(checkpoint),
                pause_for_verification=False,
            )
            product = {
                "link": "https://www.lazada.com.ph/products/sample-i1-s2.html",
                "name": "Sample",
                "rating": 4.5,
                "discovery_source": category_url,
                "discovery_page": 2,
            }
            scraper._upsert_discovery_product(product, "discovered")
            scraper.discovery_completed_pages[category_url] = 1
            scraper._save_discovery_checkpoint()

            resumed = LazadaScraper(
                category_urls=[category_url],
                output=str(output),
                discovery_checkpoint=str(checkpoint),
                pause_for_verification=False,
            )

            self.assertIn(product["link"], resumed.discovery_products)
            self.assertEqual(
                resumed.discovery_products[product["link"]]["discovery_status"],
                "discovered",
            )
            self.assertEqual(resumed.discovery_completed_pages[category_url], 1)

    def test_discovery_checkpoint_with_different_source_is_ignored(self):
        first_url = "https://www.lazada.com.ph/shop-computer-accessories/"
        second_url = "https://www.lazada.com.ph/shop-mobiles-tablets/"
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "queue.json"
            first = LazadaScraper(
                category_urls=[first_url],
                discovery_checkpoint=str(checkpoint),
                pause_for_verification=False,
            )
            product = {
                "link": "https://www.lazada.com.ph/products/sample-i1-s2.html",
                "discovery_source": first_url,
                "discovery_page": 1,
            }
            first._upsert_discovery_product(product, "discovered")
            first._save_discovery_checkpoint()

            second = LazadaScraper(
                category_urls=[second_url],
                discovery_checkpoint=str(checkpoint),
                pause_for_verification=False,
            )

            self.assertEqual(second.discovery_products, {})
            self.assertEqual(second.discovery_completed_pages, {})

    def test_validates_qualification_thresholds(self):
        with self.assertRaises(ValueError):
            LazadaScraper(keyword="lcd", min_rating=5.1)
        with self.assertRaises(ValueError):
            LazadaScraper(keyword="lcd", min_rating=-0.1)
        with self.assertRaises(ValueError):
            LazadaScraper(keyword="lcd", min_reviews=-1)

    def test_exact_qualification_boundaries_are_accepted(self):
        scraper = LazadaScraper(
            keyword="lcd",
            min_rating=4.0,
            min_reviews=100,
            pause_for_verification=False,
        )
        product = {"rating": 4.0, "total_rating": 100}

        self.assertTrue(scraper._qualify_product(product))
        self.assertEqual(product["qualification"]["status"], "qualified")
        self.assertEqual(product["qualification"]["reasons"], [])
        self.assertEqual(product["reported_review_count"], 100)
        self.assertEqual(product["written_reviews_collected"], 0)

    def test_products_below_qualification_boundaries_are_rejected(self):
        scraper = LazadaScraper(
            keyword="lcd",
            min_rating=4.0,
            min_reviews=100,
            pause_for_verification=False,
        )
        product = {
            "link": "https://www.lazada.com.ph/products/sample-i1-s2.html",
            "rating": 3.9,
            "total_rating": 99,
            "comments": [{"content": "old result"}],
        }

        self.assertFalse(scraper._qualify_product(product))
        self.assertEqual(product["qualification"]["status"], "rejected")
        self.assertEqual(
            product["qualification"]["reasons"],
            ["rating_below_minimum", "review_count_below_minimum"],
        )
        self.assertEqual(product["comments"], [])
        self.assertEqual(product["written_reviews_collected"], 0)
        self.assertEqual(product["review_status"], "not_qualified")
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "not_qualified")
        self.assertTrue(scraper._saved_qualification_rejection_matches(product))

    def test_rejected_result_is_rechecked_when_thresholds_change(self):
        product = {
            "qualification": {
                "status": "rejected",
                "min_rating": 4.0,
                "min_reviews": 100,
            }
        }
        same = LazadaScraper(keyword="lcd", min_rating=4.0, min_reviews=100)
        changed = LazadaScraper(keyword="lcd", min_rating=3.5, min_reviews=50)

        self.assertTrue(same._saved_qualification_rejection_matches(product))
        self.assertFalse(changed._saved_qualification_rejection_matches(product))

    def test_builds_ph_search_url(self):
        parts = urlsplit(self.scraper._build_search_url(2))
        query = parse_qs(parts.query)
        self.assertEqual(parts.netloc, "www.lazada.com.ph")
        self.assertEqual(parts.path, "/catalog/")
        self.assertEqual(query["q"], ["laptop stand"])
        self.assertEqual(query["page"], ["2"])

    def test_normalizes_lazada_ph_product_url(self):
        url = "//www.lazada.com.ph/products/sample-i123456-s789012.html?spm=tracking#reviews"
        self.assertEqual(
            self.scraper._normalize_product_url(url),
            "https://www.lazada.com.ph/products/sample-i123456-s789012.html",
        )

    def test_rejects_other_markets_and_non_product_pages(self):
        self.assertEqual(
            self.scraper._normalize_product_url(
                "https://www.lazada.co.id/products/sample-i1-s2.html"
            ),
            "",
        )
        self.assertEqual(
            self.scraper._normalize_product_url("https://www.lazada.com.ph/catalog/?q=laptop"),
            "",
        )

    def test_parses_ph_counts(self):
        self.assertEqual(self.scraper._parse_compact_number("1,234 Ratings"), 1234)
        self.assertEqual(self.scraper._parse_compact_number("2.5K sold"), 2500)
        self.assertEqual(self.scraper._parse_compact_number("1,2k"), 1200)
        self.assertEqual(self.scraper._parse_compact_number(""), 0)

    def test_parses_rating(self):
        self.assertEqual(self.scraper._parse_rating("4.8/5"), 4.8)
        self.assertEqual(self.scraper._parse_rating("Rated 1 star"), 1.0)
        self.assertEqual(self.scraper._parse_rating("unrated"), 0.0)

    def test_distinguishes_reported_reviews_from_unique_written_comments(self):
        product = {
            "total_rating": 353,
            "comments": [
                {"comment_id": "one", "content": "Good"},
                {"comment_id": "one", "content": "Good"},
                {"comment_id": "two", "content": "Bad"},
            ],
        }

        self.scraper._sync_review_counts(product)

        self.assertEqual(product["total_rating"], 353)
        self.assertEqual(product["reported_review_count"], 353)
        self.assertEqual(product["written_reviews_collected"], 2)

    def test_repairs_common_browser_text_encoding(self):
        self.assertEqual(self.scraper._clean_text("â‚±44.00"), "₱44.00")
        self.assertEqual(self.scraper._clean_text("Color Family:Â Wood"), "Color Family: Wood")
        self.assertEqual(
            self.scraper._clean_text("⭐️😊 Hindi sira — don’t replace"),
            "⭐️😊 Hindi sira — don’t replace",
        )

    def test_direct_url_mode_uses_compatible_schema(self):
        url = "https://www.lazada.com.ph/products/sample-i123-s456.html"
        scraper = LazadaScraper(product_urls=[url], pause_for_verification=False)
        product = scraper._product_from_url(url)
        self.assertEqual(product["link"], url)
        self.assertEqual(product["platform"], "lazada")
        self.assertEqual(product["market"], "PH")
        self.assertEqual(product["currency"], "PHP")

    def test_normalizes_review_snapshot(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script):
                return [
                    {
                        "comment_id": "review-1",
                        "author": "A****a",
                        "rating_label": "Rated 2 stars",
                        "active_stars": 0,
                        "time": "13 Aug 2026",
                        "content": "  Sira agad\nnot sturdy  ",
                        "variation": "Color: Silver",
                        "seller_respond": "Sorry about that",
                        "helpful": "Helpful (1.2K)",
                    }
                ]

        self.scraper.driver = FakeDriver()
        reviews = self.scraper._reviews_from_current_page()
        self.assertEqual(reviews[0]["comment_id"], "review-1")
        self.assertEqual(reviews[0]["rating"], 2)
        self.assertEqual(reviews[0]["content"], "Sira agad not sturdy")
        self.assertEqual(reviews[0]["like_count"], 1200)

    def test_skips_empty_written_review_by_default(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script):
                return [{"rating_label": "5 stars", "content": ""}]

        self.scraper.driver = FakeDriver()
        self.assertEqual(self.scraper._reviews_from_current_page(), [])

    def test_deduplicates_reviews_across_star_filters(self):
        reviews = [
            {"comment_id": "a", "content": "Good", "rating": 5},
            {"comment_id": "a", "content": "Good", "rating": 5},
            {"comment_id": "b", "content": "Sira", "rating": 1},
        ]
        unique = self.scraper._deduplicate_reviews(reviews)
        self.assertEqual([review["comment_id"] for review in unique], ["a", "b"])

    def test_verifies_returned_review_rating_before_labeling_filter(self):
        self.assertTrue(
            self.scraper._reviews_match_rating_filter(
                [{"rating": 2}, {"rating": 2}], 2
            )
        )
        self.assertFalse(
            self.scraper._reviews_match_rating_filter(
                [{"rating": 5}, {"rating": 5}], 2
            )
        )
        self.assertFalse(self.scraper._reviews_match_rating_filter([], 2))

    def test_orders_reviews_from_lowest_to_highest_rating(self):
        reviews = [
            {"comment_id": "high", "rating": 5},
            {"comment_id": "low", "rating": 1},
            {"comment_id": "middle", "rating": 3},
        ]
        ordered = self.scraper._order_reviews_low_to_high(reviews)
        self.assertEqual(
            [review["comment_id"] for review in ordered],
            ["low", "middle", "high"],
        )

    def test_default_review_limit_fills_from_lowest_rating_upward(self):
        product = {"comments": []}
        clicked = []
        requested = []
        available = {
            1: [
                {"comment_id": "one-a", "rating": 1},
                {"comment_id": "one-b", "rating": 1},
            ],
            2: [
                {"comment_id": "two-a", "rating": 2},
                {"comment_id": "two-b", "rating": 2},
                {"comment_id": "two-c", "rating": 2},
            ],
        }
        self.scraper._click_rating_filter = lambda star: clicked.append(star) or True

        def collect(limit, product=None, source_filter=None):
            requested.append((source_filter, limit))
            return available.get(source_filter, [])[:limit]

        self.scraper._collect_reviews = collect
        reviews = self.scraper._collect_reviews_lowest_first(product, 5)

        self.assertEqual(clicked, [1, 2])
        self.assertEqual(requested, [(1, 5), (2, 3)])
        self.assertEqual([review["rating"] for review in reviews], [1, 1, 2, 2, 2])
        self.assertEqual(product["applied_rating_filters"], [1, 2])

    def test_lowest_first_discards_mismatched_filter_results(self):
        product = {"comments": []}
        self.scraper._click_rating_filter = lambda _star: True
        self.scraper._collect_reviews = lambda *_args, **_kwargs: [
            {"comment_id": "wrong", "rating": 5}
        ]

        reviews = self.scraper._collect_reviews_lowest_first(product, 3)

        self.assertEqual(reviews, [])
        self.assertIn("mismatched results were discarded", product["review_filter_warning"])

    def test_supports_current_iweb_review_pagination(self):
        self.assertEqual(
            self.scraper.REVIEW_NEXT_SELECTORS[:2],
            ('.iweb-pagination-next button', '.iweb-pagination-next'),
        )

    def test_disabled_next_button_stops_without_retrying(self):
        class DisabledButton:
            @staticmethod
            def is_displayed():
                return True

            @staticmethod
            def get_attribute(name):
                return "disabled" if name == "class" else None

        class FakeDriver:
            find_calls = 0

            def find_elements(self, _by, _selector):
                self.find_calls += 1
                return [DisabledButton()]

        driver = FakeDriver()
        self.scraper.driver = driver
        self.scraper.pagination_retries = 3
        self.scraper._current_review_signature = lambda: "same-page"

        self.assertFalse(self.scraper._next_review_page("same-page"))
        self.assertEqual(self.scraper._last_pagination_stop_reason, "next_button_disabled")
        self.assertEqual(driver.find_calls, 1)

    def test_cookie_save_is_atomic_and_preserves_previous_file_on_browser_failure(self):
        class BrokenDriver:
            @staticmethod
            def get_cookies():
                raise RuntimeError("browser closed")

        with TemporaryDirectory() as directory:
            cookie_path = Path(directory) / "cookies.dat"
            original = [{"name": "session", "value": "existing"}]
            cookie_path.write_bytes(pickle.dumps(original))
            self.scraper.cookies_file = str(cookie_path)
            self.scraper.driver = BrokenDriver()
            self.scraper._save_cookies()
            self.assertEqual(pickle.loads(cookie_path.read_bytes()), original)

    def test_prepares_session_before_checking_login(self):
        events = []
        self.scraper._safe_get = lambda url, check_verification=True: events.append(
            ("open", url, check_verification)
        )
        self.scraper._restore_or_request_login = lambda: events.append(("restore",)) or True

        self.assertTrue(self.scraper._prepare_session())
        self.assertEqual(
            events,
            [("open", self.scraper.BASE_URL, False), ("restore",)],
        )

    def test_login_detection_accepts_byte_url(self):
        class FakeDriver:
            current_url = b"https://www.lazada.com.ph/catalog/?q=test"

            @staticmethod
            def find_elements(_by, _selector):
                return []

        self.scraper.driver = FakeDriver()
        self.assertFalse(self.scraper._login_required())

    def test_records_no_reviews_as_a_terminal_product_result(self):
        product = {
            "link": "https://www.lazada.com.ph/products/empty-i1-s2.html",
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "empty.json")
            self.scraper._record_no_reviews(product)
        self.assertEqual(product["review_status"], "no_reviews")
        self.assertEqual(product["comments"], [])
        self.assertEqual(product["reported_review_count"], 0)
        self.assertEqual(product["written_reviews_collected"], 0)
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "no_reviews")
        self.assertTrue(self.scraper._review_checkpoint_is_terminal(product))

    def test_filtered_empty_page_is_terminal_only_for_the_requested_rating(self):
        product = {
            "review_status": "complete",
            "comments": [{"rating": 2, "source_rating_filter": 2}],
            "review_checkpoint": {
                "context": "2",
                "stop_reason": "empty_review_page",
            },
        }
        rating_two = LazadaScraper(keyword="lcd", rating=2, pause_for_verification=False)
        rating_three = LazadaScraper(keyword="lcd", rating=3, pause_for_verification=False)

        self.assertTrue(rating_two._review_checkpoint_is_terminal(product))
        self.assertFalse(rating_three._review_checkpoint_is_terminal(product))

    def test_retryable_checkpoint_is_not_terminal(self):
        product = {
            "review_status": "collecting",
            "review_checkpoint": {
                "context": "2",
                "stop_reason": "max_review_pages_reached",
            },
        }
        scraper = LazadaScraper(keyword="lcd", rating=2, pause_for_verification=False)

        self.assertFalse(scraper._review_checkpoint_is_terminal(product))

    def test_reported_reviews_with_empty_widget_are_not_marked_no_reviews(self):
        product = {
            "link": "https://www.lazada.com.ph/products/reviews-i1-s2.html",
            "total_rating": 353,
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "missing-widget.json")
            self.scraper._record_review_widget_missing(product)
        self.assertEqual(product["review_status"], "review_widget_missing")
        self.assertEqual(
            product["review_checkpoint"]["stop_reason"],
            "review_widget_missing",
        )

    def test_rating_filter_reacquires_after_stale_dom(self):
        class RebuildingDriver:
            option_attempts = 0

            def execute_script(self, script, *_args):
                if "const rating = String(arguments[0])" in script:
                    self.option_attempts += 1
                    if self.option_attempts == 1:
                        raise StaleElementReferenceException("menu rebuilt")
                    return "1 Star"
                return False

        driver = RebuildingDriver()
        self.scraper.driver = driver
        self.scraper.pagination_retries = 2
        self.assertTrue(self.scraper._click_rating_filter(1))
        self.assertEqual(driver.option_attempts, 2)

    def test_review_javascript_reads_lazada_star_masks(self):
        script = self.scraper._review_javascript()

        self.assertIn("half_([\\d.]+)%", script)
        self.assertIn("Number(percentage[1]) / 100", script)
        self.assertIn("if (paths.length) return total", script)

    def test_resumes_past_duplicate_review_pages(self):
        old_review = {"comment_id": "old", "author": "a", "time": "1", "content": "old", "rating": 5}
        new_review = {"comment_id": "new", "author": "b", "time": "2", "content": "new", "rating": 4}
        pages = [[old_review], [new_review]]
        state = {"page": 0}
        self.scraper._review_container = lambda: object()
        self.scraper._reviews_from_current_page = lambda: pages[state["page"]]
        self.scraper._current_review_signature = lambda: str(state["page"])

        def advance(_signature):
            if state["page"] + 1 >= len(pages):
                return False
            state["page"] += 1
            return True

        self.scraper._next_review_page = advance
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "reviews.json")
            product = {"link": "https://www.lazada.com.ph/products/test-i1-s2.html", "comments": [old_review]}
            reviews = self.scraper._collect_reviews(2, product=product)
        self.assertEqual([review["comment_id"] for review in reviews], ["old", "new"])
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "target_reached")


if __name__ == "__main__":
    unittest.main()
