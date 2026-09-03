import unittest
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit

from src.temu_scraper import TemuScraper, build_parser


class TestTemuScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = TemuScraper(
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
        self.assertEqual(self.scraper.out_file, "temu_ph_laptop_stand.json")
        self.assertIn("--no-first-run", self.scraper.options.arguments)
        self.assertIn("--disable-search-engine-choice-screen", self.scraper.options.arguments)
        self.assertIn("--window-size=900,700", self.scraper.options.arguments)
        self.assertNotIn("--start-maximized", self.scraper.options.arguments)

    def test_cookie_save_is_atomic_and_preserves_previous_session_on_browser_failure(self):
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

            class EmptyDriver:
                @staticmethod
                def get_cookies():
                    return []

            self.scraper.driver = EmptyDriver()
            self.scraper._save_cookies()
            self.assertEqual(pickle.loads(cookie_path.read_bytes()), original)

    def test_prepares_session_before_checking_login(self):
        events = []
        self.scraper._safe_get = lambda url, check_verification=True: events.append(
            ("open", url, check_verification)
        )
        self.scraper._load_cookies = lambda: events.append(("cookies",)) or False
        self.scraper._check_verification = lambda: events.append(("verify",)) or False

        self.assertTrue(self.scraper._prepare_session())
        self.assertEqual(
            events,
            [("open", self.scraper.MARKET_HOME, False), ("cookies",), ("verify",)],
        )

    def test_requires_keyword_or_product_url(self):
        with self.assertRaises(ValueError):
            TemuScraper()

    def test_rejects_list_containing_only_invalid_urls(self):
        with self.assertRaises(ValueError):
            TemuScraper(product_urls=["https://example.com/product/123"])

    def test_builds_ph_search_url(self):
        parts = urlsplit(self.scraper._build_search_url(2))
        query = parse_qs(parts.query)
        self.assertEqual(parts.netloc, "www.temu.com")
        self.assertEqual(parts.path, "/ph-en/search_result.html")
        self.assertEqual(query["search_key"], ["laptop stand"])
        self.assertEqual(query["page"], ["2"])

    def test_normalizes_goods_query_url(self):
        url = "https://www.temu.com/goods.html?_bg_fs=1&goods_id=601099521225879#reviews"
        self.assertEqual(
            self.scraper._normalize_product_url(url),
            "https://www.temu.com/goods.html?goods_id=601099521225879",
        )

    def test_normalizes_slug_product_url(self):
        url = "https://m.temu.com/ph-en/portable-laptop-stand-g-601099588578152.html?share=1"
        self.assertEqual(
            self.scraper._normalize_product_url(url),
            "https://www.temu.com/goods.html?goods_id=601099588578152",
        )

    def test_preserves_localized_slug_for_navigation_without_tracking_query(self):
        url = (
            "https://www.temu.com/ph/wireless-earbuds-g-601100261015054.html"
            "?search_key=headphones&_x_ads_channel=google"
        )
        scraper = TemuScraper(product_urls=[url], pause_for_verification=False)
        canonical = "https://www.temu.com/goods.html?goods_id=601100261015054"
        self.assertEqual(scraper.product_urls, [canonical])
        self.assertEqual(
            scraper.product_navigation_urls[canonical],
            "https://www.temu.com/ph/wireless-earbuds-g-601100261015054.html",
        )

    def test_rejects_other_hosts_and_search_pages(self):
        self.assertEqual(
            self.scraper._normalize_product_url(
                "https://example.com/goods.html?goods_id=601099521225879"
            ),
            "",
        )
        self.assertEqual(
            self.scraper._normalize_product_url(
                "https://www.temu.com/ph-en/search_result.html?search_key=laptop"
            ),
            "",
        )

    def test_parses_compact_counts(self):
        self.assertEqual(self.scraper._parse_compact_number("1,234 reviews"), 1234)
        self.assertEqual(self.scraper._parse_compact_number("2.5K+ sold"), 2500)
        self.assertEqual(self.scraper._parse_compact_number("1,2k bought"), 1200)
        self.assertEqual(self.scraper._parse_compact_number("none"), 0)

    def test_parses_rating(self):
        self.assertEqual(self.scraper._parse_rating("4.8 out of five stars"), 4.8)
        self.assertEqual(self.scraper._parse_rating("Rated 1 star"), 1.0)
        self.assertEqual(self.scraper._parse_rating("unrated"), 0.0)

    def test_normalizes_product_name_and_removes_temu_title_suffix(self):
        self.assertEqual(
            self.scraper._normalize_product_name(
                "  earphones new wireless   in ear - Temu Philippines"
            ),
            "earphones new wireless in ear",
        )

    def test_reads_structured_temu_price_range_from_page_source(self):
        source = r'{"minToMaxPriceStr":"\u20b1173-\u20b1508","currency":"PHP"}'
        self.assertEqual(self.scraper._price_from_page_source(source), "₱173-₱508")

    def test_direct_url_mode_uses_compatible_schema(self):
        url = "https://www.temu.com/goods.html?goods_id=601099521225879"
        scraper = TemuScraper(product_urls=[url], pause_for_verification=False)
        product = scraper._product_from_url(url)
        self.assertEqual(product["goods_id"], "601099521225879")
        self.assertEqual(product["platform"], "temu")
        self.assertEqual(product["market"], "PH")
        self.assertEqual(product["currency"], "PHP")

    def test_applies_json_ld_product(self):
        product = self.scraper._product_from_url(
            "https://www.temu.com/goods.html?goods_id=601099521225879"
        )
        self.scraper._apply_json_ld(
            product,
            {
                "@type": "Product",
                "name": "Portable laptop stand",
                "brand": {"name": "Example"},
                "image": ["https://img.kwcdn.com/product/example.jpg"],
                "offers": {"price": "199.00", "priceCurrency": "PHP"},
                "aggregateRating": {"ratingValue": "4.7", "reviewCount": "1.2K"},
            },
        )
        self.assertEqual(product["name"], "Portable laptop stand")
        self.assertEqual(product["price"], "₱199.00")
        self.assertEqual(product["rating"], 4.7)
        self.assertEqual(product["total_rating"], 1200)

    def test_normalizes_review_snapshot(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script, _root):
                return [
                    {
                        "comment_id": "review-1",
                        "author": "A****a",
                        "rating_label": "2 out of 5 stars",
                        "time": "14 Aug 2026",
                        "content": "  Sira agad\nnot sturdy  ",
                        "variation": "Color: Silver",
                        "helpful": "Helpful (1.2K)",
                    }
                ]

        self.scraper.driver = FakeDriver()
        reviews = self.scraper._reviews_from_current_view(object())
        self.assertEqual(reviews[0]["comment_id"], "review-1")
        self.assertEqual(reviews[0]["rating"], 2)
        self.assertEqual(reviews[0]["content"], "Sira agad not sturdy")
        self.assertEqual(reviews[0]["like_count"], 1200)

    def test_skips_aggregate_rating_boilerplate_and_date_only_rows(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script, _root):
                return [
                    {
                        "rating_label": "4.4 out of 5 stars",
                        "content": "All reviews are from verified purchases",
                    },
                    {
                        "rating_label": "5 out of 5 stars",
                        "time": "on Aug 11, 2026",
                        "content": "on Aug 11, 2026",
                    },
                    {
                        "rating_label": "5 out of 5 stars",
                        "time": "on Aug 23, 2026",
                        "content": "Excellent headphones, I love them!",
                    },
                ]

        self.scraper.driver = FakeDriver()
        reviews = self.scraper._reviews_from_current_view(object())
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["rating"], 5)
        self.assertEqual(reviews[0]["content"], "Excellent headphones, I love them!")

    def test_normalizes_network_reviews_and_rejects_product_records(self):
        payload = {
            "data": {
                "reviews": [
                    {
                        "goodsId": "601100261015054",
                        "commentId": "review-1",
                        "userName": "A****a",
                        "starRating": 2,
                        "reviewContent": "Sira agad",
                        "reviewTime": "Aug 27, 2026",
                    },
                    {
                        "goodsId": "601100261015054",
                        "commentId": "review-2",
                        "name": "B****b",
                        "score": 4,
                        "comment": "Vale la pena",
                        "concatTimeLang": "on Aug 28, 2026",
                        "reviewLang": {"translateComment": "Worth the price"},
                    },
                    {
                        "goodsId": "601100261015054",
                        "id": "product-1",
                        "goodsScore": 5,
                        "content": "Wireless headphones product description",
                    },
                    {
                        "goodsId": "999999999999999",
                        "commentId": "wrong-product",
                        "starRating": 1,
                        "reviewContent": "Not for the requested product",
                    },
                ]
            }
        }
        reviews = self.scraper._reviews_from_network_payload(
            payload, expected_goods_id="601100261015054"
        )
        self.assertEqual(
            [review["comment_id"] for review in reviews],
            ["review-1", "review-2"],
        )
        self.assertEqual(reviews[0]["content"], "Sira agad")
        self.assertEqual(reviews[0]["extraction_source"], "network")
        self.assertEqual(reviews[1]["author"], "B****b")
        self.assertEqual(reviews[1]["time"], "on Aug 28, 2026")
        self.assertEqual(reviews[1]["content"], "Worth the price")

    def test_merges_translated_dom_and_network_versions_without_double_counting(self):
        dom_reviews = [
            {
                "comment_id": "",
                "author": "Marcela Lagos",
                "rating": 5,
                "time": "on Aug 19, 2026",
                "content": "They have clear, high-quality sound",
            },
            {
                "comment_id": "",
                "author": "Wrong card association",
                "rating": 5,
                "time": "on Jul 28, 2026",
                "content": "They have clear, high-quality sound",
            },
        ]
        network_reviews = [
            {
                "comment_id": "2664868561840656",
                "author": "Marcela Lagos",
                "rating": 5,
                "time": "on Aug 19, 2026",
                "content": "They have clear, high-quality sound",
                "extraction_source": "network",
            }
        ]
        merged = self.scraper._merge_review_sources(dom_reviews, network_reviews)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["comment_id"], "2664868561840656")

    def test_captures_reviews_from_browser_network_responses(self):
        request_id = "request-1"
        request_url = (
            "https://www.temu.com/api/review/list?goods_id=601100261015054"
        )
        messages = [
            {
                "message": json.dumps(
                    {
                        "message": {
                            "method": "Network.requestWillBeSent",
                            "params": {
                                "requestId": request_id,
                                "request": {"url": request_url},
                            },
                        }
                    }
                )
            },
            {
                "message": json.dumps(
                    {
                        "message": {
                            "method": "Network.responseReceived",
                            "params": {
                                "requestId": request_id,
                                "type": "XHR",
                                "response": {"url": request_url},
                            },
                        }
                    }
                )
            },
        ]

        class FakeDriver:
            def __init__(self):
                self.drained = False

            def get_log(self, _kind):
                if self.drained:
                    return []
                self.drained = True
                return messages

            @staticmethod
            def execute_cdp_cmd(command, params):
                if command != "Network.getResponseBody" or params != {
                    "requestId": request_id
                }:
                    raise AssertionError("unexpected CDP command")
                return {
                    "body": json.dumps(
                        {
                            "comments": [
                                {
                                    "goodsId": "601100261015054",
                                    "commentId": "network-review",
                                    "starRating": 4,
                                    "commentContent": "Works as expected",
                                }
                            ]
                        }
                    ),
                    "base64Encoded": False,
                }

        self.scraper.driver = FakeDriver()
        product = {"goods_id": "601100261015054"}
        reviews = self.scraper._capture_review_network(product)
        self.assertEqual([review["comment_id"] for review in reviews], ["network-review"])

    def test_collects_network_reviews_when_dom_cards_are_empty(self):
        network_review = {
            "comment_id": "network-review",
            "author": "A****a",
            "time": "Aug 27, 2026",
            "content": "Captured from the review response",
            "rating": 4,
            "extraction_source": "network",
        }
        self.scraper.review_limit = 1
        self.scraper._reviews_from_current_view = lambda _root: []
        self.scraper._capture_review_network = lambda _product: [network_review]
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=601100261015054",
            "goods_id": "601100261015054",
            "name": "Sample product",
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "network.json")
            reviews = self.scraper._collect_reviews(object(), product=product)
        self.assertEqual(reviews, [network_review])
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "target_reached")
        self.assertEqual(
            set(product["comments"][0]),
            {"comment_id", "rating", "content"},
        )

    def test_saved_reviews_exclude_identifying_and_extra_metadata(self):
        review = {
            "comment_id": "review-1",
            "author": "Named shopper",
            "rating": 2,
            "time": "on Aug 27, 2026",
            "content": "Stopped working",
            "variation": "Color: White",
            "seller_respond": "Contact support",
            "like_count": 12,
            "extraction_source": "network",
            "source_rating_filter": 2,
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "minimal.json")
            product = {
                "link": "https://www.temu.com/goods.html?goods_id=601100261015054",
                "comments": [],
            }
            self.scraper._checkpoint_reviews(
                product,
                [review],
                batch_number=1,
                stop_reason="target_reached",
            )
            saved = json.loads(Path(self.scraper.out_file).read_text(encoding="utf-8"))
        self.assertEqual(
            product["comments"],
            [
                {
                    "comment_id": "review-1",
                    "rating": 2,
                    "content": "Stopped working",
                }
            ],
        )
        self.assertEqual(saved[0]["comments"], product["comments"])
        self.assertEqual(
            saved[0]["review_checkpoint"]["collected_reviews"],
            len(product["comments"]),
        )

    def test_saved_reviews_require_a_structured_comment_id(self):
        reviews = [
            {"comment_id": "", "rating": 1, "content": "Reviewer Name"},
            {"comment_id": "review-2", "rating": 2, "content": "Stopped working"},
        ]
        self.assertEqual(
            self.scraper._reviews_for_output(reviews),
            [
                {
                    "comment_id": "review-2",
                    "rating": 2,
                    "content": "Stopped working",
                }
            ],
        )

    def test_product_redirect_during_reviews_is_retryable(self):
        class RedirectedDriver:
            current_url = "https://www.temu.com/ph-en/?no_cache_id=redirected"

        self.scraper.driver = RedirectedDriver()
        self.scraper._reviews_from_current_view = lambda _root: self.fail(
            "reviews must not be read after leaving the requested product"
        )
        self.scraper._save_debug_artifacts = lambda _kind: None
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=601100261015054",
            "goods_id": "601100261015054",
            "name": "Sample product",
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "redirect.json")
            reviews = self.scraper._collect_reviews(object(), product=product)
        self.assertEqual(reviews, [])
        self.assertEqual(
            product["review_checkpoint"]["stop_reason"],
            "product_redirected_during_reviews",
        )
        self.assertFalse(self.scraper._existing_result_complete(product))

    def test_current_temu_review_dom_uses_accessible_card_fallbacks(self):
        script = self.scraper._review_javascript()
        self.assertIn('[aria-label="avatar" i]', script)
        self.assertIn('img[alt="avatar" i]', script)
        self.assertIn('[role="text"][aria-label*=" on " i]', script)
        self.assertIn("item.querySelectorAll('section')", script)
        self.assertIn("[role=\"link\"], a", script)

    def test_opens_full_review_list_from_preview_widget(self):
        state = {"expanded": False}

        class Button:
            text = "See all reviews"

            @staticmethod
            def get_attribute(_name):
                return None

            @staticmethod
            def is_displayed():
                return True

            @staticmethod
            def click():
                state["expanded"] = True

        class Root:
            def __init__(self, name):
                self.name = name

            @staticmethod
            def find_elements(_by, _selector):
                return [Button()]

        class Driver:
            @staticmethod
            def execute_script(_script, _element):
                return None

        preview = Root("preview")
        expanded = Root("expanded")
        self.scraper.driver = Driver()
        self.scraper._review_card_count = lambda root: 14 if root is expanded else 4
        self.scraper._review_root = lambda: expanded if state["expanded"] else preview
        self.assertIs(self.scraper._expand_review_widget(preview), expanded)
        self.assertTrue(state["expanded"])

    def test_stops_when_all_rendered_review_cards_have_been_scanned(self):
        written = [
            {
                "comment_id": f"review-{index}",
                "author": f"author-{index}",
                "time": f"on Aug {index + 1}, 2026",
                "content": f"Written review {index}",
                "rating": 5,
            }
            for index in range(10)
        ]
        self.scraper.review_limit = 20
        self.scraper._reviews_from_current_view = lambda _root: written
        self.scraper._review_card_count = lambda _root: 14
        self.scraper._advance_reviews = lambda *_args: self.fail(
            "pagination should not run after every displayed card was scanned"
        )
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=1",
            "name": "Sample",
            "total_rating": 14,
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "rendered.json")
            reviews = self.scraper._collect_reviews(object(), product=product)
        self.assertEqual(len(reviews), 10)
        self.assertEqual(
            product["review_checkpoint"]["stop_reason"],
            "all_rendered_reviews_scanned",
        )

    def test_skips_empty_written_review_by_default(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script, _root):
                return [{"rating_label": "5 stars", "content": ""}]

        self.scraper.driver = FakeDriver()
        self.assertEqual(self.scraper._reviews_from_current_view(object()), [])

    def test_skips_purchase_variation_that_is_not_a_written_review(self):
        self.assertEqual(
            self.scraper._written_review_content(
                "Purchased: White / Single Pack - Lightning"
            ),
            "",
        )
        script = self.scraper._review_javascript()
        self.assertIn("purchased|size|color|style|model|type", script)

    def test_skips_author_name_misidentified_as_review_content(self):
        class FakeDriver:
            @staticmethod
            def execute_script(_script, _root):
                return [
                    {
                        "comment_id": "",
                        "author": "Ramona Alcanrara",
                        "rating_label": "5 stars",
                        "time": "on Aug 27, 2026",
                        "content": "Ramona Alcanrara",
                    }
                ]

        self.scraper.driver = FakeDriver()
        self.assertEqual(self.scraper._reviews_from_current_view(object()), [])

    def test_product_filters(self):
        self.scraper.min_rating = 4.0
        self.scraper.min_reviews = 100
        self.assertTrue(self.scraper._qualifies({"rating": 4.0, "total_rating": 100}))
        self.assertFalse(self.scraper._qualifies({"rating": 3.9, "total_rating": 1000}))
        self.assertFalse(self.scraper._qualifies({"rating": 5.0, "total_rating": 99}))

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

    def test_selects_exact_lowest_rating_sort_option(self):
        events = []
        state = {"menu_open": False}

        class Element:
            def __init__(self, text="", aria_label="", visible=True, click=None):
                self.text = text
                self.aria_label = aria_label
                self.visible = visible
                self.on_click = click

            def get_attribute(self, name):
                return self.aria_label if name == "aria-label" else None

            def is_displayed(self):
                return self.visible() if callable(self.visible) else self.visible

            def click(self):
                events.append(f"click:{self.text or self.aria_label}")
                if self.on_click:
                    self.on_click()

        trigger = Element(
            aria_label="Recommended more selections",
            click=lambda: state.update(menu_open=True),
        )
        option = Element(
            text="Lowest rating",
            visible=lambda: state["menu_open"],
        )

        class Scope:
            @staticmethod
            def find_elements(_by, _selector):
                return [trigger, option]

        class Driver(Scope):
            @staticmethod
            def execute_script(_script, _element):
                return None

        product = {"link": "https://www.temu.com/goods.html?goods_id=1"}
        self.scraper.driver = Driver()
        self.scraper.pagination_timeout = 0.1
        self.scraper._reset_product_network_capture = (
            lambda selected: events.append(
                "reset:" + selected["link"].rsplit("=", 1)[-1]
            )
        )

        self.assertTrue(self.scraper._click_lowest_rating_sort(Scope(), product))
        self.assertEqual(
            events,
            [
                "click:Recommended more selections",
                "reset:1",
                "click:Lowest rating",
            ],
        )

    def test_review_scroll_targets_nearest_scrollable_modal_ancestor(self):
        script = self.scraper._review_scroll_javascript()
        self.assertIn("window.getComputedStyle", script)
        self.assertIn("node.scrollHeight > node.clientHeight + 2", script)
        self.assertIn("style.overflowY", script)
        self.assertIn("scroller.scrollTop = scroller.scrollHeight", script)
        self.assertIn("dispatchEvent(new Event('scroll'", script)

    def test_limited_collection_uses_lowest_available_star_filters_first(self):
        attempted_filters = []
        root = object()
        self.scraper.review_limit = 2
        self.scraper._review_root = lambda: root

        def click_filter(_root, rating):
            attempted_filters.append(rating)
            return rating >= 2

        self.scraper._click_rating_filter = click_filter
        self.scraper._collect_reviews = lambda _root, **kwargs: [
            {
                "comment_id": f"review-{kwargs['source_filter']}",
                "rating": kwargs["source_filter"],
                "content": f"Rating {kwargs['source_filter']}",
            }
        ]
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=601100261015054",
            "comments": [],
        }
        reviews, applied_filters, redirected = (
            self.scraper._collect_rating_filtered_reviews(root, product)
        )
        self.assertEqual(attempted_filters, [1, 2, 3])
        self.assertEqual(applied_filters, [2, 3])
        self.assertEqual([review["rating"] for review in reviews], [2, 3])
        self.assertFalse(redirected)

    def test_detects_login_redirect(self):
        class Body:
            text = "Sign in to continue"

        class FakeDriver:
            current_url = b"https://www.temu.com/login.html?from=search"
            title = "Sign in / Register"

            @staticmethod
            def find_element(_by, _value):
                return Body()

        self.scraper.driver = FakeDriver()
        self.assertTrue(self.scraper._check_verification())

    def test_detects_verification_that_appears_while_opening_reviews(self):
        events = []
        self.scraper._verification_detected = lambda: True
        self.scraper._check_verification = lambda: events.append("checked") or True
        self.scraper._review_root = lambda: self.fail(
            "review lookup should wait until verification is cleared"
        )
        self.assertIsNone(self.scraper._open_review_section())
        self.assertEqual(events, ["checked"])

    def test_parser_supports_all_reviews_and_filters(self):
        args = build_parser().parse_args(
            [
                "electronic gadgets",
                "--all-reviews",
                "--min-rating",
                "4",
                "--min-reviews",
                "100",
                "--cookies-file",
                "session.dat",
            ]
        )
        self.assertTrue(args.all_reviews)
        self.assertEqual(args.min_rating, 4.0)
        self.assertEqual(args.min_reviews, 100)
        self.assertEqual(args.cookies_file, "session.dat")

    def test_default_review_limit_is_one_hundred(self):
        args = build_parser().parse_args(["laptop stand"])
        self.assertEqual(args.review_limit, 100)
        scraper = TemuScraper(keyword="laptop stand", pause_for_verification=False)
        self.assertEqual(scraper.review_limit, 100)

    def test_resumes_past_duplicate_review_batches(self):
        self.scraper.review_limit = 2
        old_review = {"comment_id": "old", "author": "a", "time": "1", "content": "old", "rating": 5}
        new_review = {"comment_id": "new", "author": "b", "time": "2", "content": "new", "rating": 4}
        pages = [[old_review], [new_review]]
        state = {"page": 0}
        root = object()
        self.scraper._reviews_from_current_view = lambda _root: pages[state["page"]]
        self.scraper._review_signature = lambda _root: str(state["page"])

        def advance(_root, _signature):
            if state["page"] + 1 >= len(pages):
                return False
            state["page"] += 1
            return True

        self.scraper._advance_reviews = advance
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "reviews.json")
            product = {"link": "https://www.temu.com/goods.html?goods_id=1", "comments": [old_review]}
            reviews = self.scraper._collect_reviews(root, product=product)
        self.assertEqual([review["comment_id"] for review in reviews], ["old", "new"])
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "target_reached")

    def test_records_explicit_no_reviews_as_a_terminal_product_result(self):
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=1",
            "name": "Sample product",
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "empty.json")
            self.scraper._record_no_reviews(product, "0 customer reviews")
        self.assertEqual(product["review_status"], "no_reviews")
        self.assertTrue(product["no_reviews_confirmed"])
        self.assertEqual(product["no_reviews_evidence"], "0 customer reviews")
        self.assertEqual(product["comments"], [])
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "no_reviews")
        self.assertTrue(self.scraper._existing_result_complete(product))

    def test_old_extractor_checkpoint_is_reprocessed(self):
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=1",
            "name": "Sample product",
            "comments": [{"rating": 5, "content": "Old result"}],
            "review_checkpoint": {
                "context": "limited",
                "stop_reason": "no_additional_reviews",
            },
        }
        self.assertFalse(self.scraper._existing_result_complete(product))
        product["review_extractor_version"] = self.scraper.REVIEW_EXTRACTOR_VERSION
        self.assertTrue(self.scraper._existing_result_complete(product))

    def test_zero_review_claim_requires_explicit_page_evidence(self):
        self.assertEqual(
            self.scraper._explicit_no_review_evidence("4.8 stars and fast delivery"),
            "",
        )
        self.assertEqual(
            self.scraper._explicit_no_review_evidence("Be the first to write a review"),
            "be the first to write a review",
        )
        with self.assertRaises(ValueError):
            self.scraper._record_no_reviews(
                {"link": "https://example.test", "comments": []}, ""
            )

    def test_missing_review_widget_is_retryable(self):
        product = {
            "link": "https://www.temu.com/goods.html?goods_id=1",
            "name": "Sample product",
            "comments": [],
        }
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "retry.json")
            self.scraper._record_review_widget_missing(product, "Widget was not identified")
        self.assertEqual(product["review_status"], "review_widget_missing")
        self.assertEqual(
            product["review_checkpoint"]["stop_reason"], "review_widget_missing"
        )
        self.assertFalse(self.scraper._existing_result_complete(product))

    def test_legacy_blank_no_review_checkpoint_is_retried(self):
        legacy = {
            "link": "https://www.temu.com/goods.html?goods_id=606434481785322",
            "name": "",
            "comments": [],
            "review_status": "no_reviews",
            "review_checkpoint": {"context": "all", "stop_reason": "no_reviews"},
        }
        self.assertFalse(self.scraper._existing_result_complete(legacy))

    def test_existing_exhausted_result_gets_clean_title_and_review_warning(self):
        record = {
            "link": "https://www.temu.com/goods.html?goods_id=606434481785322",
            "name": "  Wireless   earphones - Temu Philippines",
            "total_rating": 14,
            "comments": [
                {
                    "comment_id": "existing-review",
                    "rating": 5,
                    "content": "Excellent",
                }
            ],
            "review_checkpoint": {
                "context": "limited",
                "stop_reason": "no_additional_reviews",
            },
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text(json.dumps([record]), encoding="utf-8")
            scraper = TemuScraper(
                product_urls=[record["link"]],
                output=str(output),
                pause_for_verification=False,
            )
        restored = scraper.output_data[record["link"]]
        self.assertEqual(restored["name"], "Wireless earphones")
        self.assertIn("14 ratings", restored["review_collection_warning"])
        self.assertIn("1 unique written review", restored["review_collection_warning"])

    def test_missing_widget_without_zero_review_text_is_not_marked_no_reviews(self):
        class Body:
            text = "Wireless earphones with noise cancellation"

        class FakeDriver:
            current_url = "https://www.temu.com/goods.html?goods_id=606434481785322"

            @staticmethod
            def find_elements(_by, _value):
                return []

            @staticmethod
            def find_element(_by, _value):
                return Body()

        product = self.scraper._product_from_url(FakeDriver.current_url)
        self.scraper.driver = FakeDriver()
        self.scraper._safe_get = lambda _url: False
        self.scraper._dismiss_popups = lambda: None
        self.scraper._open_review_section = lambda: None
        self.scraper._save_debug_artifacts = lambda _kind: None
        with TemporaryDirectory() as directory:
            self.scraper.out_file = str(Path(directory) / "missing.json")
            self.scraper._scrape_details(product)
        self.assertEqual(product["review_status"], "review_widget_missing")
        self.assertEqual(
            product["review_checkpoint"]["stop_reason"], "review_widget_missing"
        )
        self.assertNotIn("no_reviews_confirmed", product)


if __name__ == "__main__":
    unittest.main()
