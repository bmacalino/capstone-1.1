import unittest

from src.lazada_scraper import LazadaScraper

PRODUCT = "https://www.lazada.com.ph/products/pdp-i4644812363-s27518600306.html"


class FakeDriver:
    """Just enough of a browser for the login check."""

    def __init__(self, url, cookies=("lzd_uid", "lzd_sid"), header_login_visible=True):
        self.current_url = url
        self.cookies = set(cookies)
        self.header_login_visible = header_login_visible

    def get_cookie(self, name):
        return {"name": name, "value": "x"} if name in self.cookies else None

    def find_elements(self, _by, selector):
        class Element:
            @staticmethod
            def is_displayed():
                return True

        return [Element()] if self.header_login_visible and "login" in selector.casefold() else []


def scraper(**options):
    return LazadaScraper(product_urls=[PRODUCT], pause_for_verification=False, **options)


class TestLoginCheck(unittest.TestCase):
    def test_a_restored_session_is_not_blocked_by_the_header_login_link(self):
        # Regression: Lazada shows "LOGIN" in the header even with a restored
        # session, so every scan waited until someone scrolled it away.
        subject = scraper()
        subject.driver = FakeDriver(PRODUCT, header_login_visible=True)
        self.assertFalse(subject._login_required())

    def test_no_session_cookies_means_a_login_is_needed(self):
        subject = scraper()
        subject.driver = FakeDriver(PRODUCT, cookies=())
        self.assertTrue(subject._login_required())

    def test_half_a_session_is_not_a_session(self):
        subject = scraper()
        subject.driver = FakeDriver(PRODUCT, cookies=("lzd_uid",))
        self.assertTrue(subject._login_required())

    def test_the_login_page_always_needs_a_login(self):
        subject = scraper()
        subject.driver = FakeDriver("https://member.lazada.com.ph/user/login?redirect=x")
        self.assertTrue(subject._login_required())


class TestCoveredWindow(unittest.TestCase):
    def test_chrome_keeps_a_covered_window_live(self):
        # Measured 28 Sept 2026: with the scraper window covered, the review
        # section took over 50 s to load (it waited for someone to scroll);
        # with these switches it loaded in 1.6 s, the same as uncovered.
        arguments = scraper().options.arguments
        for switch in (
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-background-timer-throttling",
            "--disable-features=CalculateNativeWinOcclusion",
        ):
            self.assertIn(switch, arguments)


class TestStarDistribution(unittest.TestCase):
    def test_feed_scores_are_listed_from_five_stars_down(self):
        # Measured on the Anker charger listing, 27 Sept 2026.
        ratings = {"scores": [13704, 102, 42, 9, 35], "rateCount": 13892}
        self.assertEqual(
            LazadaScraper._star_distribution_from_feed(ratings, 13892),
            {"1": 35, "2": 9, "3": 42, "4": 102, "5": 13704},
        )

    def test_counts_that_do_not_add_up_to_the_listing_are_rejected(self):
        self.assertIsNone(LazadaScraper._star_distribution_from_feed({"scores": [100, 0, 0, 0, 0]}, 13892))

    def test_malformed_feeds_are_rejected(self):
        for ratings in (None, {}, {"scores": [1, 2, 3]}, {"scores": ["a", 0, 0, 0, 0]}, {"scores": [5, -1, 0, 0, 0]}):
            with self.subTest(ratings=ratings):
                self.assertIsNone(LazadaScraper._star_distribution_from_feed(ratings, 0))

    def test_item_id_comes_from_the_product_link(self):
        self.assertEqual(LazadaScraper._item_id(PRODUCT), "4644812363")
        self.assertEqual(LazadaScraper._item_id("https://www.lazada.com.ph/products/x-i123456789.html"), "123456789")
        self.assertEqual(LazadaScraper._item_id("https://www.lazada.com.ph/"), "")


class TestCollectByStar(unittest.TestCase):
    """The per-star loop, with the browser replaced by canned results."""

    def run_collection(self, distribution, pages):
        subject = scraper(all_star_types=True, star_limit_per_type=60)
        product = {"link": PRODUCT, "comments": []}
        subject._read_star_distribution = lambda _product: distribution
        clicked = []

        def click(star):
            clicked.append(star)
            return True

        def collect(limit, product=None, source_filter=None, listed=None):
            reviews, stop_reason = pages[source_filter]
            product["review_checkpoint"] = {"context": str(source_filter), "stop_reason": stop_reason}
            return [{"rating": source_filter, "content": f"r{source_filter}-{i}", "author": str(i)} for i in range(reviews)]

        subject._click_rating_filter = click
        subject._collect_reviews = collect
        subject._collect_by_star(product)
        return product, clicked

    def test_an_empty_star_level_is_skipped_not_fatal(self):
        # The old loop stopped at the first star level with no reviews, so a
        # product with no 2-star reviews never had its 1- and 5-star ones read.
        distribution = {"1": 35, "2": 0, "3": 42, "4": 102, "5": 13704}
        pages = {
            3: (20, "next_button_disabled"),
            4: (60, "target_reached"),
            1: (18, "next_button_disabled"),
            5: (60, "target_reached"),
        }
        product, clicked = self.run_collection(distribution, pages)
        self.assertEqual(clicked, [3, 4, 1, 5])
        self.assertEqual(product["applied_rating_filters"], [3, 2, 4, 1, 5])
        self.assertEqual(product["star_collection"]["2"]["stop_reason"], "none_listed")
        self.assertTrue(all(item["complete"] for item in product["star_collection"].values()))
        self.assertEqual(len(product["comments"]), 158)
        self.assertEqual(product["sampling_method"], "stratified_by_star")
        self.assertEqual(product["star_quota"], 60)

    def test_a_level_cut_short_is_marked_incomplete(self):
        distribution = {"1": 35, "2": 9, "3": 42, "4": 102, "5": 13704}
        pages = {
            3: (20, "next_button_disabled"),
            2: (4, "next_button_disabled"),
            4: (60, "target_reached"),
            1: (10, "page_signature_timeout_attempt_3"),
            5: (60, "target_reached"),
        }
        product, _ = self.run_collection(distribution, pages)
        self.assertFalse(product["star_collection"]["1"]["complete"])
        self.assertTrue(product["star_collection"]["2"]["complete"])

    def test_wrong_star_results_still_stop_collection(self):
        subject = scraper(all_star_types=True, star_limit_per_type=60)
        product = {"link": PRODUCT, "comments": []}
        subject._read_star_distribution = lambda _product: None
        subject._click_rating_filter = lambda star: True

        def collect(limit, product=None, source_filter=None, listed=None):
            product["review_checkpoint"] = {"context": str(source_filter), "stop_reason": "target_reached"}
            return [{"rating": 5, "content": "wrong star", "author": "a"}]

        subject._collect_reviews = collect
        subject._collect_by_star(product)
        self.assertIn("did not return verified 3-star", product["review_filter_warning"])
        self.assertEqual(product["comments"], [])


class TestRatingOnlyPages(unittest.TestCase):
    """Live scan, 27 Sept 2026: 3- and 4-star collection stopped at the first
    page whose reviews had no text, several pages before the end."""

    def test_a_rating_only_page_is_turned_not_treated_as_the_end(self):
        import tempfile
        from pathlib import Path

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        subject = LazadaScraper(
            product_urls=[PRODUCT], pause_for_verification=False,
            output=str(Path(folder.name) / "out.json"),
        )
        pages = [
            [{"content": "maganda", "author": "a", "time": "1", "rating": 3},
             {"content": "ok lang", "author": "b", "time": "1", "rating": 3}],
            [],  # rating-only reviews: items on the page, none with text
            [{"content": "medyo mahina", "author": "c", "time": "2", "rating": 3}],
        ]
        state = {"page": 0}
        subject._review_container = lambda timeout=3: object()
        subject._reviews_from_current_page = lambda: [dict(item) for item in pages[state["page"]]]
        subject._page_has_only_unwritten_reviews = lambda: not pages[state["page"]]
        subject._current_review_signature = lambda: f"page-{state['page']}"

        def next_page(_signature):
            if state["page"] + 1 < len(pages):
                state["page"] += 1
                return True
            subject._last_pagination_stop_reason = "next_button_disabled"
            return False

        subject._next_review_page = next_page
        product = {"link": PRODUCT, "comments": []}
        collected = subject._collect_reviews(60, product=product, source_filter=3)
        self.assertEqual([item["content"] for item in collected], ["maganda", "ok lang", "medyo mahina"])
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "next_button_disabled")

    def test_a_level_that_fits_on_one_page_is_not_paged(self):
        # Measured 28 Sept 2026: each one-page star level spent ~6 s retrying a
        # next button that does not exist.
        import tempfile
        from pathlib import Path

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        subject = LazadaScraper(
            product_urls=[PRODUCT], pause_for_verification=False,
            output=str(Path(folder.name) / "out.json"),
        )
        subject._review_container = lambda timeout=3: object()
        subject._reviews_from_current_page = lambda: [
            {"content": "sira agad", "author": "a", "time": "1", "rating": 1},
            {"content": "hindi gumana", "author": "b", "time": "1", "rating": 1},
        ]
        subject._next_review_page = lambda _signature: self.fail("a one-page level must not be paged")
        product = {"link": PRODUCT, "comments": []}
        collected = subject._collect_reviews(60, product=product, source_filter=1, listed=5)
        self.assertEqual(len(collected), 2)
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "single_page")

    def test_a_one_page_level_counts_as_complete(self):
        subject = scraper(all_star_types=True, star_limit_per_type=60)
        product = {"link": PRODUCT, "comments": []}
        subject._read_star_distribution = lambda _product: {"1": 5, "2": 0, "3": 3, "4": 2, "5": 139}
        subject._click_rating_filter = lambda star: True
        seen = {}

        def collect(limit, product=None, source_filter=None, listed=None):
            seen[source_filter] = listed
            stop = "single_page" if listed is not None and listed <= 5 else "next_button_disabled"
            product["review_checkpoint"] = {"context": str(source_filter), "stop_reason": stop}
            return [{"rating": source_filter, "content": f"r{source_filter}", "author": str(source_filter)}]

        subject._collect_reviews = collect
        subject._collect_by_star(product)
        self.assertEqual(seen, {3: 3, 4: 2, 1: 5, 5: 139})
        self.assertTrue(all(entry["complete"] for entry in product["star_collection"].values()))

    def test_only_pages_with_items_and_no_text_count_as_rating_only(self):
        subject = scraper()

        class Driver:
            def __init__(self, records):
                self.records = records

            def execute_script(self, _script):
                return self.records

        subject.driver = Driver([{"content": ""}, {"content": "  "}])
        self.assertTrue(subject._page_has_only_unwritten_reviews())
        subject.driver = Driver([])
        self.assertFalse(subject._page_has_only_unwritten_reviews())
        subject.driver = Driver([{"content": ""}, {"content": "sira"}])
        self.assertFalse(subject._page_has_only_unwritten_reviews())


if __name__ == "__main__":
    unittest.main()
