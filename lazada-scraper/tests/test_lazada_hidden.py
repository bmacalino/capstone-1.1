import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.lazada_scraper import LazadaScraper, build_parser

PRODUCT = "https://www.lazada.com.ph/products/pdp-i4644812363-s27518600306.html"
PUNISH_PAGE = (
    '<script>var url = "//my.lazada.com.ph//pdp/review/getReviewList/_____tmd_____/punish'
    '?x5secdata=abc";window._config_ = {"action":"captcha"};</script>'
)


FOLDER = tempfile.TemporaryDirectory()
OUTPUTS = itertools.count()


def tearDownModule():
    FOLDER.cleanup()


def scraper(**options):
    # Each scraper saves to its own file, so no test reads another's scan.
    return LazadaScraper(
        product_urls=[PRODUCT], pause_for_verification=False,
        output=str(Path(FOLDER.name) / f"out{next(OUTPUTS)}.json"), **options,
    )


def item(review_id, rating, content="ok naman", **extra):
    return {
        "reviewRateId": review_id, "buyerName": "J***n", "rating": rating,
        "reviewTime": "31 May 2026", "reviewContent": content,
        "skuInfo": "Variation:ZQS1202 SPEAKER", "likeCount": 2, "replies": [], **extra,
    }


def model(items, total_pages=1, scores=(10, 4, 3, 2, 1)):
    return {
        "items": items,
        "paging": {"totalItems": sum(scores), "totalPages": total_pages, "currentPage": 1},
        "ratings": {"average": 4.6, "rateCount": sum(scores), "scores": list(scores)},
    }


class TestWindowModes(unittest.TestCase):
    def test_a_scan_is_visible_and_reads_the_page_unless_asked(self):
        subject = scraper()
        self.assertEqual(subject.window_mode, "visible")
        self.assertEqual(subject.reviews_from, "page")
        self.assertFalse(any(arg.startswith("--window-position") for arg in subject.options.arguments))

    def test_an_offscreen_window_opens_outside_the_screen(self):
        # Measured 12 Oct 2026: with this switch Chrome starts at x = -20000,
        # so the window never flashes up before it can be moved.
        subject = scraper(window_mode="offscreen")
        self.assertIn("--window-position=-20000,0", subject.options.arguments)

    def test_a_hidden_scan_reads_the_feed_unless_told_to_read_the_page(self):
        self.assertEqual(scraper(window_mode="offscreen").reviews_from, "feed")
        self.assertEqual(scraper(window_mode="headless").reviews_from, "feed")
        self.assertEqual(scraper(window_mode="headless", reviews_from="page").reviews_from, "page")
        self.assertEqual(scraper(reviews_from="feed").reviews_from, "feed")

    def test_unknown_modes_are_refused(self):
        with self.assertRaises(ValueError):
            scraper(window_mode="minimised")
        with self.assertRaises(ValueError):
            scraper(reviews_from="api")

    def test_the_command_line_carries_both_choices(self):
        args = build_parser().parse_args(["--product-url", PRODUCT])
        self.assertEqual((args.window, args.reviews_from), ("visible", None))
        args = build_parser().parse_args(["--product-url", PRODUCT, "--window", "headless", "--reviews-from", "page"])
        self.assertEqual((args.window, args.reviews_from), ("headless", "page"))

    def start(self, subject):
        with (
            patch("src.lazada_scraper.SafeChrome") as chrome,
            patch("src.lazada_scraper._position_scraper_window") as position,
            patch.object(LazadaScraper, "_detect_chrome_major", return_value=None),
        ):
            driver = subject._start_browser()
        return chrome, position, driver

    def test_only_headless_starts_chrome_headless(self):
        for mode, headless in (("visible", False), ("offscreen", False), ("headless", True)):
            with self.subTest(mode=mode):
                chrome, position, _ = self.start(scraper(window_mode=mode))
                self.assertIs(chrome.call_args.kwargs["headless"], headless)
                # Only a visible window is placed beside the app.
                self.assertEqual(position.called, mode == "visible")

    def test_a_second_chrome_gets_fresh_options(self):
        # undetected-chromedriver refuses an options object it has used.
        subject = scraper(window_mode="headless")
        first, _, _ = self.start(subject)
        second, _, _ = self.start(subject)
        self.assertIsNot(first.call_args.kwargs["options"], second.call_args.kwargs["options"])

    def test_the_first_scan_with_no_saved_session_is_visible(self):
        subject = scraper(window_mode="offscreen")
        subject._has_saved_session = lambda: False
        modes = []

        def start():
            modes.append(subject.window_mode)
            raise KeyboardInterrupt

        subject._start_browser = start
        with self.assertRaises(KeyboardInterrupt):
            subject.execute()
        self.assertEqual(modes, ["visible"])
        self.assertFalse(any(arg.startswith("--window-position") for arg in subject.options.arguments))

    def test_a_saved_session_keeps_the_scan_hidden(self):
        subject = scraper(window_mode="headless")
        subject._has_saved_session = lambda: True
        modes = []

        def start():
            modes.append(subject.window_mode)
            raise KeyboardInterrupt

        subject._start_browser = start
        with self.assertRaises(KeyboardInterrupt):
            subject.execute()
        self.assertEqual(modes, ["headless"])


class TestShowingTheWindow(unittest.TestCase):
    def test_an_offscreen_window_is_moved_onto_the_screen_once(self):
        subject = scraper(window_mode="offscreen")
        subject.driver = Mock()
        with patch("src.lazada_scraper._position_scraper_window") as position:
            subject._show_window("test")
            subject._show_window("test")
        position.assert_called_once_with(subject.driver)
        self.assertEqual(subject.window_mode, "visible")

    def test_a_headless_scan_carries_on_in_a_visible_chrome(self):
        subject = scraper(window_mode="headless")
        hidden, shown = Mock(current_url=PRODUCT), Mock()
        subject.driver = hidden
        steps = []
        subject._save_cookies = lambda: steps.append("save cookies")
        subject._load_cookies = lambda: steps.append("load cookies")
        subject._start_browser = lambda: steps.append(f"start {subject.window_mode}") or shown
        subject._safe_get = lambda url, check_verification=True: steps.append((url, check_verification))
        subject._show_window("test")
        hidden.quit.assert_called_once()
        self.assertIs(subject.driver, shown)
        self.assertEqual(steps, [
            "save cookies", "start visible",
            (LazadaScraper.BASE_URL, False), "load cookies", (PRODUCT, False),
        ])

    def test_a_verification_page_brings_a_hidden_scan_on_screen(self):
        subject = scraper(window_mode="headless")
        subject._verification_detected = lambda: True
        shown = []
        subject._show_window = shown.append
        self.assertTrue(subject._check_verification())
        self.assertEqual(len(shown), 1)

    def test_no_verification_page_leaves_the_scan_hidden(self):
        subject = scraper(window_mode="headless")
        subject._verification_detected = lambda: False
        subject._show_window = lambda reason: self.fail("the window must stay hidden")
        self.assertFalse(subject._check_verification())

    def test_a_needed_login_brings_a_hidden_scan_on_screen(self):
        subject = scraper(window_mode="offscreen")
        subject._load_cookies = lambda: False
        subject._login_required = lambda: True
        shown = []
        subject._show_window = shown.append
        self.assertFalse(subject._restore_or_request_login())
        self.assertEqual(len(shown), 1)


class TestFeedReviews(unittest.TestCase):
    def test_a_feed_entry_becomes_the_record_the_page_reader_writes(self):
        entry = item(
            901, 2, "  Mahina  ang tunog\n🔋Battery Life:mabilis malowbat ",
            replies=[{"reviewContent": "Pasensya na po"}],
        )
        self.assertEqual(scraper()._review_from_feed_item(entry), {
            "comment_id": "901",
            "author": "J***n",
            "rating": 2,
            "time": "2026-05-31",
            "content": "Mahina ang tunog 🔋Battery Life:mabilis malowbat",
            "variation": "Variation:ZQS1202 SPEAKER",
            "seller_respond": "Pasensya na po",
            "like_count": 2,
        })

    def test_recent_reviews_keep_their_relative_date(self):
        self.assertEqual(LazadaScraper._feed_date("2 weeks ago"), "2 weeks ago")
        self.assertEqual(LazadaScraper._feed_date("7 Jun 2026"), "2026-06-07")
        self.assertEqual(LazadaScraper._feed_date(None), "")

    def fetch(self, reply):
        subject = scraper()
        subject.driver = Mock()
        if isinstance(reply, Exception):
            subject.driver.execute_async_script.side_effect = reply
        else:
            subject.driver.execute_async_script.return_value = reply
        return subject, subject._fetch_review_feed("4644812363", star=3, page=2)

    def test_the_feed_is_asked_for_one_page_of_one_star_level(self):
        subject, (found, problem) = self.fetch(json.dumps({"model": model([item(1, 3)])}))
        self.assertEqual((len(found["items"]), problem), (1, ""))
        url = subject.driver.execute_async_script.call_args.args[1]
        for part in ("itemId=4644812363", "filter=3", "pageNo=2", "pageSize=50"):
            self.assertIn(part, url)

    def test_a_verification_reply_is_recognised_and_nothing_is_read(self):
        # The reply a plain request received on 12 Oct 2026.
        self.assertEqual(self.fetch(PUNISH_PAGE)[1], (None, "verification"))
        self.assertEqual(self.fetch(json.dumps({"rgv587_flag": "sm", "url": "x"}))[1], (None, "verification"))

    def test_anything_else_that_is_not_reviews_is_unavailable(self):
        for reply in ('{"fetchError": "TypeError: Failed to fetch"}', "", "<html></html>", '{"model": {}}', RuntimeError("timeout")):
            with self.subTest(reply=reply):
                self.assertEqual(self.fetch(reply)[1], (None, "unavailable"))

    def collect(self, pages, limit=60, star=None, **options):
        subject = scraper(**options)
        asked = []

        def feed_page(_product, _item_id, star=0, page=1):
            asked.append((star, page))
            return pages[page - 1] if page <= len(pages) else (model([], total_pages=0), "")

        subject._feed_page = feed_page
        with patch("src.lazada_scraper.time.sleep"):
            result = subject._collect_reviews_from_feed({"link": PRODUCT}, limit, star=star)
        return result, asked

    def test_the_two_star_list_also_holds_one_star_reviews(self):
        # Measured 12 Oct 2026: filter 2 returned ratings [2, 1, 1, 2, 1].
        pages = [(model([item(1, 2), item(2, 1), item(3, 1), item(4, 2), item(5, 1)]), "")]
        (reviews, stop_reason, _), asked = self.collect(pages, star=2)
        self.assertEqual([review["comment_id"] for review in reviews], ["1", "4"])
        self.assertTrue(all(review["source_rating_filter"] == 2 for review in reviews))
        self.assertEqual(stop_reason, "feed_ended")
        self.assertEqual(asked, [(2, 1)])

    def test_collection_stops_at_the_quota_without_asking_for_more(self):
        pages = [(model([item(i, 5) for i in range(1, 51)], total_pages=9), ""),
                 (model([item(i, 5) for i in range(51, 101)], total_pages=9), "")]
        (reviews, stop_reason, pages_read), asked = self.collect(pages, limit=60, star=5)
        self.assertEqual((len(reviews), stop_reason, pages_read), (60, "target_reached", 2))
        self.assertEqual(asked, [(5, 1), (5, 2)])

    def test_an_empty_page_is_the_end_of_the_list(self):
        # Measured 12 Oct 2026: 64 four-star ratings, 32 with text, came as
        # one page of 32 and then an empty page.
        pages = [(model([item(i, 4) for i in range(32)], total_pages=2), "")]
        (reviews, stop_reason, pages_read), _ = self.collect(pages, star=4)
        self.assertEqual((len(reviews), stop_reason, pages_read), (32, "feed_ended", 2))

    def test_the_feed_lists_only_the_first_three_hundred_entries(self):
        # Measured 12 Oct 2026 on a listing with 2,111 one- and two-star
        # ratings: page 6 of 50 was served, page 7 came back with no model,
        # and about one entry in five was a 2-star review.
        pages = [(model([item(page * 100 + i, 2 if i < 10 else 1) for i in range(50)], total_pages=43), "")
                 for page in range(6)]
        (reviews, stop_reason, pages_read), asked = self.collect(pages, limit=60, star=2)
        self.assertEqual((len(reviews), stop_reason, pages_read), (60, "target_reached", 6))
        (reviews, stop_reason, pages_read), asked = self.collect(pages, limit=100, star=2)
        self.assertEqual((len(reviews), stop_reason, pages_read), (60, "feed_limit", 6))
        # The page past the limit is not asked for.
        self.assertEqual(asked[-1], (2, 6))

    def test_a_reply_with_no_model_is_the_end_of_what_the_feed_lists(self):
        past_the_limit = '{"success":true,"model":null,"msgCode":null,"httpStatusCode":200}'
        found, problem = self.fetch(past_the_limit)[1]
        self.assertEqual((found["items"], found["beyond_depth"], problem), ([], True, ""))
        beyond = ({"items": [], "paging": {}, "beyond_depth": True}, "")
        (reviews, stop_reason, _), _ = self.collect([(model([item(1, 5)], total_pages=9), ""), beyond], star=5)
        self.assertEqual((len(reviews), stop_reason), (1, "feed_limit"))
        # No list at all on a first page is a failure, not an end.
        (_, stop_reason, _), _ = self.collect([beyond], star=5)
        self.assertEqual(stop_reason, "feed_unavailable")

    def test_ratings_without_text_and_repeats_are_left_out(self):
        pages = [(model([item(1, 3), item(2, 3, "   "), item(1, 3)]), "")]
        (reviews, _, _), _ = self.collect(pages, star=3)
        self.assertEqual([review["comment_id"] for review in reviews], ["1"])
        (reviews, _, _), _ = self.collect(pages, star=3, include_empty=True)
        self.assertEqual([review["comment_id"] for review in reviews], ["1", "2"])

    def test_a_refused_feed_stops_collection_and_says_why(self):
        pages = [(model([item(1, 1)], total_pages=3), ""), (None, "verification")]
        (reviews, stop_reason, _), _ = self.collect(pages, star=1)
        self.assertEqual((len(reviews), stop_reason), (1, "feed_verification"))

    def test_without_a_star_level_every_review_is_taken(self):
        pages = [(model([item(1, 5), item(2, 1)]), "")]
        (reviews, _, _), asked = self.collect(pages)
        self.assertEqual(len(reviews), 2)
        self.assertNotIn("source_rating_filter", reviews[0])
        self.assertEqual(asked, [(0, 1)])


class TestFeedRefusal(unittest.TestCase):
    def test_a_refusal_shows_the_window_once_and_asks_again(self):
        subject = scraper(window_mode="headless")
        replies = [(None, "verification"), (model([item(1, 5)]), ""), (None, "verification")]
        subject._fetch_review_feed = lambda item_id, star=0, page=1: replies.pop(0)
        steps = []
        subject._show_window = lambda reason: steps.append("show")
        subject._safe_get = lambda url, check_verification=True: steps.append((url, check_verification))
        found, _ = subject._feed_page({"link": PRODUCT}, "4644812363")
        self.assertEqual(len(found["items"]), 1)
        # The product page is reopened with the usual verification wait.
        self.assertEqual(steps, ["show", (PRODUCT, True)])
        self.assertEqual(subject._feed_page({"link": PRODUCT}, "4644812363"), (None, "verification"))
        self.assertEqual(steps, ["show", (PRODUCT, True)])


class TestFeedScan(unittest.TestCase):
    def by_star(self, results, distribution):
        subject = scraper(all_star_types=True, star_limit_per_type=60, window_mode="headless")

        def collect(_product, limit, star=None):
            count, stop_reason = results[star]
            return [subject._review_from_feed_item(item(star * 1000 + i, star)) for i in range(count)], stop_reason, 1

        subject._collect_reviews_from_feed = collect
        product = {"link": PRODUCT, "comments": []}
        return subject._collect_by_star_from_feed(product, distribution), product

    def test_each_star_level_is_recorded_the_way_the_page_reader_records_it(self):
        results = {3: (20, "feed_ended"), 4: (60, "target_reached"), 1: (18, "feed_ended"), 5: (60, "target_reached")}
        done, product = self.by_star(results, {"1": 35, "2": 0, "3": 42, "4": 102, "5": 13704})
        self.assertTrue(done)
        self.assertEqual(product["applied_rating_filters"], [3, 2, 4, 1, 5])
        self.assertEqual(product["star_collection"]["2"]["stop_reason"], "none_listed")
        self.assertEqual(product["star_collection"]["3"], {
            "listed": 42, "collected": 20, "stop_reason": "feed_ended", "complete": True,
        })
        self.assertTrue(all(entry["complete"] for entry in product["star_collection"].values()))
        self.assertEqual(len(product["comments"]), 158)
        self.assertEqual([review["rating"] for review in product["comments"]][:19], [1] * 18 + [3])
        self.assertEqual(product["sampling_method"], "stratified_by_star")
        self.assertEqual(product["star_quota"], 60)
        self.assertEqual(product["review_order"], "review_feed_default_within_star")
        self.assertEqual(product["review_status"], "complete")

    def test_a_level_cut_short_is_marked_incomplete(self):
        results = {3: (20, "feed_ended"), 2: (4, "max_review_pages_reached"), 4: (9, "feed_ended"),
                   1: (3, "feed_ended"), 5: (60, "target_reached")}
        _, product = self.by_star(results, {"1": 35, "2": 9, "3": 42, "4": 102, "5": 13704})
        self.assertFalse(product["star_collection"]["2"]["complete"])

    def test_a_level_cut_at_the_feeds_depth_counts_if_it_holds_any_review(self):
        results = {3: (20, "feed_ended"), 2: (58, "feed_limit"), 4: (60, "target_reached"),
                   1: (60, "target_reached"), 5: (60, "target_reached")}
        done, product = self.by_star(results, {"1": 1662, "2": 449, "3": 799, "4": 977, "5": 37797})
        self.assertTrue(done)
        self.assertEqual(product["star_collection"]["2"], {
            "listed": 449, "collected": 58, "stop_reason": "feed_limit", "complete": True,
        })
        results[2] = (0, "feed_limit")
        _, product = self.by_star(results, {"1": 1662, "2": 449, "3": 799, "4": 977, "5": 37797})
        self.assertFalse(product["star_collection"]["2"]["complete"])

    def test_a_feed_that_stops_answering_hands_over_to_the_page(self):
        results = {3: (20, "feed_ended"), 2: (0, "feed_verification")}
        done, _ = self.by_star(results, {"1": 35, "2": 9, "3": 42, "4": 102, "5": 13704})
        self.assertFalse(done)

    def feed_scan(self, first_reply, **options):
        subject = scraper(window_mode="headless", **options)
        subject._feed_page = lambda product, item_id, star=0, page=1: first_reply
        return subject

    def test_the_listing_figures_come_from_the_feed(self):
        subject = self.feed_scan((model([item(1, 5)], scores=(2782, 64, 87, 72, 235)), ""), all_star_types=True)
        seen = {}
        subject._collect_by_star_from_feed = lambda product, distribution: seen.update(distribution) or True
        product = {"link": PRODUCT}
        self.assertTrue(subject._scrape_reviews_from_feed(product))
        self.assertEqual(seen, {"1": 235, "2": 72, "3": 87, "4": 64, "5": 2782})
        self.assertEqual((product["rating"], product["total_rating"]), (4.6, 3240))
        self.assertEqual(product["review_star_distribution_source"], "lazada_review_feed")
        self.assertEqual(product["review_source"], "lazada_review_feed")

    def test_no_feed_means_the_page_is_read(self):
        self.assertFalse(self.feed_scan((None, "verification"))._scrape_reviews_from_feed({"link": PRODUCT}))
        self.assertFalse(self.feed_scan((model([]), ""))._scrape_reviews_from_feed({"link": "https://www.lazada.com.ph/"}))
        broken = model([item(1, 5)])
        broken["ratings"] = {"scores": [1, 2, 3]}
        self.assertFalse(self.feed_scan((broken, ""))._scrape_reviews_from_feed({"link": PRODUCT}))

    def test_a_listing_with_no_ratings_is_recorded_as_having_no_reviews(self):
        subject = self.feed_scan((model([], total_pages=0, scores=(0, 0, 0, 0, 0)), ""), all_star_types=True)
        product = {"link": PRODUCT}
        self.assertTrue(subject._scrape_reviews_from_feed(product))
        self.assertEqual(product["review_status"], "no_reviews")

    def test_a_plain_scan_takes_the_feed_in_its_own_order(self):
        subject = self.feed_scan((model([item(1, 5), item(2, 1, "sira agad")]), ""), review_limit=30)
        product = {"link": PRODUCT}
        with patch("src.lazada_scraper.time.sleep"):
            self.assertTrue(subject._scrape_reviews_from_feed(product))
        self.assertEqual([review["comment_id"] for review in product["comments"]], ["1", "2"])
        self.assertEqual(product["review_order"], "review_feed_default")
        self.assertEqual(product["review_status"], "complete")
        self.assertEqual(product["review_checkpoint"]["stop_reason"], "feed_ended")
        self.assertTrue(subject._review_checkpoint_is_terminal(product))

    def details(self, subject, product):
        subject.driver = Mock()
        subject._safe_get = lambda url, check_verification=True: None
        subject._first_text = lambda root, selectors: ""
        subject._product_image_url = lambda root: ""
        subject._scrape_details(product)

    def test_a_feed_scan_never_scrolls_to_the_review_section(self):
        subject = scraper(window_mode="headless")
        subject._scrape_reviews_from_feed = lambda product: True
        subject._scroll_to_reviews = lambda: self.fail("the page's review section must not be needed")
        self.details(subject, {"link": PRODUCT})

    def test_reviews_that_will_not_load_bring_a_hidden_scan_on_screen(self):
        # Seen 12 Oct 2026 with a signed-out browser: Lazada held the review
        # list back for a check shown on the page, which a hidden window hides.
        subject = scraper(window_mode="headless", reviews_from="page")
        steps = []
        subject._show_window = lambda reason: steps.append("show")

        def scroll():
            steps.append("scroll")
            if steps.count("scroll") == 2:
                raise KeyboardInterrupt
            return False

        subject._scroll_to_reviews = scroll
        subject.driver = Mock()
        subject._safe_get = lambda url, check_verification=True: steps.append("open")
        subject._first_text = lambda root, selectors: ""
        subject._product_image_url = lambda root: ""
        with self.assertRaises(KeyboardInterrupt):
            subject._scrape_details({"link": PRODUCT})
        self.assertEqual(steps, ["open", "scroll", "show", "open", "scroll"])

    def test_feed_reviews_are_dropped_before_the_page_is_read_instead(self):
        subject = scraper(window_mode="headless")

        def refused(product):
            product["review_source"] = "lazada_review_feed"
            product["comments"] = [{"comment_id": "1", "content": "from the feed"}]
            return False

        subject._scrape_reviews_from_feed = refused
        seen = []

        def scroll():
            seen.append(list(product["comments"]))
            raise KeyboardInterrupt

        subject._scroll_to_reviews = scroll
        product = {"link": PRODUCT}
        with self.assertRaises(KeyboardInterrupt):
            self.details(subject, product)
        self.assertEqual(seen, [[]])
        self.assertNotIn("review_source", product)


if __name__ == "__main__":
    unittest.main()
