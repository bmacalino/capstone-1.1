import csv
import json
import tempfile
import unittest
from pathlib import Path

from research.build_sentiment_training_set import (
    build,
    product_id,
    render_csv,
    scrape_files,
    star_label,
)

TEST_PRODUCT_URL = "https://www.lazada.com.ph/products/pdp-i3034831887-s14941077788.html"


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestProductId(unittest.TestCase):
    def test_links_and_ids_reduce_to_the_same_item_id(self):
        # Regression: the test CSV stores full URLs and the scrapes store links
        # or bare ids. Comparing them unreduced silently excluded nothing.
        self.assertEqual(product_id(TEST_PRODUCT_URL), "3034831887")
        self.assertEqual(product_id("https://www.lazada.com.ph/products/pdp-i5084402409.html"), "5084402409")
        self.assertEqual(product_id("https://www.temu.com/goods.html?goods_id=601100344333196"), "601100344333196")
        self.assertEqual(product_id("601100344333196"), "601100344333196")
        self.assertEqual(product_id("https://shopee.ph/Baseus-HZ11-i.1106380971.28805738530"), "28805738530")
        self.assertEqual(product_id(""), "")


class TestStarLabel(unittest.TestCase):
    def test_three_stars_are_dropped(self):
        self.assertEqual([star_label(r) for r in (1, 2, 3, 4, 5)],
                         ["negative", "negative", None, "positive", "positive"])


class TestBuild(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        root = Path(self._temp.name)
        firecs = root / "firecs"
        firecs.mkdir()
        write_csv(firecs / "FiReCS_train_set.csv", ["review", "label"], [{"review": "already in firecs train", "label": "2.0"}])
        write_csv(firecs / "FiReCS_test_set.csv", ["review", "label"], [{"review": "a firecs test review", "label": "0.0"}])
        self.market = root / "market.csv"
        write_csv(
            self.market,
            ["review_text", "product_id", "rating", "adjudicated_label"],
            [
                {"review_text": "the marketplace test review", "product_id": "https://www.lazada.com.ph/products/pdp-i999999999.html", "rating": "1", "adjudicated_label": "negative"},
                {"review_text": "another test review", "product_id": TEST_PRODUCT_URL, "rating": "5", "adjudicated_label": "positive"},
                {"review_text": "a meh one", "product_id": TEST_PRODUCT_URL, "rating": "3", "adjudicated_label": "negative"},
            ],
        )
        scrapes = root / "scrapes"
        scrapes.mkdir()

        def product(link, comments):
            return {"link": link, "comments": [{"content": text, "rating": rating} for text, rating in comments]}

        (scrapes / "lazada_a.json").write_text(json.dumps([
            product("https://www.lazada.com.ph/products/pdp-i1111111111-s1.html", [
                ("sira agad after 2 days", 1),
                ("ok naman sulit", 5),
                ("pwede na", 3),
                ("the marketplace test review", 1),   # leaked test text
                ("a firecs test review", 2),          # leaked FiReCS test text
                ("already in firecs train", 4),
                ("", 5),
                ("no rating given", None),
            ]),
            # A different review of a product that is in the marketplace test set.
            product(TEST_PRODUCT_URL, [("same product, different review", 4)]),
        ]), encoding="utf-8")
        (scrapes / "lazada_b.json").write_text(json.dumps([
            product("https://www.lazada.com.ph/products/pdp-i2222222222.html", [("OK naman, sulit!", 4), ("mahina battery", 2)]),
        ]), encoding="utf-8")
        (scrapes / "lazada_a_quality.json").write_text("[]", encoding="utf-8")
        self.files = scrape_files([scrapes])
        self.firecs = firecs

    def run_build(self):
        return build(self.files, self.market, firecs_dir=self.firecs)

    def test_side_files_are_skipped(self):
        self.assertEqual([path.name for path in self.files], ["lazada_a.json", "lazada_b.json"])

    def test_no_test_text_or_test_product_reaches_the_training_set(self):
        rows, manifest = self.run_build()
        texts = {row["text"] for row in rows}
        self.assertNotIn("the marketplace test review", texts)
        self.assertNotIn("a firecs test review", texts)
        self.assertNotIn("same product, different review", texts)
        self.assertEqual(manifest["exclusions"]["marketplace_test_text"], 1)
        self.assertEqual(manifest["exclusions"]["marketplace_test_product"], 1)
        self.assertEqual(manifest["exclusions"]["firecs_test_text"], 1)
        self.assertEqual(manifest["marketplace_test_products_excluded"], 2)

    def test_labels_follow_the_star_rule(self):
        rows, manifest = self.run_build()
        labels = {row["text"]: row["label"] for row in rows}
        self.assertEqual(labels, {
            "sira agad after 2 days": "negative",
            "ok naman sulit": "positive",
            "mahina battery": "negative",
        })
        self.assertEqual(manifest["exclusions"]["three_star_dropped"], 1)

    def test_duplicates_firecs_train_and_empty_rows_are_removed(self):
        _, manifest = self.run_build()
        # "OK naman, sulit!" normalizes to the same key as "ok naman sulit".
        self.assertEqual(manifest["exclusions"]["duplicate"], 1)
        self.assertEqual(manifest["exclusions"]["already_in_firecs_train"], 1)
        self.assertEqual(manifest["exclusions"]["no_text_or_rating"], 2)

    def test_star_agreement_is_reported_per_band(self):
        _, manifest = self.run_build()
        agreement = manifest["star_agreement_with_human_labels"]
        self.assertEqual(agreement["1-2"]["agreement"], 1.0)
        self.assertEqual(agreement["3"]["agreement"], 0.0)
        self.assertEqual(agreement["4-5"]["human_agrees"], 1)

    def test_a_rerun_reproduces_the_csv_byte_for_byte(self):
        first = render_csv(self.run_build()[0])
        second = render_csv(self.run_build()[0])
        self.assertEqual(first, second)
        self.assertNotIn(b"\r\n", first)


if __name__ == "__main__":
    unittest.main()
