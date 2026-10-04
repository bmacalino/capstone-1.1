import unittest
from pathlib import Path

from evaluate_firecs import load_firecs
from research.finetune_transformer import product_holdout, raw_firecs_rows
from train_firecs import DEFAULT_TEST, DEFAULT_TRAIN, audit_training_records


class TestRawFirecsRows(unittest.TestCase):
    def test_raw_rows_are_the_same_rows_as_the_normalized_audit(self):
        # The raw-text model must train on exactly the rows the TF-IDF model
        # did: same duplicates and test overlaps removed, same order.
        train, test = load_firecs(Path(DEFAULT_TRAIN)), load_firecs(Path(DEFAULT_TEST))
        normalized, normalized_labels, audit = audit_training_records(train, test)
        raw, raw_labels = raw_firecs_rows(train, test)

        self.assertEqual(len(raw), audit["clean_train_rows"])
        self.assertEqual(raw_labels, normalized_labels)
        # Raw means untouched: every kept text is a review exactly as written.
        originals = {record["review"] for record in train}
        self.assertTrue(all(text in originals for text in raw))
        self.assertNotEqual(raw[:50], normalized[:50])


class TestProductHoldout(unittest.TestCase):
    def rows(self):
        return [
            {"text": f"review {product}-{index}", "label": "positive", "product_id": product}
            for product in ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10")
            for index in range(3)
        ]

    def test_whole_products_are_held_out(self):
        fit, validation, held = product_holdout(self.rows(), 0.2, seed=42)
        self.assertEqual(len(held), 2)
        self.assertEqual({row["product_id"] for row in validation}, set(held))
        self.assertFalse({row["product_id"] for row in fit} & set(held))
        self.assertEqual(len(fit) + len(validation), 30)

    def test_the_split_is_reproducible(self):
        self.assertEqual(product_holdout(self.rows(), 0.2, 42)[2], product_holdout(self.rows(), 0.2, 42)[2])

    def test_at_least_one_product_is_held_out(self):
        self.assertEqual(len(product_holdout(self.rows()[:3], 0.1, 42)[2]), 1)



class TestServedTextMatchesTraining(unittest.TestCase):
    def test_the_app_feeds_a_normalized_model_exactly_its_training_text(self):
        # Training uses train_firecs.normalized_text(); the app passes the
        # sentiment model's normalizer output and the classifier adds the
        # feature markers. Any drift here would silently degrade the model.
        from defaketive_sentiment.model import DefaketiveSentimentModel
        from defaketive_sentiment.trained import feature_text
        from train_firecs import normalized_text

        train, _ = load_firecs(Path(DEFAULT_TRAIN)), None
        samples = [record["review"] for record in train[:2000]] + [
            "  Hndi GUMANA agad!!  ", "Sira agad 😡😡 pero mabilis delivery", "10 out of 10",
        ]
        for review in samples:
            served = feature_text(DefaketiveSentimentModel.normalize(review).strip())
            self.assertEqual(served, normalized_text(review), review)


if __name__ == "__main__":
    unittest.main()
