import csv
import tempfile
import unittest
from pathlib import Path

from train_firecs import (
    audit_training_records,
    audit_key,
    candidate_factories,
    load_sentitaglish,
    load_tagasenti,
    normalized_text,
)


class TestTrainFiReCS(unittest.TestCase):
    def test_audit_removes_duplicates_and_cross_split_overlap(self):
        train = [
            {"review": "Maganda!", "label": "positive"},
            {"review": "  MAGANDA!  ", "label": "positive"},
            {"review": "Sira agad", "label": "negative"},
            {"review": "Okay lamang", "label": "neutral"},
        ]
        test = [{"review": "sira agad", "label": "negative"}]

        texts, labels, audit = audit_training_records(train, test)

        self.assertEqual(texts, [normalized_text("Maganda!"), normalized_text("Okay lamang")])
        self.assertEqual(labels, ["positive", "neutral"])
        self.assertEqual(audit["duplicate_train_rows_removed"], 1)
        self.assertEqual(audit["cross_split_train_rows_removed"], 1)
        self.assertEqual(audit["conflicting_duplicate_texts"], 0)

    def test_audit_key_ignores_punctuation_and_case(self):
        self.assertEqual(audit_key("  MAGANDA!!!  "), audit_key("maganda"))
        self.assertEqual(audit_key("Hindi-gumagana"), audit_key("hindi gumagana"))

    def test_training_candidates_are_reproducibly_declared(self):
        self.assertEqual(
            set(candidate_factories()),
            {
                "word_linear_svc",
                "character_linear_svc",
                "word_character_linear_svc",
                "word_character_logistic_regression",
            },
        )

    def test_sentitaglish_loader_excludes_mixed_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sentitaglish.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["review", "sentiment"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"review": "Pangit", "sentiment": "1"},
                        {"review": "Okay lang", "sentiment": "2"},
                        {"review": "Maganda", "sentiment": "3"},
                        {"review": "Maganda pero sira", "sentiment": "4"},
                    ]
                )

            records, audit = load_sentitaglish(path)

        self.assertEqual(
            [record["label"] for record in records],
            ["negative", "neutral", "positive"],
        )
        self.assertEqual(audit["mixed_rows_excluded"], 1)
        self.assertTrue(all(record["source"] == "sentitaglish_human" for record in records))

    def test_tagasenti_loader_records_mixed_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tagasenti.csv"
            with path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["sentence", "label"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"sentence": "Hindi gumagana", "label": "0"},
                        {"sentence": "Karaniwan lang", "label": "1"},
                        {"sentence": "Sulit", "label": "2"},
                    ]
                )

            records, audit = load_tagasenti(path)

        self.assertEqual(len(records), 3)
        self.assertIn("no row-level source field", audit["provenance_warning"])
        self.assertTrue(
            all(record["source"] == "tagasenti_mixed_provenance" for record in records)
        )


if __name__ == "__main__":
    unittest.main()
