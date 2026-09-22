from __future__ import annotations

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

from research.create_lazada_hard_example_round1 import normalized_text


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lazada_hard_example_annotation_round_1"


class LazadaHardExampleRound1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = json.loads(
            (OUTPUT / "lazada_hard_example_round1_queue.json").read_text(
                encoding="utf-8"
            )
        )
        with (ROOT / "outputs/lazada_gold_test_v1/lazada_round1_all_adjudicated.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            cls.used = list(csv.DictReader(handle))

    def test_queue_has_expected_product_disjoint_shape(self) -> None:
        self.assertEqual(len(self.rows), 250)
        counts = Counter(row["product_id"] for row in self.rows)
        self.assertEqual(len(counts), 10)
        self.assertEqual(set(counts.values()), {25})
        self.assertFalse(
            set(counts) & {row["product_id"] for row in self.used}
        )

    def test_queue_has_unique_ids_and_reviews(self) -> None:
        self.assertEqual(
            len({row["review_id"] for row in self.rows}), len(self.rows)
        )
        normalized = [normalized_text(row["review_text"]) for row in self.rows]
        self.assertEqual(len(set(normalized)), len(normalized))
        self.assertFalse(
            set(normalized)
            & {normalized_text(row["review_text"]) for row in self.used}
        )

    def test_queue_is_blind(self) -> None:
        forbidden = {
            "model_probability",
            "model_prediction",
            "distance_to_threshold",
            "selection_bucket",
        }
        self.assertTrue(all(not (set(row) & forbidden) for row in self.rows))
        self.assertTrue(all(row["human_risk_detected"] == "" for row in self.rows))


if __name__ == "__main__":
    unittest.main()
