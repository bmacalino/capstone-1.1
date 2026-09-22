from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.select_firecs_risk_round2 import (
    ANNOTATED_DATASET,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE,
    load_candidates,
    select_sample,
)


class FirecsRiskRound2SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidates = load_candidates(DEFAULT_SOURCE, ANNOTATED_DATASET)
        cls.selected, cls.summary = select_sample(
            candidates, DEFAULT_ARTIFACT, seed=42
        )

    def test_sample_is_balanced_by_sentiment_and_bucket(self):
        self.assertEqual(len(self.selected), 400)
        self.assertEqual(
            Counter(row["firecs_sentiment"] for row in self.selected),
            {"negative": 150, "neutral": 170, "positive": 80},
        )
        self.assertEqual(
            Counter(row["sampling_bucket"] for row in self.selected),
            {"decision_boundary": 200, "likely_risk": 101, "likely_no_risk": 99},
        )

    def test_sample_has_unique_ids_and_normalized_reviews(self):
        self.assertEqual(len({row["review_id"] for row in self.selected}), 400)
        self.assertEqual(
            len({row["normalized_sha256"] for row in self.selected}), 400
        )

    def test_source_is_train_only(self):
        self.assertEqual({row["source_split"] for row in self.selected}, {"train"})


if __name__ == "__main__":
    unittest.main()
