from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.build_firecs_risk_round2_resolved import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_EXISTING,
    DEFAULT_SELECTION,
    USER_RESOLVED_IDS,
    build_datasets,
    normalized_review,
)


class FirecsRiskRound2ResolvedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.round2, cls.combined, cls.manifest = build_datasets(
            DEFAULT_SELECTION, DEFAULT_ANNOTATIONS, DEFAULT_EXISTING
        )

    def test_round2_is_fully_resolved(self):
        self.assertEqual(len(self.round2), 400)
        self.assertEqual(
            Counter(row["risk_detected"] for row in self.round2),
            {"yes": 256, "no": 144},
        )
        self.assertEqual({row["source_split"] for row in self.round2}, {"train"})

    def test_user_resolutions_are_recorded(self):
        user_rows = {
            row["review_id"]
            for row in self.round2
            if row["annotation_source"] == "user_resolved"
        }
        self.assertEqual(user_rows, USER_RESOLVED_IDS)

    def test_combined_dataset_has_no_duplicate_ids_or_reviews(self):
        self.assertEqual(len(self.combined), 1400)
        self.assertEqual(
            len({row["review_id"] for row in self.combined}), len(self.combined)
        )
        self.assertEqual(
            len({normalized_review(row["review_text"]) for row in self.combined}),
            len(self.combined),
        )

    def test_manifest_discloses_non_independent_labels(self):
        self.assertEqual(
            self.manifest["status"],
            "development_labels_not_independent_gold_test",
        )
        self.assertEqual(self.manifest["integrity"]["unresolved_labels"], 0)


if __name__ == "__main__":
    unittest.main()
