import unittest

import numpy as np

from research.train_domain_adapted_product_risk_round3 import (
    build_rows, checked_splits, connected_groups, training_indices,
)


class Round3TrainingTests(unittest.TestCase):
    def test_duplicate_text_links_whole_products_transitively(self):
        rows = [
            {"product_id": "a", "text": "shared"},
            {"product_id": "a", "text": "unique a"},
            {"product_id": "b", "text": "shared"},
            {"product_id": "b", "text": "unique b"},
            {"product_id": "", "text": "unique b"},
            {"product_id": "c", "text": "separate"},
        ]
        groups = connected_groups(rows)
        self.assertEqual(len(set(groups[:5])), 1)
        self.assertNotEqual(groups[0], groups[5])

    def test_baseline_cannot_use_hard_examples_or_outer_test(self):
        cohorts = np.array(["lazada_hard", "firecs", "lazada_legacy", "lazada_hard", "firecs"])
        outer_train = np.array([0, 1, 2])
        np.testing.assert_array_equal(training_indices(outer_train, cohorts, augmented=False), [1, 2])
        np.testing.assert_array_equal(training_indices(outer_train, cohorts, augmented=True), [0, 1, 2])

    def test_group_folds_cover_each_row_once(self):
        texts = np.array([f"review {i}" for i in range(40)])
        labels = np.tile([0, 1], 20)
        groups = np.repeat(np.arange(20), 2)
        seen = []
        for train, test in checked_splits(texts, labels, groups, np.arange(40), folds=5, seed=42):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            seen.extend(test)
        self.assertEqual(sorted(seen), list(range(40)))

    def test_real_training_sources_include_approved_labels(self):
        rows = build_rows()
        self.assertEqual(len(rows), 2424)
        self.assertTrue(all(r.get("review_text") for r in rows))
        lookup = {r["review_id"]: r for r in rows}
        self.assertEqual(lookup["lazada-hard-r1-0015"]["risk_detected"], 0)
        self.assertEqual(lookup["lazada-hard-r1-0154"]["risk_detected"], 1)
        self.assertEqual(sum(r["cohort"] == "lazada_hard" for r in rows), 250)
        self.assertEqual(len({r["product_id"] for r in rows if r["product_id"]}), 27)


if __name__ == "__main__":
    unittest.main()
