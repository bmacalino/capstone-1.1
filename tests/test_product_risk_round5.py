import unittest
from collections import Counter

import numpy as np

from research.compare_product_risk_round5 import (
    build_rows, reconcile, calibration_splits, fit_model, connected_groups,
)


class Round5Tests(unittest.TestCase):
    def test_overlay_preserves_originals_and_pending_decisions(self):
        original = build_rows()
        before = [dict(r) for r in original]
        rows, audit = reconcile(original)
        self.assertEqual(before, original)
        self.assertEqual(len(rows), 2424)
        self.assertEqual(sum(a['risk_detected'] != b['risk_detected'] for a,b in zip(original, rows)), 29)
        old = {r['review_id']: r for r in original}
        new = {r['review_id']: r for r in rows}
        for review_id in audit['pending_ids']:
            self.assertEqual(old[review_id]['risk_detected'], new[review_id]['risk_detected'])
        for review_id, expected in [('lazada-r1-0040', 0), ('lazada-r1-0048', 1), ('lazada-r1-0105', 0)]:
            self.assertEqual(new[review_id]['risk_detected'], expected)
        np.testing.assert_array_equal(connected_groups(original), connected_groups(rows))

    def test_calibration_remaps_subset_and_keeps_groups_disjoint(self):
        texts = np.array([f'unique review {i}' for i in range(60)], dtype=object)
        labels = np.array([i % 2 for i in range(60)])
        groups = np.array([f'group-{i // 2}' for i in range(60)])
        indices = np.arange(12, 56)
        splits = calibration_splits(texts, labels, groups, indices, 42)
        seen = []
        for train, test in splits:
            self.assertFalse(set(groups[indices[train]]) & set(groups[indices[test]]))
            self.assertTrue(set(train) | set(test) == set(range(len(indices))))
            seen.extend(test.tolist())
        self.assertEqual(Counter(seen), Counter(range(len(indices))))

    def test_both_models_return_binary_probabilities(self):
        texts = np.array([f'unit {i} broken defective does not work' if i % 2 else
                          f'unit {i} excellent quality works well' for i in range(36)], dtype=object)
        labels = np.array([i % 2 for i in range(36)])
        groups = np.array([f'group-{i // 2}' for i in range(36)])
        for config in ('lr', 'svm_1'):
            model = fit_model(config, texts, labels, groups, np.arange(36), 42)
            probability = model.predict_proba(['broken defective', 'excellent quality'])
            self.assertEqual(list(model.classes_), [0, 1])
            self.assertEqual(probability.shape, (2, 2))
            self.assertTrue(np.isfinite(probability).all())
            np.testing.assert_allclose(probability.sum(axis=1), 1.)
            self.assertGreater(probability[0, 1], probability[1, 1])


if __name__ == '__main__':
    unittest.main()
