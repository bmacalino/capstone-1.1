import unittest

from research.build_confirmed_product_risk_labels import (
    apply_confirmed, build_confirmed_rows, build_rows, read_json, DECISIONS,
)


class ConfirmedRiskLabelsTests(unittest.TestCase):
    def test_confirmed_labels_and_counts(self):
        rows, _, _ = build_confirmed_rows()
        labels = {r['review_id']: r['risk_detected'] for r in rows}
        self.assertEqual(labels['lazada-r1-0076'], 1)
        self.assertEqual(labels['lazada-r1-0190'], 1)
        self.assertEqual(labels['lazada-r1-0904'], 0)
        self.assertEqual(len(rows), 2424)
        self.assertEqual(sum(labels.values()), 1197)

    def test_original_rows_are_not_mutated(self):
        rows = build_rows()
        before = [dict(r) for r in rows]
        apply_confirmed(rows, read_json(DECISIONS)['decisions'])
        self.assertEqual(rows, before)

    def test_mismatched_text_and_duplicate_decisions_rejected(self):
        decisions = read_json(DECISIONS)['decisions']
        with self.assertRaises(ValueError):
            apply_confirmed(build_rows(), [decisions[0]] * 3)
        decisions[0]['review_text'] = 'another review'
        with self.assertRaises(ValueError):
            apply_confirmed(build_rows(), decisions)


if __name__ == '__main__':
    unittest.main()
