import json
import unittest

from research.compare_product_risk_round6 import (
    build_confirmed_rows, build_rows, frozen_splits, read_json, PREVIOUS, DESTINATION,
)


class Round6Tests(unittest.TestCase):
    def test_confirmed_dataset_matches_persisted_json(self):
        rows, _, _ = build_confirmed_rows()
        self.assertEqual(json.loads(json.dumps(rows)), read_json(DESTINATION / 'development_rows.json'))
        self.assertEqual(sum(r['risk_detected'] for r in rows), 1197)

    def test_frozen_folds_preserve_previous_product_holdouts(self):
        rows, _, _ = build_confirmed_rows()
        previous = read_json(PREVIOUS / 'oof_predictions.json')
        splits = frozen_splits(rows, build_rows(), previous)
        self.assertEqual(len(splits), 5)
        seen = []
        for train, test in splits:
            train_products = {rows[i]['product_id'] for i in train if rows[i]['product_id']}
            test_products = {rows[i]['product_id'] for i in test if rows[i]['product_id']}
            self.assertFalse(train_products & test_products)
            seen.extend(test.tolist())
        self.assertEqual(sorted(seen), list(range(2424)))

    def test_changed_assignment_is_rejected(self):
        rows, _, _ = build_confirmed_rows()
        previous = read_json(PREVIOUS / 'oof_predictions.json')
        previous[0]['fold'] = 99
        with self.assertRaisesRegex(ValueError, 'Outer folds differ'):
            frozen_splits(rows, build_rows(), previous)


if __name__ == '__main__':
    unittest.main()
