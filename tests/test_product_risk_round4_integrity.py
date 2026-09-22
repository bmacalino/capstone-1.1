import unittest
from collections import defaultdict

from research.train_domain_adapted_product_risk_round3 import build_rows, connected_groups
from defaketive_sentiment.product_risk_context import clean_risk_text


class Round4IntegrityTests(unittest.TestCase):
    def test_new_cleaner_does_not_create_cross_group_duplicates(self):
        rows = build_rows()
        groups = connected_groups(rows)
        seen = defaultdict(set)
        for row, group in zip(rows, groups):
            seen[clean_risk_text(row["text"])].add(group)
        conflicts = {text: sorted(ids) for text, ids in seen.items() if len(ids) > 1}
        self.assertEqual(conflicts, {}, "Cleaner creates cross-group duplicates; folds must be rebuilt")


if __name__ == "__main__":
    unittest.main()
