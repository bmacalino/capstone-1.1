from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cross_validate_firecs_product_risk import DEFAULT_DATASET
from research.train_domain_adapted_product_risk import (
    DEFAULT_LAZADA,
    build_development_rows,
)


class DomainAdaptedProductRiskTests(unittest.TestCase):
    def test_builder_excludes_final_lazada_test_split(self):
        firecs, lazada = build_development_rows(DEFAULT_DATASET, DEFAULT_LAZADA)
        self.assertEqual(len(firecs), 1000)
        self.assertEqual(len(lazada), 774)
        self.assertEqual({record["split"] for record in lazada}, {"train", "validation"})
        self.assertEqual(len({record["product_id"] for record in lazada}), 17)


if __name__ == "__main__":
    unittest.main()
