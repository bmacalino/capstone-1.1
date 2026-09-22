import csv
import json
import tempfile
import unittest
from pathlib import Path

from research.error_audit import audit, classify_error


class TestErrorAudit(unittest.TestCase):
    def test_classifies_contrast_and_aspect_signals(self):
        result = classify_error("Maganda ang packaging pero mali ang kulay at mabagal ang delivery")

        self.assertEqual(result["length_bucket"], "6-15 words")
        self.assertTrue(result["flags"]["contrast"])
        self.assertTrue(result["flags"]["order_accuracy"])
        self.assertTrue(result["flags"]["delivery"])
        self.assertTrue(result["flags"]["likely_taglish"])

    def test_audit_reports_confusions_and_feature_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "errors.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["review", "actual_label", "predicted_label"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "review": "Okay pero mali ang kulay",
                        "actual_label": "negative",
                        "predicted_label": "neutral",
                    }
                )

            result = audit(path, example_limit=1)

        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["confusions"]["negative -> neutral"], 1)
        self.assertEqual(result["features"]["contrast"]["error_count"], 1)
        self.assertEqual(result["features"]["contrast"]["examples"][0]["actual"], "negative")


if __name__ == "__main__":
    unittest.main()