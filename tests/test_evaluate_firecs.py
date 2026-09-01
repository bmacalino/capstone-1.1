import tempfile
import unittest
from pathlib import Path

from evaluate_firecs import evaluate_firecs, load_firecs


class StubModel:
    def analyze(self, text):
        label = "negative" if "sira" in text else "positive"
        return {
            "label": label,
            "scores": {"compound": -0.5 if label == "negative" else 0.5},
            "sentiment_evidence": [],
            "risk": {"evidence": []},
        }


class TestEvaluateFiReCS(unittest.TestCase):
    def test_loads_numeric_labels_and_evaluates_split(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "sample.csv"
            dataset.write_text(
                "review,label\nsira agad,0.0\nokay lamang,1.0\nmaganda,2.0\n",
                encoding="utf-8",
            )

            rows = load_firecs(dataset)
            metrics, errors = evaluate_firecs(dataset, StubModel())

        self.assertEqual(
            [row["label"] for row in rows], ["negative", "neutral", "positive"]
        )
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["error_count"], 1)
        self.assertEqual(errors[0]["actual_label"], "neutral")

    def test_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "bad.csv"
            dataset.write_text("text,sentiment\nhello,positive\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "review and label"):
                load_firecs(dataset)


if __name__ == "__main__":
    unittest.main()
