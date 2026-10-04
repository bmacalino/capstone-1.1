import json
import tempfile
import unittest
from pathlib import Path

from defaketive_sentiment import hazard_model
from defaketive_sentiment.hazard_model import HazardBadgeModel
from defaketive_sentiment.model import DefaketiveSentimentModel
from defaketive_sentiment.transformer_sentiment import (
    DEFAULT_MODEL_DIR,
    MANIFEST_NAME,
    TransformerModelError,
    TransformerSentimentClassifier,
    file_sha256,
    read_manifest,
)


class RecordingClassifier:
    """Stands in for a classifier and records the text it was given."""

    model_version = "fake-sentiment-1"
    label_source = "fake_classifier"

    def __init__(self, wants_raw_text, fail=False):
        self.wants_raw_text = wants_raw_text
        self.fail = fail
        self.seen = []

    def predict(self, text):
        if self.fail:
            raise RuntimeError("inference failed")
        self.seen.append(text)
        return {"label": "positive", "confidence": 0.9, "probabilities": {"positive": 0.9}, "evidence": []}


class TestManifestGate(unittest.TestCase):
    """The app must refuse a model it cannot prove is the one measured."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.folder = Path(self._temp.name)
        (self.folder / "model.onnx").write_bytes(b"not really a model")
        (self.folder / "tokenizer.json").write_bytes(b"{}")

    def write_manifest(self, **overrides):
        manifest = {
            "model_version": "defaketive-test-onnx-int8-00000000",
            "labels": ["negative", "neutral", "positive"],
            "files": {name: file_sha256(self.folder / name) for name in ("model.onnx", "tokenizer.json")},
            "gate": "test gate",
            "qualified_for_integration": True,
        }
        manifest.update(overrides)
        (self.folder / MANIFEST_NAME).write_bytes(json.dumps(manifest).encode("utf-8"))

    def test_a_missing_model_is_refused_with_rebuild_instructions(self):
        with self.assertRaises(TransformerModelError) as caught:
            TransformerSentimentClassifier(self.folder / "absent")
        self.assertIn("slim_sentiment_model.py", str(caught.exception))

    def test_a_changed_model_file_is_refused(self):
        self.write_manifest()
        (self.folder / "model.onnx").write_bytes(b"a different model")
        with self.assertRaises(TransformerModelError) as caught:
            read_manifest(self.folder)
        self.assertIn("SHA-256", str(caught.exception))

    def test_a_missing_file_is_refused(self):
        self.write_manifest()
        (self.folder / "tokenizer.json").unlink()
        with self.assertRaises(TransformerModelError):
            read_manifest(self.folder)

    def test_a_model_that_failed_the_gate_is_refused(self):
        self.write_manifest(qualified_for_integration=False)
        with self.assertRaises(TransformerModelError) as caught:
            read_manifest(self.folder)
        self.assertIn("gate", str(caught.exception))

    def test_an_intact_qualified_manifest_is_accepted(self):
        self.write_manifest()
        self.assertEqual(read_manifest(self.folder)["labels"][0], "negative")


class TestClassifierInput(unittest.TestCase):
    REVIEW = "  Hndi GUMANA agad!!  "

    def test_the_transformer_gets_the_review_as_written(self):
        # It was trained on raw text; normalizing would feed it input it never saw.
        classifier = RecordingClassifier(wants_raw_text=True)
        DefaketiveSentimentModel(classifier=classifier).analyze(self.REVIEW)
        self.assertEqual(classifier.seen, ["Hndi GUMANA agad!!"])

    def test_the_tfidf_path_still_gets_normalized_text(self):
        classifier = RecordingClassifier(wants_raw_text=False)
        model = DefaketiveSentimentModel(classifier=classifier)
        model.analyze(self.REVIEW)
        self.assertEqual(classifier.seen, [model.normalize(self.REVIEW)])

    def test_the_label_source_names_the_classifier(self):
        result = DefaketiveSentimentModel(classifier=RecordingClassifier(True)).analyze("ok naman")
        self.assertEqual(result["label_source"], "fake_classifier")
        self.assertEqual(result["model_version"], "fake-sentiment-1")

    def test_a_failing_injected_classifier_raises_instead_of_using_vader(self):
        model = DefaketiveSentimentModel(classifier=RecordingClassifier(True, fail=True))
        with self.assertRaises(RuntimeError):
            model.analyze("sira agad")

    def test_the_rating_rule_still_applies(self):
        class Neutral(RecordingClassifier):
            def predict(self, text):
                return {"label": "neutral", "confidence": 0.5, "probabilities": {}, "evidence": []}

        result = DefaketiveSentimentModel(classifier=Neutral(True)).analyze("dumating na", rating=1)
        self.assertEqual(result["label"], "negative")
        self.assertEqual(result["label_adjustment"]["source"], "rating_context")


class TestVersionComposition(unittest.TestCase):
    def test_the_badge_version_changes_with_the_sentiment_model(self):
        # Reports show the sentiment split, so a new sentiment model must make
        # saved analyses stale.
        sentiment = DefaketiveSentimentModel(classifier=RecordingClassifier(True))
        model = HazardBadgeModel(allow_unqualified=True, sentiment_model=sentiment)
        self.assertEqual(model.model_version, f"{hazard_model.MODEL_VERSION}+fake-sentiment-1")
        self.assertEqual(model.analyze("ok naman")["model_version"], model.model_version)


@unittest.skipUnless((DEFAULT_MODEL_DIR / MANIFEST_NAME).is_file(), "served sentiment model not installed")
class TestServedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = TransformerSentimentClassifier()

    def test_predictions_are_well_formed(self):
        result = self.classifier.predict("Sira agad after 2 days, hindi na nag charge")
        self.assertIn(result["label"], {"negative", "neutral", "positive"})
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=4)
        self.assertEqual(result["evidence"], [])

    def test_clear_reviews_get_the_obvious_label(self):
        self.assertEqual(self.classifier.predict("Sira agad after 2 days, hindi na nag charge")["label"], "negative")
        self.assertEqual(self.classifier.predict("Ang ganda! Sulit na sulit, salamat seller")["label"], "positive")

    def test_the_five_star_reviews_from_the_bug_report_are_not_read_as_complaints(self):
        # Four 5-star Lazada reviews the TF-IDF model labelled negative.
        reviews = [
            "Another quality product from a reputable brand. Did a quick charge, and it always never disappoint.",
            "10 out of 10",
            "This item possesses a quiet elegance that transcends its ordinary purpose, "
            "blending practicality with a subtle sense of refinement.",
            "Compatibility:Compatible with various USB C devices.,",
        ]
        model = DefaketiveSentimentModel(classifier=self.classifier)
        labels = [model.analyze(review, rating=5)["label"] for review in reviews]
        self.assertNotIn("negative", labels)

    def test_a_review_scores_the_same_alone_and_in_a_batch(self):
        texts = ["maganda naman", "sira agad", "ok lang, medyo matagal dumating"]
        self.assertEqual(self.classifier.predict_batch(texts), [self.classifier.predict(text) for text in texts])


if __name__ == "__main__":
    unittest.main()
