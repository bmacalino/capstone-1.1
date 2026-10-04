"""Hazard checker 0.4: learned models behind the keyword checker."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from defaketive_sentiment import hazard_learned, registry
from defaketive_sentiment.hazard_learned import (
    DEFAULT_MODEL_DIR,
    LearnedHazardScorer,
    LearnedModelError,
    fit_category_model,
    keyword_features,
)
from defaketive_sentiment.hazard_model import MODEL_VERSION, HazardBadgeModel, LexiconHazardTagger

SHIPPED = (DEFAULT_MODEL_DIR / "manifest.json").is_file()


class TestKeywordFeatures(unittest.TestCase):
    def test_counts_only_terms_that_counted(self):
        tagged = {
            "tags": ["defect"],
            "terms": [
                {"category": "defect", "negated": False},
                {"category": "defect", "negated": True},
                {"category": "counterfeit", "negated": False, "excluded": "idiom"},
                {"category": "safety", "negated": False},
            ],
        }
        self.assertEqual(keyword_features(tagged).tolist(), [1, 1, 0, 0, 1, 0])


class StubScorer:
    """Says yes to defect and no to safety, whatever the sentence."""

    needs_embedding = False
    categories = ["safety", "defect"]
    version = "stub-1"

    def __init__(self):
        self.seen = []

    def decide(self, normalized, kw, embedding):
        self.seen.append(kw.tolist())
        return {"safety": (False, 0.1), "defect": (True, 0.9)}


class TestTaggerWithLearnedModels(unittest.TestCase):
    def test_learned_decisions_override_keywords_for_their_categories(self):
        tagger = LexiconHazardTagger(learned=StubScorer())
        tagged = tagger.tag_sentence("nasunog agad pero peke din")
        self.assertNotIn("safety", tagged["tags"])      # keywords said yes; learned says no
        self.assertIn("defect", tagged["tags"])         # keywords said no; learned says yes
        self.assertIn("counterfeit", tagged["tags"])    # not learned: keywords decide
        self.assertEqual(tagged["learned"], {"safety": 0.1, "defect": 0.9})
        # The matched words are kept for highlighting.
        self.assertTrue(any(term["category"] == "safety" for term in tagged["terms"]))

    def test_features_are_taken_before_any_learned_decision(self):
        scorer = StubScorer()
        LexiconHazardTagger(learned=scorer).tag_sentence("nasunog agad pero peke din")
        safety_tag, defect_tag = scorer.seen[0][3], scorer.seen[0][4]
        self.assertEqual((safety_tag, defect_tag), (1.0, 0.0))

    def test_an_embedding_model_without_an_embedder_is_refused(self):
        scorer = StubScorer()
        scorer.needs_embedding = True
        with self.assertRaises(Exception):
            LexiconHazardTagger(learned=scorer)

    def test_the_badge_version_names_the_learned_models(self):
        model = HazardBadgeModel(allow_unqualified=True, tagger=LexiconHazardTagger(learned=StubScorer()))
        self.assertTrue(model.model_version.startswith(f"{MODEL_VERSION}+stub-1"))


class GatedScorer:
    """A defect model in or_gated mode that always wants to add a flag."""

    needs_embedding = False
    needs_sentiment = True
    categories = ["defect"]
    combine = {"defect": "or_gated"}
    version = "gated-1"

    def __init__(self, on=True):
        self.on = on

    def decide(self, normalized, kw, embedding):
        return {"defect": (self.on, 0.9 if self.on else 0.1)}


class TestComplaintGate(unittest.TestCase):
    def tagger(self, label, on=True):
        calls = []

        def read(text):
            calls.append(text)
            return label

        return LexiconHazardTagger(learned=GatedScorer(on), sentiment_label=read), calls

    def test_no_flag_is_added_to_praise(self):
        tagger, _ = self.tagger("positive")
        tagged = tagger.tag_sentence("gumagana po cya")
        self.assertNotIn("defect", tagged["tags"])
        self.assertEqual(tagged["gate_sentiment"], {"defect": "positive"})

    def test_a_flag_is_added_to_a_complaint(self):
        tagger, _ = self.tagger("negative")
        self.assertIn("defect", tagger.tag_sentence("mahina yung tunog")["tags"])

    def test_a_keyword_flag_is_never_removed(self):
        tagger, calls = self.tagger("positive", on=False)
        self.assertIn("defect", tagger.tag_sentence("sira agad")["tags"])
        self.assertEqual(calls, [])  # no sentiment run when nothing is in question

    def test_sentiment_runs_only_when_a_flag_would_be_added(self):
        tagger, calls = self.tagger("negative", on=False)
        tagger.tag_sentence("maganda naman")
        self.assertEqual(calls, [])

    def test_a_gated_model_without_a_sentiment_reader_is_refused(self):
        with self.assertRaises(Exception):
            LexiconHazardTagger(learned=GatedScorer())


class TestFitting(unittest.TestCase):
    def test_a_tk_model_fits_and_scores(self):
        texts = ["sira agad", "hindi gumagana", "maganda naman", "sulit salamat"] * 5
        kw = np.zeros((len(texts), 6))
        model = fit_category_model("TK", texts, kw, None, [1, 1, 0, 0] * 5, category="defect")
        p = model.probability(["sira na agad"], np.zeros((1, 6)), None)
        self.assertEqual(p.shape, (1,))
        self.assertFalse(model.needs_embedding)


@unittest.skipUnless(SHIPPED, "learned hazard model not installed")
class TestShippedModels(unittest.TestCase):
    def test_only_the_adopted_categories_ship(self):
        manifest = json.loads((DEFAULT_MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
        scorer = LearnedHazardScorer()
        self.assertEqual(sorted(scorer.categories), sorted(manifest["categories"]))
        for category, decision in manifest["adoption"]["decisions"].items():
            self.assertEqual(decision["adopted"], category in scorer.categories)

    def test_a_changed_model_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "model"
            shutil.copytree(DEFAULT_MODEL_DIR, copy)
            name = next(iter(json.loads((copy / "manifest.json").read_text(encoding="utf-8"))["categories"].values()))["file"]
            (copy / name).write_bytes(b"tampered")
            with self.assertRaises(LearnedModelError):
                LearnedHazardScorer(copy)

    def test_changed_keyword_lists_are_refused(self):
        with mock.patch.object(hazard_learned, "keyword_file_hashes", return_value={"risk_lexicon.tsv": "x"}):
            with self.assertRaises(LearnedModelError):
                LearnedHazardScorer()

    def test_a_missing_model_is_refused_with_instructions(self):
        with self.assertRaises(LearnedModelError) as caught:
            LearnedHazardScorer(DEFAULT_MODEL_DIR.parent / "absent")
        self.assertIn("train_hazard_classifier.py", str(caught.exception))

    def test_the_registry_ships_the_learned_checker_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(registry.HAZARD_ENVIRONMENT_VARIABLE, None)
            self.assertEqual(registry.configured_hazard(), registry.LEARNED_HAZARD)
        model = registry.load_model(registry.HAZARD_BADGES)
        self.assertIsNotNone(model.tagger.learned)
        self.assertIn("hazard-learned", model.model_version)

    def test_the_keyword_checker_stays_selectable(self):
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)
        with mock.patch.dict(os.environ, {registry.HAZARD_ENVIRONMENT_VARIABLE: "keywords"}):
            model = registry.load_model(registry.HAZARD_BADGES)
        self.assertIsNone(model.tagger.learned)
        self.assertNotIn("hazard-learned", model.model_version)

    def test_an_unknown_hazard_model_is_refused(self):
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)
        with mock.patch.dict(os.environ, {registry.HAZARD_ENVIRONMENT_VARIABLE: "bert"}):
            with self.assertRaises(registry.ModelUnavailableError):
                registry.load_model(registry.HAZARD_BADGES)

    def test_a_clear_defect_report_is_still_found(self):
        tagger = LexiconHazardTagger(learned=LearnedHazardScorer())
        self.assertIn("defect", tagger.tag_sentence("sira agad, hindi na gumagana after 2 days")["tags"])


if __name__ == "__main__":
    unittest.main()
