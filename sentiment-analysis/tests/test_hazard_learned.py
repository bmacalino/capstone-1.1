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
    is_high_star,
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


class StarScorer(GatedScorer):
    """Scores every sentence 0.6: over its ordinary threshold, under its 0.7 bar for 4-5 star reviews."""

    high_star_threshold = {"defect": 0.7}

    def decide(self, normalized, kw, embedding):
        return {"defect": (True, 0.6)}


class TestHighStarBar(unittest.TestCase):
    PLAIN = "gumagana po cya"  # no defect keyword in it

    def tagger(self, scorer=None):
        return LexiconHazardTagger(learned=scorer or StarScorer(), sentiment_label=lambda text: "negative")

    def test_which_ratings_are_high(self):
        for rating, expected in ((5, True), ("4", True), (4.0, True), ("5.0", True), (3, False), ("1", False), (None, False), ("", False), ("n/a", False)):
            with self.subTest(rating=rating):
                self.assertEqual(is_high_star(rating), expected)

    def test_a_weak_score_adds_no_flag_in_a_well_rated_review(self):
        tagged = self.tagger().tag_sentence(self.PLAIN, 5)
        self.assertNotIn("defect", tagged["tags"])
        self.assertEqual(tagged["gate_sentiment"], {"defect": "below_high_star_bar"})

    def test_the_same_score_adds_a_flag_in_a_low_rated_or_unrated_review(self):
        for rating in (1, 3, None, ""):
            with self.subTest(rating=rating):
                self.assertIn("defect", self.tagger().tag_sentence(self.PLAIN, rating)["tags"])

    def test_a_score_over_the_bar_still_adds_a_flag(self):
        scorer = StarScorer()
        scorer.high_star_threshold = {"defect": 0.55}
        self.assertIn("defect", self.tagger(scorer).tag_sentence(self.PLAIN, 5)["tags"])

    def test_a_keyword_flag_stands_whatever_the_rating(self):
        self.assertIn("defect", self.tagger().tag_sentence("sira agad", 5)["tags"])

    def test_the_rating_reaches_every_sentence_of_a_review(self):
        tagged = self.tagger().tag_review("Gumagana po cya. Sira agad.", 5)
        self.assertEqual(["defect" in sentence["tags"] for sentence in tagged], [False, True])

    def test_the_badge_model_passes_the_rating_on(self):
        model = HazardBadgeModel(allow_unqualified=True, tagger=self.tagger())
        self.assertNotIn("defect", model.analyze(self.PLAIN, 5)["review_tags"])
        self.assertIn("defect", model.analyze(self.PLAIN, 2)["review_tags"])
        self.assertIn("defect", model.analyze(self.PLAIN)["review_tags"])


class FilterScorer:
    """A safety model in "and" mode: it may only remove a keyword flag."""

    needs_embedding = False
    categories = ["safety"]
    combine = {"safety": "and"}
    version = "filter-1"

    def __init__(self, on):
        self.on = on

    def decide(self, normalized, kw, embedding):
        return {"safety": (self.on, 0.9 if self.on else 0.1)}


class TestFilterMode(unittest.TestCase):
    def test_a_keyword_flag_the_model_rejects_is_removed(self):
        tagged = LexiconHazardTagger(learned=FilterScorer(False)).tag_sentence("nasunog agad")
        self.assertNotIn("safety", tagged["tags"])

    def test_a_keyword_flag_the_model_accepts_stands(self):
        tagged = LexiconHazardTagger(learned=FilterScorer(True)).tag_sentence("nasunog agad")
        self.assertIn("safety", tagged["tags"])

    def test_the_model_never_adds_a_flag(self):
        tagged = LexiconHazardTagger(learned=FilterScorer(True)).tag_sentence("maganda naman")
        self.assertNotIn("safety", tagged["tags"])

    def test_a_serious_flag_is_not_the_filters_to_remove(self):
        # The model says no to both. Mild heat goes; a fire report stays.
        scorer = FilterScorer(False)
        scorer.filter_spares = {"safety": 2}
        tagger = LexiconHazardTagger(learned=scorer)
        self.assertNotIn("safety", tagger.tag_sentence("umiinit pa yung charger")["tags"])
        self.assertIn("safety", tagger.tag_sentence("nasunog agad")["tags"])


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
        # Safety is a filter on the keyword flags; defect is the retrained model.
        self.assertEqual(sorted(scorer.categories), ["defect", "safety"])
        self.assertEqual(scorer.combine, {"safety": "and", "defect": "or_gated"})
        # The filter may never remove a flag stronger than mild heat.
        self.assertEqual(scorer.filter_spares, {"safety": 2})
        # Written (synthetic) sentences trained it and are declared as training only.
        self.assertIn("Training only", manifest["synthetic_training_data"]["what"])
        self.assertEqual(manifest["keyword_rules"], "0.4")
        # Made the default on the owner's instruction, not by a pre-registered test.
        self.assertFalse(manifest["qualified_for_integration"])
        self.assertEqual(manifest["adoption"]["adopted_by"], "owner instruction")

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
        tagger = LexiconHazardTagger.for_scorer(LearnedHazardScorer(), sentiment_label=lambda text: "negative")
        self.assertIn("defect", tagger.tag_sentence("sira agad, hindi na gumagana after 2 days")["tags"])

    def test_box_and_seller_complaints_are_not_added_as_defects(self):
        # Guide rule 1, from the live check after round 3.
        tagger = LexiconHazardTagger.for_scorer(LearnedHazardScorer(), sentiment_label=lambda text: "negative")
        for sentence in ("Poor packaging.", "No protective packaging.", "but all I get were automated responses."):
            with self.subTest(sentence=sentence):
                self.assertNotIn("defect", tagger.tag_sentence(sentence)["tags"])

    def test_a_fault_mentioned_with_the_seller_is_still_found(self):
        # The first, blunter rule lost these five round-3 reports.
        tagger = LexiconHazardTagger.for_scorer(LearnedHazardScorer(), sentiment_label=lambda text: "negative")
        tagged = tagger.tag_sentence("seller not responding sayang pera dto 5 mins lang lowbat na")
        self.assertIn("defect", tagged["tags"])

    def test_the_shipped_folder_names_its_keyword_rules(self):
        scorer = LearnedHazardScorer()
        self.assertEqual(scorer.manifest["keyword_rules"], "0.4")
        with self.assertRaises(Exception):
            # Served with the rules it was not trained with: refused.
            LexiconHazardTagger(learned=scorer, sentiment_label=lambda text: "negative")

    def test_the_app_no_longer_flags_denied_or_imagined_defects(self):
        model = registry.load_model(registry.HAZARD_BADGES)
        self.assertEqual(model.tagger.rules, "0.4")
        self.assertIn("+rules0.4", model.model_version)
        for sentence in (
            "Wala pong sira ang mga na recieve ko.",
            "original talaga, walang ka sira sira",
            "Akala ko nung una sira na hahaha lowbat lang pala.",
            "pabili ulit ako sa susunod pag nasira na.",
        ):
            with self.subTest(sentence=sentence):
                self.assertNotIn("defect", model.tagger.tag_sentence(sentence)["tags"])

    def test_the_team_label_models_stay_selectable(self):
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)
        with mock.patch.dict(os.environ, {registry.HAZARD_ENVIRONMENT_VARIABLE: registry.TEAM_LABELS_HAZARD}):
            model = registry.load_model(registry.HAZARD_BADGES)
        self.assertEqual(model.tagger.learned.model_dir.name, registry.TEAM_LABELS_HAZARD_DIR)
        self.assertEqual(sorted(model.tagger.learned.categories), ["defect", "safety"])

    def test_every_app_checker_reports_the_two_newer_labels(self):
        # The report screens show five labels, so whichever checker is chosen must fill them.
        self.addCleanup(registry.clear_cache)
        for choice in ("", registry.TEAM_LABELS_HAZARD, registry.KEYWORD_HAZARD, registry.FIVE_LABELS_HAZARD):
            registry.clear_cache()
            with self.subTest(choice=choice or "default"), mock.patch.dict(os.environ, {registry.HAZARD_ENVIRONMENT_VARIABLE: choice}):
                model = registry.load_model(registry.HAZARD_BADGES)
                self.assertIn("quality", model.tagger.tag_sentence("sobrang hina")["extra_tags"])
                self.assertIn("+c2", model.model_version)
                self.assertEqual(model.analyze("sobrang hina")["review_extra_tags"], ["quality"])

    def test_the_shipped_defect_model_is_stricter_in_well_rated_reviews(self):
        scorer = LearnedHazardScorer()
        self.assertGreater(scorer.high_star_threshold["defect"], scorer.models["defect"].threshold)
        tagger = LexiconHazardTagger.for_scorer(scorer, sentiment_label=lambda text: "neutral")
        # The Anker check: a plain remark the earlier retrained model quoted as a defect.
        self.assertNotIn("defect", tagger.tag_sentence("Gumagana naman sya.", 5)["tags"])
        # A real report in a 5-star review is still found.
        self.assertIn("defect", tagger.tag_sentence("Ayaw na mag charge after 1 week.", 5)["tags"])

    def test_the_star_bar_is_part_of_the_version(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "model"
            shutil.copytree(DEFAULT_MODEL_DIR, copy)
            manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
            manifest["categories"]["defect"].pop("threshold_high_star")
            (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertNotEqual(LearnedHazardScorer(copy).version, LearnedHazardScorer().version)

    def test_a_star_bar_on_a_category_that_cannot_use_it_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "model"
            shutil.copytree(DEFAULT_MODEL_DIR, copy)
            manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
            manifest["categories"]["safety"]["threshold_high_star"] = 0.9
            (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(LearnedModelError):
                LearnedHazardScorer(copy)

    def test_the_shipped_safety_filter_keeps_reports_and_drops_figures_of_speech(self):
        tagger = LexiconHazardTagger.for_scorer(LearnedHazardScorer(), sentiment_label=lambda text: "negative")
        for report in ("nasunog yung charger pagka saksak", "umusok yung powerbank habang naka charge", "sobrang init ng charger hindi na mahawakan"):
            with self.subTest(report=report):
                self.assertIn("safety", tagger.tag_sentence(report)["tags"])
        for harmless in ("apoy ang tunog ng speaker solid", "hindi naman umiinit kahit matagal gamitin"):
            with self.subTest(harmless=harmless):
                self.assertNotIn("safety", tagger.tag_sentence(harmless)["tags"])

    def test_sparing_serious_flags_is_only_for_a_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "model"
            shutil.copytree(DEFAULT_MODEL_DIR, copy)
            manifest = json.loads((copy / "manifest.json").read_text(encoding="utf-8"))
            manifest["categories"]["defect"]["filter_spares_severity_above"] = 2
            (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(LearnedModelError):
                LearnedHazardScorer(copy)

    def test_the_shipped_gate_keeps_praise_unflagged(self):
        tagger = LexiconHazardTagger.for_scorer(LearnedHazardScorer(), sentiment_label=lambda text: "positive")
        self.assertNotIn("defect", tagger.tag_sentence("gumagana po cya")["tags"])


if __name__ == "__main__":
    unittest.main()
