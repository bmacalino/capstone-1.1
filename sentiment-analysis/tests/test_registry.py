import os
import unittest
from unittest import mock

from defaketive_sentiment import registry
from defaketive_sentiment.product_badges import SCHEMA_VERSION


class TestModeSelection(unittest.TestCase):
    def test_default_mode_is_the_badge_model(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(registry.ENVIRONMENT_VARIABLE, None)
            self.assertEqual(registry.configured_mode(), registry.HAZARD_BADGES)

    def test_environment_variable_selects_the_mode(self):
        with mock.patch.dict(os.environ, {registry.ENVIRONMENT_VARIABLE: "  MANUSCRIPT  "}):
            self.assertEqual(registry.configured_mode(), registry.MANUSCRIPT)

    def test_every_documented_mode_loads(self):
        for mode in registry.MODEL_MODES:
            with self.subTest(mode=mode):
                self.assertTrue(registry.load_model(mode).model_version)

    def test_lexicon_aliases_select_the_vader_baseline(self):
        versions = {registry.load_model(alias).model_version for alias in registry.MANUSCRIPT_ALIASES}
        self.assertEqual(len(versions), 1)

    def test_only_the_badge_mode_produces_badge_summaries(self):
        self.assertTrue(registry.is_badge_model(registry.load_model(registry.HAZARD_BADGES)))
        for mode in (registry.HYBRID_MANUSCRIPT, registry.MANUSCRIPT):
            with self.subTest(mode=mode):
                self.assertFalse(registry.is_badge_model(registry.load_model(mode)))

    def test_badge_model_declares_the_summary_schema(self):
        model = registry.load_model(registry.HAZARD_BADGES)
        self.assertEqual(model.schema_version, SCHEMA_VERSION)


class TestNoSilentFallback(unittest.TestCase):
    """A load failure must surface, not quietly become another model.

    With one 0-100 score a substitution only changed the number. With badges it
    changes the shape of the output, so the report would render a different
    model's verdict with nothing to indicate it had happened.
    """

    def setUp(self):
        # A cached model would hide the patched constructor below.
        registry.clear_cache()
        self.addCleanup(registry.clear_cache)

    def test_a_load_failure_raises_rather_than_substituting(self):
        with mock.patch(
            "defaketive_sentiment.hazard_model.HazardBadgeModel",
            side_effect=OSError("artifact missing"),
        ):
            with self.assertRaises(registry.ModelUnavailableError) as caught:
                registry.load_model(registry.HAZARD_BADGES)
        self.assertIn("artifact missing", str(caught.exception))

    def test_the_failure_names_the_mode_that_failed(self):
        with mock.patch(
            "defaketive_sentiment.manuscript.ManuscriptRiskModel",
            side_effect=OSError("lexicon missing"),
        ):
            with self.assertRaises(registry.ModelUnavailableError) as caught:
                registry.load_model(registry.MANUSCRIPT)
        self.assertIn("manuscript", str(caught.exception))

    def test_a_missing_transformer_refuses_with_instructions(self):
        from defaketive_sentiment import transformer_sentiment

        with mock.patch.object(transformer_sentiment, "DEFAULT_MODEL_DIR", transformer_sentiment.MODELS_ROOT / "absent"):
            with self.assertRaises(registry.ModelUnavailableError) as caught:
                registry.load_model(registry.HAZARD_BADGES, sentiment=registry.XLMR_SENTIMENT)
        self.assertIn("slim_sentiment_model.py", str(caught.exception))

    def test_a_failed_load_is_not_cached(self):
        # Installing the missing model and retrying must work without a restart.
        with mock.patch(
            "defaketive_sentiment.hazard_model.HazardBadgeModel",
            side_effect=OSError("artifact missing"),
        ):
            with self.assertRaises(registry.ModelUnavailableError):
                registry.load_model(registry.HAZARD_BADGES)
        self.assertTrue(registry.load_model(registry.HAZARD_BADGES).model_version)

    def test_an_unknown_sentiment_model_is_refused(self):
        with self.assertRaises(registry.ModelUnavailableError) as caught:
            registry.load_model(registry.HAZARD_BADGES, sentiment="bert")
        self.assertIn(registry.SENTIMENT_ENVIRONMENT_VARIABLE, str(caught.exception))


class TestCaching(unittest.TestCase):
    def test_the_same_model_instance_is_shared(self):
        # The catalog asks for the model version on every load; each call used
        # to rebuild the whole model.
        self.assertIs(registry.load_model(registry.HAZARD_BADGES), registry.load_model(registry.HAZARD_BADGES))

    def test_the_sentiment_choice_is_part_of_the_cache_key(self):
        with mock.patch.object(registry, "_build", side_effect=lambda mode, sentiment, hazard=None: object()):
            registry.clear_cache()
            self.addCleanup(registry.clear_cache)
            first = registry.load_model(registry.HAZARD_BADGES, sentiment="tfidf")
            second = registry.load_model(registry.HAZARD_BADGES, sentiment="xlmr")
        self.assertIsNot(first, second)

    def test_the_default_sentiment_model_is_the_transformer(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(registry.SENTIMENT_ENVIRONMENT_VARIABLE, None)
            self.assertEqual(registry.configured_sentiment(), registry.XLMR_SENTIMENT)

    def test_the_badge_version_names_the_sentiment_model(self):
        version = registry.model_version(registry.HAZARD_BADGES, sentiment=registry.TFIDF_SENTIMENT)
        self.assertIn("hazard-lexicon", version)
        self.assertIn("firecs-tfidf", version)


class TestVersionAgreement(unittest.TestCase):
    def test_reported_version_matches_the_model_that_would_be_loaded(self):
        # The bug this prevents: analyzing with one model while labelling the
        # results with another's version, which makes staleness detection
        # either re-analyze forever or never re-analyze.
        for mode in registry.MODEL_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(
                    registry.model_version(mode), registry.load_model(mode).model_version
                )


class TestModelDisagreement(unittest.TestCase):
    """Why the redesign exists, measured on the models themselves."""

    REVIEWS = [
        "Nasunog agad ang powerbank",
        "sumabog habang naka charge",
        "maganda naman sulit",
    ]

    def test_legacy_scores_disagree_wildly_on_an_obvious_hazard(self):
        scores = {}
        for mode in (registry.HYBRID_MANUSCRIPT, registry.MANUSCRIPT, registry.LEGACY_HYBRID):
            model = registry.load_model(mode)
            analyses = [model.analyze(review) for review in self.REVIEWS]
            scores[mode] = model.summarize(analyses)["risk_score"]
        # Two explicit fire and explosion reports in three reviews, and the
        # three weighted-sum models span roughly 17 to 67 points.
        self.assertGreater(max(scores.values()) - min(scores.values()), 40)

    def test_the_badge_model_warns_where_legacy_scores_read_as_low_risk(self):
        model = registry.load_model(registry.HAZARD_BADGES)
        summary = model.summarize(model.analyze_batch(self.REVIEWS))
        self.assertEqual(summary["verdict"]["state"], "safety_warning")
        self.assertTrue(summary["safety"]["alert"])


if __name__ == "__main__":
    unittest.main()
