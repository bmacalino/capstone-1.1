import unittest

from defaketive_sentiment import product_badges as badges
from defaketive_sentiment.hazard_model import (
    CATEGORY_MAP,
    HazardBadgeModel,
    HazardModelError,
    LexiconHazardTagger,
    hazard_subtype,
)


class TestQualificationGate(unittest.TestCase):
    def test_loading_is_refused_by_default(self):
        # The lexicon tagger has passed no evaluation gate, because there is
        # nothing to evaluate it against yet. Refusing by default stops its
        # output being quoted as validated accuracy.
        with self.assertRaises(HazardModelError):
            HazardBadgeModel()

    def test_explicit_opt_in_reports_itself_as_unqualified(self):
        model = HazardBadgeModel(allow_unqualified=True)
        self.assertFalse(model.qualified_for_integration)


class TestCategoryMapping(unittest.TestCase):
    def test_performance_and_durability_count_as_defects(self):
        # The tagging guide defines DEFECT as "the product does not work".
        # "hindi gumagana" and "mabilis masira" are filed under performance and
        # durability in the lexicon but are defects by that definition.
        self.assertEqual(CATEGORY_MAP["performance"], badges.DEFECT)
        self.assertEqual(CATEGORY_MAP["durability"], badges.DEFECT)

    def test_quality_opinions_are_not_defects(self):
        # "cheaply made" is a judgement about how a product feels, not a report
        # that it failed. Folding it in would inflate every defect rate with
        # ordinary grumbling.
        self.assertNotIn("quality", CATEGORY_MAP)


class TestTagging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HazardBadgeModel(allow_unqualified=True)

    def tags(self, text):
        return set(self.model.analyze(text)["review_tags"])

    def confident(self, text):
        return set(self.model.analyze(text)["review_high_confidence"])

    def test_tags_the_three_categories(self):
        self.assertEqual(self.tags("Nasunog agad ang unit"), {"safety"})
        self.assertEqual(self.tags("hindi gumagana pagdating"), {"defect"})
        self.assertEqual(self.tags("peke ang item, hindi original"), {"counterfeit"})

    def test_one_sentence_can_be_two_categories(self):
        self.assertEqual(
            self.tags("peke sya hindi gumagana kahit long press"),
            {"counterfeit", "defect"},
        )

    def test_quality_grumbling_is_not_tagged(self):
        self.assertEqual(self.tags("cheaply made, poor quality talaga"), set())

    def test_negated_complaints_are_not_tagged(self):
        for text in (
            "walang sira at gumagana nang maayos",
            "hindi naman umiinit kahit matagal",
            "not fake, legit talaga",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.tags(text), set())

    def test_english_negation_is_handled(self):
        # Taglish reviews negate in either language. Before English negators
        # were added, "it is not mainit" read as a heat report.
        self.assertEqual(self.tags("comfy it is not mainit sa skin"), set())

    def test_terms_carrying_their_own_negation_still_count(self):
        # "hindi gumagana" is a complaint, not a denial of one.
        self.assertEqual(self.tags("hindi gumagana ang charger"), {"defect"})


class TestHighConfidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = HazardBadgeModel(allow_unqualified=True)

    def confident(self, text):
        return set(self.model.analyze(text)["review_high_confidence"])

    def test_a_subjectless_report_is_still_confident(self):
        # Filipino is pro-drop: "Nasunog agad" is a complete report with no
        # subject. Requiring an explicit product noun would discard exactly the
        # terse, emphatic reports that matter most.
        self.assertIn("safety", self.confident("Nasunog agad"))

    def test_packaging_damage_is_not_a_product_hazard(self):
        result = self.model.analyze("yung box nasunog pero okay naman yung item")
        self.assertIn("safety", result["review_tags"])
        self.assertNotIn("safety", result["review_high_confidence"])

    def test_courier_and_seller_context_is_excluded(self):
        for text in (
            "courier dropped it, the package was on fire",
            "the seller shop burned down",
        ):
            with self.subTest(text=text):
                self.assertNotIn("safety", self.confident(text))

    def test_cosmetic_complaints_are_tagged_but_not_confident(self):
        result = self.model.analyze("gasgas lang konti")
        self.assertIn("defect", result["review_tags"])
        self.assertNotIn("defect", result["review_high_confidence"])


class TestFalseFriends(unittest.TestCase):
    """The phrasings the previous machine labels got wrong."""

    @classmethod
    def setUpClass(cls):
        cls.model = HazardBadgeModel(allow_unqualified=True)

    def test_known_false_friends_do_not_report_a_hazard(self):
        for text in (
            "not legit dual shock 4 but it is working fine",
            "Sabog ang sound. Pangit.",
            "okay naman po siya. Sabog lang po ung sounds",
        ):
            with self.subTest(text=text):
                self.assertNotIn("safety", self.model.analyze(text)["review_tags"])


class TestSubtypes(unittest.TestCase):
    def test_subtype_is_the_most_severe_match(self):
        self.assertEqual(hazard_subtype(["init", "sunog"]), "fire_explosion_swelling")
        self.assertEqual(hazard_subtype(["init"]), "overheat_burn")
        self.assertEqual(hazard_subtype(["spark"]), "shock_spark_short")

    def test_an_unrecognised_term_falls_back_to_unspecified(self):
        self.assertEqual(hazard_subtype(["delikado"]), "safety_unspecified")

    def test_severity_comes_from_the_frozen_taxonomy(self):
        model = HazardBadgeModel(allow_unqualified=True)
        self.assertEqual(model.analyze("Nasunog agad")["max_severity"], 4)
        self.assertEqual(model.analyze("sobrang init ang unit")["max_severity"], 3)


class TestEvidenceSpans(unittest.TestCase):
    def test_matched_terms_point_back_at_the_original_text(self):
        # The report highlights evidence in what the reviewer wrote, not in the
        # normalized form, so spans have to survive affix folding.
        tagger = LexiconHazardTagger()
        raw = "Yung powerbank ko UMIINIT sobra"
        result = tagger.tag_sentence(raw)
        spans = [term["span"] for term in result["terms"] if term["span"]]
        self.assertTrue(spans)
        self.assertEqual(raw[spans[0][0] : spans[0][1]], "UMIINIT")


class TestProductIntegration(unittest.TestCase):
    def test_summarize_produces_a_badge_summary(self):
        model = HazardBadgeModel(allow_unqualified=True)
        reviews = ["Nasunog agad", "sumabog ang unit"] + ["maganda naman"] * 98
        analyses = model.analyze_batch(reviews)
        summary = model.summarize(analyses, product_type="power_bank")

        self.assertEqual(summary["schema_version"], badges.SCHEMA_VERSION)
        self.assertEqual(summary["verdict"]["state"], badges.SAFETY_WARNING)
        self.assertTrue(summary["safety"]["alert"])
        self.assertNotIn("risk_score", summary)
        self.assertFalse(summary["qualified_for_integration"])

    def test_a_clean_product_reports_no_warning_signs(self):
        model = HazardBadgeModel(allow_unqualified=True)
        analyses = model.analyze_batch(["maganda at sulit"] * 100)
        summary = model.summarize(analyses)
        self.assertEqual(summary["verdict"]["state"], badges.NO_WARNING_SIGNS)
        self.assertIn("not proof", summary["verdict"]["detail"])

    def test_versions_are_recorded_for_the_manifest(self):
        model = HazardBadgeModel(allow_unqualified=True)
        summary = model.summarize(model.analyze_batch(["maganda"]))
        self.assertTrue(summary["normalizer_version"])
        self.assertTrue(summary["segmenter_version"])
        self.assertTrue(summary["model_version"])


if __name__ == "__main__":
    unittest.main()
