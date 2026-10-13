import unittest

from defaketive_sentiment import hazard_taxonomy as taxonomy
from defaketive_sentiment import product_badges as badges
from defaketive_sentiment.product_badges import (
    CategoryFlag,
    ProductBadgeAggregator,
    SafetyBadge,
)


def review(key, *tagged, rating=1, sentiment="negative", text=None):
    """Build one stub review analysis.

    `tagged` entries are (category, high_confidence, subtype, text). Using a
    stub rather than a real tagger is the point: the whole product layer is
    testable before a single sentence has been labelled.
    """
    sentences = []
    for index, item in enumerate(tagged):
        category, confident, subtype, sentence_text = item
        sentences.append(
            {
                "index": index,
                "text_raw": sentence_text or text or f"{category} report {key}",
                "tags": [category],
                "scores": {category: 20.0},
                "high_confidence": {category: confident},
                "hazard_subtype": subtype,
                "terms": [{"ngram": category, "cc": 9.0, "negated": False}],
            }
        )
    return {
        "review_key": key,
        "rating": rating,
        "sentences": sentences,
        "sentiment": {"label": sentiment},
    }


def clean(key, sentiment="positive"):
    return {"review_key": key, "sentences": [], "sentiment": {"label": sentiment}}


def fire(key, confident=True, text=None):
    return review(key, ("safety", confident, "fire_explosion_swelling", text))


BASELINES = {
    "version": "test-baselines-1",
    "product_types": {
        "power_bank": {
            "defect": {"pooled_rate": 0.03},
            "counterfeit": {"pooled_rate": 0.02},
            "safety": {
                "likelihood_bins": {
                    "low": [0.0, 0.004],
                    "medium": [0.004, 0.02],
                    "high": [0.02, 1.0],
                }
            },
        }
    },
}


def aggregator(**overrides):
    parameters = badges.load_parameters()
    parameters.update(overrides)
    return ProductBadgeAggregator(parameters=parameters, baselines=BASELINES)


class TestRollup(unittest.TestCase):
    def test_a_review_counts_once_per_category_however_many_sentences(self):
        # The fix for the old formula's double counting, applied per category.
        noisy = review(
            "r1",
            ("safety", True, "overheat_burn", "sobrang init"),
            ("safety", True, "overheat_burn", "ang init talaga"),
            ("safety", True, "overheat_burn", "napaso ako"),
        )
        summary = aggregator().summarize([noisy] + [clean(f"c{i}") for i in range(49)])
        self.assertEqual(summary["safety"]["flagged_reviews"], 1)
        self.assertAlmostEqual(summary["safety"]["rate"], 1 / 50)

    def test_one_review_can_be_flagged_in_two_categories(self):
        # "umiinit at hindi na nag-charge" is both, and must be counted in both.
        both = review(
            "r1",
            ("safety", True, "overheat_burn", "umiinit"),
            ("defect", False, None, "hindi na nag-charge"),
        )
        summary = aggregator().summarize([both] + [clean(f"c{i}") for i in range(49)])
        self.assertEqual(summary["safety"]["flagged_reviews"], 1)
        self.assertEqual(summary["flags"]["defect"]["flagged_reviews"], 1)

    def test_untagged_reviews_produce_no_flags(self):
        summary = aggregator().summarize([clean(f"c{i}") for i in range(50)])
        self.assertEqual(summary["safety"]["flagged_reviews"], 0)
        self.assertEqual(summary["safety"]["rate_lower_bound"], 0.0)
        self.assertEqual(summary["verdict"]["state"], badges.NO_WARNING_SIGNS)

    def test_sentences_analyzed_is_reported(self):
        summary = aggregator().summarize([fire("r1"), clean("c1")])
        self.assertEqual(summary["coverage"]["sentences_analyzed"], 1)

    def test_reviews_without_their_own_key_are_still_counted_separately(self):
        # Rates count DISTINCT review keys. A tagger that does not stamp a key
        # would otherwise collapse every review onto one empty identity, and a
        # product with fifty hazard reports would count as one.
        keyless = [
            {"sentences": review("", ("safety", True, "fire_explosion_swelling", f"nasunog {i}"))["sentences"]}
            for i in range(5)
        ]
        summary = aggregator().summarize(keyless + [{"sentences": []} for _ in range(95)])
        self.assertEqual(summary["safety"]["flagged_reviews"], 5)


class TestSafetySeverity(unittest.TestCase):
    def test_severity_uses_only_high_confidence_reports(self):
        # A low-confidence match must not be able to raise a product to
        # severity 4; that is what the alert path keys on.
        reviews = [fire("r1", confident=False)] + [clean(f"c{i}") for i in range(49)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["safety"]["flagged_reviews"], 1)
        self.assertEqual(summary["safety"]["severity"], 0)

    def test_severity_is_the_worst_confirmed_hazard(self):
        reviews = [
            review("r1", ("safety", True, "overheat_burn", "mainit")),
            review("r2", ("safety", True, "fire_explosion_swelling", "nasunog")),
        ] + [clean(f"c{i}") for i in range(48)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["safety"]["severity"], 4)

    def test_severity_basis_is_reported_for_the_write_up(self):
        reviews = [fire("r1")] + [clean(f"c{i}") for i in range(49)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertIn("[", summary["safety"]["severity_basis"])

    def test_subtype_counts_are_broken_out(self):
        reviews = [
            fire("r1"),
            review("r2", ("safety", True, "shock_spark_short", "nakuryente")),
        ] + [clean(f"c{i}") for i in range(48)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        subtypes = summary["safety"]["hazard_subtypes"]
        self.assertEqual(subtypes["fire_explosion_swelling"]["reviews"], 1)
        self.assertEqual(subtypes["shock_spark_short"]["reviews"], 1)


class TestAlertOverride(unittest.TestCase):
    def test_two_independent_fire_reports_warn_even_at_a_tiny_rate(self):
        # The design's central claim, and the handoff's worked example: two
        # reports in four hundred reviews is a 0.5% rate but must still warn.
        reviews = [
            fire("r1", text="nasunog yung unit habang naka-charge"),
            fire("r2", text="sumabog at nagliyab ang powerbank ko"),
        ] + [clean(f"c{i}") for i in range(398)]
        summary = aggregator().summarize(reviews, product_type="power_bank")

        self.assertTrue(summary["safety"]["alert"])
        self.assertEqual(summary["safety"]["independent_high_confidence_reports"], 2)
        self.assertAlmostEqual(summary["safety"]["rate"], 0.005)
        self.assertAlmostEqual(summary["safety"]["rate_lower_bound"], 0.0014, places=4)
        self.assertIn(summary["safety"]["tier"], {"High", "Serious"})
        self.assertEqual(summary["verdict"]["state"], badges.SAFETY_WARNING)

    def test_one_report_alone_does_not_trigger_the_alert(self):
        reviews = [fire("r1")] + [clean(f"c{i}") for i in range(399)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertFalse(summary["safety"]["alert"])

    def test_one_serious_report_on_a_large_listing_is_a_caution_not_a_warning(self):
        # Decided 12 Oct 2026. The tier table gives High for a severity-4
        # hazard however rare, so one report used to be a safety warning and
        # 14 of 19 warned listings rested on one. A single safety flag is
        # wrong about as often as it is right; a second report is asked for.
        reviews = [fire("r1")] + [clean(f"c{i}") for i in range(399)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        safety = summary["safety"]
        self.assertEqual((safety["severity"], safety["likelihood_bin"]), (4, "low"))
        self.assertEqual(safety["tier"], "Moderate")
        self.assertIn("1 separate high-confidence report of", safety["tier_held_reason"])
        self.assertIn("2 are needed", safety["tier_held_reason"])
        self.assertEqual(summary["verdict"]["state"], badges.POSSIBLE_SAFETY_ISSUE)
        self.assertEqual(summary["verdict"]["detail"], safety["tier_held_reason"])

    def test_the_same_report_copied_twice_is_still_one(self):
        duplicated = "nasunog yung unit habang naka-charge sobrang delikado"
        reviews = [fire("r1", text=duplicated), fire("r2", text=duplicated)] + [
            clean(f"c{i}") for i in range(398)
        ]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["verdict"]["state"], badges.POSSIBLE_SAFETY_ISSUE)

    def test_one_serious_report_among_few_reviews_is_not_held_back(self):
        # One report in thirty reviews is no longer rare (its lower bound is
        # 0.6%), and the hold is only for rare reports: the table's Serious
        # stands. Between 30 and about 37 reviews one report can still warn.
        reviews = [fire("r1")] + [clean(f"c{i}") for i in range(29)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        safety = summary["safety"]
        self.assertEqual((safety["likelihood_bin"], safety["tier"]), ("medium", "Serious"))
        self.assertIsNone(safety["tier_held_reason"])
        self.assertFalse(safety["alert"])
        self.assertEqual(summary["verdict"]["state"], badges.SAFETY_WARNING)

    def test_two_separate_reports_are_not_held_back(self):
        reviews = [
            fire("r1", text="nasunog yung unit habang naka-charge"),
            fire("r2", text="sumabog at nagliyab ang powerbank ko"),
        ] + [clean(f"c{i}") for i in range(398)]
        safety = aggregator().summarize(reviews, product_type="power_bank")["safety"]
        self.assertIsNone(safety["tier_held_reason"])
        self.assertEqual(safety["tier"], "High")

    def with_alert(self, **settings):
        alert = dict(badges.load_parameters()["alert"])
        alert.update({name: {"value": value} for name, value in settings.items()})
        return aggregator(alert=alert)

    def test_the_earlier_rule_can_be_restored_for_comparison(self):
        reviews = [fire("r1")] + [clean(f"c{i}") for i in range(399)]
        summary = self.with_alert(single_report_tier="High").summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["safety"]["tier"], "High")
        self.assertIsNone(summary["safety"]["tier_held_reason"])
        self.assertEqual(summary["verdict"]["state"], badges.SAFETY_WARNING)

    def test_asking_for_three_reports_holds_two(self):
        reviews = [
            fire("r1", text="nasunog yung unit habang naka-charge"),
            fire("r2", text="sumabog at nagliyab ang powerbank ko"),
        ] + [clean(f"c{i}") for i in range(398)]
        summary = self.with_alert(independent_high_confidence_reports=3).summarize(
            reviews, product_type="power_bank"
        )
        self.assertIn("2 separate high-confidence reports of", summary["safety"]["tier_held_reason"])
        self.assertEqual(summary["verdict"]["state"], badges.POSSIBLE_SAFETY_ISSUE)

    def test_a_held_tier_must_be_a_real_tier(self):
        with self.assertRaises(ValueError):
            self.with_alert(single_report_tier="Medium")

    def test_overheating_and_smoke_reports_are_judged_as_before(self):
        # Only severity 4 is held. A severity-3 report was Moderate when rare
        # and still is, by the table and with no hold recorded.
        reviews = [review("r1", ("safety", True, "smoke_smell", "amoy sunog"))] + [
            clean(f"c{i}") for i in range(399)
        ]
        safety = aggregator().summarize(reviews, product_type="power_bank")["safety"]
        self.assertEqual(safety["tier"], "Moderate")
        self.assertIsNone(safety["tier_held_reason"])

    def test_duplicated_text_counts_as_one_report(self):
        # Two copies of the same review are one piece of evidence.
        duplicated = "nasunog yung unit habang naka-charge sobrang delikado"
        reviews = [
            fire("r1", text=duplicated),
            fire("r2", text=duplicated),
        ] + [clean(f"c{i}") for i in range(398)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["safety"]["independent_high_confidence_reports"], 1)
        self.assertFalse(summary["safety"]["alert"])

    def test_severity_three_hazards_do_not_trigger_the_alert(self):
        reviews = [
            review("r1", ("safety", True, "smoke_smell", "amoy sunog")),
            review("r2", ("safety", True, "smoke_smell", "may usok na lumabas")),
        ] + [clean(f"c{i}") for i in range(398)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertFalse(summary["safety"]["alert"])

    def test_alert_survives_a_small_sample(self):
        # Count-based evidence does not need a denominator. Suppressing this
        # behind the sample-size rule would be a false negative in the most
        # severe category.
        reviews = [
            fire("r1", text="nasunog agad pagka-charge"),
            fire("r2", text="sumabog habang ginagamit ko"),
        ] + [clean(f"c{i}") for i in range(10)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertLess(summary["coverage"]["usable_reviews"], 30)
        self.assertEqual(summary["verdict"]["state"], badges.SAFETY_WARNING)

    def test_the_override_can_be_switched_off_for_comparison(self):
        parameters = badges.load_parameters()
        parameters["alert"] = dict(parameters["alert"])
        parameters["alert"]["overrides_insufficient_evidence"] = {"value": False}
        aggregate = ProductBadgeAggregator(parameters=parameters, baselines=BASELINES)
        reviews = [fire("r1", text="nasunog"), fire("r2", text="sumabog")] + [
            clean(f"c{i}") for i in range(10)
        ]
        summary = aggregate.summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["verdict"]["state"], badges.INSUFFICIENT_EVIDENCE)


class TestCategoryFlags(unittest.TestCase):
    def test_flagged_when_the_lower_bound_clears_the_baseline(self):
        reviews = [review(f"d{i}", ("defect", False, None, f"sira agad {i}")) for i in range(40)]
        reviews += [clean(f"c{i}") for i in range(60)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        defect = summary["flags"]["defect"]
        self.assertEqual(defect["state"], badges.FLAGGED)
        self.assertGreater(defect["rate_lower_bound"], defect["baseline_rate"])

    def test_not_flagged_when_the_rate_is_typical_for_the_type(self):
        # The handoff's worked example: 10 of 400 is 2.5% observed, but a 1.36%
        # lower bound, which sits below a 3% type baseline.
        reviews = [review(f"d{i}", ("defect", False, None, f"sira {i}")) for i in range(10)]
        reviews += [clean(f"c{i}") for i in range(390)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        defect = summary["flags"]["defect"]
        self.assertAlmostEqual(defect["rate"], 0.025)
        self.assertAlmostEqual(defect["rate_lower_bound"], 0.0136, places=4)
        self.assertEqual(defect["state"], badges.NOT_FLAGGED)

    def test_absolute_floor_blocks_a_flag_when_the_baseline_is_low(self):
        # Guards the known weakness of a peer-relative comparison.
        flag = aggregator()._category_flag(
            "counterfeit",
            [],
            100,
            "power_bank",
        )
        self.assertIsNotNone(flag.absolute_floor)

    def test_no_baseline_is_its_own_state(self):
        reviews = [review(f"d{i}", ("defect", False, None, "sira")) for i in range(40)]
        reviews += [clean(f"c{i}") for i in range(60)]
        summary = aggregator().summarize(reviews, product_type="unknown_gadget")
        defect = summary["flags"]["defect"]
        self.assertEqual(defect["state"], badges.NO_BASELINE)
        self.assertIsNone(defect["baseline_rate"])
        # The observed rate is still reported so the reader sees the evidence.
        self.assertGreater(defect["rate"], 0)

    def test_small_samples_report_insufficient_evidence_not_not_flagged(self):
        reviews = [review("d1", ("defect", False, None, "sira"))] + [
            clean(f"c{i}") for i in range(4)
        ]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertEqual(
            summary["flags"]["defect"]["state"], badges.INSUFFICIENT_EVIDENCE
        )


class TestVerdictOrder(unittest.TestCase):
    def test_rules_are_evaluated_in_the_documented_order(self):
        expected = [
            badges.INSUFFICIENT_EVIDENCE,
            badges.SAFETY_WARNING,
            badges.POSSIBLE_SAFETY_ISSUE,
            badges.POSSIBLE_COUNTERFEIT_OR_DEFECT,
            badges.REVIEWS_MAY_BE_UNRELIABLE,
            badges.NO_WARNING_SIGNS,
        ]
        aggregate = aggregator()
        verdict = aggregate.verdict_for(
            SafetyBadge(), {}, usable=5, scrape_complete=True
        )
        self.assertEqual(verdict["state"], expected[0])
        self.assertEqual(verdict["rule_index"], 0)

    def test_safety_outranks_a_counterfeit_flag(self):
        aggregate = aggregator()
        safety = SafetyBadge(tier="High", severity=4)
        flags = {"counterfeit": CategoryFlag("counterfeit", state=badges.FLAGGED)}
        verdict = aggregate.verdict_for(safety, flags, usable=100)
        self.assertEqual(verdict["state"], badges.SAFETY_WARNING)

    def test_partial_scrape_yields_insufficient_evidence(self):
        aggregate = aggregator()
        verdict = aggregate.verdict_for(
            SafetyBadge(), {}, usable=500, scrape_complete=False
        )
        self.assertEqual(verdict["state"], badges.INSUFFICIENT_EVIDENCE)

    def test_moderate_safety_is_a_caution_not_a_warning(self):
        aggregate = aggregator()
        verdict = aggregate.verdict_for(
            SafetyBadge(tier="Moderate", severity=3), {}, usable=100
        )
        self.assertEqual(verdict["state"], badges.POSSIBLE_SAFETY_ISSUE)

    def test_review_quality_state_ships_disabled(self):
        aggregate = aggregator()
        self.assertFalse(aggregate.review_quality_enabled)
        verdict = aggregate.verdict_for(
            SafetyBadge(), {}, usable=100, reviews_unreliable=True
        )
        self.assertEqual(verdict["state"], badges.NO_WARNING_SIGNS)


def informed(key, *labels, text="mahina yung tunog"):
    """A review the five-label checker read; `labels` are its informational labels."""
    sentence = {
        "index": 0,
        "text_raw": text,
        "tags": [],
        "extra_tags": list(labels),
        "extra_terms": [{"ngram": "mahina", "category": label, "span": [0, 6]} for label in labels],
    }
    return {
        "review_key": key,
        "review_extra_tags": list(labels),
        "sentences": [sentence] if labels else [],
        "sentiment": {"label": "negative" if labels else "positive"},
    }


class TestInformationalReports(unittest.TestCase):
    """Guide 1.4's two newer labels are counted and quoted, never judged."""

    def test_reviews_are_counted_once_per_label_and_quoted(self):
        summary = aggregator().summarize(
            [informed("a", "quality"), informed("b", "quality", "not_as_described")]
            + [informed(f"c{i}") for i in range(38)]
        )
        reports = summary["reports"]
        self.assertEqual(reports["quality"]["flagged_reviews"], 2)
        self.assertEqual(reports["not_as_described"]["flagged_reviews"], 1)
        self.assertAlmostEqual(reports["quality"]["rate"], 0.05)
        quote = reports["not_as_described"]["evidence"][0]
        self.assertEqual(quote["text"], "mahina yung tunog")
        # Only the words behind this label, so the right ones are highlighted.
        self.assertEqual(quote["terms"], [{"ngram": "mahina", "category": "not_as_described", "span": [0, 6]}])

    def test_a_label_that_was_checked_and_not_found_reports_zero(self):
        summary = aggregator().summarize([informed(f"c{i}") for i in range(40)])
        self.assertEqual(summary["reports"]["not_as_described"]["flagged_reviews"], 0)

    def test_a_checker_without_the_two_labels_reports_nothing_about_them(self):
        summary = aggregator().summarize([clean(f"c{i}") for i in range(40)])
        self.assertNotIn("reports", summary)

    def test_they_never_change_the_verdict_or_the_flags(self):
        plain = aggregator().summarize([clean(f"c{i}") for i in range(40)])
        many = aggregator().summarize(
            [informed(f"q{i}", "quality", "not_as_described") for i in range(30)]
            + [informed(f"c{i}") for i in range(10)]
        )
        self.assertEqual(many["reports"]["quality"]["flagged_reviews"], 30)
        self.assertEqual(many["verdict"], plain["verdict"])
        self.assertEqual(many["verdict"]["state"], badges.NO_WARNING_SIGNS)
        self.assertEqual(set(many["flags"]), {"counterfeit", "defect"})


class TestOutputContract(unittest.TestCase):
    def test_no_combined_risk_number_is_produced(self):
        # The whole point of the redesign. Nothing may map a badge back to a
        # single score, or the old framing returns through the back door.
        summary = aggregator().summarize(
            [fire("r1")] + [clean(f"c{i}") for i in range(49)],
            product_type="power_bank",
        )
        forbidden = {
            "risk_score",
            "risk_score_scale",
            "risk_score_weights",
            "negative_contribution_points",
            "defect_contribution_points",
            "keyword_failure_rate",
            "detected_risk_ratio",
            "reliability_score",
        }
        self.assertEqual(forbidden & set(summary), set())

    def test_positive_verdict_never_calls_a_product_safe(self):
        summary = aggregator().summarize([clean(f"c{i}") for i in range(200)])
        verdict = summary["verdict"]
        self.assertEqual(verdict["state"], badges.NO_WARNING_SIGNS)
        self.assertNotIn("safe to buy", verdict["headline"].casefold())
        self.assertIn("not proof", verdict["detail"].casefold())

    def test_limits_notice_is_always_present(self):
        for reviews in ([clean("c1")], [fire("r1")] + [clean(f"c{i}") for i in range(99)]):
            summary = aggregator().summarize(reviews, product_type="power_bank")
            self.assertTrue(summary["limits_notice"])
            self.assertIn("cannot confirm", summary["limits_notice"])

    def test_versions_are_recorded(self):
        summary = aggregator().summarize([clean("c1")])
        self.assertEqual(summary["schema_version"], badges.SCHEMA_VERSION)
        self.assertEqual(summary["taxonomy_version"], taxonomy.TAXONOMY_VERSION)
        self.assertEqual(summary["parameters_version"], "hazard-parameters-1.1.0")

    def test_recommender_ranking_value_is_a_lower_bound(self):
        summary = aggregator().summarize(
            [clean(f"c{i}") for i in range(90)]
            + [review(f"n{i}", ("defect", False, None, "sira")) for i in range(10)]
        )
        self.assertEqual(summary["sentiment_counts"]["positive"], 90)
        # 90/100 observed, but the ranking value is the conservative bound.
        self.assertLess(summary["positive_ratio_lower_bound"], 0.90)
        self.assertGreater(summary["positive_ratio_lower_bound"], 0.80)


class TestFloorWithoutBaseline(unittest.TestCase):
    """No product-type baseline exists yet, so every counterfeit and defect
    flag is NO_BASELINE. That must not mean "nothing to report"."""

    def test_the_floor_alone_can_warn(self):
        reviews = [review(f"f{i}", ("counterfeit", True, None, "peke to")) for i in range(6)]
        reviews += [clean(f"c{i}") for i in range(34)]
        summary = aggregator().summarize(reviews, product_type="unknown_gadget")

        # The flag still says honestly that it has no peer rate...
        self.assertEqual(summary["flags"]["counterfeit"]["state"], badges.NO_BASELINE)
        # ...but six of forty reviews calling it fake is not "no warning signs".
        self.assertEqual(summary["verdict"]["state"], badges.POSSIBLE_COUNTERFEIT_OR_DEFECT)
        self.assertIn("floor", summary["verdict"]["detail"])
        self.assertIn("no product-type baseline", summary["verdict"]["detail"])

    def test_an_isolated_report_below_the_floor_does_not_warn(self):
        reviews = [review("f0", ("counterfeit", True, None, "peke to"))]
        reviews += [clean(f"c{i}") for i in range(99)]
        summary = aggregator().summarize(reviews, product_type="unknown_gadget")
        self.assertEqual(summary["verdict"]["state"], badges.NO_WARNING_SIGNS)

    def test_a_baseline_when_present_still_decides(self):
        # 3 of 100 is above the 2% floor on its face, but its Wilson lower
        # bound is not, and the power_bank baseline applies: not flagged.
        reviews = [review(f"f{i}", ("counterfeit", True, None, "peke to")) for i in range(3)]
        reviews += [clean(f"c{i}") for i in range(97)]
        summary = aggregator().summarize(reviews, product_type="power_bank")
        self.assertEqual(summary["flags"]["counterfeit"]["state"], badges.NOT_FLAGGED)
        self.assertEqual(summary["verdict"]["state"], badges.NO_WARNING_SIGNS)


class TestMissingBaselines(unittest.TestCase):
    def test_aggregator_works_with_no_reference_corpus_at_all(self):
        # Phase ordering: the product layer must run before the reference
        # corpus exists, or nothing can be demoed until annotation finishes.
        aggregate = ProductBadgeAggregator(baselines={})
        summary = aggregate.summarize(
            [fire("r1"), fire("r2")] + [clean(f"c{i}") for i in range(98)],
            product_type="power_bank",
        )
        self.assertEqual(summary["verdict"]["state"], badges.SAFETY_WARNING)
        self.assertEqual(summary["flags"]["defect"]["state"], badges.NO_BASELINE)
        self.assertEqual(summary["safety"]["likelihood_basis"], "absolute_fallback")


class TestDefaultProductType(unittest.TestCase):
    """A baseline file may name the type every unnamed listing is compared with."""

    DEFAULT = {
        "version": "test-baselines-2",
        "default_product_type": "gadget",
        "product_types": {
            "gadget": {"defect": {"pooled_rate": 0.05}, "counterfeit": {"pooled_rate": 0.0}},
            "power_bank": {"defect": {"pooled_rate": 0.20}},
        },
    }

    def reviews(self, faulty):
        return [review(f"d{i}", ("defect", True, None, "sira agad")) for i in range(faulty)] + [
            clean(f"c{i}") for i in range(100 - faulty)
        ]

    def test_an_unnamed_listing_is_compared_with_the_default_type(self):
        summary = ProductBadgeAggregator(baselines=self.DEFAULT).summarize(self.reviews(6))
        flag = summary["flags"]["defect"]
        # 6 in 100 is above the old 3% floor on its face; its lower bound is
        # not above the 5% typical rate, so it is not a warning.
        self.assertEqual(flag["state"], badges.NOT_FLAGGED)
        self.assertEqual(flag["baseline_rate"], 0.05)
        self.assertEqual(flag["baseline_source"], "test-baselines-2/gadget")
        self.assertEqual(summary["verdict"]["state"], badges.NO_WARNING_SIGNS)

    def test_a_rate_clearly_above_typical_still_warns(self):
        summary = ProductBadgeAggregator(baselines=self.DEFAULT).summarize(self.reviews(20))
        self.assertEqual(summary["flags"]["defect"]["state"], badges.FLAGGED)

    def test_a_named_type_keeps_its_own_baseline(self):
        summary = ProductBadgeAggregator(baselines=self.DEFAULT).summarize(self.reviews(20), product_type="power_bank")
        self.assertEqual(summary["flags"]["defect"]["baseline_source"], "test-baselines-2/power_bank")
        self.assertEqual(summary["flags"]["defect"]["state"], badges.NOT_FLAGGED)

    def test_the_floor_still_holds_where_typical_is_zero(self):
        fakes = [review(f"f{i}", ("counterfeit", True, None, "peke to")) for i in range(3)]
        summary = ProductBadgeAggregator(baselines=self.DEFAULT).summarize(fakes + [clean(f"c{i}") for i in range(97)])
        self.assertEqual(summary["flags"]["counterfeit"]["state"], badges.NOT_FLAGGED)

    def test_without_a_default_an_unnamed_listing_has_no_baseline(self):
        summary = ProductBadgeAggregator(baselines=BASELINES).summarize(self.reviews(6))
        self.assertEqual(summary["flags"]["defect"]["state"], badges.NO_BASELINE)


class TestShippedBaselines(unittest.TestCase):
    """The measured line is in the checker's own units, so it goes stale with it."""

    def test_the_baseline_file_was_measured_with_the_served_checker(self):
        import json

        if not badges.DEFAULT_BASELINES_PATH.is_file():
            self.skipTest("no baseline file is installed")
        baselines = json.loads(badges.DEFAULT_BASELINES_PATH.read_bytes().decode("utf-8"))
        from defaketive_sentiment import registry
        from defaketive_sentiment.transformer_sentiment import TransformerModelError

        try:
            model = registry.load_model(registry.HAZARD_BADGES)
        except TransformerModelError:
            self.skipTest("the sentiment model is not installed")
        self.assertEqual(
            baselines["measured_with"], model.checker_version,
            "The checker changed since the typical rates were measured. Rebuild them: "
            "python research/verdict_check.py --star-totals, then "
            "python research/build_reference_baselines.py --write",
        )


class TestSimilarity(unittest.TestCase):
    def test_identical_text_is_maximally_similar(self):
        self.assertAlmostEqual(badges.cosine_similarity("nasunog agad", "nasunog agad"), 1.0)

    def test_unrelated_text_is_dissimilar(self):
        self.assertLess(
            badges.cosine_similarity("nasunog agad ang unit", "maganda ang packaging"),
            0.3,
        )

    def test_empty_text_is_not_similar_to_anything(self):
        self.assertEqual(badges.cosine_similarity("", "nasunog"), 0.0)


if __name__ == "__main__":
    unittest.main()
