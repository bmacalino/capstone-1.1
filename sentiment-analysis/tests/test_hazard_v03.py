"""Hazard checker 0.3.0: negation scope, guide phrasings, and exclusions.

Each case below is a sentence type seen in labelling round 1 (the development
set), written out fresh. Round 2 is the held-out test and is not used here.
"""

import unittest

from defaketive_sentiment import text_norm as tn
from defaketive_sentiment.hazard_model import MODEL_VERSION, LexiconHazardTagger


def negated(text, term):
    normalized = tn.normalize_text(text)
    return tn.is_negated_adjacent(normalized, normalized.index(term))


class TestAdjacentNegation(unittest.TestCase):
    def test_a_negator_meant_for_another_word_does_not_cancel_the_report(self):
        self.assertFalse(negated("not good sira yung isa", "sira"))
        self.assertFalse(negated("hindi maganda sira ang isang earpod", "sira"))
        self.assertFalse(negated("hindi samsung fake", "fake"))

    def test_negation_reaches_across_function_words(self):
        self.assertTrue(negated("walang sira", "sira"))
        self.assertTrue(negated("hindi po sira", "sira"))
        self.assertTrue(negated("hindi naman talaga sira", "sira"))
        self.assertTrue(negated("never had any issues with overheating", "overheat"))

    def test_negation_reaches_a_verb_tied_to_the_term_by_a_linker(self):
        # "no smoke came out": the negator governs the verb, "na" links the smoke.
        self.assertTrue(negated("wala nalabas na usok", "usok"))
        self.assertTrue(negated("hindi natagal ung init", "init"))

    def test_negation_covers_an_or_list(self):
        self.assertTrue(negated("walang dent o sira sa item", "sira"))

    def test_the_sentiment_path_keeps_the_old_rule(self):
        text = tn.normalize_text("not good sira yung isa")
        self.assertTrue(tn.is_negated(text, text.index("sira")))


class TestTaggerV03(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tagger = LexiconHazardTagger()
        cls.old = LexiconHazardTagger(rules="0.2")

    def tags(self, sentence, tagger=None):
        return set((tagger or self.tagger).tag_sentence(sentence)["tags"])

    def test_the_version_is_0_3_0(self):
        self.assertEqual(MODEL_VERSION, "defaketive-hazard-lexicon-0.3.0")

    def test_guide_defect_phrasings_are_found(self):
        for sentence in (
            "madaling malowbat, wala pang 1 hour",
            "ayaw mag charge kahit bago",
            "hindi nagana yung left side",
            "hindi na gumagana after 2 days",
            "the battery drains quickly",
            "it doesn't charge at all",
        ):
            with self.subTest(sentence=sentence):
                self.assertIn("defect", self.tags(sentence))

    def test_guide_counterfeit_phrasings_are_found(self):
        for sentence in ("this is not legit", "kaso iba sa picture yung dumating", "these are not genuine"):
            with self.subTest(sentence=sentence):
                self.assertIn("counterfeit", self.tags(sentence))

    def test_the_negation_fix_reaches_the_tags(self):
        self.assertIn("defect", self.tags("not good sira yung isa"))
        self.assertNotIn("defect", self.tags("not good sira yung isa", self.old))

    def test_a_missing_item_is_not_a_defect(self):
        # Guide rule 1: missing from the order is not a faulty product.
        self.assertNotIn("defect", self.tags("kulang ng 5 yung brick sa wallpaper"))

    def test_a_damaged_box_is_not_a_defective_product(self):
        self.assertNotIn("defect", self.tags("the box is damaged but the controller works perfectly"))
        self.assertNotIn("defect", self.tags("nasira yung kahon pero okay ang item"))
        self.assertIn("defect", self.tags("sira yung item pagdating"))

    def test_names_and_idioms_are_not_reports(self):
        self.assertNotIn("safety", self.tags("same lang ng kulay nung burnt at burgandy"))
        self.assertNotIn("safety", self.tags("mali yung spelling ng smoke removal"))
        self.assertNotIn("counterfeit", self.tags("panget, fake news, plum yung dumating"))
        self.assertNotIn("counterfeit", self.tags("ang cute ng fake nailll"))

    def test_a_name_phrase_only_drops_its_own_match(self):
        tagged = self.tagger.tag_sentence("smoke removal daw pero umusok yung unit")
        self.assertIn("safety", tagged["tags"])

    def test_a_worry_is_not_a_report(self):
        self.assertNotIn("counterfeit", self.tags("hesitant ako kasi baka fake"))
        self.assertIn("counterfeit", self.tags("fake pala ito"))

    def test_weak_heat_from_a_heating_product_is_not_a_hazard(self):
        self.assertNotIn("safety", self.tags("ang hina ng init kahit matagal nakasaksak"))

    def test_rules_0_2_reproduce_the_old_behaviour(self):
        self.assertIn("counterfeit", self.tags("panget, fake news, plum yung dumating", self.old))


class TestCounterfeitRulesC1(unittest.TestCase):
    """Round 4 (5 Oct 2026): "scam" alone is not a fake report; comparisons
    with the original are."""

    def setUp(self):
        self.tagger = LexiconHazardTagger()

    def flagged(self, sentence, tagger=None):
        return "counterfeit" in (tagger or self.tagger).tag_sentence(sentence)["tags"]

    def test_scam_alone_is_not_a_fake_report(self):
        for sentence in ("Definitely scam.", "scam po yung seller! dalawa order ko isa lang dumating"):
            with self.subTest(sentence=sentence):
                tagged = self.tagger.tag_sentence(sentence)
                self.assertNotIn("counterfeit", tagged["tags"])
                self.assertEqual(tagged["suppressed"].get("counterfeit"), "scam_alone")

    def test_scam_beside_a_fake_word_still_counts(self):
        self.assertTrue(self.flagged("scam, peke yung item"))

    def test_a_false_capacity_claim_is_not_a_fake_report(self):
        self.assertFalse(self.flagged("hindi legit na 20k mAh"))
        self.assertTrue(self.flagged("hindi legit yung brand"))

    def test_comparisons_with_the_original_are_found(self):
        for sentence in (
            "but you can feel it's not 100% original!",
            "not good hindi sya samsung hindi rin original kaya mura",
            "pero copy lang sya",
            "low quality anlayo sa original",
            "hindi sya kaamoy ng tunay na perfume",
            "item not shinny as the original.",
            "The batteries are no longer original",
        ):
            with self.subTest(sentence=sentence):
                self.assertTrue(self.flagged(sentence))

    def test_praise_and_doubt_are_left_alone(self):
        for sentence in ("parang original, ang ganda", "legit seller, original talaga", "baka hindi sya original"):
            with self.subTest(sentence=sentence):
                self.assertFalse(self.flagged(sentence))

    def test_the_old_behaviour_stays_measurable(self):
        old = LexiconHazardTagger(counterfeit_rules=None)
        self.assertTrue(self.flagged("Definitely scam.", old))
        self.assertFalse(self.flagged("pero copy lang sya", old))


class TestExtraLabelsC2(unittest.TestCase):
    """Guide 1.4: "poor quality" and "not as advertised", rules c2. The app
    serves them; the class default stays c1 so earlier measurements reproduce."""

    def setUp(self):
        self.tagger = LexiconHazardTagger(counterfeit_rules="c2")

    def tags(self, sentence):
        tagged = self.tagger.tag_sentence(sentence)
        return set(tagged["tags"]), set(tagged["extra_tags"])

    def test_the_class_default_reports_no_extra_labels(self):
        self.assertNotIn("extra_tags", LexiconHazardTagger().tag_sentence("Sobrang hina yung fan"))

    def test_a_disputed_number_is_not_as_described_and_not_a_fake(self):
        for sentence in ("Fake ung 20000mah", "hindi sya legit na 20000 mAh", "meaning nasa 2500mah lng itong powerbank.", "Hindi Legit 24 tv."):
            with self.subTest(sentence=sentence):
                main, extra = self.tags(sentence)
                self.assertNotIn("counterfeit", main)
                self.assertIn("not_as_described", extra)

    def test_a_picture_mismatch_moves_out_of_counterfeit(self):
        main, extra = self.tags("kaso iba sa picture yung pinadala nyo")
        self.assertNotIn("counterfeit", main)
        self.assertIn("not_as_described", extra)
        # Under rules c1 it is still a counterfeit report.
        self.assertIn("counterfeit", LexiconHazardTagger().tag_sentence("kaso iba sa picture yung pinadala nyo")["tags"])

    def test_a_real_fake_report_keeps_its_label(self):
        main, extra = self.tags("peke yung item, hindi gaya ng nasa picture")
        self.assertIn("counterfeit", main)
        self.assertIn("not_as_described", extra)

    def test_a_missing_feature_is_not_as_described(self):
        self.assertIn("not_as_described", self.tags("Hindi totoo na motion sensor light to.")[1])

    def test_weak_or_cosmetic_is_poor_quality(self):
        for sentence in ("Sobrang hina yung fan", "Mahina lang pala speaker niya", "may dent agad yung mouse", "Ung tunog ng speaker sabog", "poor quality"):
            with self.subTest(sentence=sentence):
                self.assertIn("quality", self.tags(sentence)[1])

    def test_negation_and_the_box_are_left_alone(self):
        for sentence in ("Wala pong gasgas", "hindi naman mahina ang sound", "yupi yupi yung box", "baka iba sa picture"):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.tags(sentence)[1], set())

    def test_the_words_behind_a_label_are_located_in_the_original_text(self):
        sentence = "Sobrang HINA yung fan, may gasgas pa."
        tagged = self.tagger.tag_sentence(sentence)
        found = {(term["category"], sentence[term["span"][0] : term["span"][1]]) for term in tagged["extra_terms"]}
        self.assertEqual(found, {("quality", "HINA"), ("quality", "gasgas")})
        # Kept apart from the terms the badge categories and the learned models read.
        self.assertFalse([term for term in tagged["terms"] if term["category"] in ("quality", "not_as_described")])

    def test_the_reviewers_other_device_is_not_a_dispute(self):
        # Seen on a whole listing: "my other charger is only 20W".
        self.assertEqual(self.tags("2nd time ko na bumili ng Anker charger, yung isa kong gamit 20W lang")[1], set())
        self.assertIn("not_as_described", self.tags("sabi 65w pero 20w lang")[1])

    def test_a_disputed_number_is_listed_once(self):
        tagged = self.tagger.tag_sentence("hindi legit na 20k mAh")
        self.assertEqual([term["category"] for term in tagged["extra_terms"]], ["not_as_described"])

    def test_a_review_carries_the_two_labels_beside_its_tags(self):
        from defaketive_sentiment.hazard_model import HazardBadgeModel

        model = HazardBadgeModel(allow_unqualified=True, tagger=self.tagger)
        analysis = model.analyze("Iba sa picture yung dumating. Mahina pa yung tunog.")
        self.assertEqual(analysis["review_extra_tags"], ["quality", "not_as_described"])
        # review_tags keeps meaning safety, defect or counterfeit wherever it is stored.
        self.assertFalse({"quality", "not_as_described"} & set(analysis["review_tags"]))
        self.assertEqual(model.analyze("Maganda, sulit.")["review_extra_tags"], [])
        # A checker without the two labels says nothing about them.
        self.assertNotIn("review_extra_tags", HazardBadgeModel(allow_unqualified=True).analyze("Mahina yung tunog."))

    def test_second_revision_of_the_poor_quality_patterns(self):
        for sentence in (
            "medyo maingay lang gamitin", "medyo delay yung mic", "parang gamit na yung dumating",
            "hnd sya ganun katibay", "hirap pindutin ng buttons",
        ):
            with self.subTest(sentence=sentence):
                self.assertIn("quality", self.tags(sentence)[1])
        # A condition is not a complaint, and slow charging is a defect, not poor quality.
        for sentence in ("pag mahina lang ang volume ok naman", "ang hina na nya mag charge"):
            with self.subTest(sentence=sentence):
                self.assertNotIn("quality", self.tags(sentence)[1])

    def test_second_revision_of_the_fake_rules(self):
        for sentence in (
            "i know di to original", "pekeng pekeng product nyo", "sd card not sandisk",
            "akala ko legit na orig ang dumating", "quite similar to the original", "buhangin ang laman ng battery",
        ):
            with self.subTest(sentence=sentence):
                self.assertIn("counterfeit", self.tags(sentence)[0])
        # "Fake" about the reviews or the seller, a question, the reviewer's own phone.
        for sentence in ("fake ba to", "fake seller", "hindi samsung ang phone ko"):
            with self.subTest(sentence=sentence):
                self.assertNotIn("counterfeit", self.tags(sentence)[0])
        # Rules c1 know none of this, so earlier measurements reproduce.
        self.assertNotIn("counterfeit", LexiconHazardTagger().tag_sentence("sd card not sandisk")["tags"])

    def test_the_not_as_advertised_patterns_were_left_as_they_were(self):
        # A revision of them found nothing new on held-out sentences and was dropped.
        self.assertEqual(self.tags("it is not ultra fast charging")[1], set())
        self.assertIn("not_as_described", self.tags("hindi totoo ang 20000mah")[1])

    def test_the_version_names_the_rules(self):
        from defaketive_sentiment.hazard_model import HazardBadgeModel

        model = HazardBadgeModel(allow_unqualified=True, tagger=self.tagger)
        self.assertIn("+c2r2", model.model_version)


class TestKeywordRules04(unittest.TestCase):
    """Rules 0.4: negation the keywords missed, mistaken beliefs, things that
    have not happened. Rules 0.3 must keep behaving as recorded."""

    def setUp(self):
        self.old = LexiconHazardTagger()
        self.new = LexiconHazardTagger(rules="0.4")

    def defect(self, tagger, sentence):
        return "defect" in tagger.tag_sentence(sentence)["tags"]

    def test_denied_defects_are_no_longer_flagged(self):
        for sentence in (
            "Wala pong sira ang mga na recieve ko.",
            "original talaga, walang ka sira sira",
            "Ni minsan, hindi rin kami naka experience ng lag nitong tv nato.",
            "sana hindi agad madaling malowbat pag ginamit ko mamaya.",
        ):
            with self.subTest(sentence=sentence):
                self.assertTrue(self.defect(self.old, sentence))
                self.assertFalse(self.defect(self.new, sentence))

    def test_a_mistaken_belief_is_not_a_report(self):
        tagged = self.new.tag_sentence("Akala ko nung una sira na hahaha lowbat lang pala.")
        self.assertNotIn("defect", tagged["tags"])
        self.assertEqual(tagged["excluded"]["defect"], ["mistaken_belief"])
        # "I thought it was fine; it turned out broken" is still a report.
        self.assertTrue(self.defect(self.new, "akala ko okay, sira pala"))

    def test_something_that_has_not_happened_is_not_a_report(self):
        for sentence in ("pabili ulit ako sa susunod pag nasira na.", "baka masira agad"):
            with self.subTest(sentence=sentence):
                self.assertFalse(self.defect(self.new, sentence))

    def test_real_reports_are_kept(self):
        # Each of these was lost by a wider first version of the rules.
        for sentence in (
            "wag na kayo bumili ang bilis masira nito",
            "paano naman ito magagamit kung sira",
            "sira agad, hindi na gumagana",
            "pag sinaksak mo sira agad",
        ):
            with self.subTest(sentence=sentence):
                self.assertTrue(self.defect(self.new, sentence))

    def test_damage_listed_up_to_the_box_is_about_the_box(self):
        # Revision g1: two reviews the owner found flagged in the app.
        tagged = self.new.tag_sentence("sira ,punit at wasak2 na ung box. 2.")
        self.assertNotIn("defect", tagged["tags"])
        self.assertEqual(tagged["excluded"]["defect"], ["packaging_only"])
        self.assertFalse(self.defect(self.new, "yung kahon yupi, punit at sira na"))
        # The list has to reach the box, and the product itself is still a report.
        self.assertTrue(self.defect(self.new, "sira, punit at ayaw gumana"))
        self.assertTrue(self.defect(self.new, "yupi at sira yung mouse mismo"))

    def test_a_unit_that_lasted_years_is_not_a_fault_report(self):
        sentence = (
            "yung una kong order tumagal ng 2 years bago na sira ang turbo and ang left click "
            "so I assume Ang life span netong mouse ay around that time"
        )
        tagged = self.new.tag_sentence(sentence)
        self.assertNotIn("defect", tagged["tags"])
        self.assertEqual(tagged["excluded"]["defect"], ["after_years_of_use"])
        self.assertFalse(self.defect(self.new, "umabot ng tatlong taon bago nasira"))
        # Months, or under a year, are reports.
        for report in ("3 months pa lang sira na yung left click", "wala pang 1 year sira na", "2 weeks bago nasira"):
            with self.subTest(sentence=report):
                self.assertTrue(self.defect(self.new, report))

    def test_the_guards_can_be_switched_off_to_reproduce_the_first_0_4(self):
        tagger = LexiconHazardTagger(rules="0.4")
        tagger.context_guards = False
        self.assertTrue(self.defect(tagger, "sira ,punit at wasak2 na ung box."))
        # Rules 0.3 never had them.
        self.assertTrue(self.defect(self.old, "tumagal ng 2 years bago nasira"))

    def test_a_safety_report_is_untouched(self):
        sentence = "akala ko original hindi namn pa nag iinit yung adaptor"
        self.assertEqual(
            "safety" in self.old.tag_sentence(sentence)["tags"], "safety" in self.new.tag_sentence(sentence)["tags"]
        )


class TestExtraLabelRules(unittest.TestCase):
    """The revised quality and not-as-described rules (extra_labels.py)."""

    def setUp(self):
        self.tagger = LexiconHazardTagger(rules="0.4", counterfeit_rules="c2")

    def extra(self, sentence):
        return set(self.tagger.tag_sentence(sentence)["extra_tags"])

    def test_not_as_described(self):
        for sentence in (
            "hindi sya 10meters",
            "it s not even 2 liter",
            "nasa description niyo 2L tas dumating 1L",
            "Hindi Naman fast charging",
            "hindi katulad nung nasa picture ni seller",
            "maganda lng sya sa pic",
            "color is also lighter than advertised",
            "This is a false advertisement",
        ):
            with self.subTest(sentence=sentence):
                self.assertIn("not_as_described", self.extra(sentence))

    def test_a_number_about_the_reviewers_own_phone_is_left_alone(self):
        for sentence in (
            "pero napakabilis malowbat, 3500mah lang battery ng cp ko",
            "hindi nka pag full charge sa cp ko kahit 3020mah lng.",
            "not legit dual shock 4",
            "mas maganda sa personal kesa sa picture",
        ):
            with self.subTest(sentence=sentence):
                self.assertNotIn("not_as_described", self.extra(sentence))

    def test_poor_quality(self):
        for sentence in (
            "hndi maganda tunog..",
            "tunog lata",
            "nakakadisappoint kasi sobrang nipis ng tela",
            "pero mabilis masira",
            "it s lightweight and not that sturdy",
            "kaso my punit sa ilalim",
            "Comfort:laging nahuhulog",
        ):
            with self.subTest(sentence=sentence):
                self.assertIn("quality", self.extra(sentence))

    def test_praise_negation_hearsay_and_the_box_are_left_alone(self):
        for sentence in (
            "d namn manipis",
            "maganda nmm ung tunog h di basag",
            "pero hindi yung tipong madaling masira",
            "may nababasa ako sa comments na mabilis nasira",
            "may yupi yung kanto ng box",
            "tangal ung hinang nung isa",
        ):
            with self.subTest(sentence=sentence):
                self.assertNotIn("quality", self.extra(sentence))


if __name__ == "__main__":
    unittest.main()
