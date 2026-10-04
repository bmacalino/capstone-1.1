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


if __name__ == "__main__":
    unittest.main()
