"""The electronic-gadget scope used for round 6 and gadget-only training."""

from __future__ import annotations

import unittest

from research.build_labelling_queue import ROUND_DEFAULTS
from research.gadget_scope import is_gadget, mentions_gadget


class GadgetScopeTests(unittest.TestCase):
    def test_an_electronics_scrape_is_in_scope_whatever_the_text(self) -> None:
        self.assertTrue(is_gadget({"category": "audio", "review_text": "ok naman po"}))

    def test_out_of_scope_and_temu_rows_are_left_out(self) -> None:
        self.assertFalse(is_gadget({"category": "out_of_scope", "review_text": "sira ang charger"}))
        self.assertFalse(is_gadget({"category": "", "source_file": "temu_ph_direct_1.json", "review_text": "sira ang charger"}))

    def test_a_review_without_a_category_is_judged_by_its_words(self) -> None:
        self.assertTrue(is_gadget({"category": "", "review_text": "Mabilis malowbat ang powerbank"}))
        self.assertFalse(is_gadget({"category": "", "review_text": "Maganda ang tela, kasya sa akin"}))

    def test_gadget_words_need_word_boundaries(self) -> None:
        self.assertTrue(mentions_gadget("sira yung mic"))
        self.assertFalse(mentions_gadget("mickey mouse-themed lang"[:6]))

    def test_round_six_is_gadget_only(self) -> None:
        self.assertEqual(ROUND_DEFAULTS[6]["scope"], "gadget")
        self.assertNotIn("firecs", ROUND_DEFAULTS[6]["cohorts"])


if __name__ == "__main__":
    unittest.main()
