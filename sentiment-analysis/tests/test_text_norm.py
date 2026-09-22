import csv
import random
import string
import unittest
from pathlib import Path

from defaketive_sentiment import text_norm as tn
from defaketive_sentiment.model import DefaketiveSentimentModel

LEXICON_DIR = Path(tn.__file__).with_name("lexicons")
EMPTY = "<>"


def _read_golden(name: str) -> list[dict[str, str]]:
    path = LEXICON_DIR / name
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = (line for line in handle if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t")
        return list(reader)


def _decode(value: str) -> str:
    return "" if value == EMPTY else value.replace("\\n", "\n")


class TestLegacyProfileIsUnchanged(unittest.TestCase):
    """The legacy profile is baked into every committed artifact and result."""

    def test_matches_the_original_normalizer_on_curated_cases(self):
        for row in _read_golden("normalizer_golden_v1.tsv"):
            source = _decode(row["input"])
            with self.subTest(source=source):
                self.assertEqual(
                    tn.normalize_text(source, profile=tn.LEGACY_VADER),
                    DefaketiveSentimentModel.normalize(source),
                )

    def test_matches_the_original_normalizer_on_random_input(self):
        # Curated cases only cover what we thought to write down. Fuzzing is
        # what catches an accidental divergence in an unusual character class.
        random.seed(20260921)
        alphabet = string.printable[:80] + "áé😡–—’` \t"
        for _ in range(3000):
            source = "".join(
                random.choice(alphabet) for _ in range(random.randint(0, 40))
            )
            with self.subTest(source=source):
                self.assertEqual(
                    tn.normalize_text(source, profile=tn.LEGACY_VADER),
                    DefaketiveSentimentModel.normalize(source),
                )


class TestHazardProfile(unittest.TestCase):
    def test_matches_the_frozen_golden_fixture(self):
        rows = _read_golden("normalizer_golden_v1.tsv")
        self.assertGreater(len(rows), 20, "golden fixture looks truncated")
        for row in rows:
            source = _decode(row["input"])
            with self.subTest(source=source, note=row["note"]):
                self.assertEqual(
                    tn.normalize_text(source, profile=tn.HAZARD_V1),
                    _decode(row["expected_hazard"]),
                )

    def test_affix_variants_of_one_root_collapse_together(self):
        # The reason the variant table exists: four surface forms, one lemma,
        # so the evidence for a rare hazard is not split four ways.
        forms = ["uminit", "umiinit", "nag-init", "mainit", "naginit"]
        normalized = {tn.normalize_text(form) for form in forms}
        self.assertEqual(normalized, {"init"})

    def test_digits_are_never_collapsed_as_elongation(self):
        # Reviews are full of capacity and price figures. A naive elongation
        # rule turns "20.000mah" into "20.0mah".
        for source in ("20.000mah", "10000mah", "php 1999", "5000 mah", "100000"):
            with self.subTest(source=source):
                self.assertEqual(tn.normalize_text(source), source)

    def test_ordinary_doubled_letters_survive(self):
        for source in ("off", "hello", "bull", "success"):
            with self.subTest(source=source):
                self.assertEqual(tn.normalize_text(source), source)

    def test_letter_elongation_collapses(self):
        self.assertEqual(tn.normalize_text("grabeee"), "grabe")
        self.assertEqual(tn.normalize_text("initttt"), "init")
        self.assertEqual(tn.normalize_text("sobranggggg"), "sobrang")

    def test_sabog_and_sumabog_stay_distinct(self):
        # Documented omission: bare "sabog" usually means muddy sound in these
        # reviews. Merging it into the explosion root would import a known false
        # friend straight into the safety class.
        self.assertNotEqual(tn.normalize_text("sabog"), tn.normalize_text("sumabog"))

    def test_negators_are_preserved(self):
        # Normalization must not destroy the words the negation check reads.
        for negator in tn.FILIPINO_NEGATORS:
            with self.subTest(negator=negator):
                self.assertIn(
                    tn.normalize_text(f"{negator} sira").split()[0],
                    tn.FILIPINO_NEGATORS,
                )

    def test_rejects_an_unknown_profile(self):
        with self.assertRaises(ValueError):
            tn.normalize_text("test", profile="whatever")


class TestSpans(unittest.TestCase):
    def test_spans_agree_with_normalize_text(self):
        for row in _read_golden("normalizer_golden_v1.tsv"):
            source = _decode(row["input"])
            with self.subTest(source=source):
                text, _ = tn.normalize_with_spans(source)
                self.assertEqual(text, tn.normalize_text(source, profile=tn.HAZARD_V1))

    def test_spans_point_at_the_original_characters(self):
        raw = "Yung PowerBank ko UMIINIT sobra!"
        text, spans = tn.normalize_with_spans(raw)
        self.assertEqual(text, "yung powerbank ko init sobra!")
        by_norm = {text[span.norm_start : span.norm_end]: span for span in spans}
        # A four-character lemma traced back to the seven characters written.
        self.assertEqual(by_norm["init"].raw_text, "UMIINIT")
        self.assertEqual(raw[by_norm["init"].raw_start : by_norm["init"].raw_end], "UMIINIT")

    def test_every_span_slices_its_own_raw_text(self):
        for row in _read_golden("normalizer_golden_v1.tsv"):
            source = _decode(row["input"])
            _, spans = tn.normalize_with_spans(source)
            for span in spans:
                with self.subTest(source=source, span=span):
                    self.assertEqual(source[span.raw_start : span.raw_end], span.raw_text)

    def test_spans_are_ordered_and_non_overlapping(self):
        _, spans = tn.normalize_with_spans("Yung PowerBank ko UMIINIT sobra!")
        for earlier, later in zip(spans, spans[1:]):
            self.assertLessEqual(earlier.norm_end, later.norm_start)
            self.assertLessEqual(earlier.raw_end, later.raw_start)


class TestSegmenter(unittest.TestCase):
    def test_matches_the_frozen_golden_fixture(self):
        rows = _read_golden("segmenter_golden_v1.tsv")
        self.assertGreater(len(rows), 12, "golden fixture looks truncated")
        for row in rows:
            source = _decode(row["input"])
            expected = row["expected_segments"]
            expected_segments = () if expected == EMPTY else tuple(expected.split(" || "))
            with self.subTest(source=source, note=row["note"]):
                self.assertEqual(tn.segment_sentences(source), expected_segments)

    def test_segments_never_lose_words(self):
        # A segmenter that drops text would silently hide evidence.
        for row in _read_golden("segmenter_golden_v1.tsv"):
            source = _decode(row["input"])
            with self.subTest(source=source):
                joined = " ".join(tn.segment_sentences(source))
                for word in source.split():
                    if any(character.isalnum() for character in word):
                        self.assertIn(word.strip(".,;!?"), joined)

    def test_contrast_keeps_the_hazard_out_of_the_praise(self):
        # The motivating case for splitting on contrast at all.
        segments = tn.segment_sentences("Maganda ang packaging pero sobrang init")
        self.assertEqual(len(segments), 2)
        self.assertIn("sobrang init", segments[1])
        self.assertNotIn("init", segments[0])

    def test_segment_count_is_capped(self):
        source = ". ".join(f"sira ang item {index}" for index in range(200))
        segments = tn.segment_sentences(source)
        self.assertLessEqual(len(segments), tn.MAX_SEGMENTS)
        # The tail is folded in rather than discarded.
        self.assertIn("199", segments[-1])

    def test_empty_input_yields_no_segments(self):
        for source in ("", "   ", None, "\n\n"):
            with self.subTest(source=source):
                self.assertEqual(tn.segment_sentences(source), ())


class TestNegation(unittest.TestCase):
    def test_negation_applies_within_a_clause(self):
        text = "walang sira"
        self.assertTrue(tn.is_negated(text, text.index("sira")))

    def test_negation_does_not_cross_a_conjunction(self):
        text = "walang sira at gumagana nang maayos"
        self.assertFalse(tn.is_negated(text, text.index("gumagana")))

    def test_negation_does_not_cross_a_sentence_boundary(self):
        text = "walang problema. sira agad"
        self.assertFalse(tn.is_negated(text, text.index("sira")))

    def test_negation_does_not_reach_back_indefinitely(self):
        text = "hindi ko alam kung bakit talaga sira"
        self.assertFalse(tn.is_negated(text, text.index("sira")))

    def test_terms_that_carry_their_own_negation_are_recognized(self):
        for term in ("hindi gumagana", "walang power", "ayaw mag-charge"):
            with self.subTest(term=term):
                self.assertTrue(tn.starts_with_negator(term))
        self.assertFalse(tn.starts_with_negator("sobrang init"))


class TestTransformer(unittest.TestCase):
    def test_transform_applies_the_profile(self):
        transformer = tn.TaglishNormalizer(profile=tn.HAZARD_V1)
        self.assertEqual(transformer.transform(["UMIINIT na"]), ["init na"])

    def test_legacy_profile_is_selectable(self):
        transformer = tn.TaglishNormalizer(profile=tn.LEGACY_VADER)
        self.assertEqual(transformer.transform(["UMIINIT na"]), ["umiinit na"])

    def test_rejects_an_unknown_profile(self):
        with self.assertRaises(ValueError):
            tn.TaglishNormalizer(profile="whatever")

    def test_exposes_sklearn_parameter_methods(self):
        transformer = tn.TaglishNormalizer()
        self.assertEqual(transformer.get_params()["profile"], tn.HAZARD_V1)
        transformer.set_params(profile=tn.LEGACY_VADER)
        self.assertEqual(transformer.profile, tn.LEGACY_VADER)


class TestVariantTable(unittest.TestCase):
    def test_table_loaded(self):
        self.assertGreater(len(tn.VARIANTS), 50)

    def test_no_variant_maps_to_itself_through_a_chain(self):
        # A lemma that is itself a variant would make normalization depend on
        # iteration order.
        for variant, lemma in tn.VARIANTS.items():
            with self.subTest(variant=variant):
                if lemma in tn.VARIANTS:
                    self.assertEqual(
                        tn.VARIANTS[lemma],
                        lemma,
                        f"{variant} -> {lemma} -> {tn.VARIANTS[lemma]} is a chain",
                    )

    def test_normalization_is_idempotent(self):
        for row in _read_golden("normalizer_golden_v1.tsv"):
            source = _decode(row["input"])
            with self.subTest(source=source):
                once = tn.normalize_text(source)
                self.assertEqual(tn.normalize_text(once), once)


if __name__ == "__main__":
    unittest.main()
