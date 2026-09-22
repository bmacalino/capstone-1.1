import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sentiment-analysis"))

from defaketive_sentiment.product_risk_context import clean_risk_text, context_signals

# Inputs chosen to exercise every replacement rule in clean_risk_text, plus the
# cases where it is known to disagree with the runtime normalizer.
FROZEN_FIXTURE = [
    "Maganda pero mabilis masira.",
    "HNDI gumagana",
    "d gumagana ang charger",
    "poor-quality item",
    "It doesnt work",
    "wont charge",
    "deffective ang unit",
    "malobat agad",
    "feature_quality maganda",
    "Walang sira at gumagana",
    "nag-init agad ang powerbank",
    "20.000mah pb",
    "l am disappointed",
    "iba’t ibang klase",
    "emoji \U0001f621 test",
    "sumabog ang adaptor",
    "no physical damage",
    "hindi original",
    "",
    "   ",
]
FROZEN_OUTPUT_SHA256 = "f38dddfe9301d745cfa6ef71a05a67aaadfa6aaea1550652fce67366933bfb9d"


class ContextTests(unittest.TestCase):
    def test_negated_damage_does_not_become_damage(self):
        for text in ["No damage, working well", "walang sira", "not defective"]:
            self.assertEqual(context_signals(text)["physical_problem"], 0, text)

    def test_negated_working_means_failure(self):
        for text in ["doesn't work", "hnd gumagana", "di nagana yung right", "couldn't pair it"]:
            self.assertEqual(context_signals(text)["functional_failure"], 1, text)

    def test_delivery_problem_does_not_create_product_failure(self):
        signals = context_signals("Late delivery but the item is working well")
        self.assertEqual(signals["concrete_product_problem"], 0)
        self.assertEqual(signals["order_issue"], 1)

    def test_packaging_damage_is_separate(self):
        signals = context_signals("The box was damaged but the product works fine")
        self.assertEqual(signals["packaging_damage"], 1)
        self.assertEqual(signals["physical_problem"], 0)

    def test_praise_does_not_erase_explicit_failure(self):
        signals = context_signals("Good quality but the switch broke after a day")
        self.assertEqual(signals["physical_problem"], 1)
        self.assertEqual(signals["complaint_with_praise"], 1)

    def test_no_training_feature_markers_leak_into_clean_text(self):
        self.assertEqual(clean_risk_text("Works feature_functionality feature_quality"), "works")


class FrozenCleanerTests(unittest.TestCase):
    """clean_risk_text is frozen at v1 and must not change.

    It is referenced by module path inside pickled RiskTextCleaner steps in the
    committed joblib artifacts, so an edit would retroactively change models
    that have already been fitted, evaluated and reported - with no version
    number moving to signal it. Every metric in sentiment-analysis/evaluation/
    assumes the current behaviour.

    If this test fails, the correct response is almost never to update the
    hash. It is to revert the edit and put the new behaviour in
    text_norm.normalize_text instead, which is versioned.
    """

    def test_output_hash_is_unchanged(self):
        produced = [clean_risk_text(value) for value in FROZEN_FIXTURE]
        digest = hashlib.sha256(
            json.dumps(produced, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            digest,
            FROZEN_OUTPUT_SHA256,
            "clean_risk_text changed; this invalidates every committed artifact "
            "that pickles it. Put new normalization in text_norm.py instead.",
        )

    def test_known_disagreement_with_the_runtime_normalizer_is_preserved(self):
        # Documented, deliberate difference: this cleaner maps a bare "d" to
        # "hindi" and the runtime normalizer does not. Recording it as a test
        # keeps it a known quantity rather than a latent surprise.
        from defaketive_sentiment.text_norm import normalize_text

        self.assertIn("hindi", clean_risk_text("d gumagana"))
        self.assertNotIn("hindi", normalize_text("d gumagana"))


if __name__ == "__main__":
    unittest.main()
