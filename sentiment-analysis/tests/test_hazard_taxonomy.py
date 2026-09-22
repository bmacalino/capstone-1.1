import unittest

from defaketive_sentiment import hazard_taxonomy as taxonomy


class TestSeverityTable(unittest.TestCase):
    def test_every_subtype_has_a_level_a_description_and_a_basis(self):
        for subtype in taxonomy.SEVERITY:
            with self.subTest(subtype=subtype):
                level = taxonomy.severity_for(subtype)
                self.assertIn(level, taxonomy.SEVERITY_LEVELS)
                self.assertIn(level, taxonomy.SEVERITY_DESCRIPTIONS)
                self.assertTrue(taxonomy.basis_for(subtype).strip())

    def test_every_basis_declares_its_provenance(self):
        # The handoff's standing rule: designed values must never read as
        # published ones. Every basis string must carry an explicit label.
        labels = (taxonomy.PUBLISHED, taxonomy.DERIVED, taxonomy.PANEL)
        for subtype, basis in taxonomy.SEVERITY_BASIS.items():
            with self.subTest(subtype=subtype):
                self.assertTrue(
                    any(f"[{label}]" in basis for label in labels),
                    f"{subtype} severity basis does not declare its provenance",
                )

    def test_shock_severity_is_declared_unsupported_by_publication(self):
        # Zaman et al. excluded electrical hazards for insufficient data, so
        # this value must not claim published support. Guarding it in a test
        # keeps a later edit from quietly upgrading the claim.
        basis = taxonomy.basis_for("shock_spark_short")
        self.assertIn(f"[{taxonomy.PANEL}]", basis)
        self.assertNotIn(f"[{taxonomy.PUBLISHED}]", basis)

    def test_unknown_subtype_is_rejected(self):
        with self.assertRaises(taxonomy.UnknownHazardSubtype):
            taxonomy.severity_for("meteor_strike")
        with self.assertRaises(taxonomy.UnknownHazardSubtype):
            taxonomy.basis_for("meteor_strike")

    def test_unspecified_hazard_cannot_reach_the_alert_path_alone(self):
        # The alert override requires severity 4. A safety complaint whose type
        # could not be determined must not be able to trigger it.
        self.assertLess(taxonomy.severity_for("safety_unspecified"), 4)


class TestDecisionMatrix(unittest.TestCase):
    def test_matrix_covers_every_severity_and_likelihood(self):
        for severity in taxonomy.SEVERITY_LEVELS:
            for likelihood in taxonomy.LIKELIHOOD_BINS:
                with self.subTest(severity=severity, likelihood=likelihood):
                    self.assertIn(taxonomy.tier_for(severity, likelihood), taxonomy.TIER_ORDER)

    def test_tier_never_falls_when_severity_rises(self):
        # The invariant a hand edit could silently break.
        for likelihood in taxonomy.LIKELIHOOD_BINS:
            ranks = [
                taxonomy.TIER_ORDER.index(taxonomy.tier_for(severity, likelihood))
                for severity in taxonomy.SEVERITY_LEVELS
            ]
            with self.subTest(likelihood=likelihood):
                self.assertEqual(ranks, sorted(ranks), f"not monotone in severity at {likelihood}")

    def test_tier_never_falls_when_likelihood_rises(self):
        for severity in taxonomy.SEVERITY_LEVELS:
            ranks = [
                taxonomy.TIER_ORDER.index(taxonomy.tier_for(severity, likelihood))
                for likelihood in taxonomy.LIKELIHOOD_BINS
            ]
            with self.subTest(severity=severity):
                self.assertEqual(ranks, sorted(ranks), f"not monotone in likelihood at {severity}")

    def test_a_rare_fatal_hazard_still_warns(self):
        # The design's central claim: two fire reports in four hundred reviews
        # is a low rate but must not produce a low tier.
        self.assertEqual(taxonomy.tier_for(4, "low"), "High")

    def test_a_common_minor_hazard_does_not_reach_the_top_tier(self):
        self.assertEqual(taxonomy.tier_for(1, "high"), "Moderate")

    def test_matrix_basis_is_declared_panel_assigned(self):
        self.assertIn(f"[{taxonomy.PANEL}]", taxonomy.MATRIX_BASIS)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            taxonomy.tier_for(5, "low")
        with self.assertRaises(ValueError):
            taxonomy.tier_for(4, "certain")


class TestTierOrdering(unittest.TestCase):
    def test_higher_tier_picks_the_more_severe_of_two(self):
        self.assertEqual(taxonomy.higher_tier("Low", "Serious"), "Serious")
        self.assertEqual(taxonomy.higher_tier("High", "Moderate"), "High")
        self.assertEqual(taxonomy.higher_tier("Low", "Low"), "Low")

    def test_higher_tier_is_order_independent(self):
        for first in taxonomy.TIER_ORDER:
            for second in taxonomy.TIER_ORDER:
                with self.subTest(first=first, second=second):
                    self.assertEqual(
                        taxonomy.higher_tier(first, second),
                        taxonomy.higher_tier(second, first),
                    )

    def test_higher_tier_rejects_an_unknown_tier(self):
        with self.assertRaises(ValueError):
            taxonomy.higher_tier("Low", "Catastrophic")


if __name__ == "__main__":
    unittest.main()
