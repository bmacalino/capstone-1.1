import unittest

from defaketive_sentiment.proportions import (
    Z_95,
    wilson_interval,
    wilson_lower_bound,
    wilson_upper_bound,
)


class TestWilsonInterval(unittest.TestCase):
    """Wilson bounds checked against published values and the design worked examples."""

    def test_quantile_matches_published_value(self):
        self.assertAlmostEqual(Z_95, 1.959963984540054, places=12)

    def test_matches_published_intervals(self):
        # Standard reference values for the 95% Wilson score interval.
        for successes, total, expected in [
            (0, 10, (0.0, 0.2775)),
            (1, 5, (0.0362, 0.6245)),
            (5, 5, (0.5655, 1.0)),
        ]:
            with self.subTest(successes=successes, total=total):
                low, high = wilson_interval(successes, total)
                self.assertAlmostEqual(low, expected[0], places=4)
                self.assertAlmostEqual(high, expected[1], places=4)

    def test_reproduces_design_document_worked_examples(self):
        # These three cases are quoted in defaketive_risk_scoring_handoff.md.
        # Keeping them as tests ties the implementation to the written design.
        self.assertAlmostEqual(wilson_lower_bound(2, 400), 0.0014, places=4)
        self.assertAlmostEqual(wilson_lower_bound(10, 400), 0.0136, places=4)
        self.assertAlmostEqual(wilson_lower_bound(1, 1), 0.2065, places=4)

    def test_single_complaint_does_not_flag_a_small_sample(self):
        # The whole point of using a lower bound: 1 of 5 is a 20% rate but only
        # a 3.6% lower bound, which sits below any plausible product baseline.
        self.assertAlmostEqual(1 / 5, 0.20)
        self.assertLess(wilson_lower_bound(1, 5), 0.05)

    def test_absence_of_complaints_is_not_proof_of_safety(self):
        # Why the positive verdict never says "safe": with no hazard reports in
        # 40 reviews the true rate could still be nearly 9%.
        self.assertEqual(wilson_lower_bound(0, 40), 0.0)
        self.assertGreater(wilson_upper_bound(0, 40), 0.08)

    def test_no_observations_returns_the_whole_range(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)

    def test_bounds_stay_inside_the_unit_range(self):
        for total in (1, 2, 5, 30, 400):
            for successes in range(total + 1):
                with self.subTest(successes=successes, total=total):
                    low, high = wilson_interval(successes, total)
                    self.assertGreaterEqual(low, 0.0)
                    self.assertLessEqual(high, 1.0)
                    self.assertLessEqual(low, high)

    def test_zero_successes_has_a_zero_lower_bound(self):
        for total in (1, 10, 400):
            self.assertEqual(wilson_lower_bound(0, total), 0.0)

    def test_all_successes_has_an_upper_bound_of_one(self):
        for total in (1, 10, 400):
            self.assertEqual(wilson_upper_bound(total, total), 1.0)

    def test_lower_bound_rises_with_more_evidence_at_a_fixed_rate(self):
        # 1/10, 10/100 and 100/1000 are all a 10% rate, but more reviews mean
        # more confidence that the rate is real. This monotonicity is what makes
        # the bound usable as a flag threshold.
        bounds = [wilson_lower_bound(n // 10, n) for n in (10, 100, 1000, 10000)]
        self.assertEqual(bounds, sorted(bounds))
        for bound in bounds:
            self.assertLess(bound, 0.10)

    def test_interval_narrows_as_evidence_grows(self):
        widths = []
        for total in (10, 100, 1000, 10000):
            low, high = wilson_interval(total // 10, total)
            widths.append(high - low)
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_lower_bound_rises_with_more_successes_at_a_fixed_total(self):
        bounds = [wilson_lower_bound(successes, 100) for successes in range(0, 101, 10)]
        self.assertEqual(bounds, sorted(bounds))

    def test_rejects_impossible_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(5, 4)
        with self.assertRaises(ValueError):
            wilson_interval(-1, 10)


if __name__ == "__main__":
    unittest.main()
