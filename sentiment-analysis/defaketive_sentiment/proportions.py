"""Confidence intervals for proportions.

Product badges are decided from review rates such as "3 of 412 reviews report a
hazard". A raw rate is unusable for that decision because it says nothing about
how much evidence supports it: one hazard report in two reviews is 50%, and one
in four hundred is 0.25%, yet the first is almost no evidence at all.

The Wilson score interval is used rather than the textbook normal approximation
because marketplace rates are small proportions over counts that are sometimes
tiny, which is exactly where the normal approximation fails - it can return a
negative lower bound, and its coverage collapses near 0 and 1. Wilson stays
inside [0, 1] by construction and keeps its nominal coverage for small n and
extreme p (Brown, Cai & DasGupta 2001, "Interval Estimation for a Binomial
Proportion", Statistical Science 16(2)).

Implemented in pure `math` on purpose. The closed form is three lines, and the
package deliberately declares only `nltk` as a dependency (see pyproject.toml)
so it degrades to the rule model on an incomplete install; adding scipy or
statsmodels for one formula would break that.

What the interval does and does not mean is stated in the methods: reviewers are
self-selected, so this describes sampling uncertainty over *reviews written*,
not over all buyers.
"""

from __future__ import annotations

import math

# Two-sided 95% normal quantile. Hard-coded rather than imported from scipy;
# tested against the published value in test_proportions.py.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for `successes` out of `total`.

    With no observations the interval is the whole unit range, because zero
    reviews support no conclusion at all. Callers must still apply their own
    minimum-sample rule; a wide interval is not the same as "insufficient
    evidence", and the two are reported separately.
    """
    if total <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be within 0..{total}, got {successes}")

    proportion = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (proportion + (z * z) / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + (z * z) / (4.0 * total * total)
    )
    half_width = spread / denominator
    low = max(0.0, centre - half_width)
    high = min(1.0, centre + half_width)

    # With no successes the lower bound is exactly zero, and with no failures
    # the upper bound is exactly one. Floating-point arithmetic leaves a residue
    # of about 1e-19 in both cases, which would otherwise surface as a nonzero
    # hazard rate for a product that has never been reported.
    if successes == 0:
        low = 0.0
    if successes == total:
        high = 1.0
    return (low, high)


def wilson_lower_bound(successes: int, total: int, z: float = Z_95) -> float:
    """Return only the lower bound of the Wilson interval.

    This is the value every flag decision uses. Taking the lower bound rather
    than the observed rate is what stops a single complaint in five reviews from
    flagging a product: the rate is 20%, but the lower bound is about 3.6%.
    """
    return wilson_interval(successes, total, z)[0]


def wilson_upper_bound(successes: int, total: int, z: float = Z_95) -> float:
    """Return only the upper bound of the Wilson interval.

    Used when reporting the interval in the UI, and for the "absence of evidence"
    caveat: with zero hazard reports in 40 reviews the upper bound is still about
    9%, which is why the positive verdict never says a product is safe.
    """
    return wilson_interval(successes, total, z)[1]
