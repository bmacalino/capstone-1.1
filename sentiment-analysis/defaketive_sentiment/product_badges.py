"""Roll per-sentence tags up into product badges and one ordered verdict.

This module replaces the weighted-sum product score. It produces no combined
number at all, deliberately: nothing in the literature says how many defect
reports equal one fire, so any single score has to invent that exchange rate.
Instead a product gets a safety badge and two category flags, each of which can
be read on its own terms.

THE CHAIN
    sentence tagged `c`
      -> review flagged `c`            (any tagged sentence; counted ONCE)
      -> product rate  = flagged / usable reviews
      -> lower bound   = Wilson lower bound of that rate
      -> badge or flag = lower bound against a matrix or a baseline

Counting each review once per category is what removes the double counting in
the old formula, where a keyword-flagged review contributed to both terms.

WHAT IS NOT LEARNED HERE
Severity comes from the frozen table in `hazard_taxonomy`, not from a model, and
the reasons are recorded there. Likelihood bins and the flag baselines come from
a reference corpus built by scoring many products of the same type. Everything
in this module is arithmetic and lookup.

The aggregator never imports a tagger, so the whole product layer, the verdict
and the UI can be built and tested against a stub before any sentence has been
labelled.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import hazard_taxonomy as taxonomy
from .extra_labels import EXTRA_TAGS
from .proportions import wilson_interval, wilson_lower_bound

SCHEMA_VERSION = "defaketive-summary-2.0"

SAFETY = "safety"
DEFECT = "defect"
COUNTERFEIT = "counterfeit"
CATEGORIES = (SAFETY, DEFECT, COUNTERFEIT)
FLAG_CATEGORIES = (COUNTERFEIT, DEFECT)

# Verdict states, in the order they are tested. First match wins.
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
SAFETY_WARNING = "safety_warning"
POSSIBLE_SAFETY_ISSUE = "possible_safety_issue"
POSSIBLE_COUNTERFEIT_OR_DEFECT = "possible_counterfeit_or_defect"
REVIEWS_MAY_BE_UNRELIABLE = "reviews_may_be_unreliable"
NO_WARNING_SIGNS = "no_warning_signs"

# Flag states.
FLAGGED = "flagged"
NOT_FLAGGED = "not_flagged"
NO_BASELINE = "no_baseline"

VERDICT_HEADLINES = {
    INSUFFICIENT_EVIDENCE: "Insufficient evidence",
    SAFETY_WARNING: "Safety warning",
    POSSIBLE_SAFETY_ISSUE: "Caution: possible safety issue",
    POSSIBLE_COUNTERFEIT_OR_DEFECT: "Caution: possible counterfeit or defect",
    REVIEWS_MAY_BE_UNRELIABLE: "Unclear: reviews may be unreliable",
    NO_WARNING_SIGNS: "No warning signs found",
}

# Shown on every result, in every state. The positive state must never read as
# a clearance: absence of complaints is not evidence of safety, and with a few
# dozen reviews the true hazard rate can still be several percent.
LIMITS_NOTICE = (
    "Based on public reviews only. This cannot confirm that a product is "
    "authentic, counterfeit, or safe."
)

PARAMETERS_PATH = Path(__file__).with_name("lexicons") / "hazard_parameters_v1.json"
DEFAULT_BASELINES_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "reference_baselines_v1.json"
)


def load_parameters(path: Path = PARAMETERS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _value(node: Any, default: Any = None) -> Any:
    """Read a parameter that may be a bare value or a {value, basis} block."""
    if isinstance(node, Mapping) and "value" in node:
        return node["value"]
    return node if node is not None else default


@dataclass(frozen=True)
class EvidenceSentence:
    """One tagged sentence, kept so the report can show why a badge appeared."""

    review_key: str
    sentence_index: int
    text: str
    category: str
    score: float = 0.0
    high_confidence: bool = False
    hazard_subtype: str | None = None
    terms: tuple[dict[str, Any], ...] = ()
    rating: int | None = None
    reviewed_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "review_key": self.review_key,
            "sentence_index": self.sentence_index,
            "text": self.text,
            "category": self.category,
            "score": round(float(self.score), 4),
            "high_confidence": self.high_confidence,
            "hazard_subtype": self.hazard_subtype,
            "terms": list(self.terms),
            "rating": self.rating,
            "reviewed_at": self.reviewed_at,
        }


@dataclass
class SafetyBadge:
    tier: str = "Low"
    severity: int = 0
    severity_basis: str | None = None
    hazard_subtypes: dict[str, dict[str, Any]] = field(default_factory=dict)
    flagged_reviews: int = 0
    high_confidence_reviews: int = 0
    independent_high_confidence_reports: int = 0
    rate: float = 0.0
    rate_ci_95: tuple[float, float] = (0.0, 1.0)
    rate_lower_bound: float = 0.0
    likelihood_bin: str = "low"
    likelihood_basis: str = "absolute_fallback"
    alert: bool = False
    alert_reason: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "severity": self.severity,
            "severity_basis": self.severity_basis,
            "hazard_subtypes": self.hazard_subtypes,
            "flagged_reviews": self.flagged_reviews,
            "high_confidence_reviews": self.high_confidence_reviews,
            "independent_high_confidence_reports": self.independent_high_confidence_reports,
            "rate": round(self.rate, 6),
            "rate_ci_95": [round(self.rate_ci_95[0], 6), round(self.rate_ci_95[1], 6)],
            "rate_lower_bound": round(self.rate_lower_bound, 6),
            "likelihood_bin": self.likelihood_bin,
            "likelihood_basis": self.likelihood_basis,
            "alert": self.alert,
            "alert_reason": self.alert_reason,
            "evidence": self.evidence,
        }


@dataclass
class CategoryFlag:
    category: str
    state: str = NOT_FLAGGED
    flagged_reviews: int = 0
    rate: float = 0.0
    rate_ci_95: tuple[float, float] = (0.0, 1.0)
    rate_lower_bound: float = 0.0
    baseline_rate: float | None = None
    absolute_floor: float | None = None
    baseline_source: str | None = None
    n_min_met: bool = False
    exploratory: bool = False
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "flagged_reviews": self.flagged_reviews,
            "rate": round(self.rate, 6),
            "rate_ci_95": [round(self.rate_ci_95[0], 6), round(self.rate_ci_95[1], 6)],
            "rate_lower_bound": round(self.rate_lower_bound, 6),
            "baseline_rate": self.baseline_rate,
            "absolute_floor": self.absolute_floor,
            "baseline_source": self.baseline_source,
            "n_min_met": self.n_min_met,
            "exploratory": self.exploratory,
            "evidence": self.evidence,
        }


def _char_ngrams(text: str, size: int = 4) -> Counter:
    cleaned = re.sub(r"\s+", " ", (text or "").casefold()).strip()
    if len(cleaned) < size:
        return Counter([cleaned] if cleaned else [])
    return Counter(cleaned[index : index + size] for index in range(len(cleaned) - size + 1))


def cosine_similarity(first: str, second: str) -> float:
    """Character 4-gram cosine, used only to decide report independence.

    Hand-rolled rather than pulled from scikit-learn so the aggregator stays
    importable without a fitted vectorizer, and so this module can be tested
    before any model exists.
    """
    left, right = _char_ngrams(first), _char_ngrams(second)
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[gram] * right[gram] for gram in shared)
    if not numerator:
        return 0.0
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    return numerator / (left_norm * right_norm)


def count_independent_reports(
    sentences: Sequence[EvidenceSentence], similarity_threshold: float
) -> int:
    """Count reports that are not near-duplicates of an earlier one.

    Two reviewers describing the same fault in their own words are two pieces of
    evidence. The same text posted twice is one. Greedy: each report is kept
    unless it closely matches one already kept.
    """
    kept: list[EvidenceSentence] = []
    seen_reviews: set[str] = set()
    for sentence in sentences:
        if sentence.review_key in seen_reviews:
            continue
        if any(
            cosine_similarity(sentence.text, other.text) >= similarity_threshold
            for other in kept
        ):
            continue
        kept.append(sentence)
        seen_reviews.add(sentence.review_key)
    return len(kept)


class ProductBadgeAggregator:
    """Turn per-review tagger output into badges, flags and a verdict."""

    def __init__(
        self,
        parameters: Mapping[str, Any] | None = None,
        baselines: Mapping[str, Any] | None = None,
        baselines_path: Path | None = None,
    ) -> None:
        self.parameters = dict(parameters or load_parameters())
        if baselines is not None:
            self.baselines = dict(baselines)
        else:
            path = baselines_path or DEFAULT_BASELINES_PATH
            # A missing reference corpus is an expected state, not an error. It
            # means every flag reports "no_baseline" until one is built.
            self.baselines = (
                json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            )

        alert = self.parameters.get("alert", {})
        self.n_min = int(_value(self.parameters.get("n_min"), 30))
        self.alert_threshold = int(
            _value(alert.get("independent_high_confidence_reports"), 2)
        )
        self.alert_minimum_severity = int(_value(alert.get("minimum_severity"), 4))
        self.independence_threshold = float(
            _value(alert.get("independence_cosine_threshold"), 0.9)
        )
        self.alert_overrides_insufficient = bool(
            _value(alert.get("overrides_insufficient_evidence"), True)
        )
        self.review_quality_enabled = bool(
            _value((self.parameters.get("review_quality") or {}).get("enabled"), False)
        )

    # -- inputs -----------------------------------------------------------

    @staticmethod
    def review_key(analysis: Mapping[str, Any], position: int = 0) -> str:
        return str(
            analysis.get("review_key") or analysis.get("review_id") or f"review-{position}"
        )

    @staticmethod
    def evidence_from_analysis(
        analysis: Mapping[str, Any], position: int = 0
    ) -> list[EvidenceSentence]:
        """Read tagged sentences out of one review's analysis dict.

        `position` supplies the review's identity when the analysis carries no
        key of its own. That fallback is load-bearing: rates are counted over
        DISTINCT review keys, so without it every review would share one empty
        key and a product with fifty hazard reports would count as one.
        """
        review_key = ProductBadgeAggregator.review_key(analysis, position)
        rating = analysis.get("rating")
        reviewed_at = analysis.get("reviewed_at")
        found: list[EvidenceSentence] = []
        for sentence in analysis.get("sentences") or []:
            tags = sentence.get("tags") or []
            confidence = sentence.get("high_confidence") or {}
            for category in tags:
                if category not in CATEGORIES:
                    continue
                found.append(
                    EvidenceSentence(
                        review_key=review_key,
                        sentence_index=int(sentence.get("index", 0)),
                        text=str(sentence.get("text_raw") or sentence.get("text") or ""),
                        category=category,
                        score=float((sentence.get("scores") or {}).get(category, 0.0)),
                        high_confidence=bool(confidence.get(category, False)),
                        hazard_subtype=sentence.get("hazard_subtype"),
                        terms=tuple(sentence.get("terms") or ()),
                        rating=rating,
                        reviewed_at=reviewed_at,
                    )
                )
        return found

    def _extra_reports(
        self,
        records: Sequence[Mapping[str, Any]],
        usable: int,
        weight_by_key: Mapping[str, float] | None,
    ) -> dict[str, Any]:
        """Counts and quotes for "poor quality" and "not as advertised".

        These two labels inform the shopper. They are returned beside `flags`,
        never inside it, so no verdict rule can read them. Empty when the
        checker did not run the five-label rules, so that a label which was
        not checked is never shown as "0 reviews".
        """
        if not any("review_extra_tags" in record for record in records):
            return {}
        found: list[EvidenceSentence] = []
        for position, record in enumerate(records):
            review_key = self.review_key(record, position)
            for sentence in record.get("sentences") or []:
                for category in sentence.get("extra_tags") or []:
                    if category not in EXTRA_TAGS:
                        continue
                    found.append(
                        EvidenceSentence(
                            review_key=review_key,
                            sentence_index=int(sentence.get("index", 0)),
                            text=str(sentence.get("text_raw") or sentence.get("text") or ""),
                            category=category,
                            terms=tuple(
                                term for term in sentence.get("extra_terms") or () if term.get("category") == category
                            ),
                            rating=record.get("rating"),
                            reviewed_at=record.get("reviewed_at"),
                        )
                    )
        reports: dict[str, Any] = {}
        for category in EXTRA_TAGS:
            matching = [item for item in found if item.category == category]
            flagged_reviews = {item.review_key for item in matching}
            rate, interval, _lower = self._rate(flagged_reviews, usable, weight_by_key)
            reports[category] = {
                "flagged_reviews": len(flagged_reviews),
                "rate": round(rate, 6),
                "rate_ci_95": [round(interval[0], 6), round(interval[1], 6)],
                "evidence": [item.as_dict() for item in matching],
            }
        return reports

    # -- rollup -----------------------------------------------------------

    def _bounds(self, flagged: int, total: int) -> tuple[float, tuple[float, float], float]:
        rate = (flagged / total) if total else 0.0
        interval = wilson_interval(flagged, total)
        return rate, interval, wilson_lower_bound(flagged, total)

    @staticmethod
    def effective_sample_size(weights: Iterable[float]) -> float:
        """Kish's effective sample size, (sum w)^2 / sum w^2.

        Unequal weights carry less information than the same number of equally
        weighted reviews; this is the equally weighted sample worth as much.
        """
        values = [float(weight) for weight in weights]
        squares = sum(value * value for value in values)
        return (sum(values) ** 2 / squares) if squares else 0.0

    def _rate(
        self,
        flagged_keys: set[str],
        usable: int,
        weight_by_key: Mapping[str, float] | None,
    ) -> tuple[float, tuple[float, float], float]:
        """Share of reviews flagged, with its Wilson interval.

        Unweighted, this is exactly `_bounds`. Weighted (a per-star sample), the
        share is weighted back to the listing's star mix and the interval uses
        the effective sample size in place of the review count.
        """
        if not weight_by_key:
            return self._bounds(len(flagged_keys), usable)
        total = sum(weight_by_key.values())
        if total <= 0:
            return 0.0, (0.0, 1.0), 0.0
        share = sum(weight_by_key.get(key, 0.0) for key in flagged_keys) / total
        n_eff = self.effective_sample_size(weight_by_key.values())
        interval = wilson_interval(min(n_eff, share * n_eff), n_eff)
        return share, interval, interval[0]

    def _type_baselines(self, product_type: str | None) -> dict[str, Any]:
        if not product_type:
            return {}
        return (self.baselines.get("product_types") or {}).get(product_type, {}) or {}

    def _likelihood_bin(
        self, lower_bound: float, product_type: str | None
    ) -> tuple[str, str]:
        """Place a rate in a likelihood bin, saying which definition was used."""
        bins = (self._type_baselines(product_type).get(SAFETY) or {}).get(
            "likelihood_bins"
        )
        basis = "percentile_of_product_type"
        if not bins:
            fallback = (
                self.parameters.get("reference_baselines", {}).get(
                    "fallback_likelihood_bins", {}
                )
                or {}
            )
            bins = fallback.get(SAFETY)
            basis = "absolute_fallback"
        if not bins:
            return "low", "unavailable"
        for name in ("high", "medium", "low"):
            edges = bins.get(name)
            if edges and lower_bound >= float(edges[0]):
                return name, basis
        return "low", basis

    def _safety_badge(
        self,
        evidence: Sequence[EvidenceSentence],
        usable: int,
        product_type: str | None,
        weight_by_key: Mapping[str, float] | None = None,
    ) -> SafetyBadge:
        hazard = [item for item in evidence if item.category == SAFETY]
        flagged_reviews = {item.review_key for item in hazard}
        confident = [item for item in hazard if item.high_confidence]
        confident_reviews = {item.review_key for item in confident}

        badge = SafetyBadge(flagged_reviews=len(flagged_reviews))
        badge.high_confidence_reviews = len(confident_reviews)
        badge.rate, badge.rate_ci_95, badge.rate_lower_bound = self._rate(
            flagged_reviews, usable, weight_by_key
        )
        badge.likelihood_bin, badge.likelihood_basis = self._likelihood_bin(
            badge.rate_lower_bound, product_type
        )

        # Severity is the highest level among HIGH-CONFIDENCE reports only. A
        # low-confidence match must not be able to raise a product to severity 4.
        subtypes: dict[str, dict[str, Any]] = {}
        for item in hazard:
            name = item.hazard_subtype or "safety_unspecified"
            entry = subtypes.setdefault(
                name,
                {
                    "reviews": 0,
                    "high_confidence_reviews": 0,
                    "severity": taxonomy.SEVERITY.get(name, 0),
                },
            )
            entry["reviews"] += 1
            if item.high_confidence:
                entry["high_confidence_reviews"] += 1

        severities = [
            taxonomy.severity_for(item.hazard_subtype or "safety_unspecified")
            for item in confident
            if (item.hazard_subtype or "safety_unspecified") in taxonomy.SEVERITY
        ]
        badge.severity = max(severities) if severities else 0
        badge.hazard_subtypes = subtypes
        if badge.severity:
            worst = max(
                (item for item in confident),
                key=lambda item: taxonomy.severity_for(
                    item.hazard_subtype or "safety_unspecified"
                ),
            )
            badge.severity_basis = taxonomy.basis_for(
                worst.hazard_subtype or "safety_unspecified"
            )

        badge.tier = (
            taxonomy.tier_for(badge.severity, badge.likelihood_bin)
            if badge.severity
            else "Low"
        )

        # Alert override: enough independent high-confidence reports of a
        # severity-4 hazard lift the tier regardless of how low the rate is.
        severe = [
            item
            for item in confident
            if taxonomy.SEVERITY.get(item.hazard_subtype or "safety_unspecified", 0)
            >= self.alert_minimum_severity
        ]
        badge.independent_high_confidence_reports = count_independent_reports(
            severe, self.independence_threshold
        )
        if badge.independent_high_confidence_reports >= self.alert_threshold:
            badge.alert = True
            badge.tier = taxonomy.higher_tier(badge.tier, "High")
            badge.alert_reason = (
                f"{badge.independent_high_confidence_reports} independent "
                f"high-confidence reports of a severity-"
                f"{self.alert_minimum_severity} hazard"
            )

        badge.evidence = [item.as_dict() for item in hazard]
        return badge

    def _category_flag(
        self,
        category: str,
        evidence: Sequence[EvidenceSentence],
        usable: int,
        product_type: str | None,
        weight_by_key: Mapping[str, float] | None = None,
    ) -> CategoryFlag:
        matching = [item for item in evidence if item.category == category]
        flagged_reviews = {item.review_key for item in matching}

        flag = CategoryFlag(category=category, flagged_reviews=len(flagged_reviews))
        flag.rate, flag.rate_ci_95, flag.rate_lower_bound = self._rate(
            flagged_reviews, usable, weight_by_key
        )
        flag.n_min_met = usable >= self.n_min
        flag.evidence = [item.as_dict() for item in matching]

        floors = (self.parameters.get("flags") or {}).get("absolute_floor") or {}
        flag.absolute_floor = floors.get(category)

        baseline = (self._type_baselines(product_type).get(category) or {}).get(
            "pooled_rate"
        )
        if baseline is None:
            # No peer rate to compare against. Report the observed rate and say
            # so, rather than guessing or silently reporting "not flagged".
            flag.state = NO_BASELINE
            return flag
        flag.baseline_rate = float(baseline)
        flag.baseline_source = (
            f"{self.baselines.get('version', 'reference_baselines')}/{product_type}"
        )

        if not flag.n_min_met:
            flag.state = INSUFFICIENT_EVIDENCE
            return flag

        threshold = max(flag.baseline_rate, float(flag.absolute_floor or 0.0))
        flag.state = FLAGGED if flag.rate_lower_bound > threshold else NOT_FLAGGED
        return flag

    # -- verdict ----------------------------------------------------------

    @staticmethod
    def _warns(flag: CategoryFlag) -> bool:
        """Whether a counterfeit or defect flag warrants a warning verdict.

        With a product-type baseline, the flag's own state decides. Without
        one, the flag stays NO_BASELINE -- it genuinely has no peer rate -- but
        the absolute floor still applies on its own. Treating "no baseline" as
        "nothing to report" made every product read "No warning signs found"
        however many reviews called it fake, because no reference corpus
        exists yet. The floor is the half of max(baseline, floor) that needs no
        peers, so it is the rule the full comparison would also enforce.
        """
        if flag.state == FLAGGED:
            return True
        return (
            flag.state == NO_BASELINE
            and flag.n_min_met
            and flag.absolute_floor is not None
            and flag.rate_lower_bound > float(flag.absolute_floor)
        )

    @staticmethod
    def _warning_reason(name: str, flag: CategoryFlag) -> str:
        if flag.state == FLAGGED:
            return f"{name} reports are above the typical rate for this product type"
        return (
            f"{name} reports exceed the {float(flag.absolute_floor):.0%} floor "
            "(no product-type baseline yet)"
        )

    def verdict_for(
        self,
        safety: SafetyBadge,
        flags: Mapping[str, CategoryFlag],
        *,
        usable: int,
        scrape_complete: bool = True,
        reviews_unreliable: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the verdict rules in order and return the first match."""
        enough = usable >= self.n_min and scrape_complete

        # The alert is count-based, so it does not need a denominator: two
        # independent fire reports on a twelve-review listing is strong
        # evidence, not weak evidence. Suppressing it behind the sample-size
        # rule would be a false negative in the most severe category.
        alert_wins = safety.alert and self.alert_overrides_insufficient

        rules = [
            (
                INSUFFICIENT_EVIDENCE,
                not enough and not alert_wins,
                (
                    f"Only {usable} usable reviews; at least {self.n_min} are needed"
                    if usable < self.n_min
                    else "The review collection for this product is incomplete"
                ),
            ),
            (
                SAFETY_WARNING,
                safety.alert or safety.tier in {"High", "Serious"},
                safety.alert_reason
                or f"Safety tier {safety.tier} from {safety.flagged_reviews} reports",
            ),
            (
                POSSIBLE_SAFETY_ISSUE,
                safety.tier == "Moderate",
                f"Safety tier Moderate from {safety.flagged_reviews} reports",
            ),
            (
                POSSIBLE_COUNTERFEIT_OR_DEFECT,
                any(self._warns(flag) for flag in flags.values()),
                ", ".join(
                    self._warning_reason(name, flag)
                    for name, flag in flags.items()
                    if self._warns(flag)
                ),
            ),
            (
                REVIEWS_MAY_BE_UNRELIABLE,
                self.review_quality_enabled and reviews_unreliable,
                "Many reviews look duplicated or generic",
            ),
            (
                NO_WARNING_SIGNS,
                True,
                f"No warning signs found in {usable} reviews. "
                "Absence of complaints is not proof that a product is safe.",
            ),
        ]

        for index, (state, matched, detail) in enumerate(rules):
            if matched:
                return {
                    "state": state,
                    "headline": VERDICT_HEADLINES[state],
                    "detail": detail,
                    "rule_index": index,
                }
        raise AssertionError("verdict rules must end in an unconditional match")

    # -- entry point ------------------------------------------------------

    def summarize(
        self,
        analyses: Iterable[Mapping[str, Any]],
        *,
        reviews_available: int | None = None,
        product_type: str | None = None,
        coverage: Mapping[str, Any] | None = None,
        reviews_unreliable: bool = False,
    ) -> dict[str, Any]:
        records = [record for record in analyses if isinstance(record, Mapping)]
        usable = len(records)

        evidence: list[EvidenceSentence] = []
        sentence_count = 0
        for position, record in enumerate(records):
            sentence_count += len(record.get("sentences") or [])
            evidence.extend(self.evidence_from_analysis(record, position))

        # A per-star sample carries a weight per review (analyze_reviews sets
        # it); a census carries none, and every rate below is then unweighted.
        weighted = any("sampling_weight" in record for record in records)
        weight_by_key = (
            {
                self.review_key(record, position): float(record.get("sampling_weight") or 0.0)
                for position, record in enumerate(records)
            }
            if weighted
            else None
        )

        safety = self._safety_badge(evidence, usable, product_type, weight_by_key)
        flags = {
            category: self._category_flag(category, evidence, usable, product_type, weight_by_key)
            for category in FLAG_CATEGORIES
        }

        reports = self._extra_reports(records, usable, weight_by_key)

        coverage_block = dict(coverage or {})
        coverage_block.setdefault("reviews_analyzed", usable)
        coverage_block.setdefault("usable_reviews", usable)
        coverage_block.setdefault("reviews_available", reviews_available)
        coverage_block.setdefault("sentences_analyzed", sentence_count)
        coverage_block.setdefault("scrape_completeness", "complete")
        coverage_block.setdefault("sampling", "census_of_scraped_reviews")

        verdict = self.verdict_for(
            safety,
            flags,
            usable=usable,
            scrape_complete=coverage_block.get("scrape_completeness") != "partial",
            reviews_unreliable=reviews_unreliable,
        )

        labels = [
            str(record.get("sentiment", {}).get("label") or record.get("label") or "")
            for record in records
        ]
        sentiment = Counter(labels)
        positive = sentiment.get("positive", 0)
        positive_lower_bound = wilson_lower_bound(positive, usable)
        weighting: dict[str, Any] = {}
        if weight_by_key:
            # Counts stay counts of reviews read; the shares and the bound used
            # for recommendations describe the listing's real star mix.
            weights = [float(record.get("sampling_weight") or 0.0) for record in records]
            total = sum(weights)
            n_eff = self.effective_sample_size(weights)
            shares = Counter()
            for label, weight in zip(labels, weights):
                shares["uncertain" if label == "neutral" else label] += weight
            positive_share = shares.get("positive", 0.0) / total if total else 0.0
            positive_lower_bound = (
                wilson_interval(min(n_eff, positive_share * n_eff), n_eff)[0] if n_eff else 0.0
            )
            weighting = {
                "weighted": True,
                "effective_sample_size": round(n_eff, 1),
                "sentiment_shares": {
                    name: round(shares.get(name, 0.0) / total, 6) if total else 0.0
                    for name in ("positive", "uncertain", "negative")
                },
            }

        return {
            "schema_version": SCHEMA_VERSION,
            "taxonomy_version": taxonomy.TAXONOMY_VERSION,
            "parameters_version": self.parameters.get("version"),
            "reference_baselines_version": self.baselines.get("version"),
            "verdict": verdict,
            "coverage": coverage_block,
            "safety": safety.as_dict(),
            "flags": {name: flag.as_dict() for name, flag in flags.items()},
            **({"reports": reports} if reports else {}),
            "review_quality": {
                "enabled": self.review_quality_enabled,
                "reviews_unreliable": reviews_unreliable,
            },
            "sentiment_counts": {
                "positive": positive,
                "uncertain": sentiment.get("uncertain", 0) + sentiment.get("neutral", 0),
                "negative": sentiment.get("negative", 0),
            },
            "positive_ratio_lower_bound": round(positive_lower_bound, 6),
            **weighting,
            "limits_notice": LIMITS_NOTICE,
        }
