"""Runtime model producing safety, defect and counterfeit badges.

The tagger here is LEXICON-BASED: it matches the curated terms in
`lexicons/risk_lexicon.tsv`, which already carry a category and a severity. It
exists for two reasons, and neither of them is "placeholder":

1. It makes the whole product path runnable now. The badge aggregator, the
   verdict, the database columns and the report UI can all be built and
   demonstrated before a single sentence has been hand-labelled, which is what
   keeps the annotation work off the critical path.
2. The evaluation plan requires a lexicon-only baseline to compare the learned
   smoke-term tagger against. This is that baseline, so the work is not thrown
   away when the CC tagger lands.

It will be REPLACED as the production tagger by the CC smoke-term model, which
learns its terms from hand-labelled sentences rather than taking them from a
hand-written list. Until then this model is deliberately marked unqualified, so
`qualified_for_integration` stays false and no result computed from it can be
reported as a validated accuracy figure.

WHAT IT CANNOT DO
A hand-written lexicon only finds the phrasings someone thought to write down.
Its recall on hazards described in unanticipated Taglish is unmeasured, and
measuring it is precisely the job of the labelled recall sample. Do not quote
its numbers as model performance.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from . import hazard_taxonomy as taxonomy
from . import text_norm
from .model import DefaketiveSentimentModel
from .product_badges import (
    CATEGORIES,
    COUNTERFEIT,
    DEFECT,
    SAFETY,
    SCHEMA_VERSION,
    ProductBadgeAggregator,
)

MODEL_VERSION = "defaketive-hazard-lexicon-0.1.0"

# The risk lexicon carries six categories; the badges use three. The mapping
# follows the definitions in research/HAZARD_SENTENCE_TAGGING_GUIDE.md, which is
# what the annotators are labelling against, so the baseline and the hand labels
# mean the same thing by "defect".
#
#   performance ("hindi gumagana", "not charging", "dead on arrival") and
#   durability  ("mabilis masira", "nasira agad") both describe a product that
#   does not work, which is the guide's definition of DEFECT.
#
#   quality ("cheaply made", "flimsy", "poor quality", "palpak") is deliberately
#   DROPPED. Those are opinions about how a product feels, not reports that it
#   failed, and the guide does not treat them as defects. They are the
#   "dissatisfaction" the badge design set aside on purpose: folding them in
#   would inflate every defect rate with ordinary grumbling and put the flag
#   back in the business of measuring taste.
CATEGORY_MAP = {
    "safety": SAFETY,
    "counterfeit": COUNTERFEIT,
    "defect": DEFECT,
    "performance": DEFECT,
    "durability": DEFECT,
}

# Lexicon severity 1 means "minor concern" (gasgas, maluwag, scratched - all
# cosmetic). Those may tag a sentence but must not make it high confidence.
#
# Note this is the LEXICON's 1-3 term severity, which is a different scale from
# the taxonomy's 1-4 hazard severity. The lexicon scale says how serious a
# complaint is; it carries no signal about how certain the match is. Using it as
# a confidence proxy is the weakest part of this baseline, and it is exactly
# what the learned tagger replaces with a tuned per-category cutoff.
MIN_CONFIDENT_SEVERITY = 2

# Which hazard subtype a safety term reports. Terms are compared after hazard
# normalization, so "umiinit" arrives as "init" and "overheating" as "overheat".
SUBTYPE_TERMS = {
    "fire_explosion_swelling": (
        "explode", "sumabog", "on fire", "sunog", "bloat battery",
        "lobo battery", "swell", "liyab", "nagliyab",
    ),
    "shock_spark_short": ("electric shock", "shock", "spark", "grounded", "kuryente", "short"),
    "overheat_burn": ("overheat", "init", "paso", "burn"),
    "smoke_smell": ("smoke", "burning smell", "usok", "amoy sunog"),
}

# A hazard report only counts as high confidence when it is about the PRODUCT.
#
# The rule is stated as an exclusion rather than as a requirement for a product
# noun, and that choice matters: Filipino is pro-drop, so "Nasunog agad" is a
# complete and very common way to report that the item burned, with no subject
# at all. Demanding an explicit product word would discard exactly the terse,
# emphatic reports that matter most. Instead a segment is disqualified when it
# names a different subject - the box, the courier, the seller.
NON_PRODUCT_SUBJECT = re.compile(
    r"(?<!\w)(?:"
    r"box|boxes|kahon|package|packaging|packing|parcel|balot|wrapper|wrapping|"
    r"courier|rider|delivery|shipping|shipment|padala|"
    r"seller|shop|store|tindahan|shopee|lazada|temu"
    r")(?!\w)",
    re.IGNORECASE,
)


class HazardModelError(RuntimeError):
    """Raised when a hazard artifact must not be used for reported results."""


def _load_category_terms() -> dict[str, list[tuple[str, int, re.Pattern[str]]]]:
    """Read the risk lexicon, keeping only the three badge categories."""
    model_lexicon = DefaketiveSentimentModel.__dict__  # noqa: F841 - documents intent
    from .model import LEXICON_DIR, _load_tsv

    terms: dict[str, list[tuple[str, int, re.Pattern[str]]]] = {name: [] for name in CATEGORIES}
    for entry in _load_tsv(LEXICON_DIR / "risk_lexicon.tsv"):
        category = CATEGORY_MAP.get(entry["category"])
        if category is None:
            continue
        normalized = text_norm.normalize_text(entry["term"])
        if not normalized:
            continue
        try:
            severity = int(entry["severity"])
        except (TypeError, ValueError):
            severity = 1
        pattern = re.compile(
            r"(?<!\w)" + r"\s+".join(re.escape(part) for part in normalized.split()) + r"(?!\w)"
        )
        terms[category].append((normalized, severity, pattern))
    # Longest first so a phrase wins over one of its own words.
    for category in terms:
        terms[category].sort(key=lambda item: len(item[0]), reverse=True)
    return terms


def hazard_subtype(matched_terms: Iterable[str]) -> str:
    """Assign a hazard subtype from the safety terms that matched.

    Rule-based, and declared as such: the detector learns one merged SAFETY
    class because no subtype reaches enough positives to be evaluated on its
    own. The most severe matching subtype wins.
    """
    best = "safety_unspecified"
    best_severity = taxonomy.SEVERITY[best]
    for term in matched_terms:
        for subtype, markers in SUBTYPE_TERMS.items():
            if any(marker in term for marker in markers):
                severity = taxonomy.SEVERITY[subtype]
                if severity > best_severity:
                    best, best_severity = subtype, severity
    return best


class LexiconHazardTagger:
    """Tag review sentences with safety, defect and counterfeit categories."""

    def __init__(self) -> None:
        self.terms = _load_category_terms()
        self.normalizer_version = text_norm.NORMALIZER_VERSION
        self.segmenter_version = text_norm.SEGMENTER_VERSION

    def tag_sentence(self, raw: str) -> dict[str, Any]:
        normalized, spans = text_norm.normalize_with_spans(raw)
        scores: dict[str, float] = {}
        tags: list[str] = []
        confidence: dict[str, bool] = {}
        matches: list[dict[str, Any]] = []
        severities: dict[str, int] = {}
        safety_terms: list[str] = []

        about_the_product = not NON_PRODUCT_SUBJECT.search(normalized)
        for category, entries in self.terms.items():
            occupied: list[tuple[int, int]] = []
            total = 0.0
            severity = 0
            for term, term_severity, pattern in entries:
                for match in pattern.finditer(normalized):
                    if any(
                        match.start() < end and start < match.end()
                        for start, end in occupied
                    ):
                        continue
                    occupied.append(match.span())
                    negated = not text_norm.starts_with_negator(term) and text_norm.is_negated(
                        normalized, match.start()
                    )
                    raw_span = self._raw_span(match.span(), spans)
                    matches.append(
                        {
                            "ngram": term,
                            "category": category,
                            "severity": term_severity,
                            "negated": negated,
                            "span": list(raw_span) if raw_span else None,
                        }
                    )
                    if negated:
                        continue
                    total += float(term_severity)
                    severity = max(severity, term_severity)
                    if category == SAFETY:
                        safety_terms.append(term)
            if total > 0:
                scores[category] = total
                severities[category] = severity
                tags.append(category)
                # High confidence needs an unnegated term above the cosmetic
                # floor, in a segment that is not about the packaging, courier
                # or seller. This gates the alert path, so it is the strictest
                # rule in the tagger.
                confidence[category] = (
                    severity >= MIN_CONFIDENT_SEVERITY and about_the_product
                )

        return {
            "text_raw": raw,
            "text_normalized": normalized,
            "scores": scores,
            "tags": tags,
            "high_confidence": confidence,
            "hazard_subtype": hazard_subtype(safety_terms) if SAFETY in tags else None,
            "terms": matches,
        }

    @staticmethod
    def _raw_span(
        span: tuple[int, int], spans: Sequence[text_norm.TokenSpan]
    ) -> tuple[int, int] | None:
        """Map a match in normalized text back to the original characters."""
        covering = [
            item for item in spans if item.norm_start < span[1] and span[0] < item.norm_end
        ]
        if not covering:
            return None
        return (min(item.raw_start for item in covering), max(item.raw_end for item in covering))

    def tag_review(self, text: Any) -> list[dict[str, Any]]:
        tagged = []
        for index, sentence in enumerate(text_norm.segment_sentences(text)):
            result = self.tag_sentence(sentence)
            result["index"] = index
            tagged.append(result)
        return tagged


class HazardBadgeModel:
    """Production-shaped model: per-review tagging plus product badges.

    Duck-types the contract `analyze_reviews.analyze_product_json` relies on -
    `fingerprint`, `analyze`, `analyze_batch` and `summarize` - so swapping the
    lexicon tagger for the learned one later changes nothing outside this file.
    """

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        *,
        tagger: LexiconHazardTagger | None = None,
        aggregator: ProductBadgeAggregator | None = None,
        sentiment_model: Any | None = None,
        allow_unqualified: bool = False,
    ) -> None:
        # The lexicon tagger has not passed any evaluation gate, because there
        # is nothing to evaluate it against yet. Refusing by default keeps its
        # output from being reported as validated performance.
        self.qualified_for_integration = False
        if not allow_unqualified:
            raise HazardModelError(
                "The lexicon hazard tagger is a development baseline and is not "
                "qualified for integration. Pass allow_unqualified=True to use it "
                "knowingly; its output must not be reported as model accuracy."
            )
        self.tagger = tagger or LexiconHazardTagger()
        self.aggregator = aggregator or ProductBadgeAggregator()
        self.sentiment = sentiment_model or DefaketiveSentimentModel()
        self.normalizer_version = text_norm.NORMALIZER_VERSION
        self.segmenter_version = text_norm.SEGMENTER_VERSION
        self.model_version = MODEL_VERSION

    def fingerprint(self, review: object) -> str:
        return self.sentiment.fingerprint(review)

    def analyze(self, review: object, rating: object = None) -> dict[str, Any]:
        sentences = self.tagger.tag_review(review)
        review_tags = sorted({tag for item in sentences for tag in item["tags"]})
        confident = sorted(
            {
                tag
                for item in sentences
                for tag, value in item["high_confidence"].items()
                if value
            }
        )
        subtypes = [
            item["hazard_subtype"]
            for item in sentences
            if item.get("hazard_subtype") and item["high_confidence"].get(SAFETY)
        ]
        explanation = self.sentiment.analyze(review, rating)
        label = explanation.get("label")
        return {
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "normalizer_version": self.normalizer_version,
            "taxonomy_version": taxonomy.TAXONOMY_VERSION,
            "qualified_for_integration": self.qualified_for_integration,
            "sentences": sentences,
            "review_tags": review_tags,
            "review_high_confidence": confident,
            "max_severity": max(
                (taxonomy.SEVERITY.get(name, 0) for name in subtypes), default=0
            ),
            "sentiment": {
                "label": label,
                "probabilities": (explanation.get("classifier") or {}).get("probabilities") or {},
            },
            "label": label,
            "review_excerpt": str(review or "").strip()[:240],
        }

    def analyze_batch(
        self, reviews: list[object], ratings: list[object] | None = None
    ) -> list[dict[str, Any]]:
        ratings = list(ratings or [None] * len(reviews))
        return [self.analyze(review, rating) for review, rating in zip(reviews, ratings)]

    def summarize(
        self,
        analyses: Iterable[Mapping[str, Any]],
        *,
        reviews_available: int | None = None,
        product_type: str | None = None,
        coverage: Mapping[str, Any] | None = None,
        **_ignored: Any,
    ) -> dict[str, Any]:
        summary = self.aggregator.summarize(
            analyses,
            reviews_available=reviews_available,
            product_type=product_type,
            coverage=coverage,
        )
        summary["model_version"] = self.model_version
        summary["normalizer_version"] = self.normalizer_version
        summary["segmenter_version"] = self.segmenter_version
        summary["qualified_for_integration"] = self.qualified_for_integration
        return summary
