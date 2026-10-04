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
from pathlib import Path
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

# 0.2.0 adds 42 safety terms (the Tagalog hazard vocabulary the shipped list
# lacked) and the safety false-friend filter. 0.3.0 narrows negation to the word
# it governs, adds the defect and counterfeit phrasings the tagging guide lists,
# stops counting missing items and damaged boxes as defects, and drops matches
# inside names ("burnt orange", "smoke removal", "fake news", "fake nails").
# Changing it invalidates cached analyses, which
# dashboard_utils.analysis_is_current relies on.
MODEL_VERSION = "defaketive-hazard-lexicon-0.3.0"

# Rule sets the tagger can run. "0.2" reproduces the previous release exactly,
# for the before/after measurement; the application runs the current one.
RULES = ("0.2", "0.3")
CURRENT_RULES = "0.3"

# A mild heat word ("umiinit", "mainit") is suppressed as a safety signal when
# the sentence is about something worn, the weather or a venue, or a product
# whose job is heat. Anything stronger -- fire, smoke, sparks, a swollen
# battery -- is never suppressed by context. Round 1 labels: 10 of 26 safety
# false alarms removed with no confirmed hazard lost.
FALSE_FRIENDS_FILE = "safety_false_friends_v1.tsv"
MILD_HEAT_MAX_SEVERITY = 2

# 0.3: cue classes that are NAMES, not context. A hazard word inside one of
# these phrases ("burnt orange", "smoke removal") is not a report whatever its
# severity, so that single match is dropped; the rest of the sentence still
# counts. Context classes (apparel, weather...) stay limited to mild heat.
NAME_CUE_CLASSES = {"colour", "feature_name"}

# 0.3: phrases that name a product or an idiom rather than a report ("fake
# news", "fake nails"). Same single-match rule as NAME_CUE_CLASSES.
PHRASE_EXCLUSIONS_FILE = "hazard_phrase_exclusions_v1.tsv"

# 0.3: "baka fake" is a worry, not a report (guide rule 5).
SPECULATION_WORDS = frozenset({"baka", "bka", "maybe", "might"})

# 0.3: guide rule 1 - a damaged box is not a defective product. A damage word
# within three words of a packaging word is about the packaging.
DAMAGE_TERMS = frozenset(
    {"sira", "nasira", "damaged", "damage", "broken", "crack", "cracked", "basag"}
)
PACKAGING_WORDS = frozenset(
    {"box", "boxes", "kahon", "package", "packaging", "packing", "parcel",
     "balot", "wrapper", "wrapping"}
)

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


def _load_false_friends(path: Path | None = None) -> list[tuple[re.Pattern[str], str]]:
    """Read the safety false-friend cues as whole-word patterns.

    Whole-word matching is deliberately stricter than a substring test: a cue
    that fires inside an unrelated word would suppress a real safety report,
    and a missed hazard is the costly error in this category. `path` lets the
    measurement load a frozen earlier list.
    """
    from .model import LEXICON_DIR

    cues: list[tuple[re.Pattern[str], str]] = []
    path = path or LEXICON_DIR / FALSE_FRIENDS_FILE
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 2 or not parts[0]:
            continue
        cue = re.escape(parts[0].casefold()).replace(r"\ ", r"\s+")
        cues.append((re.compile(rf"(?<!\w){cue}(?!\w)"), parts[1]))
    return cues


def _normalized_phrase(phrase: str) -> re.Pattern[str] | None:
    """A whole-word pattern for `phrase` in the tagger's normalized text."""
    normalized = text_norm.normalize_text(phrase)
    if not normalized:
        return None
    return re.compile(
        r"(?<!\w)" + r"\s+".join(re.escape(part) for part in normalized.split()) + r"(?!\w)"
    )


def _load_name_phrases(
    false_friends_path: Path | None = None,
) -> list[tuple[re.Pattern[str], str, str]]:
    """(pattern, category, class) for phrases inside which a match is dropped.

    Safety: the NAME_CUE_CLASSES rows of the false-friend list. Other
    categories: PHRASE_EXCLUSIONS_FILE. Matched in normalized text, so
    "burnt orange" and "fake nailll" are found the way the terms are.
    """
    from .model import LEXICON_DIR

    rows: list[tuple[re.Pattern[str], str, str]] = []

    def read(path: Path, has_category: bool) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [part.strip() for part in line.split("\t")]
            if has_category:
                if len(parts) < 3 or parts[1] not in CATEGORIES:
                    continue
                phrase, category, cue_class = parts[0], parts[1], parts[2]
            else:
                if len(parts) < 2 or parts[1] not in NAME_CUE_CLASSES:
                    continue
                phrase, category, cue_class = parts[0], SAFETY, parts[1]
            pattern = _normalized_phrase(phrase)
            if pattern is not None:
                rows.append((pattern, category, cue_class))

    read(false_friends_path or LEXICON_DIR / FALSE_FRIENDS_FILE, has_category=False)
    read(LEXICON_DIR / PHRASE_EXCLUSIONS_FILE, has_category=True)
    return rows


def _near_packaging(normalized: str, span: tuple[int, int], reach: int = 3) -> bool:
    before = re.findall(r"[\w']+", normalized[: span[0]])[-reach:]
    after = re.findall(r"[\w']+", normalized[span[1]:])[:reach]
    return any(word in PACKAGING_WORDS for word in before + after)


def _speculative(normalized: str, start: int, reach: int = 3) -> bool:
    clause = re.split(r"[.!?;:,]", normalized[:start])[-1]
    return any(word in SPECULATION_WORDS for word in re.findall(r"[\w']+", clause)[-reach:])


def _load_category_terms(
    lexicon_path: Path | None = None,
) -> dict[str, list[tuple[str, int, re.Pattern[str]]]]:
    """Read the risk lexicon, keeping only the three badge categories.

    lexicon_path exists so research scripts can measure a candidate term list
    without editing the shipped one. Leaving it None loads the shipped lexicon,
    so the application's behaviour is unchanged.
    """
    model_lexicon = DefaketiveSentimentModel.__dict__  # noqa: F841 - documents intent
    from .model import LEXICON_DIR, _load_tsv

    terms: dict[str, list[tuple[str, int, re.Pattern[str]]]] = {name: [] for name in CATEGORIES}
    for entry in _load_tsv(lexicon_path or LEXICON_DIR / "risk_lexicon.tsv"):
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

    def __init__(
        self,
        lexicon_path: Path | None = None,
        *,
        false_friends: bool = True,
        false_friends_path: Path | None = None,
        rules: str = CURRENT_RULES,
        learned: Any | None = None,
        embedder: Any | None = None,
        sentiment_label: Any | None = None,
    ) -> None:
        if rules not in RULES:
            raise ValueError(f"rules must be one of {RULES}")
        self.rules = rules
        # Learned per-category models (hazard_learned.LearnedHazardScorer). For
        # each category they hold, they make the final yes/no; the keyword
        # matches still supply the highlighted words and the safety subtype.
        self.learned = learned
        self.embedder = embedder
        # Sentence -> sentiment label, for categories whose learned model may
        # only add a flag to a sentence that is not praise ("or_gated").
        self.sentiment_label = sentiment_label
        if learned is not None and learned.needs_embedding and embedder is None:
            raise HazardModelError("The learned hazard model needs sentence embeddings, but no embedder was given.")
        if learned is not None and getattr(learned, "needs_sentiment", False) and sentiment_label is None:
            raise HazardModelError("The gated hazard model needs a sentiment reading, but none was given.")
        self.terms = _load_category_terms(lexicon_path)
        # Disabling the filter exists for the before/after measurement only.
        self.false_friends = _load_false_friends(false_friends_path) if false_friends else []
        # Name phrases drop a single match at any severity (rules 0.3 only).
        self.name_phrases: list[tuple[re.Pattern[str], str, str]] = (
            _load_name_phrases(false_friends_path) if rules != "0.2" else []
        )
        self.normalizer_version = text_norm.NORMALIZER_VERSION
        self.segmenter_version = text_norm.SEGMENTER_VERSION

    def _is_negated(self, normalized: str, start: int) -> bool:
        if self.rules == "0.2":
            return text_norm.is_negated(normalized, start)
        return text_norm.is_negated_adjacent(normalized, start)

    def _excluded(
        self,
        category: str,
        term: str,
        normalized: str,
        span: tuple[int, int],
        name_spans: list[tuple[int, int, str, str]],
    ) -> str | None:
        """Why this one match is not a report, or None. Rules 0.3 only."""
        if self.rules == "0.2":
            return None
        for start, end, phrase_category, cue_class in name_spans:
            if phrase_category == category and start <= span[0] and span[1] <= end:
                return cue_class
        if category == COUNTERFEIT and _speculative(normalized, span[0]):
            return "speculation"
        if category == DEFECT and term in DAMAGE_TERMS and _near_packaging(normalized, span):
            return "packaging"
        return None

    def _false_friend(self, raw: str) -> str | None:
        """Return the cue class that makes this sentence's heat word harmless."""
        lowered = str(raw).casefold()
        for pattern, cue_class in self.false_friends:
            if pattern.search(lowered):
                return cue_class
        return None

    def tag_sentence(self, raw: str) -> dict[str, Any]:
        normalized, spans = text_norm.normalize_with_spans(raw)
        scores: dict[str, float] = {}
        tags: list[str] = []
        confidence: dict[str, bool] = {}
        matches: list[dict[str, Any]] = []
        severities: dict[str, int] = {}
        safety_terms: list[str] = []

        about_the_product = not NON_PRODUCT_SUBJECT.search(normalized)
        name_spans = [
            (found.start(), found.end(), category, cue_class)
            for pattern, category, cue_class in self.name_phrases
            for found in pattern.finditer(normalized)
        ]
        excluded: dict[str, list[str]] = {}
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
                    negated = not text_norm.starts_with_negator(term) and self._is_negated(
                        normalized, match.start()
                    )
                    raw_span = self._raw_span(match.span(), spans)
                    reason = self._excluded(category, term, normalized, match.span(), name_spans)
                    matches.append(
                        {
                            "ngram": term,
                            "category": category,
                            "severity": term_severity,
                            "negated": negated,
                            "span": list(raw_span) if raw_span else None,
                            **({"excluded": reason} if reason else {}),
                        }
                    )
                    if reason and not negated:
                        excluded.setdefault(category, []).append(reason)
                    if negated or reason:
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

        suppressed: dict[str, str] = {}
        if SAFETY in tags and severities.get(SAFETY, 0) <= MILD_HEAT_MAX_SEVERITY:
            cue_class = self._false_friend(raw)
            if cue_class:
                # The terms stay in "terms" so the decision can be audited.
                tags.remove(SAFETY)
                scores.pop(SAFETY, None)
                confidence.pop(SAFETY, None)
                severities.pop(SAFETY, None)
                suppressed[SAFETY] = cue_class

        learned: dict[str, float] = {}
        gated: dict[str, str] = {}
        if self.learned is not None:
            from .hazard_learned import keyword_features

            # Features come from the keyword result as it stands here, before
            # any learned decision, exactly as in training.
            kw = keyword_features({"terms": matches, "tags": tags})
            embedding = self.embedder(raw) if self.learned.needs_embedding else None
            combine = getattr(self.learned, "combine", {})
            for category, (on, probability) in self.learned.decide(normalized, kw, embedding).items():
                learned[category] = round(probability, 4)
                if combine.get(category) == "or_gated":
                    # Keywords' flag stands; the model may only add one, and
                    # never to a sentence read as praise. The sentiment model
                    # runs only for the sentences where that is in question.
                    if on and category not in tags:
                        label = self.sentiment_label(raw)
                        gated[category] = label
                        if label != "positive":
                            tags.append(category)
                            scores[category] = probability
                            severities[category] = MIN_CONFIDENT_SEVERITY
                            confidence[category] = about_the_product
                    continue
                if on and category not in tags:
                    tags.append(category)
                    scores[category] = probability
                    severities[category] = MIN_CONFIDENT_SEVERITY
                    confidence[category] = about_the_product
                elif not on and category in tags:
                    tags.remove(category)
                    scores.pop(category, None)
                    confidence.pop(category, None)
                    severities.pop(category, None)
            tags.sort(key=CATEGORIES.index)

        return {
            "text_raw": raw,
            "text_normalized": normalized,
            "scores": scores,
            "tags": tags,
            "high_confidence": confidence,
            "hazard_subtype": hazard_subtype(safety_terms) if SAFETY in tags else None,
            "terms": matches,
            "suppressed": suppressed,
            **({"excluded": excluded} if excluded else {}),
            **({"learned": learned} if learned else {}),
            **({"gate_sentiment": gated} if gated else {}),
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
        # Reports show the sentiment split next to the badges, so a change of
        # sentiment model must make saved analyses stale just as a change of
        # tagger does. Hence both versions in one string.
        sentiment_version = getattr(self.sentiment, "model_version", None)
        learned = getattr(self.tagger, "learned", None)
        hazard_version = f"{MODEL_VERSION}+{learned.version}" if learned is not None else MODEL_VERSION
        self.model_version = (
            f"{hazard_version}+{sentiment_version}" if sentiment_version else hazard_version
        )

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
