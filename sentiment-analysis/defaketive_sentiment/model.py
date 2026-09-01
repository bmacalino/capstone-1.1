"""Explainable English, Filipino, and Taglish sentiment and product-risk analysis.

The model extends NLTK's VADER rules with a locally maintained seed lexicon.
The lexicon is intentionally stored as data so researchers can review, annotate,
version, and validate every entry without changing the scoring code.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - exercised only in an incomplete install
    SentimentIntensityAnalyzer = None


MODEL_VERSION = "defaketive-vader-seed-0.1.2"
LEXICON_DIR = Path(__file__).with_name("lexicons")

FILIPINO_NEGATORS = {
    "ayaw",
    "di",
    "hindi",
    "huwag",
    "ndi",
    "wala",
    "walang",
    "wag",
}

FILIPINO_BOOSTERS = {
    "grabe": 0.293,
    "napaka": 0.293,
    "sobra": 0.293,
    "sobrang": 0.293,
    "super": 0.293,
    "talagang": 0.293,
    "medyo": -0.293,
    "kaunti": -0.293,
    "konti": -0.293,
}

SPELLING_NORMALIZATION = {
    "dizapoytedd": "disappointed",
    "hnd": "hindi",
    "hndi": "hindi",
    "ndi": "hindi",
    "wlang": "walang",
    "sobranggg": "sobrang",
}

DIRECT_FAILURE_PREFIXES = (
    "ayaw ",
    "di ",
    "does not ",
    "hindi ",
    "no ",
    "not ",
    "walang ",
)


class ModelSetupError(RuntimeError):
    """Raised when NLTK or its VADER data is not installed."""


def _load_tsv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        content = (line for line in file if line.strip() and not line.startswith("#"))
        reader = csv.reader(content, delimiter="\t")
        for values in reader:
            if path.name == "sentiment_lexicon.tsv":
                term, valence, language, category = values
                rows.append(
                    {
                        "term": term.casefold(),
                        "valence": valence,
                        "language": language,
                        "category": category,
                    }
                )
            else:
                term, category, severity, language = values
                rows.append(
                    {
                        "term": term.casefold(),
                        "category": category,
                        "severity": severity,
                        "language": language,
                    }
                )
    return rows


def _term_pattern(term: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in term.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


class DefaketiveSentimentModel:
    """NLTK VADER with Filipino/Taglish rules and an independent risk layer."""

    def __init__(self, positive_threshold: float = 0.05, negative_threshold: float = -0.05):
        if negative_threshold >= positive_threshold:
            raise ValueError("negative_threshold must be lower than positive_threshold")
        if SentimentIntensityAnalyzer is None:
            raise ModelSetupError("NLTK is missing. Install the project requirements first.")

        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except LookupError as exc:
            raise ModelSetupError(
                "NLTK's vader_lexicon is missing. Run: "
                ".\\.venv\\Scripts\\python.exe -m nltk.downloader vader_lexicon"
            ) from exc

        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.sentiment_entries = _load_tsv(LEXICON_DIR / "sentiment_lexicon.tsv")
        self.risk_entries = _load_tsv(LEXICON_DIR / "risk_lexicon.tsv")

        self.analyzer.constants.NEGATE.update(FILIPINO_NEGATORS)
        self.analyzer.constants.BOOSTER_DICT.update(FILIPINO_BOOSTERS)

        self.phrase_tokens: dict[str, str] = {}
        for index, entry in enumerate(self.sentiment_entries):
            term = entry["term"]
            # VADER strips ASCII punctuation while creating its token map, so
            # phrase and symbol-only placeholders must contain only letters
            # and digits. VADER otherwise drops standalone emoji tokens.
            needs_placeholder = " " in term or not re.search(r"\w", term, re.UNICODE)
            token = f"defaketivephrase{index}" if needs_placeholder else term
            self.analyzer.lexicon[token] = float(entry["valence"])
            if needs_placeholder:
                self.phrase_tokens[term] = token

        self._sentiment_patterns = [
            (entry, _term_pattern(entry["term"]))
            for entry in sorted(self.sentiment_entries, key=lambda item: len(item["term"]), reverse=True)
        ]
        self._risk_patterns = [
            (entry, _term_pattern(entry["term"]))
            for entry in sorted(self.risk_entries, key=lambda item: len(item["term"]), reverse=True)
        ]

    @staticmethod
    def normalize(text: Any) -> str:
        value = unicodedata.normalize("NFKC", str(text or ""))
        value = value.replace("’", "'").replace("`", "'")
        # Marketplace translations occasionally use a hyphen where a space is
        # expected (for example, "poor-quality"). VADER keeps that as one
        # unknown token, so normalize word-joining hyphens before matching.
        value = re.sub(r"(?<=\w)[‐‑‒–—-](?=\w)", " ", value)
        value = re.sub(r"\s+", " ", value).strip().casefold()
        # Temu review translations sometimes confuse a capital I with a lower
        # case L. VADER treats the standalone token "l" as positive slang.
        value = re.sub(
            r"(?<!\w)l(?=\s+(?:am|bought|can|could|did|do|have|ordered|received|was|will|would)\b)",
            "i",
            value,
        )
        parts = []
        for part in re.split(r"(\W+)", value):
            parts.append(SPELLING_NORMALIZATION.get(part, part))
        return "".join(parts)

    def fingerprint(self, text: Any) -> str:
        normalized = self.normalize(text)
        # Retain Unicode symbols so emoji-only reviews have stable, non-empty
        # fingerprints and participate in duplicate detection and summaries.
        characters = (
            character
            if character.isalnum()
            or character == "_"
            or unicodedata.category(character).startswith(("S", "M"))
            else " "
            for character in normalized
        )
        return re.sub(r"\s+", " ", "".join(characters)).strip()

    def _prepare_for_vader(self, normalized: str) -> str:
        prepared = normalized
        for phrase, token in sorted(self.phrase_tokens.items(), key=lambda item: len(item[0]), reverse=True):
            # Surround replacements so adjacent emojis become separate VADER
            # tokens instead of one concatenated unknown token.
            prepared = _term_pattern(phrase).sub(f" {token} ", prepared)
        # VADER's contrast rule recognizes "but". "Kaso" is excluded because
        # it can also be a noun meaning "case".
        return re.sub(r"(?<!\w)(pero|subalit|ngunit)(?!\w)", "but", prepared)

    @staticmethod
    def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
        return any(span[0] < end and start < span[1] for start, end in occupied)

    def _is_negated(self, normalized: str, start: int) -> bool:
        prefix = normalized[:start]
        # Do not carry negation across a sentence or a coordination boundary.
        # This keeps "walang sira at gumagana" from negating "gumagana".
        clause = re.split(
            r"[.!?;:]|(?<!\w)(?:at|and|but|pero|subalit|ngunit)(?!\w)",
            prefix,
            flags=re.IGNORECASE,
        )[-1]
        previous = re.findall(r"[\w']+", clause, flags=re.UNICODE)[-3:]
        return any(
            word in FILIPINO_NEGATORS or word in self.analyzer.constants.NEGATE
            for word in previous
        )

    def _sentiment_evidence(self, normalized: str) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []

        for entry, pattern in self._sentiment_patterns:
            for match in pattern.finditer(normalized):
                if self._overlaps(match.span(), occupied):
                    continue
                occupied.append(match.span())
                evidence.append(
                    {
                        "term": match.group(0),
                        "valence": float(entry["valence"]),
                        "language": entry["language"],
                        "category": entry["category"],
                        "negated": self._is_negated(normalized, match.start())
                        and not entry["term"].startswith(DIRECT_FAILURE_PREFIXES),
                    }
                )

        for match in re.finditer(r"[\w']+", normalized, flags=re.UNICODE):
            if self._overlaps(match.span(), occupied):
                continue
            term = match.group(0)
            if term not in self.analyzer.lexicon:
                continue
            evidence.append(
                {
                    "term": term,
                    "valence": float(self.analyzer.lexicon[term]),
                    "language": "en",
                    "category": "vader",
                    "negated": self._is_negated(normalized, match.start()),
                }
            )

        return sorted(evidence, key=lambda item: normalized.find(item["term"]))

    def _risk_evidence(self, normalized: str) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        occupied: list[tuple[int, int]] = []
        for entry, pattern in self._risk_patterns:
            for match in pattern.finditer(normalized):
                if self._overlaps(match.span(), occupied):
                    continue
                term = entry["term"]
                negated = self._is_negated(normalized, match.start())
                if term.startswith(DIRECT_FAILURE_PREFIXES):
                    negated = False
                occupied.append(match.span())
                matches.append(
                    {
                        "term": match.group(0),
                        "category": entry["category"],
                        "severity": int(entry["severity"]),
                        "language": entry["language"],
                        "negated": negated,
                    }
                )
        return matches

    @staticmethod
    def _rating_value(rating: Any) -> float | None:
        """Return a marketplace star rating when it is usable as context."""
        try:
            value = float(rating)
        except (TypeError, ValueError):
            return None
        return value if 1 <= value <= 5 else None

    def analyze(self, text: Any, rating: Any = None) -> dict[str, Any]:
        normalized = self.normalize(text)
        if not normalized:
            return {
                "model_version": MODEL_VERSION,
                "label": "neutral",
                "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
                "sentiment_evidence": [],
                "label_adjustment": None,
                "risk": {"detected": False, "score": 0.0, "categories": [], "evidence": []},
            }

        scores = self.analyzer.polarity_scores(self._prepare_for_vader(normalized))
        compound = scores["compound"]
        if compound >= self.positive_threshold:
            label = "positive"
        elif compound <= self.negative_threshold:
            label = "negative"
        else:
            label = "neutral"

        sentiment_evidence = self._sentiment_evidence(normalized)
        risk_evidence = self._risk_evidence(normalized)
        active_risks = [item for item in risk_evidence if not item["negated"]]
        max_severity = max((item["severity"] for item in active_risks), default=0)

        # Text remains the primary signal, but a one- or two-star rating is a
        # useful tie-breaker for terse complaints that VADER cannot understand
        # ("won't charge", "volume goes down"). It can also resolve a false
        # positive when strong negative text/risk evidence is present alongside
        # incidental positive words such as "lucky", "free", or "appreciate".
        # Clearly positive low-star text is left positive instead of blindly
        # copying the rating into the sentiment label.
        rating_value = self._rating_value(rating)
        active_strong_negative = any(
            not item["negated"] and item["valence"] <= -1.5
            for item in sentiment_evidence
        )
        label_adjustment = None
        if rating_value is not None and rating_value <= 2:
            if label == "neutral":
                label = "negative"
                label_adjustment = {
                    "source": "rating_context",
                    "rating": rating_value,
                    "reason": "low_rating_neutral_text",
                }
            elif label == "positive" and (active_strong_negative or active_risks):
                label = "negative"
                label_adjustment = {
                    "source": "rating_context",
                    "rating": rating_value,
                    "reason": "low_rating_with_negative_evidence",
                }

        return {
            "model_version": MODEL_VERSION,
            "label": label,
            "scores": scores,
            "sentiment_evidence": sentiment_evidence,
            "label_adjustment": label_adjustment,
            "risk": {
                "detected": bool(active_risks),
                "score": round(max_severity / 3, 4),
                "categories": sorted({item["category"] for item in active_risks}),
                "evidence": risk_evidence,
            },
        }

    @staticmethod
    def summarize(analyses: Iterable[dict[str, Any]]) -> dict[str, Any]:
        records = list(analyses)
        total = len(records)
        if total == 0:
            return {
                "review_count": 0,
                "sentiment_counts": {"positive": 0, "neutral": 0, "negative": 0},
                "sentiment_ratios": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
                "average_compound": 0.0,
                "risk_review_count": 0,
                "keyword_failure_rate": 0.0,
                "risk_score": 0.0,
                "risk_score_scale": "percent",
                "risk_categories": {},
            }

        counts = Counter(record["label"] for record in records)
        ratios = {label: counts[label] / total for label in ("positive", "neutral", "negative")}
        risky = [record for record in records if record["risk"]["detected"]]
        failure_rate = len(risky) / total
        category_counts = Counter(
            category for record in risky for category in record["risk"]["categories"]
        )
        # Preliminary capstone weighting; validate before treating it as calibrated.
        risk_score = 100 * ((0.3 * ratios["negative"]) + (0.7 * failure_rate))

        return {
            "review_count": total,
            "sentiment_counts": {label: counts[label] for label in ("positive", "neutral", "negative")},
            "sentiment_ratios": {label: round(ratios[label], 4) for label in ratios},
            "average_compound": round(
                sum(record["scores"]["compound"] for record in records) / total, 4
            ),
            "risk_review_count": len(risky),
            "keyword_failure_rate": round(failure_rate, 4),
            "risk_score": round(risk_score, 2),
            "risk_score_scale": "percent",
            "risk_categories": dict(sorted(category_counts.items())),
        }
