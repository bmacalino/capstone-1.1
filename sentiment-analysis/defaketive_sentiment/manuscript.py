"""Manuscript-aligned product-risk aggregation models.

RESEARCH BASELINE ONLY as of the badge redesign. These models are no longer the
production risk path; `hazard_model.HazardBadgeModel` is. They are kept, and
kept working, because the validation plan compares the new design against the
original weighted-sum formula, and removing them would make that comparison
impossible to reproduce.

Both models aggregate products with the manuscript's fixed weighted-sum:

    0.30 * negative review ratio + 0.70 * detected-risk review ratio

The weights were never calibrated - `docs/model.md` says they "must be confirmed
with the research advisers before being presented as validated product risk" -
and the formula double counts, because a keyword-flagged review is almost always
also a negative one. Those are among the reasons for the replacement.

``ManuscriptRiskModel`` preserves the strict VADER/lexicon baseline.
``HybridManuscriptRiskModel`` uses the trained sentiment and binary-risk heads
with the same weighted-sum product score. Severity is not used by either.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import joblib

from .model import DefaketiveSentimentModel
from .trained import feature_text


MODEL_VERSION = "defaketive-manuscript-vader-wsm-1.0.0"
RISK_SCORE_METHOD = "manuscript_vader_keyword_wsm"
HYBRID_RISK_SCORE_METHOD = "hybrid_classifier_manuscript_wsm"
NEGATIVE_WEIGHT = 0.30
DEFECT_WEIGHT = 0.70
DEFAULT_HYBRID_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "models" / "hybrid_manuscript_v1.joblib"
)


class ManuscriptModelError(RuntimeError):
    """Raised when a manuscript artifact must not be used for reported results."""


class ManuscriptRiskModel(DefaketiveSentimentModel):
    """Apply the exact sentiment/keyword weighted-sum design in the manuscript."""

    def __init__(self) -> None:
        # The manuscript specifies a lexicon-based sentiment model. Disable the
        # optional supervised FiReCS classifier so VADER remains authoritative.
        super().__init__(use_trained=False)
        self.model_version = MODEL_VERSION

    def analyze(self, review: object, rating: object = None) -> dict[str, Any]:
        # Star ratings are deliberately not passed to the base model. Otherwise
        # a one- or two-star rating could override VADER's text classification,
        # which would no longer match the documented method.
        result = super().analyze(review, rating=None)
        result["model_version"] = self.model_version
        result["label_source"] = (
            "empty_text" if result.get("label_source") == "empty_text" else "vader_rules"
        )
        # The application calls VADER's neutral band "uncertain" to avoid
        # implying that an ambiguous or factual review has no opinion. This
        # naming change does not affect the negative ratio used by the score.
        if result.get("label") == "neutral":
            result["label"] = "uncertain"

        risk = result.get("risk") or {}
        detected = bool(risk.get("detected"))
        # A review counts once in the failure-rate numerator regardless of how
        # many active defect phrases it contains. Severity is not part of the
        # manuscript score, so do not expose it as a scoring output.
        risk["score"] = 1.0 if detected else 0.0
        risk["score_basis"] = "binary_defect_indicator"
        risk["evidence"] = [
            {key: value for key, value in item.items() if key != "severity"}
            for item in risk.get("evidence") or []
        ]
        result["risk"] = risk
        return result

    @staticmethod
    def summarize(
        analyses: Iterable[dict[str, Any]],
        *,
        reviews_available: int | None = None,
        sampling_representative: bool | None = None,
        star_distribution_known: bool | None = None,
    ) -> dict[str, Any]:
        records = list(analyses)
        total = len(records)
        available = max(total, int(reviews_available or total))
        if not total:
            return {
                "review_count": 0,
                "sentiment_counts": {"positive": 0, "uncertain": 0, "negative": 0},
                "sentiment_ratios": {"positive": 0.0, "uncertain": 0.0, "negative": 0.0},
                "average_compound": 0.0,
                "risk_review_count": 0,
                "affected_review_count": 0,
                "keyword_failure_rate": 0.0,
                "detected_risk_ratio": 0.0,
                "risk_review_rate": 0.0,
                "negative_contribution_points": 0.0,
                "defect_contribution_points": 0.0,
                "risk_score": 0.0,
                "risk_score_scale": "percent",
                "risk_score_method": RISK_SCORE_METHOD,
                "risk_score_weights": {
                    "negative_ratio": NEGATIVE_WEIGHT,
                    "defect_review_ratio": DEFECT_WEIGHT,
                },
                "risk_categories": {},
                "risk_evidence_terms": {},
                "reviews_available": available,
                "coverage_ratio": 0.0,
                "sampling_representative": sampling_representative,
                "star_distribution_known": star_distribution_known,
                "evidence_status": "no_usable_reviews",
                "evidence_warnings": ["no_usable_reviews"],
            }

        weights = [
            max(0.0, float((record.get("risk") or {}).get("sampling_weight") or 1.0))
            for record in records
        ]
        weight_total = sum(weights) or float(total)
        labels = ("positive", "uncertain", "negative")
        counts = Counter(str(record.get("label") or "uncertain") for record in records)
        weighted_counts = {
            label: sum(
                weight
                for record, weight in zip(records, weights)
                if str(record.get("label") or "uncertain") == label
            )
            for label in labels
        }
        sentiment_ratios = {
            label: weighted_counts[label] / weight_total for label in labels
        }
        risky = [record for record in records if (record.get("risk") or {}).get("detected")]
        weighted_risk_reviews = sum(
            weight
            for record, weight in zip(records, weights)
            if (record.get("risk") or {}).get("detected")
        )
        failure_rate = weighted_risk_reviews / weight_total
        negative_points = 100 * NEGATIVE_WEIGHT * sentiment_ratios["negative"]
        defect_points = 100 * DEFECT_WEIGHT * failure_rate
        risk_score = negative_points + defect_points

        category_counts = Counter(
            category
            for record in risky
            for category in (record.get("risk") or {}).get("categories") or []
        )
        evidence_terms = Counter(
            str(item.get("term") or "").strip().casefold()
            for record in records
            for item in (record.get("risk") or {}).get("evidence") or []
            if item.get("term") and not item.get("negated")
        )
        warnings: list[str] = []
        if total < 30:
            warnings.append("limited_sample")
        if sampling_representative is False:
            warnings.append("biased_sample")
        if available > total and star_distribution_known is not True:
            warnings.append("star_distribution_unknown")

        return {
            "review_count": total,
            "sentiment_counts": {label: counts[label] for label in labels},
            "sentiment_ratios": {
                label: round(sentiment_ratios[label], 4) for label in labels
            },
            "weighted_sentiment_counts": {
                label: round(weighted_counts[label], 4) for label in labels
            },
            "average_compound": round(
                sum(
                    weight * float((record.get("scores") or {}).get("compound") or 0)
                    for record, weight in zip(records, weights)
                )
                / weight_total,
                4,
            ),
            "risk_review_count": len(risky),
            "affected_review_count": len(risky),
            "weighted_risk_review_count": round(weighted_risk_reviews, 4),
            "keyword_failure_rate": round(failure_rate, 4),
            "detected_risk_ratio": round(failure_rate, 4),
            "risk_review_rate": round(failure_rate, 4),
            "negative_contribution_points": round(negative_points, 2),
            "defect_contribution_points": round(defect_points, 2),
            "risk_score": round(risk_score, 2),
            "risk_score_scale": "percent",
            "risk_score_method": RISK_SCORE_METHOD,
            "risk_score_weights": {
                "negative_ratio": NEGATIVE_WEIGHT,
                "defect_review_ratio": DEFECT_WEIGHT,
            },
            "risk_categories": dict(sorted(category_counts.items())),
            "risk_evidence_terms": dict(evidence_terms.most_common()),
            "reviews_available": available,
            "coverage_ratio": round(total / available, 4) if available else 0.0,
            "sampling_representative": sampling_representative,
            "star_distribution_known": star_distribution_known,
            "evidence_status": warnings[0] if warnings else "sufficient",
            "evidence_warnings": warnings,
        }


class HybridManuscriptRiskModel:
    """Use trained review classifiers with the manuscript's 30/70 score.

    The artifact contains only the two heads used by the application. Product
    score depends only on the negative-review ratio and the binary detected-risk
    ratio; risk categories and severity are not trained or inferred.
    """

    def __init__(
        self,
        path: Path | str = DEFAULT_HYBRID_ARTIFACT,
        *,
        allow_unqualified: bool = False,
    ) -> None:
        self.path = Path(path)
        self.artifact = joblib.load(self.path)
        required = {"metadata", "sentiment_pipeline", "risk_detection_pipeline"}
        missing = required.difference(self.artifact)
        if missing:
            raise ValueError(
                f"Invalid hybrid-manuscript artifact; missing: {sorted(missing)}"
            )
        metadata = self.artifact["metadata"]
        # Every other artifact loader in this package refuses an artifact whose
        # own metadata says it did not pass its evaluation gates - see
        # trained.py and marketplace_v2.py. This class did not check, and the
        # artifact it loads by default is marked qualified_for_integration:
        # false, so the application has been serving a model its own gate would
        # have rejected. Research scripts that need the unqualified artifact for
        # a baseline comparison must now say so explicitly.
        if not allow_unqualified and not metadata.get("qualified_for_integration"):
            raise ManuscriptModelError(
                f"{self.path.name} is marked qualified_for_integration=false and "
                "must not be used for reported results. Pass allow_unqualified=True "
                "to load it deliberately for a research baseline."
            )
        self.qualified_for_integration = bool(metadata.get("qualified_for_integration"))
        self.base_model_version = str(metadata["model_version"])
        self.model_version = f"{self.base_model_version}+manuscript-wsm-1.0.0"
        self.explainer = DefaketiveSentimentModel()

    @staticmethod
    def _probabilities(pipeline: Any, text: str) -> dict[str, float]:
        values = pipeline.predict_proba([text])[0]
        return {
            str(label): round(float(probability), 6)
            for label, probability in zip(pipeline.classes_, values)
        }

    @staticmethod
    def _probabilities_batch(pipeline: Any, texts: list[str]) -> list[dict[str, float]]:
        """Score many reviews in one call.

        scikit-learn's per-call overhead dominates at these text lengths: one
        review at a time costs about 6.6 ms, while the same reviews scored in a
        single call cost about 0.55 ms each - roughly twelve times faster. That
        gap did not matter when analysis was capped at 100 reviews per product;
        with the cap removed and listings running to thousands of reviews, it is
        the difference between a usable scan and an unusable one.
        """
        if not texts:
            return []
        matrix = pipeline.predict_proba(texts)
        labels = [str(label) for label in pipeline.classes_]
        return [
            {label: round(float(probability), 6) for label, probability in zip(labels, row)}
            for row in matrix
        ]

    def fingerprint(self, review: object) -> str:
        return self.explainer.fingerprint(review)

    def _detection_threshold(self) -> float:
        return float(
            self.artifact.get(
                "risk_detection_threshold",
                getattr(self.artifact["risk_detection_pipeline"], "threshold", 0.5),
            )
        )

    def analyze_batch(
        self, reviews: list[object], ratings: list[object] | None = None
    ) -> list[dict[str, Any]]:
        """Analyze many reviews, batching the classifier calls.

        Produces exactly what `analyze` produces for each review; only the
        scheduling of the sklearn calls differs. The lexicon explainer stays
        per-review because VADER and the regex matchers have no batch form.
        """
        if not reviews:
            return []
        ratings = list(ratings or [None] * len(reviews))
        texts = [feature_text(DefaketiveSentimentModel.normalize(item)) for item in reviews]
        sentiment_rows = self._probabilities_batch(self.artifact["sentiment_pipeline"], texts)
        risk_rows = self._probabilities_batch(self.artifact["risk_detection_pipeline"], texts)
        threshold = self._detection_threshold()
        return [
            self._assemble(
                review,
                self.explainer.analyze(review, rating),
                sentiment_probabilities,
                risk_probabilities,
                threshold,
            )
            for review, rating, sentiment_probabilities, risk_probabilities in zip(
                reviews, ratings, sentiment_rows, risk_rows
            )
        ]

    def analyze(self, review: object, rating: object = None) -> dict[str, Any]:
        text = feature_text(DefaketiveSentimentModel.normalize(review))
        explanation = self.explainer.analyze(review, rating)
        return self._assemble(
            review,
            explanation,
            self._probabilities(self.artifact["sentiment_pipeline"], text),
            self._probabilities(self.artifact["risk_detection_pipeline"], text),
            self._detection_threshold(),
        )

    def _assemble(
        self,
        review: object,
        explanation: dict[str, Any],
        sentiment_probabilities: dict[str, float],
        risk_probabilities: dict[str, float],
        detection_threshold: float,
    ) -> dict[str, Any]:
        sentiment = max(sentiment_probabilities, key=sentiment_probabilities.get)
        risk_probability = risk_probabilities.get("1", 0.0)
        detected = risk_probability >= detection_threshold
        evidence = [
            {key: value for key, value in item.items() if key != "severity"}
            for item in (explanation.get("risk") or {}).get("evidence") or []
        ]
        return {
            "model_version": self.model_version,
            "qualified_for_integration": bool(
                self.artifact["metadata"].get("qualified_for_integration")
            ),
            "label": sentiment,
            "label_source": "marketplace_direct_three_class",
            "scores": explanation.get("scores") or {},
            "classifier": {
                "label": sentiment,
                "confidence": max(sentiment_probabilities.values()),
                "probabilities": sentiment_probabilities,
            },
            "sentiment_evidence": explanation.get("sentiment_evidence") or [],
            "sentiment": {
                "label": sentiment,
                "probabilities": sentiment_probabilities,
                "uncertainty_threshold": None,
            },
            "risk": {
                "detected": detected,
                "confidence": risk_probability,
                "score": 1.0 if detected else 0.0,
                "score_basis": "binary_trained_risk_indicator",
                "probabilities": risk_probabilities,
                "categories": [],
                "category_probabilities": {},
                "detection_threshold": detection_threshold,
                "evidence": evidence,
            },
            "review_excerpt": str(review or "").strip()[:240],
        }

    @staticmethod
    def summarize(
        analyses: Iterable[dict[str, Any]],
        *,
        reviews_available: int | None = None,
        sampling_representative: bool | None = None,
        star_distribution_known: bool | None = None,
    ) -> dict[str, Any]:
        summary = ManuscriptRiskModel.summarize(
            analyses,
            reviews_available=reviews_available,
            sampling_representative=sampling_representative,
            star_distribution_known=star_distribution_known,
        )
        summary["risk_score_method"] = HYBRID_RISK_SCORE_METHOD
        summary["risk_detection_source"] = "trained_binary_classifier"
        summary["sentiment_source"] = "trained_direct_three_class_classifier"
        return summary
