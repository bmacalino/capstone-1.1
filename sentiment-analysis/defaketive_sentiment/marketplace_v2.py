"""Safe inference interface for the experimental marketplace-v2 artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from collections import Counter

from .model import DefaketiveSentimentModel
from .risk_heads import rating_feature_text
from .trained import feature_text


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "lazada_round1_binary_uncertain.joblib"
RISK_PRIOR = 0.15
RISK_EVIDENCE_REVIEWS = 30
SEVERITY_WEIGHTS = {1: 0.33, 2: 0.67, 3: 1.0}
BOOTSTRAP_SAMPLES = 1_000
RISK_SCORING_VERSION = "risk-score-v1"


class MarketplaceV2Model:
    """Load a trusted local candidate and keep unqualified models opt-in."""

    def __init__(self, path: Path | str = DEFAULT_ARTIFACT, *, allow_unqualified: bool = False):
        self.path = Path(path)
        artifact = joblib.load(self.path)
        required = {
            "metadata",
            "sentiment_pipeline",
            "risk_detection_pipeline",
            "risk_severity_pipeline",
        }
        missing = required.difference(artifact)
        if missing:
            raise ValueError(f"Invalid marketplace-v2 artifact; missing: {sorted(missing)}")
        if not artifact["metadata"].get("qualified_for_integration") and not allow_unqualified:
            raise RuntimeError(
                "Marketplace-v2 has not passed its evaluation gates. "
                "Pass allow_unqualified=True only for research testing."
            )
        self.artifact = artifact
        self.base_model_version = str(artifact["metadata"]["model_version"])
        self.model_version = f"{self.base_model_version}+{RISK_SCORING_VERSION}"
        # The rule model supplies human-readable evidence while the trained
        # heads remain authoritative for risk probability and severity.
        self.explainer = DefaketiveSentimentModel()

    @staticmethod
    def _probabilities(pipeline: Any, text: str) -> dict[str, float]:
        values = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        return {str(label): round(float(probability), 6) for label, probability in zip(classes, values)}

    def fingerprint(self, review: object) -> str:
        return self.explainer.fingerprint(review)

    def analyze(self, review: object, rating: object = None) -> dict[str, Any]:
        text = feature_text(DefaketiveSentimentModel.normalize(review))
        explanation = self.explainer.analyze(review, rating)
        sentiment_pipeline = self.artifact["sentiment_pipeline"]
        risk_pipeline = self.artifact["risk_detection_pipeline"]
        severity_pipeline = self.artifact["risk_severity_pipeline"]

        sentiment_probabilities = self._probabilities(sentiment_pipeline, text)
        sentiment = max(sentiment_probabilities, key=sentiment_probabilities.get)
        uncertainty_threshold = self.artifact.get("sentiment_uncertainty_threshold")
        if (
            uncertainty_threshold is not None
            and sentiment_probabilities[sentiment] < float(uncertainty_threshold)
        ):
            sentiment = "uncertain"
        risk_probabilities = self._probabilities(risk_pipeline, text)
        risk_probability = risk_probabilities.get("1", 0.0)
        detection_threshold = float(
            self.artifact.get(
                "risk_detection_threshold",
                getattr(risk_pipeline, "threshold", 0.5),
            )
        )
        risk_detected = risk_probability >= detection_threshold
        severity_text = (
            rating_feature_text(text, rating)
            if self.artifact["metadata"].get("severity_uses_rating_token")
            else text
        )
        severity_probabilities = self._probabilities(severity_pipeline, severity_text)
        expected_severity = sum(
            int(level) * probability
            for level, probability in severity_probabilities.items()
        )
        expected_severity_weight = sum(
            SEVERITY_WEIGHTS[int(level)] * probability
            for level, probability in severity_probabilities.items()
        )
        review_contribution = risk_probability * expected_severity_weight
        result: dict[str, Any] = {
            "model_version": self.model_version,
            "qualified_for_integration": bool(
                self.artifact["metadata"].get("qualified_for_integration")
            ),
            "label": sentiment,
            "label_source": "marketplace_binary_with_uncertainty",
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
                "uncertainty_threshold": uncertainty_threshold,
            },
            "risk": {
                "detected": risk_detected,
                "confidence": risk_probability,
                "score": round(review_contribution, 4),
                "probabilities": risk_probabilities,
                "categories": [],
                "category_probabilities": {},
                "severity": int(severity_pipeline.predict([severity_text])[0]),
                "severity_probabilities": severity_probabilities,
                "expected_severity": round(expected_severity, 4),
                "expected_severity_weight": round(expected_severity_weight, 4),
                "detection_threshold": detection_threshold,
                "evidence": (explanation.get("risk") or {}).get("evidence") or [],
            },
            "review_excerpt": str(review or "").strip()[:240],
        }
        return result

    @staticmethod
    def summarize(
        analyses: list[dict[str, Any]],
        *,
        reviews_available: int | None = None,
        sampling_representative: bool | None = None,
        star_distribution_known: bool | None = None,
        bootstrap_samples: int = BOOTSTRAP_SAMPLES,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        records = list(analyses)
        total = len(records)
        if not total:
            return {
                "review_count": 0,
                "sentiment_counts": {"positive": 0, "uncertain": 0, "negative": 0},
                "sentiment_ratios": {"positive": 0.0, "uncertain": 0.0, "negative": 0.0},
                "risk_review_count": 0,
                "keyword_failure_rate": 0.0,
                "risk_score": 0.0,
                "risk_score_scale": "percent",
                "risk_categories": {},
                "risk_score_method": "calibrated_probability_expected_severity",
                "evidence_confidence": 0.0,
                "reviews_available": max(0, int(reviews_available or 0)),
                "coverage_ratio": 0.0,
                "risk_score_ci_95": [0.0, 0.0],
                "evidence_status": "no_usable_reviews",
                "evidence_warnings": ["no_usable_reviews"],
            }

        counts = Counter(record.get("label", "uncertain") for record in records)
        sentiment_labels = ("positive", "uncertain", "negative")
        ratios = {label: counts[label] / total for label in sentiment_labels}
        risky = [record for record in records if (record.get("risk") or {}).get("detected")]
        risk_review_rate = len(risky) / total
        evidence_terms = Counter(
            str(item.get("term") or "").strip().casefold()
            for record in records
            for item in (record.get("risk") or {}).get("evidence") or []
            if item.get("term") and not item.get("negated")
        )
        impacts = [float((record.get("risk") or {}).get("score") or 0) for record in records]
        weights = [
            max(0.0, float((record.get("risk") or {}).get("sampling_weight") or 1.0))
            for record in records
        ]
        weight_total = sum(weights) or float(total)
        raw_risk = sum(weight * impact for weight, impact in zip(weights, impacts)) / weight_total
        effective_sample_size = (
            weight_total**2 / sum(weight**2 for weight in weights)
            if any(weights)
            else float(total)
        )
        evidence_confidence = effective_sample_size / (
            effective_sample_size + RISK_EVIDENCE_REVIEWS
        )
        calibrated_risk = (
            evidence_confidence * raw_risk
            + (1 - evidence_confidence) * RISK_PRIOR
        )
        detected_confidences = [
            float((record.get("risk") or {}).get("confidence") or 0)
            for record in records
        ]
        expected_severities = [
            float((record.get("risk") or {}).get("expected_severity") or 0)
            for record in records
        ]
        available = max(total, int(reviews_available or total))
        coverage = total / available if available else 0.0
        evidence_warnings = []
        if total < 30:
            evidence_warnings.append("limited_sample")
        if sampling_representative is False:
            evidence_warnings.append("biased_sample")
        if available > total and star_distribution_known is not True:
            evidence_warnings.append("star_distribution_unknown")
        evidence_status = evidence_warnings[0] if evidence_warnings else "sufficient"

        rng = np.random.default_rng(random_seed)
        bootstrap_scores: list[float] = []
        if bootstrap_samples > 0:
            impact_array = np.asarray(impacts, dtype=float)
            weight_array = np.asarray(weights, dtype=float)
            for _ in range(int(bootstrap_samples)):
                chosen = rng.integers(0, total, total)
                chosen_weights = weight_array[chosen]
                denominator = float(chosen_weights.sum())
                sampled_raw = (
                    float(np.dot(chosen_weights, impact_array[chosen]) / denominator)
                    if denominator
                    else 0.0
                )
                bootstrap_scores.append(
                    100
                    * (
                        evidence_confidence * sampled_raw
                        + (1 - evidence_confidence) * RISK_PRIOR
                    )
                )
        ci_low, ci_high = (
            np.percentile(bootstrap_scores, [2.5, 97.5]).tolist()
            if bootstrap_scores
            else [100 * calibrated_risk, 100 * calibrated_risk]
        )
        top_evidence = sorted(
            (
                {
                    "review_index": index,
                    "excerpt": str(record.get("review_excerpt") or "")[:180],
                    "contribution": round(float((record.get("risk") or {}).get("score") or 0), 4),
                    "risk_probability": round(float((record.get("risk") or {}).get("confidence") or 0), 4),
                    "expected_severity": round(float((record.get("risk") or {}).get("expected_severity") or 0), 4),
                }
                for index, record in enumerate(records)
            ),
            key=lambda item: item["contribution"],
            reverse=True,
        )[:5]
        return {
            "review_count": total,
            "sentiment_counts": {label: counts[label] for label in sentiment_labels},
            "sentiment_ratios": {label: round(ratios[label], 4) for label in sentiment_labels},
            "average_compound": round(
                sum(float((record.get("scores") or {}).get("compound") or 0) for record in records)
                / total,
                4,
            ),
            "risk_review_count": len(risky),
            "affected_review_count": len(risky),
            "keyword_failure_rate": round(risk_review_rate, 4),
            "risk_review_rate": round(risk_review_rate, 4),
            "average_risk_probability": round(
                sum(weight * probability for weight, probability in zip(weights, detected_confidences))
                / weight_total,
                4,
            ),
            "average_risk_severity": round(
                sum(weight * severity for weight, severity in zip(weights, expected_severities))
                / weight_total,
                4,
            ),
            "raw_risk_score": round(100 * raw_risk, 2),
            "evidence_confidence": round(evidence_confidence, 4),
            "effective_sample_size": round(effective_sample_size, 2),
            "risk_prior": RISK_PRIOR,
            "risk_score": round(100 * calibrated_risk, 2),
            "risk_score_scale": "percent",
            "risk_score_method": "calibrated_probability_expected_severity",
            "risk_categories": {},
            "risk_evidence_terms": dict(evidence_terms.most_common()),
            "reviews_available": available,
            "coverage_ratio": round(coverage, 4),
            "sampling_representative": sampling_representative,
            "star_distribution_known": star_distribution_known,
            "risk_score_ci_95": [round(float(ci_low), 2), round(float(ci_high), 2)],
            "evidence_status": evidence_status,
            "evidence_warnings": evidence_warnings,
            "representative_risk_evidence": top_evidence,
        }
