"""One place that decides which model the application runs.

WHY THIS EXISTS
The mode was previously read in two places in `dashboard_utils` - once to pick
the model and once to report its version - and each site did its own branching
and its own exception handling. Two copies of the same decision can disagree,
and when they do, the app analyzes reviews with one model while labelling the
results with another's version string. That mismatch then makes
`analysis_is_current` re-analyze forever, or worse, not re-analyze when it
should.

WHY THE SILENT FALLBACK IS GONE
The old code caught every exception from loading the trained model and quietly
substituted the VADER baseline. With one 0-100 score that merely changed the
number; with badges it would change the SHAPE of the output, so the report would
render a different model's verdict with no indication anything had happened. A
load failure now propagates, and the caller is expected to show "analysis
unavailable" rather than a badge computed by something else.

WHY MODELS ARE CACHED
`analysis_is_current` asks for the model version on every catalog load, and
each call used to build the whole model again. With the transformer that is
~200 MB and a second or two per call; on a 512 MB host two concurrent builds
would not fit. One instance per (mode, sentiment) is built, under a lock, and
shared: `analyze` keeps no per-call state on the instance. A failed load is not
cached, so installing the missing model and retrying works without a restart.
"""

from __future__ import annotations

import os
import threading
from typing import Any

HAZARD_BADGES = "hazard_badges"
HYBRID_MANUSCRIPT = "hybrid_manuscript"
MANUSCRIPT = "manuscript"
LEGACY_HYBRID = "legacy_hybrid"

# Aliases accepted for the lexicon/VADER baseline.
MANUSCRIPT_ALIASES = {MANUSCRIPT, "lexicon", "vader"}

ENVIRONMENT_VARIABLE = "DEFAKETIVE_MODEL_MODE"
DEFAULT_MODE = HAZARD_BADGES

MODEL_MODES = (HAZARD_BADGES, HYBRID_MANUSCRIPT, MANUSCRIPT, LEGACY_HYBRID)

# Which sentiment classifier the badge model uses. The transformer is the
# served model; TF-IDF stays selectable for the unit tests and as the research
# baseline. Neither is ever substituted for the other when one fails to load.
XLMR_SENTIMENT = "xlmr"
TFIDF_SENTIMENT = "tfidf"
SENTIMENT_MODELS = (XLMR_SENTIMENT, TFIDF_SENTIMENT)
SENTIMENT_ENVIRONMENT_VARIABLE = "DEFAKETIVE_SENTIMENT_MODEL"
DEFAULT_SENTIMENT = XLMR_SENTIMENT

# Which hazard checker the badge model uses. "learned" is the shipped one: the
# keyword checker with the learned models adopted on labelling round 2
# (research/PREREGISTRATION_hazard_v4.md) deciding their categories.
# "keywords" is the keyword checker alone, kept for research comparison.
LEARNED_HAZARD = "learned"
KEYWORD_HAZARD = "keywords"
HAZARD_MODELS = (LEARNED_HAZARD, KEYWORD_HAZARD)
HAZARD_ENVIRONMENT_VARIABLE = "DEFAKETIVE_HAZARD_MODEL"
DEFAULT_HAZARD = LEARNED_HAZARD

_CACHE: dict[tuple[str, str | None, str | None], Any] = {}
_CACHE_LOCK = threading.Lock()


class ModelUnavailableError(RuntimeError):
    """Raised when the configured model cannot be loaded.

    Deliberately NOT caught inside this module. The caller decides how to tell
    the user that analysis is unavailable; silently swapping in another model is
    what this design is meant to prevent.
    """


def configured_mode() -> str:
    return (os.environ.get(ENVIRONMENT_VARIABLE) or DEFAULT_MODE).strip().casefold()


def configured_hazard() -> str:
    return (os.environ.get(HAZARD_ENVIRONMENT_VARIABLE) or DEFAULT_HAZARD).strip().casefold()


def configured_sentiment() -> str:
    return (os.environ.get(SENTIMENT_ENVIRONMENT_VARIABLE) or DEFAULT_SENTIMENT).strip().casefold()


def clear_cache() -> None:
    """Forget loaded models, e.g. after the model files were replaced."""
    with _CACHE_LOCK:
        _CACHE.clear()


def load_model(mode: str | None = None, sentiment: str | None = None) -> Any:
    """Return the shared model for `mode`, or raise ModelUnavailableError.

    `sentiment` picks the badge model's sentiment classifier and is ignored by
    the legacy modes, which carry their own.
    """
    selected = (mode or configured_mode()).strip().casefold()
    legacy = selected in MANUSCRIPT_ALIASES or selected in (LEGACY_HYBRID, HYBRID_MANUSCRIPT)
    choice = None if legacy else (sentiment or configured_sentiment()).strip().casefold()
    hazard = None if legacy else configured_hazard()
    key = (selected, choice, hazard)
    with _CACHE_LOCK:
        model = _CACHE.get(key)
        if model is None:
            model = _CACHE[key] = _build(selected, choice, hazard)
    return model


def _hazard_tagger(choice: str, sentiment_model: Any) -> Any:
    from .hazard_model import LexiconHazardTagger

    if choice == KEYWORD_HAZARD:
        return LexiconHazardTagger()
    if choice != LEARNED_HAZARD:
        raise ModelUnavailableError(
            f"Unknown hazard model {choice!r}. Set {HAZARD_ENVIRONMENT_VARIABLE} "
            f"to one of: {', '.join(HAZARD_MODELS)}."
        )
    from .hazard_learned import LearnedHazardScorer

    classifier = getattr(sentiment_model, "trained_classifier", None)
    encoder_sha = (getattr(classifier, "manifest", None) or {}).get("files", {}).get("model.onnx")
    scorer = LearnedHazardScorer(embedding_model_sha256=encoder_sha)
    embedder = getattr(classifier, "embed", None) if scorer.needs_embedding else None
    return LexiconHazardTagger(learned=scorer, embedder=embedder)


def _sentiment_model(choice: str) -> Any:
    from .model import DefaketiveSentimentModel

    if choice == TFIDF_SENTIMENT:
        model = DefaketiveSentimentModel()
        if model.trained_classifier is None:
            # DefaketiveSentimentModel quietly uses VADER when the artifact is
            # missing; here that would label VADER output as TF-IDF.
            raise ModelUnavailableError(
                f"The TF-IDF sentiment model could not be loaded: {model.trained_model_error}"
            )
        return model
    if choice == XLMR_SENTIMENT:
        from .transformer_sentiment import TransformerSentimentClassifier

        return DefaketiveSentimentModel(classifier=TransformerSentimentClassifier())
    raise ModelUnavailableError(
        f"Unknown sentiment model {choice!r}. Set {SENTIMENT_ENVIRONMENT_VARIABLE} "
        f"to one of: {', '.join(SENTIMENT_MODELS)}."
    )


def _build(selected: str, sentiment: str | None, hazard: str | None = None) -> Any:
    try:
        if selected in MANUSCRIPT_ALIASES:
            from .manuscript import ManuscriptRiskModel

            return ManuscriptRiskModel()
        if selected == LEGACY_HYBRID:
            from .marketplace_v2 import MarketplaceV2Model

            return MarketplaceV2Model(allow_unqualified=True)
        if selected == HYBRID_MANUSCRIPT:
            from .manuscript import HybridManuscriptRiskModel

            # The shipped artifact is marked qualified_for_integration=false.
            # Loading it knowingly is the point of keeping this mode: it is the
            # research baseline the badge design is compared against.
            return HybridManuscriptRiskModel(allow_unqualified=True)

        from .hazard_model import HazardBadgeModel

        # The current tagger is the lexicon baseline, which has passed no
        # evaluation gate because there is nothing yet to evaluate it against.
        # It is loaded knowingly and reports qualified_for_integration=false in
        # every summary, so the interface can show that its output is a
        # development result rather than a validated one.
        sentiment_model = _sentiment_model(sentiment)
        return HazardBadgeModel(
            allow_unqualified=True,
            sentiment_model=sentiment_model,
            tagger=_hazard_tagger(hazard or DEFAULT_HAZARD, sentiment_model),
        )
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            f"Could not load the {selected!r} model: {error}"
        ) from error


def model_version(mode: str | None = None, sentiment: str | None = None) -> str:
    """Return the active model's version string.

    Loads the same model `load_model` would, so the version reported can never
    describe a different model from the one that did the analysis.
    """
    return str(load_model(mode, sentiment).model_version)


def is_badge_model(model: Any) -> bool:
    """Whether a model produces badge summaries rather than a 0-100 score."""
    return getattr(model, "schema_version", None) is not None
