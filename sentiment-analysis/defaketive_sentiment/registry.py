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
"""

from __future__ import annotations

import os
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


class ModelUnavailableError(RuntimeError):
    """Raised when the configured model cannot be loaded.

    Deliberately NOT caught inside this module. The caller decides how to tell
    the user that analysis is unavailable; silently swapping in another model is
    what this design is meant to prevent.
    """


def configured_mode() -> str:
    return (os.environ.get(ENVIRONMENT_VARIABLE) or DEFAULT_MODE).strip().casefold()


def load_model(mode: str | None = None) -> Any:
    """Return the model for `mode`, or raise ModelUnavailableError."""
    selected = (mode or configured_mode()).strip().casefold()
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
        return HazardBadgeModel(allow_unqualified=True)
    except ModelUnavailableError:
        raise
    except Exception as error:
        raise ModelUnavailableError(
            f"Could not load the {selected!r} model: {error}"
        ) from error


def model_version(mode: str | None = None) -> str:
    """Return the active model's version string.

    Loads the same model `load_model` would, so the version reported can never
    describe a different model from the one that did the analysis.
    """
    return str(load_model(mode).model_version)


def is_badge_model(model: Any) -> bool:
    """Whether a model produces badge summaries rather than a 0-100 score."""
    return getattr(model, "schema_version", None) is not None
