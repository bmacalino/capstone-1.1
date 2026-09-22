"""Frozen hazard severity table and the severity x likelihood decision matrix.

Severity is assigned by rule from the hazard type. It is NOT predicted by a
model, and the distinction must survive into the write-up. Two reasons:

1. There is not enough data to learn it. Across every labelled row in this
   project there are roughly 100-140 true safety reviews; split across hazard
   subtypes and four severity levels that is well under 20 positives per cell.
2. The published precedent did the same and reported that learning it failed.
   Zaman et al. (2024) used expert-assigned weights per hazard type rather than
   a learned severity model, and their severity-vs-expert correlation was still
   only 0.06/0.19 after calibration, against 0.51 for likelihood. They conclude
   human judgement is required for severity.

Every number here carries an explicit provenance label, because the handoff's
standing instruction is that designed values must never be presented as
published ones. `PUBLISHED` means a paper or official source states it.
`DERIVED` means it is argued from a published value but is not itself published.
`PANEL` means this project's rater panel assigned it and no published weight
exists. Most of this table is `PANEL`. That is the study's gap, and it is
stated as such rather than hidden.

Anchors used in the argument below:
- Zaman et al. (2024), Risk Analysis 44(3):705-723 - expert hazard weights, mean
  of 5 experts: fire 0.9, thermal 0.4, chemical 0.5, choking 1.0, suffocation
  1.0. Electrical and biological hazards were EXCLUDED for insufficient data,
  which is precisely why shock and sparking have no published weight here.
- EU RAPEX / Safety Gate four-level severity scale: 1 reversible minor,
  2 reversible serious, 3 irreversible major, 4 death.
- CPSC power-bank recalls (Casely, Yiisonger, NEWDERY): reports of swelling,
  ignition, property-damage fires, and one fatality.
"""

from __future__ import annotations

TAXONOMY_VERSION = "hazard-taxonomy-1.0.0"

# Provenance labels, used in SEVERITY_BASIS below.
PUBLISHED = "PUBLISHED"
DERIVED = "DERIVED"
PANEL = "PANEL"

# EU RAPEX severity levels, lowest to highest.
SEVERITY_LEVELS = (1, 2, 3, 4)
SEVERITY_DESCRIPTIONS = {
    1: "Reversible minor injury",
    2: "Reversible serious injury",
    3: "Irreversible major injury",
    4: "Risk of death",
}

# Product risk tiers, lowest to highest. Order matters: `max` over tiers and the
# monotonicity property test both rely on TIER_ORDER.
TIER_ORDER = ("Low", "Moderate", "High", "Serious")

# Hazard subtypes, most severe first. A review's subtype is assigned after
# detection by matching the smoke terms that fired; the detector itself learns a
# single merged SAFETY class, because no subtype reaches 20 positives.
SEVERITY = {
    "fire_explosion_swelling": 4,
    "shock_spark_short": 4,
    "overheat_burn": 3,
    "smoke_smell": 3,
    "safety_unspecified": 2,
}

SEVERITY_BASIS = {
    "fire_explosion_swelling": (
        f"[{DERIVED}] Zaman et al. assign fire the second-highest expert weight "
        "(0.9 of 1.0). CPSC power-bank recalls record ignition, swelling, "
        "property-damage fires and one fatality, which maps to RAPEX level 4 "
        "(risk of death). The level itself is this project's assignment."
    ),
    "shock_spark_short": (
        f"[{PANEL}] No published severity weight exists. Zaman et al. explicitly "
        "EXCLUDED electrical hazards for insufficient data, so there is nothing "
        "to derive from. Assigned level 4 by this project's rater panel on the "
        "argument that mains-voltage electric shock can be fatal. This is the "
        "weakest-supported value in the table and the study's stated gap."
    ),
    "overheat_burn": (
        f"[{DERIVED}] Zaman et al. assign thermal hazards 0.4, used here only as "
        "a floor. Contact burns are painful and can scar, which is RAPEX level 3 "
        "(irreversible major) rather than level 4. The level is this project's "
        "assignment."
    ),
    "smoke_smell": (
        f"[{PANEL}] No published weight. Assigned level 3 because smoke and a "
        "burning smell are precursors to ignition rather than injuries in "
        "themselves; rated below fire but above an unspecified concern."
    ),
    "safety_unspecified": (
        f"[{PANEL}] A safety complaint whose hazard type could not be "
        "determined from the review text. Assigned level 2 so it cannot by "
        "itself raise a product to the alert path."
    ),
}

# Likelihood bins, least to most likely. Bin edges are NOT defined here; they
# come from percentiles of the per-product Wilson lower bound within a product
# type, stored in reference_baselines_v1.json. Deriving them from the data
# rather than fixing them here is deliberate: a 0.5% hazard rate means something
# different for a power bank than for a phone case.
LIKELIHOOD_BINS = ("low", "medium", "high")

# Severity x likelihood -> product safety tier. Structurally EU RAPEX, but the
# individual cells are this project's assignment, not published values.
#
#           likelihood:  low        medium      high
SEVERITY_LIKELIHOOD_MATRIX = {
    4: {"low": "High",     "medium": "Serious",  "high": "Serious"},
    3: {"low": "Moderate", "medium": "High",     "high": "Serious"},
    2: {"low": "Low",      "medium": "Moderate", "high": "High"},
    1: {"low": "Low",      "medium": "Low",      "high": "Moderate"},
}

MATRIX_BASIS = (
    f"[{PANEL}] The severity x likelihood structure follows the EU RAPEX / "
    "Safety Gate risk matrix, which is published. The individual cell values "
    "are assigned by this project and are not published. They are constrained "
    "to be monotone - raising either severity or likelihood never lowers the "
    "tier - and are tested for that property. Critics of RAPEX note that its "
    "probability estimates are subjective; the same caveat applies here."
)


class UnknownHazardSubtype(KeyError):
    """Raised when a subtype is not in the frozen taxonomy."""


def severity_for(subtype: str) -> int:
    """Return the frozen severity level for a hazard subtype."""
    try:
        return SEVERITY[subtype]
    except KeyError as exc:
        raise UnknownHazardSubtype(
            f"Unknown hazard subtype {subtype!r}. Known subtypes: {sorted(SEVERITY)}"
        ) from exc


def basis_for(subtype: str) -> str:
    """Return the written provenance for a subtype's severity level.

    Shown in the report's methodology panel so a reader can see that shock
    severity is panel-assigned while fire severity is argued from a paper.
    """
    try:
        return SEVERITY_BASIS[subtype]
    except KeyError as exc:
        raise UnknownHazardSubtype(f"Unknown hazard subtype {subtype!r}") from exc


def tier_for(severity: int, likelihood_bin: str) -> str:
    """Return the safety tier for a severity level and likelihood bin."""
    if severity not in SEVERITY_LIKELIHOOD_MATRIX:
        raise ValueError(
            f"severity must be one of {sorted(SEVERITY_LIKELIHOOD_MATRIX)}, got {severity!r}"
        )
    if likelihood_bin not in LIKELIHOOD_BINS:
        raise ValueError(
            f"likelihood_bin must be one of {LIKELIHOOD_BINS}, got {likelihood_bin!r}"
        )
    return SEVERITY_LIKELIHOOD_MATRIX[severity][likelihood_bin]


def higher_tier(first: str, second: str) -> str:
    """Return whichever tier is higher, used by the alert override."""
    for tier in (first, second):
        if tier not in TIER_ORDER:
            raise ValueError(f"tier must be one of {TIER_ORDER}, got {tier!r}")
    return max(first, second, key=TIER_ORDER.index)
