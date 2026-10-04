"""One canonical text normalizer and sentence segmenter, shared by training,
annotation and runtime.

WHY THIS MODULE EXISTS
Before it, the project had two normalizers that disagreed: `model.normalize()`
ran at runtime, and `product_risk_context.clean_risk_text()` ran inside fitted
pipelines. They mapped different things (bare "d" became "hindi" in one only),
so a review could be preprocessed one way during training and another way during
inference. That class of bug is silent - it costs accuracy without raising
anything.

`clean_risk_text` cannot simply be fixed. It is referenced by module path inside
pickled `RiskTextCleaner` steps in the committed joblib artifacts, so editing it
would retroactively change the behaviour of models that have already been
evaluated and reported. It is frozen instead, and all new work uses this module.

TWO PROFILES
- `legacy_vader` reproduces `model.normalize()` byte for byte, so the existing
  VADER and FiReCS paths keep their exact behaviour.
- `hazard_v1` adds Taglish affix folding and elongation collapse for the hazard
  tagger.

VERSIONING IS LOAD-BEARING
`NORMALIZER_VERSION` and `SEGMENTER_VERSION` are written into the annotation
manifest and the model artifact. Every sentence label is keyed by
`sha256(review_id | segment_index | segment_text)`, so changing either version
invalidates every label collected under the old one. `HazardBadgeModel` refuses
to load an artifact whose normalizer version does not match this module's, which
turns a silent accuracy loss into a startup failure.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, NamedTuple

NORMALIZER_VERSION = "taglish-norm-1.0.0"
SEGMENTER_VERSION = "taglish-seg-1.0.0"

LEGACY_VADER = "legacy_vader"
HAZARD_V1 = "hazard_v1"
PROFILES = (LEGACY_VADER, HAZARD_V1)

_LEXICON_DIR = Path(__file__).with_name("lexicons")
VARIANTS_PATH = _LEXICON_DIR / "taglish_variants_v1.tsv"

# Verb affixes that attach to a root with a hyphen in casual Filipino writing:
# "nag-init", "na-overheat", "i-charge". A hyphen between one of these and a
# root is an affix boundary and is closed up, so the token reaches the variant
# table as one word. Any other word-joining hyphen becomes a space, which is the
# existing behaviour for cases like "poor-quality".
AFFIX_PREFIXES = frozenset(
    {
        "i", "in", "ipa", "ipag", "ka", "ma", "mag", "magka", "magpa", "maka",
        "mang", "mapa", "na", "naka", "nakaka", "nag", "nagka", "nagpa",
        "nang", "napa", "pa", "pag", "pagka", "pang", "pinag", "um",
    }
)

FILIPINO_NEGATORS = frozenset(
    {"ayaw", "di", "hindi", "huwag", "ndi", "wala", "walang", "wag"}
)

# These reviews are Taglish, so negation arrives in either language and often
# both in one sentence. The original runtime check reached VADER's English
# negator set through the analyzer; this module has no analyzer, so the English
# negators are listed here. Without them "it is not mainit" reads as a heat
# report.
ENGLISH_NEGATORS = frozenset(
    {
        "not", "no", "never", "none", "nothing", "neither", "nor", "nowhere",
        "cannot", "cant", "can't", "dont", "don't", "doesnt", "doesn't",
        "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "arent",
        "aren't", "werent", "weren't", "wont", "won't", "wouldnt", "wouldn't",
        "shouldnt", "shouldn't", "couldnt", "couldn't", "hasnt", "hasn't",
        "havent", "haven't", "hadnt", "hadn't", "without", "aint", "ain't",
    }
)

NEGATORS = FILIPINO_NEGATORS | ENGLISH_NEGATORS

# Lexicon terms that already begin with a negator. Without this carve-out the
# negation check would read "hindi gumagana" as a negated risk term and cancel
# a genuine failure report.
DIRECT_FAILURE_PREFIXES = (
    "ayaw ", "di ", "does not ", "hindi ", "no ", "not ", "walang ",
)

_QUOTES = {"’": "'", "`": "'"}
_WORD_JOINING_HYPHEN = re.compile(r"(?<=\w)[‐‑‒–—-](?=\w)")
_WHITESPACE = re.compile(r"\s+")

# Temu's machine translations render a capital I as a lower case L. VADER reads
# the standalone token "l" as positive slang, so "l am disappointed" would score
# as praise. Carried over from the original normalizer.
_TEMU_OCR_L = re.compile(
    r"(?<!\w)l(?=\s+(?:am|bought|can|could|did|do|have|ordered|received|was|will|would)\b)"
)

# Three or more identical LETTERS in a row collapse to one. No English or
# Tagalog word contains a tripled letter, so this needs no exception list, and
# it leaves ordinary doubles ("off", "hello") untouched. Tagalog reduplication
# is syllabic ("bibili", "araw-araw"), never a letter run, so nothing real is
# lost. Runs of two are deliberately not touched.
#
# Digits are excluded, and that exclusion is load-bearing: these reviews are
# full of capacity and price figures, and a naive rule turns "20.000mah" into
# "20.0mah" and "10000" into "10". Punctuation is excluded for the same reason -
# "!!!" carries emphasis and costs nothing to keep, since word n-grams drop it.
_ELONGATION = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)


class TokenSpan(NamedTuple):
    """Where a normalized token came from in the original review text.

    The UI highlights evidence in the text the reviewer actually wrote, not in
    the normalized form, so a matched smoke term has to be traceable back to its
    original characters.
    """

    norm_start: int
    norm_end: int
    raw_start: int
    raw_end: int
    raw_text: str


def _load_variants(path: Path = VARIANTS_PATH) -> dict[str, str]:
    """Load the variant table, skipping comments and the header row."""
    variants: dict[str, str] = {}
    if not path.is_file():
        return variants
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = (line for line in handle if line.strip() and not line.startswith("#"))
        for values in csv.reader(rows, delimiter="\t"):
            if len(values) < 2 or values[0] == "variant":
                continue
            variant, lemma = values[0].strip().casefold(), values[1].strip().casefold()
            if variant and lemma:
                variants[variant] = lemma
    return variants


VARIANTS = _load_variants()


# --------------------------------------------------------------------------
# legacy profile - must stay byte-identical to the original model.normalize()
# --------------------------------------------------------------------------

_LEGACY_SPELLING = {
    "dizapoytedd": "disappointed",
    "hnd": "hindi",
    "hndi": "hindi",
    "ndi": "hindi",
    "wlang": "walang",
    "sobranggg": "sobrang",
}


def _normalize_legacy(text: Any) -> str:
    """Reproduce the original normalizer exactly.

    Do not 'improve' this function. Its output is baked into every committed
    model artifact and every recorded evaluation result.
    """
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("’", "'").replace("`", "'")
    value = _WORD_JOINING_HYPHEN.sub(" ", value)
    value = _WHITESPACE.sub(" ", value).strip().casefold()
    value = _TEMU_OCR_L.sub("i", value)
    parts = []
    for part in re.split(r"(\W+)", value):
        parts.append(_LEGACY_SPELLING.get(part, part))
    return "".join(parts)


# --------------------------------------------------------------------------
# hazard profile
# --------------------------------------------------------------------------


def _structural_clean(text: str) -> tuple[str, list[int]]:
    """Apply the character-level passes, tracking where each character came from.

    Returns the cleaned text and a parallel list giving, for each cleaned
    character, its index in the original string. NFKC and casefold can turn one
    character into several; every output character maps back to the single input
    character that produced it.
    """
    # NFKC, per character so the index map stays exact.
    folded: list[str] = []
    sources: list[int] = []
    for index, character in enumerate(text):
        expanded = unicodedata.normalize("NFKC", character)
        folded.append(expanded)
        sources.extend([index] * len(expanded))
    value = "".join(folded)

    # Quote folding is one-to-one, so the map is unchanged.
    value = "".join(_QUOTES.get(character, character) for character in value)

    # Casefold, again per character.
    cased: list[str] = []
    cased_sources: list[int] = []
    for character, source in zip(value, sources):
        lowered = character.casefold()
        cased.append(lowered)
        cased_sources.extend([source] * len(lowered))
    value = "".join(cased)
    sources = cased_sources

    # Hyphens: close up affix boundaries, otherwise separate the words.
    out: list[str] = []
    out_sources: list[int] = []
    for match in _iter_with_hyphen_decisions(value):
        kind, start, end = match
        if kind == "keep":
            out.append(value[start:end])
            out_sources.extend(sources[start:end])
        elif kind == "drop":
            pass  # affix hyphen: emit nothing, joining the two sides
        else:  # "space"
            out.append(" ")
            out_sources.append(sources[start])
    value = "".join(out)
    sources = out_sources

    # Collapse whitespace runs to a single space and trim.
    collapsed: list[str] = []
    collapsed_sources: list[int] = []
    previous_was_space = True  # leading whitespace is dropped
    for character, source in zip(value, sources):
        if character.isspace():
            if previous_was_space:
                continue
            collapsed.append(" ")
            collapsed_sources.append(source)
            previous_was_space = True
        else:
            collapsed.append(character)
            collapsed_sources.append(source)
            previous_was_space = False
    while collapsed and collapsed[-1] == " ":
        collapsed.pop()
        collapsed_sources.pop()
    return "".join(collapsed), collapsed_sources


def _iter_with_hyphen_decisions(value: str) -> Iterable[tuple[str, int, int]]:
    """Walk the text, deciding for each word-joining hyphen whether to drop it."""
    position = 0
    for match in _WORD_JOINING_HYPHEN.finditer(value):
        if match.start() > position:
            yield ("keep", position, match.start())
        preceding = re.search(r"[\w']+$", value[: match.start()])
        is_affix = preceding is not None and preceding.group(0) in AFFIX_PREFIXES
        yield ("drop" if is_affix else "space", match.start(), match.end())
        position = match.end()
    if position < len(value):
        yield ("keep", position, len(value))


def _normalize_token(token: str) -> str:
    """Collapse elongation, then fold the token to its lemma if one is known."""
    collapsed = _ELONGATION.sub(r"\1", token)
    # Look the variant up on the token's word characters so trailing punctuation
    # does not defeat the match, then reattach whatever was stripped.
    match = re.match(r"^(\W*)([\w']*)(\W*)$", collapsed, flags=re.UNICODE)
    if match is None:
        return collapsed
    prefix, core, suffix = match.groups()
    return f"{prefix}{VARIANTS.get(core, core)}{suffix}"


def normalize_with_spans(text: Any) -> tuple[str, tuple[TokenSpan, ...]]:
    """Normalize with the hazard profile, returning token provenance.

    This is the primary implementation of the hazard profile;
    `normalize_text(value, profile="hazard_v1")` returns its first element, so
    the two can never drift apart.
    """
    value, sources = _structural_clean(str(text or ""))
    value = _TEMU_OCR_L.sub("i", value)
    # The OCR substitution is one character for one character, so the map holds.

    pieces: list[str] = []
    spans: list[TokenSpan] = []
    cursor = 0
    for match in re.finditer(r"\S+", value):
        if match.start() > cursor:
            pieces.append(value[cursor : match.start()])
        token = match.group(0)
        normalized_token = _normalize_token(token)
        norm_start = sum(len(piece) for piece in pieces)
        pieces.append(normalized_token)
        raw_start = sources[match.start()] if match.start() < len(sources) else 0
        raw_end = (sources[match.end() - 1] + 1) if match.end() - 1 < len(sources) else raw_start
        spans.append(
            TokenSpan(
                norm_start=norm_start,
                norm_end=norm_start + len(normalized_token),
                raw_start=raw_start,
                raw_end=raw_end,
                raw_text=str(text or "")[raw_start:raw_end],
            )
        )
        cursor = match.end()
    if cursor < len(value):
        pieces.append(value[cursor:])
    return "".join(pieces), tuple(spans)


def normalize_text(text: Any, profile: str = HAZARD_V1) -> str:
    """Normalize review text under the named profile."""
    if profile == LEGACY_VADER:
        return _normalize_legacy(text)
    if profile == HAZARD_V1:
        return normalize_with_spans(text)[0]
    raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")


# --------------------------------------------------------------------------
# segmentation
# --------------------------------------------------------------------------

_LINE_BREAK = re.compile(r"[\r\n  ]+")
_SEMICOLON = re.compile(r";")

# Only CONTRASTIVE connectives split a segment. The additive ones ("at", "and")
# are deliberately excluded even though the negation-scope rule treats them as
# boundaries: splitting on every "at" would shred ordinary noun lists
# ("charger at cable") into fragments too short to carry an n-gram. Contrast is
# what actually hides a hazard - "maganda ang packaging pero sobrang init" must
# not have its hazard averaged away by the praise in front of it.
_CONTRAST = re.compile(r"(?<!\w)(?:pero|subalit|ngunit|kaso|but|however)(?!\w)", re.IGNORECASE)

# A period only ends a segment when whitespace or the end of the text follows.
# That single condition already protects decimals ("20.000mah") and URLs
# ("shopee.ph/x"), where the period is followed by an alphanumeric character.
_TERMINAL = re.compile(r"[.!?…]+(?=\s|$)")

# Abbreviations whose trailing period is not a segment boundary.
_ABBREVIATIONS = frozenset(
    {
        "pc", "pcs", "no", "nos", "mr", "mrs", "ms", "dr", "etc", "vs", "inc",
        "co", "ltd", "approx", "min", "max", "ea", "pr", "qty", "st", "rd",
    }
)

MAX_SEGMENTS = 40
MIN_SEGMENT_CHARACTERS = 3


def _is_emoji(character: str) -> bool:
    return unicodedata.category(character) == "So"


def _emoji_boundaries(text: str) -> list[int]:
    """Return cut positions after emoji used as terminal punctuation.

    Reviewers routinely end a thought with an emoji instead of a full stop, so a
    run of emoji, or a single emoji followed by a letter, ends a segment.
    """
    cuts: list[int] = []
    index = 0
    length = len(text)
    while index < length:
        if _is_emoji(text[index]):
            start = index
            while index < length and (_is_emoji(text[index]) or text[index].isspace()):
                index += 1
            run = text[start:index].strip()
            trailing = text[index] if index < length else ""
            if len(run) >= 2 or (trailing and trailing.isalnum()):
                cuts.append(index)
        else:
            index += 1
    return cuts


def _terminal_boundaries(text: str) -> list[int]:
    cuts = []
    for match in _TERMINAL.finditer(text):
        preceding = re.search(r"([\w']+)\s*$", text[: match.start()])
        if preceding is not None and preceding.group(1).casefold() in _ABBREVIATIONS:
            continue
        cuts.append(match.end())
    return cuts


def _split_chunk(text: str) -> list[str]:
    """Split one line into segments at every boundary rule."""
    cuts = set(_terminal_boundaries(text))
    cuts.update(match.end() for match in _SEMICOLON.finditer(text))
    cuts.update(_emoji_boundaries(text))
    # Contrastive connectives cut BEFORE the word, so the connective stays with
    # the clause it introduces and the raw text is reproduced exactly.
    cuts.update(match.start() for match in _CONTRAST.finditer(text) if match.start() > 0)

    ordered = sorted(cut for cut in cuts if 0 < cut < len(text))
    segments = []
    previous = 0
    for cut in ordered:
        segments.append(text[previous:cut])
        previous = cut
    segments.append(text[previous:])
    return segments


def segment_sentences(text: Any) -> tuple[str, ...]:
    """Split a review into segments for tagging.

    These are clauses, not linguistic sentences: a contrastive connective starts
    a new segment. The methods section says so explicitly, because it means the
    segment count is higher than a sentence count would be.
    """
    value = str(text or "")
    if not value.strip():
        return ()

    segments: list[str] = []
    for line in _LINE_BREAK.split(value):
        if line.strip():
            segments.extend(_split_chunk(line))

    merged: list[str] = []
    for segment in segments:
        stripped = segment.strip()
        if not stripped or not re.search(r"\w", stripped, re.UNICODE):
            # No word characters: punctuation or a lone emoji. Attach it to the
            # previous segment rather than dropping evidence the reviewer wrote.
            if merged and stripped:
                merged[-1] = f"{merged[-1]} {stripped}".strip()
            continue
        if merged and len(stripped) < MIN_SEGMENT_CHARACTERS:
            merged[-1] = f"{merged[-1]} {stripped}".strip()
            continue
        merged.append(stripped)

    if len(merged) > MAX_SEGMENTS:
        # Defensive: a keyword-stuffed review must not generate unbounded work.
        # The tail is folded into the last kept segment so no text is lost.
        head = merged[: MAX_SEGMENTS - 1]
        head.append(" ".join(merged[MAX_SEGMENTS - 1 :]))
        merged = head
    return tuple(merged)


# --------------------------------------------------------------------------
# negation
# --------------------------------------------------------------------------


def is_negated(normalized: str, start: int, extra_negators: Iterable[str] = ()) -> bool:
    """Report whether the term beginning at `start` sits under a negation.

    Scope is the current clause only, looking back at most three words. Without
    the clause limit, "walang sira at gumagana nang maayos" would read the
    negation as cancelling "gumagana".
    """
    negators = set(NEGATORS)
    negators.update(word.casefold() for word in extra_negators)
    prefix = normalized[:start]
    clause = re.split(
        r"[.!?;:]|(?<!\w)(?:at|and|but|pero|subalit|ngunit)(?!\w)",
        prefix,
        flags=re.IGNORECASE,
    )[-1]
    previous = re.findall(r"[\w']+", clause, flags=re.UNICODE)[-3:]
    return any(word in negators for word in previous)


# Words a negator reaches across in `is_negated_adjacent`: particles, pronouns,
# determiners, and the nouns of "no issues with", "walang problema sa".
NEGATION_TRANSPARENT = frozenset(
    {
        # Filipino particles, linkers, pronouns and demonstratives
        "po", "pa", "na", "nang", "naman", "nman", "nmn", "man", "ng", "ang",
        "yung", "ung", "yong", "iyong", "sa", "si", "ni", "ay", "ba", "din",
        "rin", "ring", "daw", "raw", "lang", "lng", "lamang", "talaga",
        "masyado", "gaano", "ganun", "ganon", "kasi", "ka", "ko", "mo", "siya",
        "sya", "xa", "nya", "niya", "nila", "namin", "natin", "ito", "to", "yan",
        "iyan", "yun", "iyon", "kahit", "pati", "nag", "mag", "ma", "problema",
        # English function words
        "the", "a", "an", "it", "its", "is", "was", "are", "were", "be", "been",
        "being", "very", "really", "too", "so", "even", "at", "all", "ever",
        "any", "anything", "still", "yet", "of", "with", "that", "this", "there",
        "has", "have", "had", "got", "get", "gets", "getting", "single", "more",
        "much", "issue", "issues", "problem", "problems", "sign", "signs", "trace",
        "feel", "feels", "felt", "become", "becomes", "seem", "seems", "run", "runs",
    }
)

# Tagalog linkers. "wala nalabas na usok" (no smoke came out): the negator
# governs the verb, and the linker "na" makes the smoke its subject. One verb
# may therefore sit between a negator and the term when a linker follows it.
NEGATION_LINKERS = frozenset({"na", "ng", "nang", "yung", "ung", "yong", "iyong", "ang"})

# "walang dent o sira": the negator covers every item of an "or" list.
NEGATION_COORDINATORS = frozenset({"o", "or", "nor"})


def is_negated_adjacent(
    normalized: str, start: int, extra_negators: Iterable[str] = ()
) -> bool:
    """Report whether a negator governs the term beginning at `start`.

    Used by the hazard tagger. `is_negated` looks up to three words back, so in
    "not good sira yung isa" the "not" that belongs to "good" also cancels
    "sira", and "hindi samsung fake" loses its "fake". Here a negator reaches
    the term across function words ("hindi po sira", "no issues with
    overheating"), across one verb that a linker ties to the term ("wala
    nalabas na usok"), and across an "or" list ("walang dent o sira"). Any
    other word ends its reach, so "not good sira" is not negated. The clause
    limit is the same as `is_negated`'s. `is_negated` itself is left unchanged
    because the sentiment path and its recorded results depend on it.
    """
    negators = set(NEGATORS)
    negators.update(word.casefold() for word in extra_negators)
    clause = re.split(
        r"[.!?;:]|(?<!\w)(?:at|and|but|pero|subalit|ngunit)(?!\w)",
        normalized[:start],
        flags=re.IGNORECASE,
    )[-1]
    words = re.findall(r"[\w']+", clause, flags=re.UNICODE)
    linker_seen = False
    verb_allowance = 1
    skip_coordinated = False
    for word in reversed(words[-8:]):
        if word in negators:
            return True
        if word in NEGATION_COORDINATORS:
            skip_coordinated = True
            continue
        if skip_coordinated:
            skip_coordinated = False
            continue
        if word in NEGATION_LINKERS:
            linker_seen = True
            continue
        if word in NEGATION_TRANSPARENT:
            continue
        if linker_seen and verb_allowance:
            verb_allowance -= 1
            linker_seen = False
            continue
        return False
    return False


def starts_with_negator(term: str) -> bool:
    """Report whether a lexicon term already contains its own negation."""
    return term.casefold().startswith(DIRECT_FAILURE_PREFIXES)


# --------------------------------------------------------------------------
# sklearn transformer
# --------------------------------------------------------------------------


class TaglishNormalizer:
    """Pipeline step applying `normalize_text`.

    Defined as a plain class calling a module-level function, matching the
    existing `product_risk_context.RiskTextCleaner` pattern, so a fitted pipeline
    pickles by reference and stays loadable.
    """

    def __init__(self, profile: str = HAZARD_V1) -> None:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")
        self.profile = profile

    def fit(self, X: Any, y: Any = None) -> "TaglishNormalizer":  # noqa: N803
        return self

    def transform(self, X: Iterable[Any]) -> list[str]:  # noqa: N803
        return [normalize_text(value, self.profile) for value in X]

    def fit_transform(self, X: Iterable[Any], y: Any = None) -> list[str]:  # noqa: N803
        return self.transform(X)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"profile": self.profile}

    def set_params(self, **params: Any) -> "TaglishNormalizer":
        for key, value in params.items():
            setattr(self, key, value)
        return self
