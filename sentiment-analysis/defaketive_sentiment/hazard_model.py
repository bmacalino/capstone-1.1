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
from .extra_labels import (
    EXTRA_RULES_REVISION,
    EXTRA_TAGS,
    NOT_AS_DESCRIBED,
    QUALITY,
    not_as_described_spans,
    quality_spans,
)
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
RULES = ("0.2", "0.3", "0.4")
CURRENT_RULES = "0.3"
# What the application serves. The class default stays 0.3 so that research
# code which builds a tagger without naming the rules reproduces 0.3.0.
SHIPPED_RULES = "0.4"

# Rules 0.4 (7 Oct 2026) fix keyword flags that reading the round-6
# disagreements showed to be wrong: "Wala pong sira", "walang ka sira sira",
# "hindi rin kami naka experience ng lag", "sana hindi agad madaling
# malowbat" (negation that did not reach the term); "akala ko sira, lowbat
# lang pala" (a mistaken belief); "pabili ulit pag nasira na" (something that
# has not happened). They change the keyword features, so a learned model
# names the rule set it was trained with and is refused under another.
# Kept short on purpose. Pronouns ("kayo", "ako") and "namn" were tried and
# cancelled real reports on the team's rounds ("wag na kayo bumili ang bilis
# masira"), as did "kung" among the hypothetical words ("paano ito magagamit
# kung sira").
NEGATION_TRANSPARENT_04 = frozenset({"pong", "ho", "hong", "nga", "ngang", "namang", "agad"})
# "hindi rin kami naka experience ng lag", "wala pa naman ako na encounter na defective"
NEVER_EXPERIENCED = re.compile(
    r"(?<!\w)(?:hindi|di|never|wala|walang|no|not)\s+(?:[\w']+\s+){0,5}"
    r"(?:experience[ds]?|naranasan|encounter(?:ed)?|naencounter)\s+(?:(?:ng|na|of|any|a)\s+){0,2}$",
    re.IGNORECASE,
)
MISTAKEN_BELIEF_WORDS = frozenset({"akala", "kala", "inakala", "akla", "thought"})
HYPOTHETICAL_WORDS = frozenset({"pag", "kapag", "if", "once", "baka", "bka"})
# Two more readings of a damage word that are not a fault of the product
# (11 Oct 2026, from two reviews the owner found flagged in the app; revision
# "g1" of rules 0.4, part of the version string).
#   "sira, punit at wasak2 na ung box": a list of damage words that ends at the
#   box. The three-word reach of the packaging rule stops short of it.
#   "tumagal ng 2 years bago nasira": the earlier unit lasted years. Years
#   only: "3 months pa lang sira na" is a report.
KEYWORD_GUARDS_REVISION = "g1"
DAMAGE_LIST_REACH = 8
AFTER_YEARS_OF_USE = re.compile(
    r"(?<!\w)(?:\d+|isang|dalawang|tatlong|apat na|limang|ilang|one|two|three|four|five|several|a few|few)\s*"
    r"(?:years?|yrs?|taon)\s+(?:[\w']+\s+){0,2}(?:bago|before|until)\s+(?:[\w']+\s+){0,2}$",
    re.IGNORECASE,
)
DENIED_REASONS = frozenset({"mistaken_belief", "hypothetical", "packaging_only", "after_years_of_use"})

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
# Words a list of damage may run through on its way to the box (revision g1).
DAMAGE_LIST_WORDS = DAMAGE_TERMS | frozenset(
    {"punit", "wasak", "yupi", "nayupi", "gasgas", "butas", "lukot", "durog", "dent", "dents", "dented", "torn",
     "at", "and", "na", "ung", "yung", "ang", "mga", "pati", "din", "rin", "pa", "po", "the", "tapos", "tas"}
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


# Guide rule 1 for flags a learned model ADDS: a sentence about the box, the
# courier, the seller or their replies is not a product defect. Added 5 Oct
# 2026 after round 3, from the live check; its effect on round 3 is therefore
# reported as post-hoc, not as a held-out result. Keyword flags are unaffected.
SERVICE_SUBJECT = re.compile(
    r"(?<!\w)(?:"
    r"response|responses|respond|responds|responding|reply|replies|replied|"
    r"refund|refunds|refunded|customer service|chat|"
    r"arrived late|late delivery|na deliver|nadeliver|dumating"
    r")(?!\w)",
    re.IGNORECASE,
)
LEARNED_ADD_RULES = "r1b"
NON_PRODUCT_WINDOW = 2

# Counterfeit rules "c1", added 5 Oct 2026 from labelling round 4 (391
# sentences, AI-labelled and owner-adjudicated; used to improve this check,
# never to score it). They run after the keyword result and after any learned
# model, so the keyword features the learned models were trained on are
# unchanged. None reproduces 0.3.0 for the before/after measurement.
COUNTERFEIT_RULES = "c1"
# "scam" was 5 real reports against 75 false flags in round 4: reviewers use
# it for a seller who did not deliver, a missing item or a broken product.
# It no longer makes a counterfeit flag by itself.
WEAK_COUNTERFEIT_TERMS = frozenset({"scam"})
# "hindi legit na 20k mAh": a false size or capacity claim, not a copy.
SPEC_CLAIM_AFTER = re.compile(r"\s*(?:(?:na|ng|ung|yung|ang|sinasabi|the|a)\s+){0,3}\d")
_NOT = r"(?:hindi|di|nd|ndi|not|isn't|isnt|wasn't|wasnt)"
_REAL = r"(?:original|orig|legit|authentic|genuine|tunay)"
_FILLER = r"(?:rin|din|sya|siya|sia|ito|naman|po|talaga|yung|ung|ang|the|a|really|kasing|kcng)"
COUNTERFEIT_PHRASES = tuple(
    re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE)
    for pattern in (
        # "hindi rin original", "nd sia orig", "di ito yung original"
        rf"{_NOT}\s+(?:{_FILLER}\s+){{1,3}}{_REAL}",
        # "not 100% original"
        rf"{_NOT}\s+\d+\s*%?\s*{_REAL}",
        # "no longer original"
        rf"no longer\s+{_REAL}",
        # "not shiny as the original", "not like original", "not the same sa original"
        rf"{_NOT}\s+(?:\w+\s+){{0,2}}(?:as|like|sa)\s+(?:the\s+)?{_REAL}",
        # "anlayo sa original", "ang layu sa orig scent"
        rf"(?:malayo|anlayo|layo|layu)\s+sa\s+{_REAL}",
        # "hindi sya kaamoy ng tunay"
        rf"{_NOT}\s+(?:\w+\s+){{0,2}}(?:kaamoy|katulad|kagaya|kamukha|kapareho)\s+(?:ng|nung|sa)\s+{_REAL}",
        # "copy lang sya"
        r"(?:copy|kopya)\s+lang",
    )
)


# Rules "c2" (guide 1.4, 7 Oct 2026): two more labels, reported beside the three
# badge categories in a separate "extra_tags" field. The badge categories, the
# keyword features and the learned models are untouched; c2 only stops a
# picture mismatch or a disputed number from counting as a counterfeit report,
# because those now have their own label. The class default stays c1 so
# earlier measurements reproduce; the app serves SHIPPED_COUNTERFEIT_RULES.
EXTRA_RULES = "c2"
COUNTERFEIT_RULE_SETS = (COUNTERFEIT_RULES, EXTRA_RULES)
PICTURE_TERM = "iba sa picture"
# Served by the app since 8 Oct 2026, when the report screens gained the two
# labels (research/RESULTS_TABLES.md, Table 11i).
SHIPPED_COUNTERFEIT_RULES = EXTRA_RULES

# Rules c2, second revision (8 Oct 2026): more ways of calling something a
# copy, and three uses of "fake" that are not about the product. Written from
# two thirds of the labelled sentences and checked on the third
# (research/improve_label_rules.py). Used only under c2.
_BRAND = (
    r"(?:samsung|realme|sandisk|jbl|apple|anker|xiaomi|oppo|vivo|sony|lenovo|ugreen|baseus|logitech|kingston|nokia"
    r"|iphone|airpods)"
)
# "hindi samsung", "sd card not sandisk", "di realme to"
BRAND_DENIAL = re.compile(rf"(?<!\w)(?:{_NOT}\s+(?:(?:sya|siya|to|ito|naman|po)\s+)?{_BRAND})(?!\w)", re.IGNORECASE)
# ...unless it is the reviewer's own device: "hindi samsung ang phone ko"
BRAND_OF_OWN_DEVICE = re.compile(r"\s+(?:(?:ang|yung|ung)\s+)?(?:phone|cp|cellphone|gamit|unit|charger)\s+ko(?!\w)", re.IGNORECASE)
COUNTERFEIT_PHRASES_C2 = tuple(
    re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE)
    for pattern in (
        # "i know di to original"
        rf"{_NOT}\s+to\s+{_REAL}",
        # "akala ko legit na orig ang dumating"
        rf"(?:akala|kala)\s*(?:ko|q)\s+(?:pa naman\s+)?{_REAL}",
        # "quite similar to the original", "almost original"
        rf"(?:similar to|almost)\s+(?:the\s+)?{_REAL}",
        r"copy ?cut|copycat|parang xerox|xerox (?:copy )?(?:lang|lng)",
        # "pekeng pekeng product nyo", but not "pekeng reviews"
        r"pekeng(?:\s+pekeng)?\s+(?!reviews?\b|account\b|seller\b|news\b)\w+",
        # a battery filled with sand
        r"buhangin\s+(?:lang\s+)?(?:ang\s+)?laman",
    )
) + (BRAND_DENIAL,)
# "fake reviews", "fake seller", "fake ba to?", "it seems fake", "did i get a fake one?"
FAKE_NOT_THE_PRODUCT = re.compile(
    r"(?<!\w)(?:(?:fake|peke)\s+(?:\w+\s+){0,2}(?:reviews?|seller|account|news)|reviews?\s+(?:\w+\s+){0,2}(?:fake|peke)"
    r"|(?:fake|peke)\s+ba|seems?\s+(?:\w+\s+)?(?:fake|peke)|(?:fake|peke)\s+one\s*\?)(?!\w)",
    re.IGNORECASE,
)


def _merged(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Overlapping matches joined, so one phrase is highlighted once."""
    out: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def without_non_product_words(normalized: str) -> str | None:
    """The sentence with box/courier/seller words and their two neighbours on
    each side removed, or None when it has no such word.

    A first version dropped every sentence naming the seller or the box; on
    round 3 that removed five real reports ("seller not responding ... 5 mins
    lang lowbat na") and no false one. So the sentence is kept when what
    remains still reads as a defect to the learned model.
    """
    hits = [m.span() for pattern in (NON_PRODUCT_SUBJECT, SERVICE_SUBJECT) for m in pattern.finditer(normalized)]
    if not hits:
        return None
    words = [(m.start(), m.end()) for m in re.finditer(r"[\w']+", normalized)]
    drop: set[int] = set()
    for start, end in hits:
        inside = [i for i, (a, b) in enumerate(words) if a < end and start < b]
        if inside:
            drop.update(range(max(0, inside[0] - NON_PRODUCT_WINDOW), min(len(words), inside[-1] + NON_PRODUCT_WINDOW + 1)))
    return " ".join(normalized[a:b] for i, (a, b) in enumerate(words) if i not in drop)


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


def _damage_list_about_packaging(normalized: str, span: tuple[int, int]) -> bool:
    """A run of damage words and joining words that reaches a packaging word."""
    for words in (
        re.findall(r"[\w']+", normalized[span[1]:])[:DAMAGE_LIST_REACH],
        re.findall(r"[\w']+", normalized[: span[0]])[::-1][:DAMAGE_LIST_REACH],
    ):
        for word in words:
            if word in PACKAGING_WORDS:
                return True
            # "wasak2", "yupi2x": written reduplication.
            if re.sub(r"\d+x?$", "", word) not in DAMAGE_LIST_WORDS:
                break
    return False


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
        counterfeit_rules: str | None = COUNTERFEIT_RULES,
    ) -> None:
        if rules not in RULES:
            raise ValueError(f"rules must be one of {RULES}")
        if counterfeit_rules not in (None, *COUNTERFEIT_RULE_SETS):
            raise ValueError(f"counterfeit_rules must be None or one of {COUNTERFEIT_RULE_SETS}")
        self.rules = rules
        # Research switch: measure the effect of the denied-defect block (rules 0.4).
        self.block_denied = True
        # Revision g1 of rules 0.4 (KEYWORD_GUARDS_REVISION); False reproduces 0.4 as first served.
        self.context_guards = True
        self.counterfeit_rules = counterfeit_rules if rules != "0.2" else None
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
        trained_with = (getattr(learned, "manifest", None) or {}).get("keyword_rules", rules if learned is None or not hasattr(learned, "manifest") else "0.3")
        if learned is not None and trained_with != rules:
            raise HazardModelError(
                f"The learned hazard model was trained with keyword rules {trained_with}, not {rules}; its keyword features would differ."
            )
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

    @classmethod
    def for_scorer(cls, scorer: Any, counterfeit_rules: str | None = None, **kwargs: Any) -> "LexiconHazardTagger":
        """A tagger served with the rule sets a learned-model folder was built for.

        `counterfeit_rules` overrides the folder's own choice; the keyword rules
        cannot be overridden, because the model's features depend on them."""
        manifest = getattr(scorer, "manifest", None) or {}
        return cls(
            rules=manifest.get("keyword_rules", "0.3"),
            counterfeit_rules=counterfeit_rules or manifest.get("counterfeit_rules", COUNTERFEIT_RULES),
            learned=scorer,
            **kwargs,
        )

    def _is_negated(self, normalized: str, start: int) -> bool:
        if self.rules == "0.2":
            return text_norm.is_negated(normalized, start)
        if self.rules == "0.4":
            return text_norm.is_negated_adjacent(
                normalized, start, extra_transparent=NEGATION_TRANSPARENT_04
            ) or bool(NEVER_EXPERIENCED.search(normalized[:start]))
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
        if self.rules == "0.4" and self.context_guards and category == DEFECT and term in DAMAGE_TERMS:
            if _damage_list_about_packaging(normalized, span):
                return "packaging_only"
            if AFTER_YEARS_OF_USE.search(normalized[: span[0]]):
                return "after_years_of_use"
        if self.rules == "0.4" and category == DEFECT:
            before = re.findall(r"[\w']+", re.split(r"[.!?;:,]", normalized[: span[0]])[-1])
            after = re.findall(r"[\w']+", normalized[span[1] :])[:2]
            if before and before[-1] in HYPOTHETICAL_WORDS:
                return "hypothetical"
            if any(word in MISTAKEN_BELIEF_WORDS for word in before[-4:]) and "pala" not in after:
                return "mistaken_belief"
        return None

    def _false_friend(self, raw: str) -> str | None:
        """Return the cue class that makes this sentence's heat word harmless."""
        lowered = str(raw).casefold()
        for pattern, cue_class in self.false_friends:
            if pattern.search(lowered):
                return cue_class
        return None

    def tag_sentence(self, raw: str, rating: object = None) -> dict[str, Any]:
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
                repeated: dict[str, Any] = {}
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
                    # "walang ka sira sira": a repeated word shares the first one's negation.
                    if self.rules == "0.4" and not negated and repeated.get("end") is not None:
                        gap = normalized[repeated["end"] : match.start()]
                        negated = bool(repeated["negated"]) and not gap.strip()
                    repeated = {"end": match.end(), "negated": negated}
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
            from .hazard_learned import is_high_star, keyword_features

            # Features come from the keyword result as it stands here, before
            # any learned decision, exactly as in training.
            kw = keyword_features({"terms": matches, "tags": tags})
            embedding = self.embedder(raw) if self.learned.needs_embedding else None
            combine = getattr(self.learned, "combine", {})
            # In a 4-5 star review the model must clear a higher bar to add a flag.
            star_bars = getattr(self.learned, "high_star_threshold", {}) if is_high_star(rating) else {}
            for category, (on, probability) in self.learned.decide(normalized, kw, embedding).items():
                learned[category] = round(probability, 4)
                if combine.get(category) == "or_gated":
                    bar = star_bars.get(category)
                    if on and category not in tags and bar is not None and probability < bar:
                        gated[category] = "below_high_star_bar"
                        continue
                    # Keywords' flag stands; the model may only add one, and
                    # never to a sentence read as praise. The sentiment model
                    # runs only for the sentences where that is in question.
                    if on and category not in tags:
                        # Rules 0.4: the keywords read a defect word here as imagined
                        # or mistaken; the model may not put the flag back. Plain
                        # negation is NOT blocked: on the labelled rounds that cost
                        # three real reports ("wla na, sira na") for two false flags.
                        if self.rules == "0.4" and self.block_denied and any(
                            match["category"] == category and match.get("excluded") in DENIED_REASONS
                            for match in matches
                        ):
                            gated[category] = "denied_by_keywords"
                            continue
                        rest = without_non_product_words(normalized)
                        model = getattr(self.learned, "models", {}).get(category)
                        if rest is not None and model is not None and (
                            not rest
                            or float(model.probability([rest], kw.reshape(1, -1), None)[0])
                            < (model.threshold if bar is None else bar)
                        ):
                            gated[category] = "not_about_the_product"
                            continue
                        label = self.sentiment_label(raw)
                        gated[category] = label
                        if label != "positive":
                            tags.append(category)
                            scores[category] = probability
                            severities[category] = MIN_CONFIDENT_SEVERITY
                            confidence[category] = about_the_product
                    continue
                if combine.get(category) == "and":
                    # A filter: it can drop a keyword flag, never add one. Where the
                    # folder says so, a flag above a given severity is not its to drop.
                    spare = getattr(self.learned, "filter_spares", {}).get(category)
                    serious = spare is not None and severities.get(category, 0) > spare
                    if not on and category in tags and not serious:
                        tags.remove(category)
                        scores.pop(category, None)
                        confidence.pop(category, None)
                        severities.pop(category, None)
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

        described: list[tuple[int, int]] = []
        if self.counterfeit_rules in COUNTERFEIT_RULE_SETS:
            if self.counterfeit_rules == EXTRA_RULES:
                described = self._not_as_described(normalized)
            shown = [self._raw_span(span, spans) for span in described]
            counted = [
                match
                for match in matches
                if match["category"] == COUNTERFEIT and not match["negated"] and not match.get("excluded")
            ]
            phrase = self._counterfeit_phrase(normalized)
            if phrase is not None:
                matches.append(
                    {
                        "ngram": normalized[phrase[0] : phrase[1]],
                        "category": COUNTERFEIT,
                        "severity": MIN_CONFIDENT_SEVERITY,
                        "negated": False,
                        "span": list(self._raw_span(phrase, spans) or ()) or None,
                        "rule": COUNTERFEIT_RULES,
                    }
                )
            # Rules c2: "fake reviews", "fake seller", "fake ba to?" are not about the product.
            elsewhere = self.counterfeit_rules == EXTRA_RULES and bool(FAKE_NOT_THE_PRODUCT.search(normalized))
            strong = [
                match
                for match in counted
                if match["ngram"] not in WEAK_COUNTERFEIT_TERMS
                and not (elsewhere and match["ngram"] in ("fake", "peke"))
                and not ("legit" in match["ngram"] and self._spec_claim(raw, match))
                and not self._described_term(match, shown)
            ]
            if COUNTERFEIT in tags and not strong and phrase is None:
                tags.remove(COUNTERFEIT)
                scores.pop(COUNTERFEIT, None)
                confidence.pop(COUNTERFEIT, None)
                severities.pop(COUNTERFEIT, None)
                suppressed[COUNTERFEIT] = (
                    "scam_alone"
                    if all(m["ngram"] in WEAK_COUNTERFEIT_TERMS for m in counted)
                    else NOT_AS_DESCRIBED
                    if any(self._described_term(m, shown) for m in counted)
                    else "not_about_the_product"
                    if elsewhere
                    else "spec_claim"
                )
            elif COUNTERFEIT not in tags and phrase is not None:
                tags.append(COUNTERFEIT)
                scores[COUNTERFEIT] = float(MIN_CONFIDENT_SEVERITY)
                severities[COUNTERFEIT] = MIN_CONFIDENT_SEVERITY
                confidence[COUNTERFEIT] = about_the_product
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
            **(
                self._extra_labels(raw, normalized, matches, described, spans)
                if self.counterfeit_rules == EXTRA_RULES
                else {}
            ),
        }

    def _not_as_described(self, normalized: str) -> list[tuple[int, int]]:
        """Spans where the reviewer disputes a number, a feature or the photos."""
        return not_as_described_spans(normalized)

    @staticmethod
    def _described_term(match: Mapping[str, Any], shown: Sequence[tuple[int, int] | None]) -> bool:
        """True for a counterfeit match that is really a not-as-described
        report: the picture phrase, or "fake"/"peke" inside a disputed number."""
        if not shown:
            return False
        if match["ngram"] == PICTURE_TERM:
            return True
        # "hindi photochromic ... hindi naman legit": beside a disputed feature or
        # photo, "not legit" is about the claim (second revision of rules c2).
        if "legit" in match["ngram"]:
            return True
        span = match.get("span")
        if match["ngram"] not in ("fake", "peke") or not span:
            return False
        return any(area and area[0] <= span[0] and span[1] <= area[1] for area in shown)

    def _extra_labels(
        self,
        raw: str,
        normalized: str,
        matches: Sequence[Mapping[str, Any]],
        described: Sequence[tuple[int, int]],
        spans: Sequence[text_norm.TokenSpan],
    ) -> dict[str, Any]:
        """The two labels of guide 1.4 for one sentence, with the words behind
        them. Kept apart from "tags" and "terms": the badge categories, the
        keyword features and the learned models never see them."""
        terms: list[dict[str, Any]] = []
        for category, found in ((QUALITY, quality_spans(normalized)), (NOT_AS_DESCRIBED, described)):
            for start, end in _merged(found):
                where = self._raw_span((start, end), spans)
                terms.append(
                    {"ngram": normalized[start:end], "category": category, "span": list(where) if where else None}
                )
        # "hindi legit na 20k mAh": a disputed number the keywords read as "legit".
        # Listed once: skipped when a pattern above already covers the same words.
        covered = [term["span"] for term in terms if term["category"] == NOT_AS_DESCRIBED and term["span"]]
        for match in matches:
            if match["category"] != COUNTERFEIT or "legit" not in match["ngram"] or not self._spec_claim(raw, match):
                continue
            span = match.get("span")
            if span and any(area[0] <= span[0] and span[1] <= area[1] for area in covered):
                continue
            terms.append({"ngram": match["ngram"], "category": NOT_AS_DESCRIBED, "span": span})
        present = {term["category"] for term in terms}
        return {"extra_tags": [tag for tag in EXTRA_TAGS if tag in present], "extra_terms": terms}

    @staticmethod
    def _spec_claim(raw: str, match: Mapping[str, Any]) -> bool:
        """True when the words after a "legit" match are a number: the
        reviewer is disputing a size or capacity, not the brand."""
        span = match.get("span")
        return bool(span and SPEC_CLAIM_AFTER.match(str(raw)[span[1] :]))

    def _counterfeit_phrase(self, normalized: str) -> tuple[int, int] | None:
        """The first comparison-with-the-original phrase that is a report."""
        patterns = COUNTERFEIT_PHRASES + (COUNTERFEIT_PHRASES_C2 if self.counterfeit_rules == EXTRA_RULES else ())
        for pattern in patterns:
            for found in pattern.finditer(normalized):
                if _speculative(normalized, found.start()):
                    continue
                if SPEC_CLAIM_AFTER.match(normalized[found.end() :]):
                    continue
                if pattern is BRAND_DENIAL and BRAND_OF_OWN_DEVICE.match(normalized[found.end() :]):
                    continue
                return found.span()
        return None

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

    def tag_review(self, text: Any, rating: object = None) -> list[dict[str, Any]]:
        tagged = []
        for index, sentence in enumerate(text_norm.segment_sentences(text)):
            result = self.tag_sentence(sentence, rating)
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
        rules = getattr(self.tagger, "rules", "0.3")
        base_version = MODEL_VERSION if rules == "0.3" else f"{MODEL_VERSION}+rules{rules}"
        if rules == "0.4" and getattr(self.tagger, "context_guards", False):
            base_version += KEYWORD_GUARDS_REVISION
        hazard_version = base_version
        if learned is not None:
            gated_rules = f"+{LEARNED_ADD_RULES}" if getattr(learned, "needs_sentiment", False) else ""
            hazard_version = f"{base_version}+{learned.version}{gated_rules}"
        if getattr(self.tagger, "counterfeit_rules", None):
            # The two informational labels' patterns carry their own revision.
            revision = EXTRA_RULES_REVISION if self.tagger.counterfeit_rules == EXTRA_RULES else ""
            hazard_version = f"{hazard_version}+{self.tagger.counterfeit_rules}{revision}"
        self.model_version = (
            f"{hazard_version}+{sentiment_version}" if sentiment_version else hazard_version
        )
        # The typical rates a verdict is compared with are measured with one
        # checker (research/build_reference_baselines.py). `checker_version` is
        # what they are matched against; the baseline's own version joins
        # model_version so saved reports are redone when the line moves.
        self.checker_version = self.model_version
        baselines_version = (getattr(self.aggregator, "baselines", None) or {}).get("version")
        if baselines_version:
            self.model_version = f"{self.model_version}+{baselines_version}"

    def fingerprint(self, review: object) -> str:
        return self.sentiment.fingerprint(review)

    def analyze(self, review: object, rating: object = None) -> dict[str, Any]:
        sentences = self.tagger.tag_review(review, rating)
        review_tags = sorted({tag for item in sentences for tag in item["tags"]})
        # The two informational labels travel beside review_tags, never inside it:
        # review_tags keeps meaning safety, defect or counterfeit wherever it is stored.
        five_labels = getattr(self.tagger, "counterfeit_rules", None) == EXTRA_RULES
        extra_tags = [tag for tag in EXTRA_TAGS if any(tag in (item.get("extra_tags") or ()) for item in sentences)]
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
            **({"review_extra_tags": extra_tags} if five_labels else {}),
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
