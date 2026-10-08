"""Build the Round 1 sentence-labelling queue for the three annotators.

Pulls candidate reviews from every source the project owns, splits them into
sentences with the frozen segmenter, and writes one worksheet per annotator
in the column format that research/HAZARD_SENTENCE_TAGGING_GUIDE.md describes.

The sampling design is the point of this script, so it is stated here rather
than buried:

  seed_hit    reviews matching research/lexicons/mining_seeds_v1.tsv.
              Enrichment only. Safety and counterfeit reports are roughly
              0.6% of reviews, so a purely random pool would yield almost no
              positives inside any budget a three-person team can afford.
  low_star    1-3 star reviews (or FiReCS negative) that did NOT match a
              seed. This is the anti-circularity stratum: it finds hazards
              phrased in words the seed list does not contain, which is the
              recall the seed list cannot measure about itself.
  random_all  a uniform random sample across every star and every source.
              This is the ONLY stratum on which prevalence and recall may be
              reported. Rates measured on the other two are inflated by
              construction.

Every row carries its stratum and a sampling weight so the three can be
recombined into an unbiased estimate later. The worksheets given to the
annotators deliberately omit both columns: an annotator who can see that a
row was selected for containing "nasunog" is no longer labelling blind.

ROUND 2 (--round 2) is the held-out test round for the hazard checker. It
excludes every review already used in an earlier round (queue and practice
set), and every row is blind: the checker being tested is what produces the
pre-labels, so showing them would pull the test labels towards its answers.

Usage:
    python research/build_labelling_queue.py --round 2 --annotators A1,A2,A3
    python research/build_labelling_queue.py --annotators AAA,BBB,CCC
    python research/build_labelling_queue.py --dry-run
    python research/build_labelling_queue.py --shopee-hf path/to/shopee.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for _path in (ROOT, SENTIMENT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from defaketive_sentiment.text_norm import (  # noqa: E402
    SEGMENTER_VERSION,
    normalize_text,
    segment_sentences,
)

QUEUE_VERSION = "labelling-queue-1.0.0"
SEEDS_VERSION = "mining-seeds-1.0.0"
GUIDE_VERSION = "1.0"

SEEDS_PATH = ROOT / "research" / "lexicons" / "mining_seeds_v1.tsv"
OUTPUT_DIR = ROOT / "outputs" / "labelling_queue_round_1"
OUTPUTS = ROOT / "outputs"

SCRAPE_DIRS = (ROOT / "research" / "data", ROOT)
SCRAPE_GLOBS = ("lazada_*.json", "temu_*.json", "shopee_*.json")

FIRECS_DIR = ROOT.parent
FIRECS_FILES = ("FiReCS_train_set.csv", "FiReCS_test_set.csv")
FIRECS_NEGATIVE_LABELS = {"0", "0.0", "negative"}

# Anything whose text appears in one of these has already been through an
# annotation round. Re-labelling it would contaminate a frozen split.
EXCLUSION_FILES = (
    ROOT / "outputs" / "lazada_gold_test_v1" / "lazada_round1_all_adjudicated.csv",
    ROOT / "outputs" / "lazada_gold_test_v1" / "lazada_round1_gold_test_v1.csv",
    ROOT / "FiReCS_Risk_Annotation_Round_1_Annotated.csv",
    ROOT / "complete_manual_marketplace_reviews.csv",
)
TEXT_COLUMNS = ("review_text", "review", "text", "content")

STRATA = ("seed_hit", "low_star", "random_all")
# Round 3 adds a stratum, sampled first: reviews with at least one sentence on
# which keyword checker 0.3.0 and the round-1 learned defect model disagree.
# That is where a learned defect model gains or loses against the keywords, and
# a uniform sample holds too few such sentences to measure it.
DISAGREEMENT = "defect_disagreement"
# Round 5 (research/PREREGISTRATION_final_v6.md): reviews with a sentence the
# counterfeit check flags or that uses an authenticity word, and reviews
# where the shipped checker and the retrained candidate disagree.
COUNTERFEIT_FOCUS = "counterfeit_focus"
MODEL_DISAGREEMENT = "model_disagreement"
# Round 7: reviews that talk about quality, size, capacity or the photos, for
# the two labels added in guide 1.4. A broad word test, not the checker's rules.
ASPECT_FOCUS = "aspect_focus"
ALL_STRATA = (COUNTERFEIT_FOCUS, ASPECT_FOCUS, MODEL_DISAGREEMENT, DISAGREEMENT) + STRATA
ASPECT = re.compile(
    r"(?<!\w)(\d+\s*(k\s*)?(mah|gb|tb|inch|inches|watts?|meters?)|picture|pic|photo|litrato|description|advertis\w*|"
    r"misleading|quality|mahina|hina|weak|manipis|marupok|flimsy|dents?|gasgas|scratch\w*|yupi|sabog|maluwag|lumuwag|"
    r"low class|matibay|durable)(?!\w)",
    re.IGNORECASE,
)
AUTHENTICITY = re.compile(
    r"(?<!\w)(fake|peke|orig|original|authentic|legit|genuine|replica|imitation|class a|counterfeit|"
    r"copy|kopya|oem|totoo|tunay|scam|budol)(?!\w)",
    re.IGNORECASE,
)
CANDIDATE_MODEL_DIR = SENTIMENT_ROOT / "models" / "hazard_classifier_v3"
DISAGREEMENT_MODEL = ROOT / "research" / "models" / "defect_tk_round1.joblib"
DEFAULT_BUDGET = {"seed_hit": 380, "low_star": 240, "random_all": 280}
DEFAULT_BLIND_RATE = 0.15
DEFAULT_PRACTICE_SIZE = 30
DEFAULT_SEED = 20260922

# Per-round defaults. Round 1 keeps its original settings (its queue
# regenerates exactly only while its source files are unchanged). Round 2 is
# the test round: about 600 sentences, all blind, a fresh seed.
ROUND_DEFAULTS = {
    1: {
        "budget": DEFAULT_BUDGET,
        "blind_rate": DEFAULT_BLIND_RATE,
        "seed": DEFAULT_SEED,
        "guide": GUIDE_VERSION,
    },
    2: {
        "budget": {"seed_hit": 260, "low_star": 160, "random_all": 180},
        "blind_rate": 1.0,
        "seed": 20260929,
        # 1.1 adds worked examples only; see the guide's version note.
        "guide": "1.1",
    },
    3: {
        "strata": (DISAGREEMENT,) + STRATA,
        "budget": {DISAGREEMENT: 120, "seed_hit": 220, "low_star": 130, "random_all": 130},
        "blind_rate": 1.0,
        "seed": 20261003,
        "guide": "1.1",
    },
    # The final test round: only reviews no earlier round and no earlier
    # unlabelled check could have touched (the public Shopee set).
    5: {
        "strata": (COUNTERFEIT_FOCUS, MODEL_DISAGREEMENT) + STRATA,
        "budget": {COUNTERFEIT_FOCUS: 280, MODEL_DISAGREEMENT: 110, "seed_hit": 120, "low_star": 45, "random_all": 45},
        "blind_rate": 1.0,
        "seed": 20261006,
        "guide": "1.2",
        "cohorts": ("shopee_hf",),
    },
    # Gadget-only (research/gadget_scope.py): unused own scrapes of electronics
    # listings plus the gadget reviews of the Shopee set. The own scrapes were
    # seen by earlier unlabelled spot checks, so this is development data.
    6: {
        "strata": (COUNTERFEIT_FOCUS, MODEL_DISAGREEMENT) + STRATA,
        "budget": {COUNTERFEIT_FOCUS: 280, MODEL_DISAGREEMENT: 110, "seed_hit": 120, "low_star": 45, "random_all": 45},
        "blind_rate": 1.0,
        "seed": 20261007,
        "guide": "1.2",
        "cohorts": ("own_scrape", "shopee_hf"),
        "scope": "gadget",
    },
    # More gadget-only training data, weighted towards the two new labels.
    7: {
        "strata": (COUNTERFEIT_FOCUS, ASPECT_FOCUS) + STRATA,
        "budget": {COUNTERFEIT_FOCUS: 150, ASPECT_FOCUS: 250, "seed_hit": 90, "low_star": 80, "random_all": 30},
        "blind_rate": 1.0,
        "seed": 20261008,
        "guide": "1.4",
        "cohorts": ("own_scrape", "shopee_hf"),
        "scope": "gadget",
    },
}


def round_strata(round_number: int) -> tuple[str, ...]:
    return tuple(ROUND_DEFAULTS[round_number].get("strata", STRATA))


def defect_disagreement_predicate():
    """True for a review with a sentence where keywords and the round-1 learned
    defect model disagree about a defect."""
    import joblib

    from defaketive_sentiment.hazard_learned import keyword_features
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    model = joblib.load(DISAGREEMENT_MODEL)
    # Keyword features as the app computes them: before the counterfeit rules.
    tagger = LexiconHazardTagger(counterfeit_rules=None)

    def disagrees(text: str) -> bool:
        for sentence in sentences_for(text):
            tagged = tagger.tag_sentence(sentence)
            keywords = "defect" in tagged["tags"]
            p = model.probability([tagged["text_normalized"]], keyword_features(tagged).reshape(1, -1), None)[0]
            if (p >= model.threshold) != keywords:
                return True
        return False

    return disagrees


def counterfeit_focus_predicate():
    """True for a review with a sentence either counterfeit check (0.3.0 or
    c1) flags, or that uses an authenticity word."""
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    old, new = LexiconHazardTagger(counterfeit_rules=None), LexiconHazardTagger()

    def focus(text: str) -> bool:
        for sentence in sentences_for(text):
            if AUTHENTICITY.search(sentence):
                return True
            if "counterfeit" in old.tag_sentence(sentence)["tags"] or "counterfeit" in new.tag_sentence(sentence)["tags"]:
                return True
        return False

    return focus


def aspect_focus_predicate():
    """True for a review with a sentence about quality, size, capacity or photos."""

    def focus(text: str) -> bool:
        return any(ASPECT.search(sentence) for sentence in sentences_for(text))

    return focus


def model_disagreement_predicate():
    """True for a review with a sentence where the shipped checker and the
    retrained candidate differ on safety or defect."""
    from defaketive_sentiment.hazard_learned import MODELS_ROOT, LearnedHazardScorer
    from defaketive_sentiment.hazard_model import LexiconHazardTagger
    from defaketive_sentiment.model import DefaketiveSentimentModel
    from defaketive_sentiment.transformer_sentiment import TransformerSentimentClassifier

    classifier = TransformerSentimentClassifier()
    sentiment = DefaketiveSentimentModel(classifier=classifier)
    seen: dict[str, str] = {}

    def label(text: str) -> str:
        if text not in seen:
            seen[text] = sentiment.analyze(text)["label"]
        return seen[text]

    sha = classifier.manifest["files"]["model.onnx"]
    taggers = [
        LexiconHazardTagger(learned=LearnedHazardScorer(folder, sentiment_model_sha256=sha), sentiment_label=label)
        # Pinned: the checker that shipped when rounds 5 and 6 were drawn.
        for folder in (MODELS_ROOT / "hazard_classifier_v2", CANDIDATE_MODEL_DIR)
    ]

    def disagrees(text: str) -> bool:
        for sentence in sentences_for(text):
            a, b = (set(tagger.tag_sentence(sentence)["tags"]) & {"safety", "defect"} for tagger in taggers)
            if a != b:
                return True
        return False

    return disagrees


# Files of an earlier round that hold review text, and the column holding it.
PRIOR_ROUND_FILES = ("queue_master.csv", "practice_set_master.csv")
PRIOR_ROUND_TEXT_COLUMN = "review_text_raw"

# Most severe first: when a sentence matches two safety subtypes the guide
# says to record the more severe one, and hazard_taxonomy orders them this way.
SUBTYPE_PRIORITY = (
    "fire_explosion_swelling",
    "shock_spark_short",
    "overheat_burn",
    "smoke_smell",
    "safety_unspecified",
)

# Columns the annotators actually see. Order matters: it is the reading order
# in the spreadsheet, so the sentence sits next to the boxes to be filled.
WORKSHEET_FIELDS = [
    "row_id",
    "review_id",
    "rating",
    "sentence_index",
    "sentence_text",
    "review_text_raw",
    "is_blind",
    "llm_safety",
    "llm_defect",
    "llm_counterfeit",
    "llm_hazard_subtype",
    "verifier_id",
    "verifier_action",
    "human_safety",
    "human_defect",
    "human_counterfeit",
    "human_hazard_subtype",
    "human_negated",
    "human_notes",
    "seconds_spent",
]

# The master keeps everything, including the columns withheld above.
MASTER_FIELDS = WORKSHEET_FIELDS + [
    "cohort",
    "stratum",
    "sampling_weight",
    "source_file",
    "product_id",
    "product_name",
    "category",
    "matched_terms",
    "blocked_terms",
    "assigned_to",
    "adjudicator",
    "is_practice",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_key(value: str) -> str:
    """Identity of a review for de-duplication and exclusion checks."""
    normalized = re.sub(r"\s+", " ", normalize_text(value or "")).strip().casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# seeds
# --------------------------------------------------------------------------


def _normalized_term(term: str) -> str:
    """Put a seed term through the same normalizer the review text goes through.

    The Taglish normalizer strips affixes, so "nasunog" becomes "sunog" and
    "umiinit" becomes "init" in the text being searched. A seed term written
    the way a person writes it would therefore never match. Normalizing the
    term at load time keeps the TSV readable by the annotators and correct at
    the same time.
    """
    return re.sub(r"\s+", " ", normalize_text(term)).strip().casefold()


def load_seeds(path: Path = SEEDS_PATH) -> tuple[list[dict[str, str]], list[str]]:
    """Return (seed rows, block phrases), normalized for matching."""
    seeds: list[dict[str, str]] = []
    blocks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4 or parts[0] == "term":
                continue
            term, category, subtype, language = (p.strip() for p in parts[:4])
            match_form = _normalized_term(term)
            if not match_form:
                continue
            if category == "BLOCK":
                blocks.append(match_form)
                continue
            seeds.append(
                {
                    "term": term.casefold(),
                    "match": match_form,
                    "category": category,
                    "subtype": subtype,
                    "language": language,
                }
            )
    if not seeds:
        raise ValueError(f"No seed terms loaded from {path}")
    # Affix stripping collapses some terms onto each other ("nasunog" and
    # "masunog" both become "sunog"). Keep one entry per match form.
    deduped: dict[str, dict[str, str]] = {}
    for row in seeds:
        deduped.setdefault(row["match"], row)
    seeds = list(deduped.values())
    # Longest first so "sobrang init" wins over "init" when both could fire.
    seeds.sort(key=lambda row: len(row["match"]), reverse=True)
    blocks = sorted(set(blocks), key=len, reverse=True)
    return seeds, blocks


def _boundary_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


class SeedMatcher:
    """Matches seed terms on normalized text, after masking false friends."""

    def __init__(self, seeds: Sequence[dict[str, str]], blocks: Sequence[str]) -> None:
        self._seeds = [(row, _boundary_pattern(row["match"])) for row in seeds]
        self._blocks = [(phrase, _boundary_pattern(phrase)) for phrase in blocks]

    def mask_blocked(self, normalized: str) -> tuple[str, list[str]]:
        hits: list[str] = []
        masked = normalized
        for phrase, pattern in self._blocks:
            if pattern.search(masked):
                hits.append(phrase)
                # Replace with spaces so offsets stay put and nothing new joins up.
                masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
        return masked, hits

    def match(self, text: str) -> dict[str, Any]:
        normalized = _normalized_term(text or "")
        masked, blocked = self.mask_blocked(normalized)
        categories: set[str] = set()
        subtypes: set[str] = set()
        terms: list[str] = []
        for row, pattern in self._seeds:
            if pattern.search(masked):
                terms.append(row["term"])
                categories.add(row["category"])
                if row["category"] == "safety" and row["subtype"] != "-":
                    subtypes.add(row["subtype"])
        subtype = ""
        for candidate in SUBTYPE_PRIORITY:
            if candidate in subtypes:
                subtype = candidate
                break
        if "safety" in categories and not subtype:
            subtype = "safety_unspecified"
        return {
            "categories": categories,
            "subtype": subtype,
            "terms": terms,
            "blocked": blocked,
        }


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------


def _product_id(product: dict[str, Any]) -> str:
    link = str(product.get("link") or "")
    match = re.search(r"[-/]i(\d{6,})", link)
    if match:
        return match.group(1)
    if link:
        return hashlib.sha1(link.encode("utf-8")).hexdigest()[:12]
    return ""


def _iter_scrape_files() -> Iterator[Path]:
    seen: set[Path] = set()
    for directory in SCRAPE_DIRS:
        if not directory.is_dir():
            continue
        for pattern in SCRAPE_GLOBS:
            for path in sorted(directory.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                yield path


def load_own_scrapes() -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    used: list[Path] = []
    for path in _iter_scrape_files():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        if not isinstance(payload, list):
            continue
        found = False
        for product in payload:
            if not isinstance(product, dict):
                continue
            comments = product.get("comments") or []
            if not isinstance(comments, list) or not comments:
                continue
            product_id = _product_id(product)
            research = product.get("research") or {}
            category = ""
            if isinstance(research, dict):
                category = str(research.get("stratum") or "")
            for comment in comments:
                if not isinstance(comment, dict):
                    continue
                content = str(comment.get("content") or "").strip()
                if not content:
                    continue
                found = True
                rating_raw = comment.get("rating")
                try:
                    rating: int | str = int(rating_raw)
                except (TypeError, ValueError):
                    rating = ""
                rows.append(
                    {
                        "cohort": "own_scrape",
                        "platform": str(product.get("platform") or ""),
                        "source_file": path.name,
                        "product_id": product_id,
                        "product_name": str(product.get("name") or ""),
                        "category": category,
                        "rating": rating,
                        "review_text": content,
                    }
                )
        if found:
            used.append(path)
    return rows, used


def load_firecs() -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    used: list[Path] = []
    for name in FIRECS_FILES:
        path = FIRECS_DIR / name
        if not path.is_file():
            continue
        used.append(path)
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for record in csv.DictReader(handle):
                text = str(record.get("review") or "").strip()
                if not text:
                    continue
                label = str(record.get("label") or "").strip()
                rows.append(
                    {
                        "cohort": "firecs",
                        "platform": "",
                        "source_file": name,
                        # FiReCS ships no product identity, so these rows can
                        # train the sentence tagger but can never take part in
                        # a product-level rate.
                        "product_id": "",
                        "product_name": "",
                        "category": "",
                        "rating": "",
                        "sentiment_label": label,
                        "review_text": text,
                    }
                )
    return rows, used


def load_shopee_hf(path: Path | None) -> tuple[list[dict[str, Any]], list[Path]]:
    """Load a local copy of scaredmeow/shopee-reviews-tl-stars (columns text,label)."""
    if path is None:
        return [], []
    # A folder holds the dataset's splits (train/validation/test .csv).
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files or not all(item.is_file() for item in files):
        raise FileNotFoundError(f"--shopee-hf given but not found: {path}")
    rows: list[dict[str, Any]] = []
    for item in files:
        rows += _shopee_hf_rows(item)
    return rows, files


def _shopee_hf_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for record in csv.DictReader(handle):
            text = str(record.get("text") or record.get("review") or "").strip()
            if not text:
                continue
            label = str(record.get("label") or "").strip()
            try:
                # The HF labels are 0-4 for 1-5 stars.
                rating: int | str = int(float(label)) + 1
            except (TypeError, ValueError):
                rating = ""
            rows.append(
                {
                    "cohort": "shopee_hf",
                    "platform": "shopee",
                    "source_file": path.name,
                    "product_id": "",
                    "product_name": "",
                    "category": "",
                    "rating": rating,
                    "review_text": text,
                }
            )
    return rows


def round_dir(round_number: int) -> Path:
    return OUTPUTS / f"labelling_queue_round_{round_number}"


def load_prior_rounds(round_number: int) -> tuple[set[str], list[Path]]:
    """Every review already put in front of annotators in an earlier round."""
    keys: set[str] = set()
    used: list[Path] = []
    for earlier in range(1, round_number):
        for name in PRIOR_ROUND_FILES:
            path = round_dir(earlier) / name
            if not path.is_file():
                if name == "queue_master.csv":
                    raise SystemExit(f"Round {earlier} queue not found at {path}; cannot exclude it.")
                continue
            used.append(path)
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                for record in csv.DictReader(handle):
                    value = str(record.get(PRIOR_ROUND_TEXT_COLUMN) or "").strip()
                    if value:
                        keys.add(text_key(value))
    return keys, used


def load_exclusions() -> tuple[set[str], list[Path]]:
    keys: set[str] = set()
    used: list[Path] = []
    for path in EXCLUSION_FILES:
        if not path.is_file():
            continue
        used.append(path)
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = [c for c in (reader.fieldnames or []) if c in TEXT_COLUMNS]
            if not columns:
                continue
            for record in reader:
                for column in columns:
                    value = str(record.get(column) or "").strip()
                    if value:
                        keys.add(text_key(value))
    return keys, used


# --------------------------------------------------------------------------
# pool assembly
# --------------------------------------------------------------------------


def is_low_star(row: dict[str, Any]) -> bool:
    rating = row.get("rating")
    if isinstance(rating, int) and 1 <= rating <= 3:
        return True
    return str(row.get("sentiment_label") or "") in FIRECS_NEGATIVE_LABELS


def build_pool(
    rows: Iterable[dict[str, Any]],
    matcher: SeedMatcher,
    excluded: set[str],
    first_stratum: tuple[str, Any] | list[tuple[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """De-duplicate, drop excluded rows, and assign every survivor a stratum."""
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {"total": 0, "empty": 0, "duplicate": 0, "excluded": 0}
    # Checked in order, before the seed list; the first that holds wins.
    firsts = [first_stratum] if isinstance(first_stratum, tuple) else list(first_stratum or [])
    for row in rows:
        counts["total"] += 1
        text = str(row.get("review_text") or "").strip()
        if len(text) < 3 or not re.search(r"[A-Za-z]", text):
            counts["empty"] += 1
            continue
        key = text_key(text)
        if key in excluded:
            counts["excluded"] += 1
            continue
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        hit = matcher.match(text)
        record = dict(row)
        record["review_text"] = text
        record["text_key"] = key
        record["review_id"] = f"{row['cohort'][:4]}-{key[:12]}"
        record["matched_terms"] = hit["terms"]
        record["blocked_terms"] = hit["blocked"]
        record["seed_categories"] = hit["categories"]
        record["seed_subtype"] = hit["subtype"]
        special = next((name for name, holds in firsts if holds(text)), None)
        if special is not None:
            record["stratum"] = special
        elif hit["terms"]:
            record["stratum"] = "seed_hit"
        elif is_low_star(record):
            record["stratum"] = "low_star"
        else:
            record["stratum"] = "random_all"
        pool.append(record)
    return pool, counts


def sentences_for(text: str) -> list[str]:
    out = []
    for sentence in segment_sentences(text):
        cleaned = sentence.strip()
        if len(cleaned) < 3 or not re.search(r"[A-Za-z]", cleaned):
            continue
        out.append(cleaned)
    return out


SEED_CATEGORY_ORDER = ("safety", "counterfeit", "defect")


def _seed_band(row: dict[str, Any]) -> str:
    """Which category quota a seed-hit review is drawn against.

    A review matching both safety and defect is counted against safety, the
    rarer of the two. It still carries both labels for the annotator.
    """
    categories = row.get("seed_categories") or set()
    for category in SEED_CATEGORY_ORDER:
        if category in categories:
            return category
    return "defect"


def _take_until_budget(
    order: list[dict[str, Any]], sentence_budget: int
) -> tuple[list[dict[str, Any]], int]:
    """Take whole reviews in the given order until the budget is reached."""
    chosen: list[dict[str, Any]] = []
    used = 0
    for row in order:
        sentences = sentences_for(row["review_text"])
        if not sentences:
            continue
        if used and used + len(sentences) > sentence_budget:
            continue
        row = dict(row)
        row["sentences"] = sentences
        chosen.append(row)
        used += len(sentences)
        if used >= sentence_budget:
            break
    return chosen, used


def sample_stratum(
    population: list[dict[str, Any]],
    sentence_budget: int,
    rng: random.Random,
    prioritise_rare: bool = False,
) -> list[dict[str, Any]]:
    """Take whole reviews at random until the sentence budget is reached.

    Reviews are never split across the budget line: a half-labelled review
    cannot contribute to a per-review rollup.

    With prioritise_rare, each of safety, counterfeit and defect gets its own
    share of the budget rather than competing for it. Equal shares, not
    proportional ones: this pool runs about 1,500 defect candidates to 87
    counterfeit and 67 safety, so a proportional draw would hand the
    annotators a queue of broken chargers and almost no hazards. Whatever a
    category cannot fill rolls over to the others, so the budget is still
    spent in full. The draw remains random inside each category.

    This is a stratified draw and the manifest says so. Prevalence must not be
    read off it.
    """
    if sentence_budget <= 0 or not population:
        return []
    order = list(population)
    rng.shuffle(order)
    if not prioritise_rare:
        return _take_until_budget(order, sentence_budget)[0]

    bands: dict[str, list[dict[str, Any]]] = {c: [] for c in SEED_CATEGORY_ORDER}
    for row in order:
        bands[_seed_band(row)].append(row)

    chosen: list[dict[str, Any]] = []
    taken_ids: set[str] = set()
    remaining = sentence_budget
    for index, category in enumerate(SEED_CATEGORY_ORDER):
        left = len(SEED_CATEGORY_ORDER) - index
        share = remaining if left == 1 else remaining // left
        picked, used = _take_until_budget(
            [r for r in bands[category] if r["review_id"] not in taken_ids], share
        )
        chosen.extend(picked)
        taken_ids.update(r["review_id"] for r in picked)
        remaining -= used
        if remaining <= 0:
            break
    return chosen


def prelabel(sentence: str, matcher: SeedMatcher) -> dict[str, str]:
    """The 'computer guess' the annotators check.

    This is the seed list applied at sentence level, and nothing more. It is
    recorded as prelabel_source in the manifest so no reader mistakes it for
    a trained model's prediction.
    """
    hit = matcher.match(sentence)
    categories = hit["categories"]
    return {
        "llm_safety": "yes" if "safety" in categories else "no",
        "llm_defect": "yes" if "defect" in categories else "no",
        "llm_counterfeit": "yes" if "counterfeit" in categories else "no",
        "llm_hazard_subtype": hit["subtype"],
        "matched_terms": "|".join(hit["terms"]),
        "blocked_terms": "|".join(hit["blocked"]),
    }


def blank_worksheet_fields() -> dict[str, str]:
    return {
        "verifier_id": "",
        "verifier_action": "",
        "human_safety": "",
        "human_defect": "",
        "human_counterfeit": "",
        "human_hazard_subtype": "",
        "human_negated": "",
        "human_notes": "",
        "seconds_spent": "",
    }


def expand_rows(
    sampled: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    matcher: SeedMatcher,
    rng: random.Random,
    blind_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # In the order the strata were sampled (STRATA for rounds 1 and 2, so
    # their queues regenerate unchanged).
    for stratum in sampled:
        for review in sampled.get(stratum, []):
            for index, sentence in enumerate(review["sentences"], start=1):
                guess = prelabel(sentence, matcher)
                is_blind = rng.random() < blind_rate
                row: dict[str, Any] = {
                    "row_id": "",
                    "review_id": review["review_id"],
                    "rating": review.get("rating", ""),
                    "sentence_index": index,
                    "sentence_text": sentence,
                    "review_text_raw": review["review_text"],
                    "is_blind": "yes" if is_blind else "no",
                    "llm_safety": "" if is_blind else guess["llm_safety"],
                    "llm_defect": "" if is_blind else guess["llm_defect"],
                    "llm_counterfeit": "" if is_blind else guess["llm_counterfeit"],
                    "llm_hazard_subtype": "" if is_blind else guess["llm_hazard_subtype"],
                    "cohort": review["cohort"],
                    "stratum": stratum,
                    "sampling_weight": f"{weights.get(stratum, 0.0):.4f}",
                    "source_file": review.get("source_file", ""),
                    "product_id": review.get("product_id", ""),
                    "product_name": review.get("product_name", ""),
                    "category": review.get("category", ""),
                    "matched_terms": guess["matched_terms"],
                    "blocked_terms": guess["blocked_terms"],
                    "assigned_to": "",
                    "adjudicator": "",
                    "is_practice": "no",
                }
                row.update(blank_worksheet_fields())
                rows.append(row)
    return rows


def build_practice_set(
    pool: list[dict[str, Any]],
    used_review_ids: set[str],
    matcher: SeedMatcher,
    size: int,
    rng: random.Random,
    annotators: Sequence[str],
) -> list[dict[str, Any]]:
    """A shared warm-up set all three label together before starting.

    Drawn from the pool directly rather than carved out of the queue, and
    deliberately over-weighted towards false friends: a sentence where a
    hazard word appears and there is no hazard. Those are where annotators
    disagree, and disagreeing over thirty sentences is cheaper than
    disagreeing over six hundred.

    Practice reviews are taken from outside the queue, so nothing gets
    labelled twice under two different protocols.
    """
    false_friends: list[tuple[dict[str, Any], int, str]] = []
    positives: list[tuple[dict[str, Any], int, str]] = []
    plain: list[tuple[dict[str, Any], int, str]] = []

    candidates = [r for r in pool if r["review_id"] not in used_review_ids]
    rng.shuffle(candidates)
    for review in candidates:
        if len(false_friends) >= size and len(positives) >= size and len(plain) >= size:
            break
        for index, sentence in enumerate(sentences_for(review["review_text"]), start=1):
            hit = matcher.match(sentence)
            entry = (review, index, sentence)
            if hit["blocked"]:
                false_friends.append(entry)
            elif hit["categories"]:
                positives.append(entry)
            elif len(sentence) > 20:
                plain.append(entry)

    wanted_friends = min(len(false_friends), size // 3)
    wanted_positives = min(len(positives), size // 2)
    picked = false_friends[:wanted_friends] + positives[:wanted_positives]
    picked += plain[: max(0, size - len(picked))]
    picked = picked[:size]

    rows: list[dict[str, Any]] = []
    for order, (review, index, sentence) in enumerate(picked, start=1):
        guess = prelabel(sentence, matcher)
        row: dict[str, Any] = {
            "row_id": f"p1-{order:03d}",
            "review_id": review["review_id"],
            "rating": review.get("rating", ""),
            "sentence_index": index,
            "sentence_text": sentence,
            "review_text_raw": review["review_text"],
            "is_blind": "no",
            "llm_safety": guess["llm_safety"],
            "llm_defect": guess["llm_defect"],
            "llm_counterfeit": guess["llm_counterfeit"],
            "llm_hazard_subtype": guess["llm_hazard_subtype"],
            "cohort": review["cohort"],
            "stratum": "practice",
            "sampling_weight": "",
            "source_file": review.get("source_file", ""),
            "product_id": review.get("product_id", ""),
            "product_name": review.get("product_name", ""),
            "category": review.get("category", ""),
            "matched_terms": guess["matched_terms"],
            "blocked_terms": guess["blocked_terms"],
            "assigned_to": "|".join(annotators),
            "adjudicator": "",
            "is_practice": "yes",
        }
        row.update(blank_worksheet_fields())
        rows.append(row)
    rng.shuffle(rows)
    for order, row in enumerate(rows, start=1):
        row["row_id"] = f"p1-{order:03d}"
    return rows


def assign(rows: list[dict[str, Any]], annotators: Sequence[str]) -> None:
    """Two annotators per row, third adjudicates, load spread evenly."""
    if len(annotators) < 2:
        raise ValueError("At least two annotators are required for agreement scoring.")
    if len(annotators) == 2:
        pairs = [(0, 1)]
    else:
        pairs = [(i, (i + 1) % len(annotators)) for i in range(len(annotators))]
    for index, row in enumerate(rows):
        if row["is_practice"] == "yes":
            row["assigned_to"] = "|".join(annotators)
            row["adjudicator"] = ""
            continue
        first, second = pairs[index % len(pairs)]
        assigned = [annotators[first], annotators[second]]
        row["assigned_to"] = "|".join(assigned)
        rest = [a for a in annotators if a not in assigned]
        row["adjudicator"] = rest[0] if rest else ""


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotators",
        default="A1,A2,A3",
        help="Comma-separated initials, e.g. CAB,MAC,RET. Re-run with the real "
        "initials before the team starts; the initials go in the filenames.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        choices=sorted(ROUND_DEFAULTS),
        help="Labelling round. Round 2 excludes round 1 and is fully blind.",
    )
    for stratum in ALL_STRATA:
        parser.add_argument(
            f"--budget-{stratum.replace('_', '-')}",
            type=int,
            default=None,
            dest=f"budget_{stratum}",
            help=f"Sentence budget for the {stratum} stratum (default: the round's).",
        )
    parser.add_argument("--blind-rate", type=float, default=None)
    parser.add_argument("--practice-size", type=int, default=DEFAULT_PRACTICE_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shopee-hf", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what the pool looks like and write nothing.",
    )
    args = parser.parse_args(argv)
    defaults = ROUND_DEFAULTS[args.round]
    for stratum in ALL_STRATA:
        if getattr(args, f"budget_{stratum}") is None:
            setattr(args, f"budget_{stratum}", defaults["budget"].get(stratum, 0))
    if args.blind_rate is None:
        args.blind_rate = defaults["blind_rate"]
    if args.seed is None:
        args.seed = defaults["seed"]
    if args.output_dir is None:
        args.output_dir = round_dir(args.round)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    rng = random.Random(args.seed)

    seeds, blocks = load_seeds()
    matcher = SeedMatcher(seeds, blocks)

    scrape_rows, scrape_files = load_own_scrapes()
    firecs_rows, firecs_files = load_firecs()
    shopee_rows, shopee_files = load_shopee_hf(args.shopee_hf)
    excluded, exclusion_files = load_exclusions()
    prior_keys, prior_files = load_prior_rounds(args.round)
    excluded |= prior_keys

    strata = round_strata(args.round)
    predicates = {
        COUNTERFEIT_FOCUS: counterfeit_focus_predicate,
        ASPECT_FOCUS: aspect_focus_predicate,
        MODEL_DISAGREEMENT: model_disagreement_predicate,
        DISAGREEMENT: defect_disagreement_predicate,
    }
    first = [(name, predicates[name]()) for name in strata if name in predicates]
    candidates = scrape_rows + firecs_rows + shopee_rows
    cohorts = ROUND_DEFAULTS[args.round].get("cohorts")
    if cohorts:
        if not shopee_rows:
            raise SystemExit(f"Round {args.round} draws only from {cohorts}; pass --shopee-hf.")
        candidates = [row for row in candidates if row["cohort"] in cohorts]
    scope = ROUND_DEFAULTS[args.round].get("scope")
    if scope == "gadget":
        from gadget_scope import is_gadget

        candidates = [row for row in candidates if is_gadget(row)]
    pool, drops = build_pool(candidates, matcher, excluded, first)

    by_stratum: dict[str, list[dict[str, Any]]] = {s: [] for s in strata}
    for row in pool:
        by_stratum[row["stratum"]].append(row)

    budgets = {s: getattr(args, f"budget_{s}") for s in strata}
    sampled = {
        s: sample_stratum(
            by_stratum[s], budgets[s], rng, prioritise_rare=(s == "seed_hit")
        )
        for s in strata
    }
    weights = {
        s: (len(by_stratum[s]) / len(sampled[s])) if sampled[s] else 0.0 for s in strata
    }

    rows = expand_rows(sampled, weights, matcher, rng, args.blind_rate)
    rng.shuffle(rows)  # interleave strata so the source of a row is not guessable
    assign(rows, annotators)
    for index, row in enumerate(rows, start=1):
        row["row_id"] = f"r{args.round}-{index:05d}"

    used_review_ids = {row["review_id"] for row in rows}
    practice = build_practice_set(
        pool, used_review_ids, matcher, args.practice_size, rng, annotators
    )

    cohort_counts: dict[str, int] = {}
    for row in rows:
        cohort_counts[row["cohort"]] = cohort_counts.get(row["cohort"], 0) + 1

    # How many candidate positives each category actually got. The validation
    # plan wants 20+ per category before that category is scored separately,
    # and safety is the one at risk, so the number belongs in the manifest
    # where it can be checked rather than assumed.
    def _category_totals(source: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
        totals = {"safety": 0, "counterfeit": 0, "defect": 0}
        for stratum_rows in source.values():
            for review in stratum_rows:
                for category in review.get("seed_categories") or set():
                    if category in totals:
                        totals[category] += 1
        return totals

    pool_by_category = _category_totals(by_stratum)
    sampled_by_category = _category_totals(sampled)

    summary: dict[str, Any] = {
        "queue_version": QUEUE_VERSION,
        "round": args.round,
        "seeds_version": SEEDS_VERSION,
        "segmenter_version": SEGMENTER_VERSION,
        "tagging_guide_version": ROUND_DEFAULTS[args.round]["guide"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.seed,
        "annotators": annotators,
        "prelabel_source": (
            "mining_seeds_v1 applied at sentence level. This is a keyword "
            "baseline, NOT a trained model and NOT an LLM prediction. It exists "
            "so annotators verify rather than label from scratch, and so the "
            "keyword baseline's error rate is measured as a by-product."
        ),
        "blind_rate_requested": args.blind_rate,
        "blind_rows": sum(1 for r in rows if r["is_blind"] == "yes"),
        "practice_rows": len(practice),
        "practice_composition": {
            "false_friends": sum(1 for r in practice if r["blocked_terms"]),
            "seed_positives": sum(
                1 for r in practice if r["matched_terms"] and not r["blocked_terms"]
            ),
            "plain": sum(
                1 for r in practice if not r["matched_terms"] and not r["blocked_terms"]
            ),
        },
        "practice_note": (
            "Drawn from reviews outside the queue so nothing is labelled twice, "
            "and over-weighted towards false friends on purpose. All three "
            "annotators label these, independently, then compare before "
            "starting their own worksheet."
        ),
        "sentence_rows": len(rows),
        "reviews_sampled": {s: len(sampled[s]) for s in strata},
        "sentences_by_stratum": {
            s: sum(1 for r in rows if r["stratum"] == s) for s in strata
        },
        "sentences_by_cohort": cohort_counts,
        "pool_by_stratum": {s: len(by_stratum[s]) for s in strata},
        "seed_candidates_in_pool": pool_by_category,
        "seed_candidates_sampled": sampled_by_category,
        "seed_candidate_note": (
            "Candidate counts, not confirmed positives. A candidate is a review "
            "the seed list matched; the annotators decide whether it is real. "
            "Expect attrition - 'without overheating' and 'walang kuryente' both "
            "match and neither is a hazard. Within seed_hit the draw is ordered "
            "safety, then counterfeit, then defect, so the rare categories are "
            "not crowded out; this is a stratified draw, not a simple random one."
        ),
        "sampling_weights": {s: round(weights[s], 4) for s in strata},
        "pool_size": len(pool),
        "dropped": drops,
        "budgets": budgets,
        "reportable_prevalence_stratum": "random_all",
        "sampling_note": (
            "seed_hit and low_star are enrichment strata. Prevalence, recall "
            "and any per-product rate must be computed on random_all, or on "
            "all three with sampling_weight applied. Reporting a raw rate over "
            "the whole queue would overstate hazard prevalence several-fold."
        ),
        "firecs_note": (
            "firecs and shopee_hf rows carry no product_id. They train the "
            "sentence tagger only and must be excluded from product-level "
            "rollups and from any product-level evaluation."
        ),
        "inputs": {
            "scrape_files": len(scrape_files),
            "firecs_files": [p.name for p in firecs_files],
            "shopee_hf": [p.name for p in shopee_files],
            "exclusion_files": {p.name: sha256(p) for p in exclusion_files},
            "excluded_text_keys": len(excluded),
            "prior_round_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in prior_files},
            "prior_round_reviews_excluded": len(prior_keys),
            "seeds_sha256": sha256(SEEDS_PATH),
            **(
                {"disagreement_model": str(DISAGREEMENT_MODEL.relative_to(ROOT)), "disagreement_model_sha256": sha256(DISAGREEMENT_MODEL)}
                if DISAGREEMENT in strata
                else {}
            ),
            **(
                {"candidate_model_manifest_sha256": sha256(CANDIDATE_MODEL_DIR / "manifest.json")}
                if MODEL_DISAGREEMENT in strata
                else {}
            ),
            "cohorts": list(cohorts) if cohorts else "all",
            "scope": scope or "all",
        },
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    out = args.output_dir
    write_csv(out / "queue_master.csv", MASTER_FIELDS, rows)
    write_csv(out / "practice_set.csv", WORKSHEET_FIELDS, practice)
    write_csv(out / "practice_set_master.csv", MASTER_FIELDS, practice)
    worksheet_rows: dict[str, int] = {}
    for annotator in annotators:
        assigned = [r for r in rows if annotator in r["assigned_to"].split("|")]
        write_csv(
            out / f"worksheet_{annotator}.csv",
            WORKSHEET_FIELDS,
            ({**r, "verifier_id": annotator} for r in assigned),
        )
        worksheet_rows[annotator] = len(assigned)
    summary["worksheet_rows"] = worksheet_rows

    # Bytes, not write_text: on Windows text mode turns every LF into CRLF.
    (out / "manifest.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))

    print(f"Wrote {len(rows)} sentence rows to {out}")
    for annotator in annotators:
        print(f"  worksheet_{annotator}.csv: {worksheet_rows[annotator]} rows")
    print(f"  practice_set.csv: {len(practice)} rows (all three label these together)")
    for stratum in strata:
        print(
            f"  {stratum}: {summary['sentences_by_stratum'][stratum]} sentences "
            f"from {len(sampled[stratum])} reviews "
            f"(pool {len(by_stratum[stratum])}, weight {weights[stratum]:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
