"""Data helpers shared by the web app and its tests."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent


def search_slug(keyword: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", keyword.lower()).strip("_")
    return slug or "search"


PLATFORM_NAMES = {"shopee": "Shopee PH", "lazada": "Lazada PH", "temu": "Temu PH"}

RECOMMENDATION_STOP_WORDS = {
    "a", "an", "and", "for", "from", "in", "more", "new", "of", "on",
    "original", "series", "the", "to", "with",
}

RECOMMENDATION_PRODUCT_TYPES = (
    (("power bank", "powerbank", "portable charger"), "power bank", "powerbank"),
    (("wireless earbuds", "earbuds", "earphones"), "wireless earbuds", "earbuds"),
    (("headphones", "headphone", "headset"), "headphones", "headphones"),
    (("wall charger", "charger", "charging adapter"), "charger", "charger"),
    (("usb cable", "charging cable", "cable"), "cable", "cable"),
    (("mechanical keyboard", "keyboard"), "keyboard", "keyboard"),
    (("gaming mouse", "wireless mouse", "mouse"), "mouse", "mouse"),
    (("smart watch", "smartwatch"), "smart watch", "smartwatch"),
    (("bluetooth speaker", "speaker"), "speaker", "speaker"),
    (("monitor",), "monitor", "monitor"),
    (("laptop",), "laptop", "laptop"),
    (("camera",), "camera", "camera"),
)


def detect_marketplace(value: str, default: str = "shopee") -> str:
    """Detect a supported marketplace from a URL, otherwise use the chosen default."""
    parsed = urlsplit(str(value).strip())
    host = parsed.netloc.casefold().split(":", 1)[0]
    if host == "shopee.ph" or host.endswith(".shopee.ph"):
        return "shopee"
    if host == "lazada.com.ph" or host.endswith(".lazada.com.ph"):
        return "lazada"
    if host == "temu.com" or host.endswith(".temu.com"):
        return "temu"
    if parsed.scheme in {"http", "https"} and host:
        raise ValueError("Only Shopee PH, Lazada PH, and Temu product links are supported.")
    return default if default in PLATFORM_NAMES else "shopee"


def product_platform(product: dict[str, Any]) -> str:
    platform = str(product.get("platform") or "").strip().casefold()
    if platform not in PLATFORM_NAMES:
        try:
            platform = detect_marketplace(str(product.get("link") or ""))
        except ValueError:
            # Backward-compatible fallback for imported research fixtures whose
            # source predates platform metadata.
            platform = "shopee"
    return platform


def platform_name(product: dict[str, Any]) -> str:
    return PLATFORM_NAMES[product_platform(product)]


def output_path(keyword: str, platform: str = "shopee") -> Path:
    platform = platform if platform in PLATFORM_NAMES else "shopee"
    return ROOT / f"{platform}_ph_{search_slug(keyword)}.json"


def result_files() -> list[Path]:
    return sorted(
        (
            path
            for pattern in ("shopee_ph_*.json", "lazada_ph_*.json", "temu_ph_*.json")
            for path in ROOT.glob(pattern)
            if not path.stem.endswith("_analyzed")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_products(source: str | Path | bytes) -> list[dict[str, Any]]:
    if isinstance(source, bytes):
        data = json.loads(source.decode("utf-8-sig"))
    else:
        with Path(source).open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    if isinstance(data, dict):
        data = data.get("products", [])
    if not isinstance(data, list):
        raise ValueError("The JSON must be a product list or contain a 'products' list.")
    return [item for item in data if isinstance(item, dict)]


def has_usable_products(products: list[dict[str, Any]]) -> bool:
    """Return whether a scraper result contains real listing or review data."""
    return any(
        str(product.get("name") or "").strip()
        or bool(product.get("comments"))
        or product.get("review_status") in {"complete", "no_reviews"}
        for product in products
    )


def cached_product_matches(
    products: list[dict[str, Any]], query: str, limit: int = 6
) -> list[dict[str, Any]]:
    """Find analyzed products without making a marketplace request."""
    value = str(query or "").strip()
    if not value or limit <= 0:
        return []
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        target = (parsed.netloc.casefold().removeprefix("www."), parsed.path.rstrip("/").casefold())
        matches = []
        for product in products:
            product_url = urlsplit(str(product.get("link") or ""))
            identity = (
                product_url.netloc.casefold().removeprefix("www."),
                product_url.path.rstrip("/").casefold(),
            )
            if identity == target:
                matches.append(product)
        return matches[:limit]

    terms = re.findall(r"[a-z0-9]+", value.casefold())
    if not terms:
        return []
    ranked = []
    for index, product in enumerate(products):
        name = str(product.get("name") or "").casefold()
        overlap = sum(term in name for term in terms)
        if overlap:
            exact_phrase = int(value.casefold() in name)
            ranked.append((exact_phrase, overlap / len(terms), product_reliability(product), -index, product))
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return [item[-1] for item in ranked[:limit]]


def merge_product_catalogs(*catalogs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge analyzed catalogs while preserving the first version of each listing."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str] | tuple[str, str, str]] = set()
    for catalog in catalogs:
        for product in catalog or []:
            if not isinstance(product, dict):
                continue
            parsed = urlsplit(str(product.get("link") or ""))
            if parsed.netloc and parsed.path:
                identity: tuple[str, str] | tuple[str, str, str] = (
                    parsed.netloc.casefold().removeprefix("www."),
                    parsed.path.rstrip("/").casefold(),
                )
            else:
                identity = (
                    "name",
                    str(product.get("platform") or "").casefold(),
                    str(product.get("name") or "").strip().casefold(),
                )
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(product)
    return merged


def _recommendation_brand(product: dict[str, Any]) -> str:
    brand = str(product.get("brand") or "").strip()
    if brand:
        return brand
    name_terms = [
        term for term in re.findall(r"[A-Za-z0-9]+", str(product.get("name") or ""))
        if term.casefold() not in RECOMMENDATION_STOP_WORDS
    ]
    return name_terms[0] if name_terms else ""


def _recommendation_terms(product: dict[str, Any]) -> set[str]:
    text = " ".join(
        str(product.get(field) or "").casefold()
        for field in ("name", "category", "brand")
    )
    terms = {
        term for term in re.findall(r"[a-z0-9]+", text)
        if term not in RECOMMENDATION_STOP_WORDS
    }
    if "powerbank" in terms or {"power", "bank"}.issubset(terms):
        terms.add("powerbank")
    product_type = _recommendation_type(product)
    if product_type:
        terms.add(product_type)
    return terms


def _recommendation_type(product: dict[str, Any]) -> str:
    text = " ".join(
        str(product.get(field) or "").casefold()
        for field in ("name", "category")
    )
    for variants, _label, canonical in RECOMMENDATION_PRODUCT_TYPES:
        if any(variant in text for variant in variants):
            return canonical
    return ""


def rank_recommendations(
    product: dict[str, Any],
    products: list[dict[str, Any]],
    limit: int = 3,
    min_positive_ratio: float = 0.5,
    min_review_count: int = 5,
) -> list[dict[str, Any]]:
    """Rank similar products whose reviews show no warning signs.

    Only products analysed with the badge model qualify: a legacy risk score
    cannot vouch for a product under the current checks. A candidate needs the
    "no warning signs" verdict -- which already requires enough reviews -- and
    not a single safety report, because suggesting a product with a fire report
    as the safer choice would be worse than suggesting nothing. Candidates rank
    by similarity, then by the lower bound of their positive-review ratio,
    which discounts products whose praise rests on a small sample.
    """
    if limit <= 0:
        return []
    target_terms = _recommendation_terms(product)
    target_type = _recommendation_type(product)
    target_category = str(product.get("category") or "").strip().casefold()
    target_brand = _recommendation_brand(product).casefold()
    ranked: list[tuple[float, float, int, int, dict[str, Any]]] = []
    for index, candidate in enumerate(products):
        if candidate.get("link") == product.get("link"):
            continue
        summary = candidate.get("sentiment_summary") or {}
        if not is_badge_summary(summary):
            continue
        view = badge_view(candidate)
        if view["state"] != "no_warning_signs" or view["counts"]["safety"]:
            continue
        positive = float(summary.get("positive_ratio_lower_bound") or 0)
        review_count = view["reviews_analyzed"]
        rating = rating_value(candidate.get("rating"))
        if (
            positive < min_positive_ratio
            or review_count < min_review_count
            or (rating is not None and rating < 4.0)
        ):
            continue

        candidate_terms = _recommendation_terms(candidate)
        candidate_type = _recommendation_type(candidate)
        if target_type and candidate_type and target_type != candidate_type:
            continue
        overlap = target_terms & candidate_terms
        candidate_category = str(candidate.get("category") or "").strip().casefold()
        category_match = bool(target_category and target_category == candidate_category)
        type_match = bool(target_type and target_type == candidate_type)
        lexical_similarity = len(overlap) / max(1, min(len(target_terms), len(candidate_terms)))
        if not category_match and not type_match and (
            len(overlap) < 3 or lexical_similarity < 0.35
        ):
            continue
        brand_match = bool(
            target_brand and target_brand == _recommendation_brand(candidate).casefold()
        )
        similarity = min(
            1.0,
            lexical_similarity + (0.25 if category_match else 0) + (0.08 if brand_match else 0),
        )
        evidence = min(1.0, math.log10(review_count + 1) / 2)
        recommendation_score = 0.6 * similarity + 0.3 * positive + 0.1 * evidence
        ranked.append((recommendation_score, positive, review_count, -index, candidate))
    ranked.sort(key=lambda item: item[:4], reverse=True)
    return [item[-1] for item in ranked[:limit]]


def price_value(value: Any) -> float | None:
    text = str(value or "").replace("â‚±", "₱")
    numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    parsed = [float(number.replace(",", "")) for number in numbers[:2]]
    return sum(parsed) / len(parsed)


def rating_value(value: Any) -> float | None:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    return rating if 0 <= rating <= 5 else None


def product_rows(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for product in products:
        comments = product.get("comments")
        view = badge_view(product)
        rows.append(
            {
                "Image": product.get("img", ""),
                "Product": product.get("name", ""),
                "Price": str(product.get("price", "")).replace("â‚±", "₱"),
                "Price (PHP)": price_value(product.get("price")),
                "Rating": rating_value(product.get("rating")),
                "Reviews": len(comments) if isinstance(comments, list) else 0,
                "Total ratings": int(product.get("total_rating") or 0),
                "Location": product.get("location", ""),
                "Platform": platform_name(product),
                "Verdict": view["headline"],
                "Reviews analyzed": view["reviews_analyzed"],
                "Safety reports": view["counts"]["safety"],
                "Counterfeit reports": view["counts"]["counterfeit"],
                "Defect reports": view["counts"]["defect"],
                "Link": product.get("link", ""),
            }
        )
    return rows


def normalized_risk_score(value: Any, scale: Any | None = None) -> float:
    """Return a risk score on 0..1 using explicit units when available.

    Older imported datasets did not record a unit, so their values retain the
    historical heuristic: values above 1 are percentages and values up to 1
    are ratios. New analyses always include ``risk_score_scale`` to remove the
    otherwise unavoidable ambiguity around low percentage scores such as 1.0.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    unit = str(scale or "").strip().casefold()
    if unit in {"percent", "percentage", "0-100"}:
        score /= 100
    elif unit not in {"ratio", "proportion", "fraction", "0-1"} and score > 1:
        score /= 100
    return min(1.0, max(0.0, score))


def reliability_score(
    risk: Any,
    positive_ratio: Any | None = None,
    risk_scale: Any | None = None,
) -> float:
    """Return the manuscript reliability score on a bounded 0..100 scale.

    With a positive ratio this implements Positive Ratio - Risk Score. The
    one-argument form is retained for compatibility with older saved results.
    """
    normalized_risk = normalized_risk_score(risk, risk_scale)
    if positive_ratio is None:
        return round(100 * (1 - normalized_risk), 1)
    try:
        positive = float(positive_ratio)
    except (TypeError, ValueError):
        positive = 0.0
    if positive > 1:
        positive /= 100
    return round(100 * max(0.0, min(1.0, positive) - normalized_risk), 1)


def product_reliability(product: dict[str, Any]) -> float:
    summary = product.get("sentiment_summary") or {}
    positive = (summary.get("sentiment_ratios") or {}).get("positive")
    return reliability_score(
        summary.get("risk_score"), positive, summary.get("risk_score_scale")
    )


# ---------------------------------------------------------------------------
# Badge presentation (summary schema 2.0)
#
# The badge model does not produce a score. It records which reviews mention a
# safety hazard, a counterfeit or a defect, and quotes them. The check is
# imperfect -- on round 1 it caught 21 of 27 confirmed safety reports, and 18
# of its 39 safety flags were harmless -- so the screen shows shoppers the
# reviews and says what they report, rather than asserting that a product is
# safe or unsafe. Everything user-facing about badges is worded here, once.
# ---------------------------------------------------------------------------

BADGE_SCHEMA_PREFIX = "defaketive-summary-2"
EVIDENCE_SHOWN_PER_BADGE = 5

# Held-out figures for the shipped safety check (keyword checker 0.3.0 with the
# learned safety model adopted on labelling round 2): 569 sentences whose safety
# label two annotators agreed on, none seen while either was built. Source:
# research/hazard_tagger_measurement_round2.json, configuration "v4_learned".
SAFETY_CHECK_ACCURACY = {
    "sentences_checked": 569,
    "reports_confirmed": 16,
    "reports_caught": 10,
    "flags_raised": 24,
    "flags_confirmed": 10,
}

# Tone drives colour only. It is never shown as a word.
VERDICT_TONES = {
    "safety_warning": "high",
    "possible_safety_issue": "moderate",
    "possible_counterfeit_or_defect": "moderate",
    "no_warning_signs": "low",
    "insufficient_evidence": "unknown",
    "reviews_may_be_unreliable": "unknown",
}

# Headlines say what reviewers report, never that a product IS unsafe.
VERDICT_HEADLINES = {
    "safety_warning": "Reviewers report safety problems",
    "possible_safety_issue": "Some reviewers mention safety concerns",
    "possible_counterfeit_or_defect": "Reviewers report fakes or defects",
    "reviews_may_be_unreliable": "These reviews may not be trustworthy",
    "insufficient_evidence": "Not enough reviews to judge",
    "no_warning_signs": "No warning signs found",
}

HAZARD_LABELS = {
    "fire_explosion_swelling": "fire, explosion or a swollen battery",
    "shock_spark_short": "electric shocks, sparks or short circuits",
    "overheat_burn": "overheating or burns",
    "smoke_smell": "smoke or a burning smell",
    "safety_unspecified": "a safety problem",
}

# "verb" is (singular, plural): "1 review says", "3 reviews say".
BADGE_COPY = {
    "safety": {
        "label": "Safety",
        "verb": ("mentions", "mention"),
        "none": "No reviews mention fire, smoke, shocks or overheating",
    },
    "counterfeit": {
        "label": "Genuine product",
        "verb": ("says", "say"),
        "claim": "it may not be genuine",
        "none": "No reviews say it is fake",
    },
    "defect": {
        "label": "Works as expected",
        "verb": ("says", "say"),
        "claim": "it broke or did not work",
        "none": "No reviews say it broke or stopped working",
    },
}


def is_badge_summary(summary: Any) -> bool:
    return isinstance(summary, dict) and str(summary.get("schema_version") or "").startswith(
        BADGE_SCHEMA_PREFIX
    )


def _reviews(count: int) -> str:
    return f"{count} review{'' if count == 1 else 's'}"


def _report_phrase(category: str, mentions: int, claim: str) -> str:
    """The verb phrase after a count: "mentions fire", "say it broke"."""
    singular, plural = BADGE_COPY[category]["verb"]
    return f"{singular if mentions == 1 else plural} {claim}"


def _usable_reviews(summary: dict[str, Any]) -> int:
    coverage = summary.get("coverage") or {}
    return int(summary.get("usable_review_count") or coverage.get("usable_reviews") or 0)


def _evidence_items(evidence: Any) -> tuple[list[dict[str, Any]], int]:
    """Return the strongest distinct quotes for a badge, and how many were cut.

    Each quote carries the character ranges of the words that triggered it, so
    the screen can highlight them without re-matching the text.
    """
    if not isinstance(evidence, list):
        return [], 0
    ranked = sorted(
        (item for item in evidence if isinstance(item, dict) and str(item.get("text") or "").strip()),
        key=lambda item: (bool(item.get("high_confidence")), float(item.get("score") or 0)),
        reverse=True,
    )
    seen: set[str] = set()
    distinct: list[dict[str, Any]] = []
    for item in ranked:
        text = str(item["text"]).strip()
        key = " ".join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        highlights = sorted(
            [int(term["span"][0]), int(term["span"][1])]
            for term in item.get("terms") or []
            if isinstance(term, dict)
            and not term.get("negated")
            and isinstance(term.get("span"), list)
            and len(term["span"]) == 2
        )
        distinct.append(
            {
                "text": text,
                "rating": item.get("rating"),
                "hazard": HAZARD_LABELS.get(str(item.get("hazard_subtype") or "")),
                "highlights": highlights,
            }
        )
    shown = distinct[:EVIDENCE_SHOWN_PER_BADGE]
    return shown, len(distinct) - len(shown)


def _hazard_phrase(safety: dict[str, Any]) -> str:
    """Name the most serious hazard reported, noting when there are others.

    Each quote also carries its own hazard label, so naming one type here loses
    nothing, and joining two "X or Y" labels reads as "X or Y and A or B".
    """
    subtypes = safety.get("hazard_subtypes") or {}
    ranked = sorted(
        (
            (int(detail.get("severity") or 0), int(detail.get("reviews") or 0), name)
            for name, detail in subtypes.items()
            if isinstance(detail, dict) and name in HAZARD_LABELS
        ),
        reverse=True,
    )
    if not ranked:
        return HAZARD_LABELS["safety_unspecified"]
    phrase = HAZARD_LABELS[ranked[0][2]]
    return f"{phrase}, and other hazards" if len(ranked) > 1 else phrase


def _safety_badge(safety: dict[str, Any], usable: int) -> dict[str, Any]:
    mentions = int(safety.get("flagged_reviews") or 0)
    evidence, more = _evidence_items(safety.get("evidence"))
    return {
        "category": "safety",
        "label": BADGE_COPY["safety"]["label"],
        "mentions": mentions,
        "reviews_analyzed": usable,
        "phrase": _report_phrase("safety", mentions, _hazard_phrase(safety)),
        "summary": (
            f"{_reviews(mentions)} {_report_phrase('safety', mentions, _hazard_phrase(safety))}"
            if mentions
            else BADGE_COPY["safety"]["none"]
        ),
        "evidence": evidence,
        "more_evidence": more,
    }


def _flag_badge(category: str, flag: Any, usable: int) -> dict[str, Any]:
    flag = flag if isinstance(flag, dict) else {}
    mentions = int(flag.get("flagged_reviews") or 0)
    evidence, more = _evidence_items(flag.get("evidence"))
    copy_text = BADGE_COPY[category]
    phrase = _report_phrase(category, mentions, copy_text["claim"])
    return {
        "category": category,
        "label": copy_text["label"],
        "mentions": mentions,
        "reviews_analyzed": usable,
        "phrase": phrase,
        "summary": f"{_reviews(mentions)} {phrase}" if mentions else copy_text["none"],
        "evidence": evidence,
        "more_evidence": more,
    }


def _verdict_detail(state: str, verdict: dict[str, Any], badges: list[dict[str, Any]], usable: int) -> str:
    """Explain the verdict in terms of what the reviews say."""
    reported = [badge for badge in badges if badge["mentions"]]
    if reported and state in {
        "safety_warning",
        "possible_safety_issue",
        "possible_counterfeit_or_defect",
    }:
        clauses = "; ".join(f"{badge['mentions']} {badge['phrase']}" for badge in reported)
        return f"Of {_reviews(usable)}, {clauses}. Read them below and judge for yourself."
    detail = str(verdict.get("detail") or "").strip().rstrip(".")
    if state == "insufficient_evidence" and reported:
        # Too few reviews for an overall judgement must not hide what the few
        # reviews say: a single fire report is still worth reading.
        clauses = "; ".join(
            [reported[0]["summary"]]
            + [f"{badge['mentions']} {badge['phrase']}" for badge in reported[1:]]
        )
        pronoun = "it" if sum(badge["mentions"] for badge in reported) == 1 else "them"
        return f"{detail}. Even so, {clauses}. Read {pronoun} below."
    if state == "no_warning_signs" and reported:
        # The aggregator's own detail says "absence of complaints", which is
        # false when an isolated complaint sits just below the warning level.
        clauses = "; ".join(
            [reported[0]["summary"]]
            + [f"{badge['mentions']} {badge['phrase']}" for badge in reported[1:]]
        )
        pronoun = "it" if sum(badge["mentions"] for badge in reported) == 1 else "them"
        return (
            f"No problem reaches a warning level in {_reviews(usable)}, but {clauses}. "
            f"Read {pronoun} below. This is not proof that a product is safe."
        )
    return f"{detail}." if detail else ""


def badge_view(product: dict[str, Any]) -> dict[str, Any]:
    """Turn a product's badge summary into what the screen shows."""
    summary = product.get("sentiment_summary") or {}
    if not is_badge_summary(summary):
        return {
            "available": False,
            "state": "not_analyzed",
            "tone": "unknown",
            "headline": "Not analyzed yet",
            "detail": "This product has not been checked with the current review model.",
            "reviews_analyzed": 0,
            "counts": {"safety": 0, "counterfeit": 0, "defect": 0},
            "badges": [],
        }

    usable = _usable_reviews(summary)
    verdict = summary.get("verdict") or {}
    state = str(verdict.get("state") or "insufficient_evidence")
    flags = summary.get("flags") or {}
    badges = [
        _safety_badge(summary.get("safety") or {}, usable),
        _flag_badge("counterfeit", flags.get("counterfeit"), usable),
        _flag_badge("defect", flags.get("defect"), usable),
    ]
    return {
        "available": True,
        "state": state,
        "tone": VERDICT_TONES.get(state, "unknown"),
        "headline": VERDICT_HEADLINES.get(state, "Not enough reviews to judge"),
        "detail": _verdict_detail(state, verdict, badges, usable),
        "reviews_analyzed": usable,
        "counts": {badge["category"]: badge["mentions"] for badge in badges},
        "badges": badges,
        "limits_notice": summary.get("limits_notice"),
        "safety_check_accuracy": SAFETY_CHECK_ACCURACY,
    }


def with_badge_view(product: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of a product carrying its display view."""
    return {**product, "badge_view": badge_view(product)}


def review_rows(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for product in products:
        for review in product.get("comments") or []:
            if not isinstance(review, dict):
                continue
            if review.get("included_in_analysis") is False:
                continue
            analysis = review.get("sentiment_analysis") or {}
            risk = analysis.get("risk") or {}
            scores = analysis.get("scores") or {}
            rows.append(
                {
                    "Product": product.get("name", ""),
                    "Author": review.get("author", ""),
                    "Stars": int(review.get("rating") or 0),
                    "Date": review.get("time", ""),
                    "Review": review.get("content", ""),
                    "Likes": int(review.get("like_count") or 0),
                    "Sentiment": analysis.get("label", "unanalyzed"),
                    "Compound": scores.get("compound"),
                    "Risk": bool(risk.get("detected", False)),
                    "Risk categories": ", ".join(risk.get("categories") or []),
                    "Duplicate": bool(review.get("is_duplicate", False)),
                    "Product link": product.get("link", ""),
                }
            )
    return rows


def _ensure_sentiment_path() -> None:
    sentiment_root = ROOT / "sentiment-analysis"
    if str(sentiment_root) not in sys.path:
        sys.path.insert(0, str(sentiment_root))


def sentiment_model_version() -> str:
    """Return the version of the model the application actually runs.

    Delegates to the registry so this can never name a different model from the
    one `analyze_products` loads. When they disagree, `analysis_is_current`
    either re-analyzes on every request or fails to re-analyze when the model
    really has changed.
    """
    _ensure_sentiment_path()
    from defaketive_sentiment.registry import ModelUnavailableError, model_version

    try:
        return model_version()
    except ModelUnavailableError:
        # Only used to decide whether stored analyses are stale. An unloadable
        # model means nothing can be current, and analyze_products will raise
        # with the real reason when it is called.
        return "unavailable"


def model_unavailable_error() -> type[Exception]:
    """The exception raised when the configured model cannot be loaded."""
    _ensure_sentiment_path()
    from defaketive_sentiment.registry import ModelUnavailableError

    return ModelUnavailableError


def require_model() -> None:
    """Raise ModelUnavailableError now if the model cannot load.

    Called before a live scan, so a missing model is reported before minutes of
    marketplace collection rather than after them.
    """
    _ensure_sentiment_path()
    from defaketive_sentiment.registry import load_model

    load_model()


def analysis_is_current(products: list[dict[str, Any]]) -> bool:
    """Return whether every saved analysis uses the active model version."""
    if not products:
        return False
    expected_version = sentiment_model_version()
    for product in products:
        if not isinstance(product, dict):
            return False
        summary = product.get("sentiment_summary") or {}
        if not (is_badge_summary(summary) or summary.get("risk_score_scale")):
            return False
        for review in product.get("comments") or []:
            if not isinstance(review, dict):
                continue
            if review.get("included_in_analysis") is False:
                continue
            analysis = review.get("sentiment_analysis") or {}
            if analysis.get("model_version") != expected_version:
                return False
    return True


def analyze_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze products with the configured model.

    Raises ModelUnavailableError if that model cannot be loaded. It deliberately
    does not fall back to a different model: with badges, a substitution would
    change the shape of the output, and the report would render another model's
    verdict with nothing to show that it had happened.
    """
    _ensure_sentiment_path()
    from defaketive_sentiment.analyze_reviews import analyze_product_json
    from defaketive_sentiment.registry import load_model

    analyzed = copy.deepcopy(products)
    return analyze_product_json(analyzed, load_model())


def json_bytes(products: list[dict[str, Any]]) -> bytes:
    return json.dumps(products, ensure_ascii=False, indent=2).encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")
