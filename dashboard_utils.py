"""Data helpers shared by the Streamlit dashboard and its tests."""

from __future__ import annotations

import copy
from collections import Counter
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

PRODUCT_IMAGE_FIELDS = (
    "img",
    "image",
    "image_url",
    "imageUrl",
    "thumbnail",
    "thumbnail_url",
    "thumbnailUrl",
    "picture",
    "picture_url",
    "images",
    "pictures",
    "media",
)

NESTED_IMAGE_FIELDS = (
    "url",
    "secure_url",
    "src",
    "image",
    "image_url",
    "imageUrl",
    "original",
    "large",
    "main",
    "thumbnail",
    "thumbnail_url",
    "thumbnailUrl",
    "images",
    "pictures",
)

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


def product_image_url(product: dict[str, Any]) -> str:
    """Return the first valid web image URL found in common product fields.

    Marketplace exports are inconsistent: an image may be a string, a
    protocol-relative URL, a list, or a nested mapping such as
    ``{"image": {"url": "..."}}``. Local paths and non-web schemes are not
    safe for browser image markup, so they deliberately resolve to an empty
    string.
    """
    if not isinstance(product, dict):
        return ""
    for field in PRODUCT_IMAGE_FIELDS:
        if field not in product:
            continue
        resolved = _resolve_product_image_value(product[field], set())
        if resolved:
            return resolved
    return ""


def _resolve_product_image_value(value: Any, seen: set[int]) -> str:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("//"):
            candidate = f"https:{candidate}"
        if not candidate or any(character in candidate for character in "\r\n\t"):
            return ""
        parsed = urlsplit(candidate)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            return ""
        return candidate

    if not isinstance(value, (dict, list, tuple)):
        return ""
    identity = id(value)
    if identity in seen:
        return ""
    seen.add(identity)

    if isinstance(value, dict):
        ordered_values = [value[field] for field in NESTED_IMAGE_FIELDS if field in value]
        ordered_fields = set(NESTED_IMAGE_FIELDS)
        ordered_values.extend(item for field, item in value.items() if field not in ordered_fields)
    else:
        ordered_values = value
    for item in ordered_values:
        resolved = _resolve_product_image_value(item, seen)
        if resolved:
            return resolved
    return ""


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


def recommendation_search_query(product: dict[str, Any]) -> str:
    """Build a compact marketplace query for finding comparable products."""
    name = re.sub(r"\s+", " ", str(product.get("name") or "").strip().casefold())
    brand = _recommendation_brand(product)
    for variants, label, _canonical in RECOMMENDATION_PRODUCT_TYPES:
        if any(variant in name for variant in variants):
            return " ".join(part for part in (brand, label) if part).strip()
    terms = [
        term
        for term in re.findall(r"[a-z0-9]+", name)
        if term not in RECOMMENDATION_STOP_WORDS
    ]
    if brand and (not terms or terms[0] != brand.casefold()):
        terms.insert(0, brand)
    return " ".join(terms[:4]) or brand or "similar product"


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
    min_reliability: float = 60.0,
    min_positive_ratio: float = 0.55,
    min_review_count: int = 5,
) -> list[dict[str, Any]]:
    """Rank similar, well-reviewed Low Risk products with explainable signals."""
    if limit <= 0:
        return []
    target_terms = _recommendation_terms(product)
    target_type = _recommendation_type(product)
    target_category = str(product.get("category") or "").strip().casefold()
    target_brand = _recommendation_brand(product).casefold()
    ranked: list[tuple[float, float, float, int, int, dict[str, Any]]] = []
    for index, candidate in enumerate(products):
        if candidate.get("link") == product.get("link") or not candidate.get("sentiment_summary"):
            continue
        if product_risk_level(candidate) != "Low":
            continue
        summary = candidate.get("sentiment_summary") or {}
        positive = float((summary.get("sentiment_ratios") or {}).get("positive") or 0)
        if positive > 1:
            positive /= 100
        reliability = product_reliability(candidate)
        review_count = int(summary.get("review_count") or len(candidate.get("comments") or []))
        rating = rating_value(candidate.get("rating"))
        if (
            reliability < min_reliability
            or positive < min_positive_ratio
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
        recommendation_score = (
            0.55 * similarity
            + 0.25 * (reliability / 100)
            + 0.15 * positive
            + 0.05 * evidence
        )
        ranked.append((recommendation_score, reliability, positive, review_count, -index, candidate))
    ranked.sort(key=lambda item: item[:5], reverse=True)
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
        summary = product.get("sentiment_summary") or {}
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
                "Risk score": risk_score_percent(
                    summary.get("risk_score"), summary.get("risk_score_scale")
                ),
                "Risk level": product_risk_level(product),
                "Reliability score": product_reliability(product),
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


def risk_score_percent(value: Any, scale: Any | None = None) -> float:
    """Return a bounded risk score on the dashboard's 0..100 display scale."""
    return round(100 * normalized_risk_score(value, scale), 2)


def risk_level(value: Any, scale: Any | None = None) -> str:
    score = normalized_risk_score(value, scale)
    if score >= 0.61:
        return "High"
    if score >= 0.31:
        return "Moderate"
    return "Low"


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


def product_risk_percent(product: dict[str, Any]) -> float:
    summary = product.get("sentiment_summary") or {}
    return risk_score_percent(
        summary.get("risk_score"), summary.get("risk_score_scale")
    )


def product_risk_level(product: dict[str, Any]) -> str:
    summary = product.get("sentiment_summary") or {}
    return risk_level(summary.get("risk_score"), summary.get("risk_score_scale"))


def risk_keyword_counts(product: dict[str, Any]) -> dict[str, int]:
    """Count unique sampled reviews containing each active risk term."""
    counts: Counter[str] = Counter()
    for review in product.get("comments") or []:
        if review.get("is_duplicate"):
            continue
        analysis = review.get("sentiment_analysis") or {}
        review_terms = {
            str(evidence["term"]).strip().casefold()
            for evidence in (analysis.get("risk") or {}).get("evidence") or []
            if evidence.get("term") and not evidence.get("negated")
        }
        counts.update(term for term in review_terms if term)
    return dict(counts.most_common())


def review_signal_counts(product: dict[str, Any]) -> dict[str, int]:
    """Count active sentiment-language signals once per unique review.

    These terms are a descriptive fallback for the dashboard when no defect or
    fraud terms are present. They must not be presented as risk evidence.
    """
    counts: Counter[str] = Counter()
    for review in product.get("comments") or []:
        if review.get("is_duplicate"):
            continue
        analysis = review.get("sentiment_analysis") or {}
        review_terms = {
            str(evidence["term"]).strip().casefold()
            for evidence in analysis.get("sentiment_evidence") or []
            if evidence.get("term") and not evidence.get("negated")
        }
        counts.update(term for term in review_terms if term)
    return dict(counts.most_common())


def weighted_risk_breakdown(product: dict[str, Any]) -> dict[str, float | int]:
    """Return the WSM inputs and their contributions on a 0..100 scale."""
    summary = product.get("sentiment_summary") or {}
    unique_reviews = [
        review
        for review in product.get("comments") or []
        if isinstance(review, dict) and not review.get("is_duplicate")
    ]
    try:
        review_count = max(0, int(summary.get("review_count")))
    except (TypeError, ValueError):
        review_count = len(unique_reviews)
    if not review_count:
        review_count = len(unique_reviews)

    sentiment_counts = summary.get("sentiment_counts") or {}
    ratios = summary.get("sentiment_ratios") or {}

    def bounded_ratio(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    negative_count_value = sentiment_counts.get("negative")
    try:
        negative_reviews = max(0, int(negative_count_value))
    except (TypeError, ValueError):
        negative_reviews = sum(
            1
            for review in unique_reviews
            if (review.get("sentiment_analysis") or {}).get("label") == "negative"
        )
    negative_reviews = min(review_count, negative_reviews) if review_count else 0
    if negative_count_value is not None and review_count:
        negative_ratio = negative_reviews / review_count
    elif "negative" in ratios:
        negative_ratio = bounded_ratio(ratios.get("negative"))
    else:
        negative_ratio = negative_reviews / review_count if review_count else 0.0

    risk_review_value = summary.get("risk_review_count")
    try:
        risk_reviews = max(0, int(risk_review_value))
    except (TypeError, ValueError):
        risk_reviews = sum(
            1
            for review in unique_reviews
            if ((review.get("sentiment_analysis") or {}).get("risk") or {}).get("detected")
        )
    risk_reviews = min(review_count, risk_reviews) if review_count else 0
    if risk_review_value is not None and review_count:
        defect_review_ratio = risk_reviews / review_count
    elif "keyword_failure_rate" in summary:
        defect_review_ratio = bounded_ratio(summary.get("keyword_failure_rate"))
    else:
        defect_review_ratio = risk_reviews / review_count if review_count else 0.0

    negative_points = 30.0 * negative_ratio
    defect_points = 70.0 * defect_review_ratio
    return {
        "review_count": review_count,
        "negative_reviews": negative_reviews,
        "risk_reviews": risk_reviews,
        "negative_ratio": negative_ratio,
        "defect_review_ratio": defect_review_ratio,
        "negative_weight": 0.30,
        "defect_weight": 0.70,
        "negative_points": negative_points,
        "defect_points": defect_points,
        "total_points": negative_points + defect_points,
    }


def review_rows(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for product in products:
        for review in product.get("comments") or []:
            if not isinstance(review, dict):
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


def sentiment_model_version() -> str:
    sentiment_root = ROOT / "sentiment-analysis"
    if str(sentiment_root) not in sys.path:
        sys.path.insert(0, str(sentiment_root))
    from defaketive_sentiment.model import MODEL_VERSION

    return MODEL_VERSION


def analysis_is_current(products: list[dict[str, Any]]) -> bool:
    """Return whether every saved analysis uses the active model version."""
    if not products:
        return False
    expected_version = sentiment_model_version()
    for product in products:
        if not isinstance(product, dict):
            return False
        if not (product.get("sentiment_summary") or {}).get("risk_score_scale"):
            return False
        for review in product.get("comments") or []:
            if not isinstance(review, dict):
                continue
            analysis = review.get("sentiment_analysis") or {}
            if analysis.get("model_version") != expected_version:
                return False
    return True


def analyze_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sentiment_root = ROOT / "sentiment-analysis"
    if str(sentiment_root) not in sys.path:
        sys.path.insert(0, str(sentiment_root))
    from defaketive_sentiment.analyze_reviews import analyze_product_json
    from defaketive_sentiment.model import DefaketiveSentimentModel

    analyzed = copy.deepcopy(products)
    return analyze_product_json(analyzed, DefaketiveSentimentModel())


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
