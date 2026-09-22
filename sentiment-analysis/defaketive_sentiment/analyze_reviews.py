"""Apply the DeFaketive model to scraped JSON or review CSV files.

REVIEW COVERAGE
Every usable review is analyzed. The old behaviour capped analysis at 100
reviews per product and sampled them in proportion to the star distribution,
which is a reasonable way to estimate a common rate but the wrong tool for a
rare one: hazard reports run at roughly half a percent of reviews, so a 100
review sample of a 2,000 review listing would be expected to contain zero of
them even when the product has ten. Reading every review is the only way the
safety badge can be trusted, and it is also what makes the completeness check
below meaningful.

The star-proportional sampler is kept, unused by default, so the research
baseline can still be reproduced exactly.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model import DefaketiveSentimentModel, ModelSetupError
from .manuscript import HybridManuscriptRiskModel


TEXT_FIELDS = ("content", "comment", "review", "text")

# 0 means "analyze every usable review". Kept as a parameter rather than
# removed so the capped behaviour remains reproducible for the comparison.
DEFAULT_REVIEW_LIMIT = 0

# The cap the manuscript results were produced under. Referenced by the
# research baseline only; never applied to a production analysis.
LEGACY_ANALYSIS_LIMIT = 100

# Fields a scraper may use for the review timestamp, most specific first.
DATE_FIELDS = ("reviewed_at", "create_time", "ctime", "review_time", "date", "time")

# Scraper stop reasons that mean the collection is known to be incomplete.
PARTIAL_STOP_REASONS = {
    "max_review_pages_reached",
    "time_budget_exceeded",
    "hard_ceiling_reached",
    "review_limit_reached",
}


def _review_text(review: dict[str, Any], preferred: str | None = None) -> str:
    if preferred and preferred in review:
        return str(review.get(preferred) or "")
    for field in TEXT_FIELDS:
        if field in review:
            return str(review.get(field) or "")
    return ""


def _rating_bucket(review: dict[str, Any]) -> int:
    try:
        rating = int(float(review.get("rating") or review.get("stars") or 0))
    except (TypeError, ValueError):
        return 0
    return rating if rating in {1, 2, 3, 4, 5} else 0


def _evenly_spaced(items: list[tuple[int, dict[str, Any]]], count: int) -> list[tuple[int, dict[str, Any]]]:
    """Select across the full captured order instead of taking only its front."""
    if count >= len(items):
        return items
    if count <= 0:
        return []
    return [items[min(len(items) - 1, math.floor((index + 0.5) * len(items) / count))] for index in range(count)]


def deduplicate_reviews(
    comments: list[dict[str, Any]], model: DefaketiveSentimentModel
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Return every distinct review, keeping the first copy of each.

    Duplicates are dropped because a review posted twice is one piece of
    evidence, and because templated review farms inflate a rate otherwise.
    """
    unique: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, comment in enumerate(comments):
        if not isinstance(comment, dict):
            continue
        fingerprint = model.fingerprint(_review_text(comment))
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append((index, comment))
    return unique, len(unique)


def representative_review_sample(
    comments: list[dict[str, Any]],
    model: DefaketiveSentimentModel,
    limit: int = LEGACY_ANALYSIS_LIMIT,
) -> tuple[list[tuple[int, dict[str, Any], float]], int]:
    """Deduplicate and select a deterministic star-proportional review sample.

    LEGACY. Not used by a production analysis, which reads every review. Kept so
    the capped manuscript results stay reproducible.
    """
    unique, _ = deduplicate_reviews(comments, model)
    if limit <= 0 or len(unique) <= limit:
        return [(index, comment, 1.0) for index, comment in unique], len(unique)

    buckets: dict[int, list[tuple[int, dict[str, Any]]]] = {rating: [] for rating in range(6)}
    for item in unique:
        buckets[_rating_bucket(item[1])].append(item)
    exact = {rating: len(items) * limit / len(unique) for rating, items in buckets.items()}
    quotas = {rating: min(len(items), int(math.floor(exact[rating]))) for rating, items in buckets.items()}
    remaining = limit - sum(quotas.values())
    order = sorted(
        buckets,
        key=lambda rating: (exact[rating] - quotas[rating], len(buckets[rating]), -rating),
        reverse=True,
    )
    for rating in order:
        if remaining <= 0:
            break
        if quotas[rating] < len(buckets[rating]):
            quotas[rating] += 1
            remaining -= 1

    sampled: list[tuple[int, dict[str, Any], float]] = []
    for rating, items in buckets.items():
        selected = _evenly_spaced(items, quotas[rating])
        weight = len(items) / len(selected) if selected else 0.0
        sampled.extend((index, comment, weight) for index, comment in selected)
    sampled.sort(key=lambda item: item[0])
    return sampled, len(unique)


def _review_date(review: dict[str, Any]) -> str | None:
    """Best-effort ISO date for one review, across the three scrapers' fields."""
    for field in DATE_FIELDS:
        value = review.get(field)
        if not value:
            continue
        if isinstance(value, (int, float)) or str(value).isdigit():
            seconds = float(value)
            # Shopee and Temu report milliseconds; Lazada reports seconds.
            if seconds > 1e11:
                seconds /= 1000.0
            try:
                return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
            except (OverflowError, OSError, ValueError):
                continue
        text = str(value).strip()
        if not text:
            continue
        for pattern in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d %b %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(text[: len(pattern) + 8], pattern).date().isoformat()
            except ValueError:
                continue
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text[:10]
    return None


def _date_range(comments: list[dict[str, Any]]) -> dict[str, str | None]:
    dates = [
        date
        for date in (_review_date(comment) for comment in comments if isinstance(comment, dict))
        if date
    ]
    if not dates:
        return {"first": None, "last": None}
    return {"first": min(dates), "last": max(dates)}


def _star_counts(comments: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        bucket = _rating_bucket(comment)
        key = str(bucket) if bucket else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _completeness(
    product: dict[str, Any],
    collected_by_star: dict[str, int],
    collected_total: int,
    listed_total: int,
    tolerance: float,
) -> tuple[str, str | None]:
    """Decide whether the scrape captured what the listing says exists.

    A partial scrape forces the insufficient-evidence verdict, because the
    reviews most likely to be missing are the low-star ones, which is exactly
    where hazard reports live. This cannot detect reviews the platform itself
    hides - stated as a limitation.
    """
    stop_reason = str(product.get("stop_reason") or "").strip().casefold()
    if stop_reason in PARTIAL_STOP_REASONS:
        return "partial", f"collection stopped early: {stop_reason}"

    if product.get("applied_rating_filters"):
        return "partial", "only selected star ratings were collected"

    listed_by_star = product.get("review_star_distribution") or product.get("displayed_by_star")
    if isinstance(listed_by_star, dict) and listed_by_star:
        for star, listed in listed_by_star.items():
            try:
                expected = int(float(listed))
            except (TypeError, ValueError):
                continue
            if expected <= 0:
                continue
            got = int(collected_by_star.get(str(star), 0))
            if got < expected * (1.0 - tolerance):
                return (
                    "partial",
                    f"{got} of {expected} {star}-star reviews were collected",
                )
        return "complete", None

    if listed_total > 0 and collected_total < listed_total * (1.0 - tolerance):
        return (
            "partial",
            f"{collected_total} of {listed_total} listed reviews were collected",
        )
    if listed_total <= 0:
        return "unknown", "the listing did not report a review count"
    return "complete", None


def _completeness_tolerance() -> float:
    try:
        from .product_badges import _value, load_parameters

        coverage = load_parameters().get("coverage") or {}
        return float(_value(coverage.get("completeness_tolerance"), 0.05))
    except Exception:
        return 0.05


def _available_review_count(product: dict[str, Any], usable_count: int) -> int:
    for field in ("reported_review_count", "total_rating", "review_count"):
        try:
            value = int(float(product.get(field) or 0))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return max(value, usable_count)
    return usable_count


def _analyze_all(
    model: Any, texts: list[str], ratings: list[Any]
) -> list[dict[str, Any]]:
    """Analyze many reviews, using the model's batch path when it has one."""
    if not texts:
        return []
    batch = getattr(model, "analyze_batch", None)
    if callable(batch):
        return batch(texts, ratings)
    return [model.analyze(text, rating=rating) for text, rating in zip(texts, ratings)]


def _summarize(model: Any, analyses: list[dict[str, Any]], **candidates: Any) -> dict[str, Any]:
    """Call the model's summarize with whichever keywords it actually accepts.

    The badge model and the legacy weighted-sum models take different keywords.
    The parameters are inspected rather than the call being wrapped in
    `except TypeError`, because that would also swallow a genuine TypeError
    raised inside summarize and silently retry with fewer arguments.
    """
    try:
        parameters = inspect.signature(model.summarize).parameters
    except (TypeError, ValueError):
        return model.summarize(analyses)
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        accepted = candidates
    else:
        accepted = {name: value for name, value in candidates.items() if name in parameters}
    return model.summarize(analyses, **accepted)


def analyze_product_json(
    data: Any,
    model: DefaketiveSentimentModel,
    *,
    review_limit: int = DEFAULT_REVIEW_LIMIT,
) -> Any:
    """Analyze every usable review of every product, in place.

    `review_limit` defaults to 0, meaning no cap. A positive value restores the
    legacy star-proportional sampling and is only for reproducing the capped
    baseline results.
    """
    products = data if isinstance(data, list) else data.get("products", [])
    if not isinstance(products, list):
        raise ValueError("JSON must be a product list or an object containing a 'products' list")

    tolerance = _completeness_tolerance()

    for product in products:
        if not isinstance(product, dict):
            continue
        comments = product.get("comments", [])
        if not isinstance(comments, list):
            continue

        if review_limit and review_limit > 0:
            selected, usable_count = representative_review_sample(
                comments, model, limit=review_limit
            )
            weights = {index: weight for index, _comment, weight in selected}
        else:
            unique, usable_count = deduplicate_reviews(comments, model)
            # Every distinct review is analyzed, so each one carries its own
            # weight of 1. The legacy summarize path multiplies by this, and 1
            # is the identity, which keeps the baseline comparison valid.
            weights = {index: 1.0 for index, _comment in unique}

        seen: dict[str, int] = {}
        pending: list[tuple[int, dict[str, Any], str]] = []
        for index, comment in enumerate(comments):
            if not isinstance(comment, dict):
                continue
            text = _review_text(comment)
            fingerprint = model.fingerprint(text)
            duplicate_of = seen.get(fingerprint) if fingerprint else None
            if fingerprint and duplicate_of is None:
                seen[fingerprint] = index

            included = index in weights and duplicate_of is None
            comment["included_in_analysis"] = included
            comment["is_duplicate"] = duplicate_of is not None
            comment["duplicate_of"] = duplicate_of
            if included:
                pending.append((index, comment, text))
            else:
                comment.pop("sentiment_analysis", None)

        # Prefer a batch call where the model offers one. With the review cap
        # removed a single product can carry thousands of reviews, and scoring
        # them one at a time is about twelve times slower.
        analyses = _analyze_all(
            model,
            [text for _index, _comment, text in pending],
            [comment.get("rating") for _index, comment, _text in pending],
        )

        unique_analyses: list[dict[str, Any]] = []
        analyzed_comments: list[dict[str, Any]] = []
        for (index, comment, _text), analysis in zip(pending, analyses):
            if isinstance(analysis.get("risk"), dict):
                analysis["risk"]["sampling_weight"] = round(weights[index], 6)
            comment["sentiment_analysis"] = analysis
            unique_analyses.append(analysis)
            analyzed_comments.append(comment)

        available = _available_review_count(product, usable_count)
        collected_by_star = _star_counts(analyzed_comments)
        completeness, reason = _completeness(
            product, collected_by_star, usable_count, available, tolerance
        )
        duplicates = sum(
            1 for comment in comments if isinstance(comment, dict) and comment.get("is_duplicate")
        )
        listed_by_star = product.get("review_star_distribution") or product.get("displayed_by_star")

        coverage = {
            "reviews_analyzed": len(unique_analyses),
            "reviews_scraped": len(comments),
            "reviews_listed": available,
            "usable_reviews": usable_count,
            "duplicate_reviews": duplicates,
            "coverage_ratio": round(usable_count / available, 4) if available else None,
            "scrape_completeness": completeness,
            "completeness_reason": reason,
            "collected_by_star": collected_by_star,
            "listed_by_star": listed_by_star if isinstance(listed_by_star, dict) else None,
            "review_date_range": _date_range(analyzed_comments),
            "sampling": (
                "star_proportional_sample"
                if review_limit and review_limit > 0
                else "census_of_scraped_reviews"
            ),
        }

        targeted_order = str(product.get("review_order") or "").casefold()
        explicitly_representative = product.get("sampling_representative")
        if "sampling_representative" in product:
            representative = (
                None
                if explicitly_representative is None
                else bool(explicitly_representative)
            )
        else:
            representative = not product.get("applied_rating_filters") and not any(
                token in targeted_order for token in ("lowest", "three_star", "low_star")
            )

        summary = _summarize(
            model,
            unique_analyses,
            reviews_available=available,
            sampling_representative=representative,
            star_distribution_known=bool(listed_by_star) or available <= usable_count,
            product_type=product.get("product_type") or product.get("category"),
            coverage=coverage,
        )
        summary["usable_review_count"] = usable_count
        summary["duplicate_review_count"] = duplicates
        summary.setdefault("coverage", coverage)
        if review_limit and review_limit > 0:
            summary["analysis_limit"] = review_limit
        product["sentiment_summary"] = summary
    return data


def analyze_csv_rows(
    rows: list[dict[str, str]], model: DefaketiveSentimentModel, text_column: str | None
) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    output = []
    for index, row in enumerate(rows):
        text = _review_text(row, text_column)
        analysis = model.analyze(text, rating=row.get("rating") or row.get("stars"))
        fingerprint = model.fingerprint(text)
        duplicate_of = seen.get(fingerprint) if fingerprint else None
        if fingerprint and duplicate_of is None:
            seen[fingerprint] = index

        enriched: dict[str, Any] = dict(row)
        enriched.update(
            {
                "sentiment_label": analysis["label"],
                "sentiment_positive": analysis["scores"]["pos"],
                "sentiment_neutral": analysis["scores"]["neu"],
                "sentiment_negative": analysis["scores"]["neg"],
                "sentiment_compound": analysis["scores"]["compound"],
                "risk_detected": analysis["risk"]["detected"],
                "risk_score": analysis["risk"]["score"],
                "risk_categories": ",".join(analysis["risk"]["categories"]),
                "sentiment_evidence": json.dumps(
                    analysis["sentiment_evidence"], ensure_ascii=False
                ),
                "risk_evidence": json.dumps(analysis["risk"]["evidence"], ensure_ascii=False),
                "is_duplicate": duplicate_of is not None,
                "duplicate_of": "" if duplicate_of is None else duplicate_of,
                "model_version": analysis["model_version"],
            }
        )
        output.append(enriched)
    return output


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_analyzed{input_path.suffix.lower()}")


def run_file(input_path: Path, output_path: Path, text_column: str | None = None) -> None:
    model = HybridManuscriptRiskModel(allow_unqualified=True)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        analyzed = analyze_product_json(data, model)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(analyzed, file, ensure_ascii=False, indent=2)
        return
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        analyzed_rows = analyze_csv_rows(rows, model, text_column)
        if not analyzed_rows:
            raise ValueError("CSV contains no data rows")
        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(analyzed_rows[0]))
            writer.writeheader()
            writer.writerows(analyzed_rows)
        return
    raise ValueError("Input must be a .json or .csv file")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Shopee or Lazada English, Filipino, and Taglish reviews locally"
        )
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="Shopee/Lazada JSON or review CSV"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON/CSV path")
    parser.add_argument("--text-column", help="CSV column containing the review text")
    parser.add_argument("--text", help="Analyze one review and print JSON")
    args = parser.parse_args()

    try:
        if args.text is not None:
            result = HybridManuscriptRiskModel(allow_unqualified=True).analyze(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.input is None:
            parser.error("provide an input file or --text")
        output = args.output or _default_output(args.input)
        if args.input.resolve() == output.resolve():
            parser.error("output must be different from input to preserve the original data")
        run_file(args.input, output, args.text_column)
        print(f"Analysis saved to {output}")
        return 0
    except (OSError, ValueError, ModelSetupError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
