"""What verdict does each listing get from the app as it runs today?

Every listing is put through the app's own path (registry.load_model,
analyze_reviews.analyze_product_json) and its verdict, the share of reviews
flagged for each label and the review count are recorded. Nothing is changed.

Two sets of listings:

  default   research/data/reference_scans/: fresh scans with per-star counts
            (research/scan_reference_listings.py). Their rates are weighted
            back to the listing, so these are the figures to trust.
  --saved   every scan file already on disk. Most were collected one star
            level at a time and carry no per-star counts, so the app marks
            them incomplete and answers "Not enough reviews to judge". For
            them a second verdict is computed AS IF the sample were complete.
            It is too harsh, because low-star reviews are over-represented;
            it is there to show how often the warning line would be crossed,
            not to judge any listing.

  --star-totals   the saved scans again, each joined with its listing's
            ratings per star level read afterwards
            (research/fetch_star_totals.py), so every review is weighted by
            how many ratings its star level stands for: the same weights the
            app gives a star-by-star scan (analyze_reviews._stratum_weights).
            A listing is used only if the star levels that were sampled hold
            at least 90 in 100 of its ratings and its rating count has not
            moved by more than a quarter since the scan.

    python research/verdict_check.py
    python research/verdict_check.py --saved
    python research/verdict_check.py --star-totals
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

WEIGHTED = "--star-totals" in sys.argv
SAVED = "--saved" in sys.argv or WEIGHTED
REFERENCE = ROOT / "research" / "data" / "reference_scans"
STAR_TOTALS = REFERENCE / "star_totals.json"
REPORT = ROOT / "research" / (
    "verdict_check_weighted.json" if WEIGHTED else "verdict_check_saved_scans.json" if SAVED else "verdict_check.json"
)
SAMPLED_SHARE_AT_LEAST = 0.90
TOTAL_MAY_MOVE = 0.25
MINIMUM_REVIEWS = 30
FLAGS = ("defect", "counterfeit")


def scan_files() -> list[Path]:
    if not SAVED:
        return sorted(path for path in REFERENCE.glob("lazada_*.json"))
    from build_labelling_queue import _iter_scrape_files

    return list(_iter_scrape_files())


def listings() -> list[dict[str, Any]]:
    """One product per link: the scan of it that holds the most written reviews."""
    best: dict[str, dict[str, Any]] = {}
    for path in scan_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        products = data if isinstance(data, list) else data.get("products", []) if isinstance(data, dict) else []
        for product in products:
            if not isinstance(product, dict) or not product.get("link"):
                continue
            written = sum(
                1 for comment in product.get("comments") or []
                if isinstance(comment, dict) and str(comment.get("content") or "").strip()
            )
            if written >= MINIMUM_REVIEWS and written > best.get(product["link"], {}).get("_written", 0):
                best[product["link"]] = {**product, "_written": written, "_file": path.name}
    return list(best.values())


def weighted_row(model, product: dict[str, Any], analysed: dict[str, Any], totals: dict[str, Any] | None) -> dict[str, Any]:
    """The listing's verdict and rates with each review weighted by its star level."""
    from defaketive_sentiment.analyze_reviews import _stratum, _stratum_weights

    if not totals or totals.get("status") != "read":
        return {"usable": False, "why_not": "no star totals"}
    listed = totals["ratings_by_star"]
    comments = [
        comment for comment in analysed.get("comments") or []
        if isinstance(comment, dict) and comment.get("included_in_analysis") and comment.get("sentiment_analysis")
    ]
    weights = _stratum_weights(listed, comments)
    total = sum(int(value) for value in listed.values())
    sampled_share = sum(int(listed[str(star)]) for star in weights) / total if total else 0.0
    out: dict[str, Any] = {
        "ratings_by_star": listed, "analysed_by_star": dict(sorted(Counter(str(_stratum(c)) for c in comments).items())),
        "sampled_share_of_ratings": round(sampled_share, 4),
    }
    try:
        then = int(float(product.get("total_rating") or product.get("reported_review_count") or 0))
    except (TypeError, ValueError):
        then = 0
    out["ratings_at_scan"] = then
    if sampled_share < SAMPLED_SHARE_AT_LEAST:
        return {**out, "usable": False, "why_not": "the sampled star levels hold too few of the listing's ratings"}
    if then and abs(total - then) > TOTAL_MAY_MOVE * then:
        return {**out, "usable": False, "why_not": "the rating count moved by more than a quarter since the scan"}
    analyses = [
        {**comment["sentiment_analysis"], "sampling_weight": round(weights.get(_stratum(comment), 0.0), 6)}
        for comment in comments
    ]
    summary = model.summarize(analyses)
    out.update(usable=True, verdict_weighted=summary["verdict"]["state"], weighted=bool(summary.get("weighted")))
    for name in FLAGS:
        for key in ("rate", "rate_lower_bound", "rate_ci_95"):
            out[f"{name}_{key}_weighted"] = summary["flags"][name].get(key)
    out["safety_reviews"] = (summary.get("safety") or {}).get("flagged_reviews")
    for name, value in (summary.get("reports") or {}).items():
        out[f"{name}_rate_weighted"] = value.get("rate")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import registry
    from defaketive_sentiment.analyze_reviews import analyze_product_json

    model = registry.load_model(registry.HAZARD_BADGES)
    star_totals = json.loads(STAR_TOTALS.read_bytes().decode("utf-8"))["listings"] if WEIGHTED else {}
    rows = []
    for product in listings():
        source = {key: value for key, value in product.items() if not key.startswith("_")}
        analysed = analyze_product_json([copy.deepcopy(source)], model)[0]
        summary = analysed.get("sentiment_summary") or {}
        coverage = summary.get("coverage") or {}
        flags = summary.get("flags") or {}
        row: dict[str, Any] = {
            "name": str(product.get("name") or "")[:70], "link": product["link"], "file": product["_file"],
            "reviews_analysed": coverage.get("reviews_analyzed"),
            "collection": coverage.get("scrape_completeness"), "sampling": coverage.get("sampling"),
            "weighted": bool(summary.get("weighted")),
            "verdict": (summary.get("verdict") or {}).get("state"),
            "safety_reviews": (summary.get("safety") or {}).get("flagged_reviews"),
            **{
                f"{name}_{key}": (flags.get(name) or {}).get(key)
                for name in FLAGS for key in ("flagged_reviews", "rate", "rate_lower_bound")
            },
            **{f"{name}_reviews": value.get("flagged_reviews") for name, value in (summary.get("reports") or {}).items()},
        }
        if SAVED:
            analyses = [
                comment["sentiment_analysis"] for comment in analysed.get("comments") or []
                if isinstance(comment, dict) and comment.get("included_in_analysis") and comment.get("sentiment_analysis")
            ]
            as_if = model.summarize([{k: v for k, v in item.items() if k != "sampling_weight"} for item in analyses])
            row["verdict_as_if_complete"] = as_if["verdict"]["state"]
            for name in FLAGS:
                row[f"{name}_rate_as_if_complete"] = as_if["flags"][name]["rate"]
                row[f"{name}_lower_bound_as_if_complete"] = as_if["flags"][name]["rate_lower_bound"]
        if WEIGHTED:
            row.update(weighted_row(model, product, analysed, star_totals.get(product["link"])))
            print(f"{'USED' if row['usable'] else 'left out':8} {str(row.get('verdict_weighted', row.get('why_not')))[:44]:44} "
                  f"faulty {row.get('defect_rate_weighted')} fake {row.get('counterfeit_rate_weighted')} | {row['name'][:36]}", flush=True)
            rows.append(row)
            continue
        rows.append(row)
        print(f"{row['verdict']:30} {str(row.get('verdict_as_if_complete', '')):30} n {row['reviews_analysed']:4} "
              f"faulty {row['defect_flagged_reviews']:3} fake {row['counterfeit_flagged_reviews']:3} | {row['name'][:44]}", flush=True)
    verdicts = Counter(row["verdict"] for row in rows)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": getattr(model, "checker_version", model.model_version), "listings": len(rows),
        "which": "saved scans, mostly star-filtered and unweighted" if SAVED else "fresh reference scans, weighted to the listing",
        "verdicts": dict(verdicts),
    }
    if WEIGHTED:
        used = [row for row in rows if row.get("usable")]
        report["which"] = "saved scans weighted by star totals read afterwards (research/fetch_star_totals.py)"
        report["usable_listings"] = len(used)
        report["left_out"] = dict(Counter(row["why_not"] for row in rows if not row.get("usable")))
        report["verdicts_weighted"] = dict(Counter(row["verdict_weighted"] for row in used))
        report["rule"] = {"sampled_share_at_least": SAMPLED_SHARE_AT_LEAST, "total_may_move": TOTAL_MAY_MOVE}
    elif SAVED:
        report["verdicts_as_if_complete"] = dict(Counter(row["verdict_as_if_complete"] for row in rows))
        report["warning"] = (
            "The as-if-complete verdicts treat star-filtered samples as whole listings. Low-star reviews are "
            "over-represented in them, so they overstate how often a warning would show."
        )
    report["rows"] = rows
    REPORT.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print({key: report[key] for key in report if key.startswith("verdicts")})
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
