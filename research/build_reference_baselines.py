"""What share of reviews reports a fault, or a fake, on an ordinary gadget listing?

The app's verdict warns when a listing's rate is clearly above a line. Until
now the line was a placeholder (3% faulty, 2% fake;
lexicons/hazard_parameters_v1.json). This measures it.

Input: research/verdict_check_weighted.json, written by
`python research/verdict_check.py --star-totals`: every saved gadget listing
through the app's own checker, each review weighted by how many ratings its
star level has on the listing (totals read by research/fetch_star_totals.py).

Fixed before the leave-one-out counts below were computed:
  * one entry per Lazada item (a listing scanned under two links counts once,
    the scan with more reviews);
  * "typical" is the median listing's rate: half of the listings are below
    it. The mean is pulled up by a few bad listings and is reported only;
  * the app's rule is unchanged: warn when the LOWER bound of a listing's
    rate is above max(typical, floor). The floor stays as a minimum;
  * leave-one-out: each listing is judged against the median of the others,
    so no listing helps set its own line.

Limits, stated: the rates are the checker's flags, not verified faults, so
the line is in the checker's own units and must be rebuilt when the checker
changes (the file records the version). Star totals were read on 8 October
2026 and the reviews were collected in September. Each listing's 5-star
level is about 40 reviews standing for thousands of ratings, so one
listing's rate is rough; the median over listings is not.

    python research/build_reference_baselines.py            # report only
    python research/build_reference_baselines.py --write    # also write the baseline file

With --feed the input is research/verdict_check_feed.json instead
(`python research/verdict_check.py --feed-scans`): the same listings scanned
again on 12 October 2026 from Lazada's review feed, as the app scans with it
(research/scan_baseline_listings.py). The feed carries whole reviews where
the page reader kept the first part of a review written in several parts,
and the checker flags more in the whole text, so a listing read from the feed
needs a line measured the same way. Every choice above is unchanged; the
report goes to reference_baselines_feed_report.json and the baseline file is
written as version 2.0.0. In these scans the reviews and the ratings per star
level are read in the same half minute, and each star level holds up to 60
reviews, not about 40.

    python research/build_reference_baselines.py --feed
    python research/build_reference_baselines.py --feed --write
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEED = "--feed" in sys.argv
PAGE_SOURCE = ROOT / "research" / "verdict_check_weighted.json"
SOURCE = ROOT / "research" / "verdict_check_feed.json" if FEED else PAGE_SOURCE
REPORT = ROOT / "research" / ("reference_baselines_feed_report.json" if FEED else "reference_baselines_report.json")
BASELINES = ROOT / "sentiment-analysis" / "models" / "reference_baselines_v1.json"
PARAMETERS = ROOT / "sentiment-analysis" / "defaketive_sentiment" / "lexicons" / "hazard_parameters_v1.json"
PRODUCT_TYPE = "gadget"
FLAGS = ("defect", "counterfeit")
MINIMUM_LISTINGS = 10
SEED = 20261008


def item_id(link: str) -> str:
    match = re.search(r"-i(\d{5,})", link)
    return match.group(1) if match else link


def listings() -> tuple[list[dict[str, Any]], str]:
    report = json.loads(SOURCE.read_bytes().decode("utf-8"))
    best: dict[str, dict[str, Any]] = {}
    for row in report["rows"]:
        if not row.get("usable"):
            continue
        key = item_id(row["link"])
        if key not in best or (row.get("reviews_analysed") or 0) > (best[key].get("reviews_analysed") or 0):
            best[key] = row
    return list(best.values()), report["model_version"]


def against_page_scans(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The same listings as read off the page (Table 12c), listing by listing."""
    earlier = {
        item_id(row["link"]): row
        for row in json.loads(PAGE_SOURCE.read_bytes().decode("utf-8"))["rows"] if row.get("usable")
    }
    pairs = [(earlier[item_id(row["link"])], row) for row in rows if item_id(row["link"]) in earlier]
    result: dict[str, Any] = {"listings_in_both": len(pairs)}
    for name in FLAGS:
        before = [float(old[f"{name}_rate_weighted"]) for old, _ in pairs]
        after = [float(new[f"{name}_rate_weighted"]) for _, new in pairs]
        change = [b - a for a, b in zip(before, after)]
        result[name] = {
            "median_rate_page_scans": round(statistics.median(before), 6),
            "median_rate_feed_scans": round(statistics.median(after), 6),
            "median_change": round(statistics.median(change), 6),
            "listings_higher": sum(1 for value in change if value > 0),
            "listings_lower": sum(1 for value in change if value < 0),
            "listings_same": sum(1 for value in change if value == 0),
        }
    result["reviews_analysed_median"] = {
        "page_scans": statistics.median(old.get("reviews_analysed") or 0 for old, _ in pairs),
        "feed_scans": statistics.median(new.get("reviews_analysed") or 0 for _, new in pairs),
    }
    return result


def median_interval(values: list[float]) -> list[float]:
    rng = np.random.default_rng(SEED)
    data = np.array(values)
    medians = [float(np.median(rng.choice(data, size=len(data), replace=True))) for _ in range(5000)]
    return [round(float(np.percentile(medians, 2.5)), 6), round(float(np.percentile(medians, 97.5)), 6)]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows, version = listings()
    floors = json.loads(PARAMETERS.read_bytes().decode("utf-8"))["flags"]["absolute_floor"]
    result: dict[str, Any] = {}
    for name in FLAGS:
        floor = float(floors[name]["value"] if isinstance(floors[name], dict) else floors[name])
        rates = [float(row[f"{name}_rate_weighted"]) for row in rows]
        lower = [float(row[f"{name}_rate_lower_bound_weighted"]) for row in rows]
        typical = statistics.median(rates)
        above_floor = sum(1 for value in lower if value > floor)
        above_line = 0
        for index, value in enumerate(lower):
            others = rates[:index] + rates[index + 1:]
            above_line += value > max(statistics.median(others), floor)
        result[name] = {
            "listings": len(rates), "typical_rate_median": round(typical, 6), "median_ci_95": median_interval(rates),
            "mean": round(statistics.mean(rates), 6),
            "percentiles": {str(p): round(float(np.percentile(rates, p)), 6) for p in (25, 50, 75, 90)},
            "listings_with_rate_zero": sum(1 for value in rates if value == 0),
            "placeholder_floor": floor,
            "warned_under_placeholder_floor": above_floor,
            "warned_under_measured_line_leave_one_out": int(above_line),
            "line_in_use_if_written": round(max(typical, floor), 6),
        }
        r = result[name]
        print(f"{name:12} listings {r['listings']} | typical (median) {typical:.3%} CI {r['median_ci_95']} | mean {r['mean']:.3%}"
              f" | 75th {r['percentiles']['75']:.3%} 90th {r['percentiles']['90']:.3%} | zero on {r['listings_with_rate_zero']}"
              f" | warned: floor {floor:.0%} -> {above_floor}, measured line -> {above_line}")
    enough = len(rows) >= MINIMUM_LISTINGS
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measured_with": version, "listings": len(rows), "enough_listings": enough,
        "rule": "typical = median listing rate; warn when a listing's lower bound is above max(typical, floor); leave-one-out",
        "flags": result,
        "safety": {
            "listings_with_any_safety_flagged_review": sum(1 for row in rows if (row.get("safety_reviews") or 0) > 0),
            "weighted_verdicts": {
                state: sum(1 for row in rows if row.get("verdict_weighted") == state)
                for state in sorted({row.get("verdict_weighted") for row in rows})
            },
        },
    }
    if FEED:
        report["against_page_scans"] = against_page_scans(rows)
        print(f"against the page scans: {report['against_page_scans']}")
        # The scans were analysed with the earlier line in force. A safety verdict uses no rate and
        # stays; the others follow the app's rule (lower bound above the line) with this line.
        lines = {name: result[name]["line_in_use_if_written"] for name in FLAGS}
        under: dict[str, int] = {}
        for row in rows:
            verdict = row["verdict_weighted"]
            if verdict in ("no_warning_signs", "possible_counterfeit_or_defect"):
                warned = any(float(row[f"{name}_rate_lower_bound_weighted"]) > lines[name] for name in FLAGS)
                verdict = "possible_counterfeit_or_defect" if warned else "no_warning_signs"
            under[verdict] = under.get(verdict, 0) + 1
        report["safety"]["verdicts_under_this_line"] = dict(sorted(under.items()))
    REPORT.write_bytes((json.dumps(report, indent=1) + "\n").encode("utf-8"))
    print(f"safety: {report['safety']}")
    print(f"wrote {REPORT.name}; enough listings: {enough}")
    if "--write" in sys.argv and enough:
        baselines = {
            "version": "reference-baselines-2.0.0" if FEED else "reference-baselines-1.0.0",
            "built_at": report["generated_at"],
            "measured_with": version,
            "source": f"research/build_reference_baselines.py (research/{REPORT.name})",
            "note": (
                "Median flagged-review rate over gadget listings scanned from Lazada's review feed as the app scans them, "
                "star-weighted. In the checker's own units, and for whole review text: rebuild when the checker or the "
                "review source changes."
                if FEED else
                "Median flagged-review rate over saved gadget listings, star-weighted. In the checker's own units: rebuild when the checker changes."
            ),
            "default_product_type": PRODUCT_TYPE,
            "product_types": {
                PRODUCT_TYPE: {
                    name: {"pooled_rate": result[name]["typical_rate_median"], "listings": result[name]["listings"],
                           "median_ci_95": result[name]["median_ci_95"]}
                    for name in FLAGS
                }
            },
        }
        BASELINES.write_bytes((json.dumps(baselines, indent=1) + "\n").encode("utf-8"))
        print(f"wrote {BASELINES.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
