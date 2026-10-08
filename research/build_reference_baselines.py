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
SOURCE = ROOT / "research" / "verdict_check_weighted.json"
REPORT = ROOT / "research" / "reference_baselines_report.json"
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
    REPORT.write_bytes((json.dumps(report, indent=1) + "\n").encode("utf-8"))
    print(f"safety: {report['safety']}")
    print(f"wrote {REPORT.name}; enough listings: {enough}")
    if "--write" in sys.argv and enough:
        baselines = {
            "version": "reference-baselines-1.0.0",
            "built_at": report["generated_at"],
            "measured_with": version,
            "source": "research/build_reference_baselines.py (research/reference_baselines_report.json)",
            "note": "Median flagged-review rate over saved gadget listings, star-weighted. In the checker's own units: rebuild when the checker changes.",
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
