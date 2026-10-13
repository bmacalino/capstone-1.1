"""How many listings get the safety warning, by how many separate reports are asked for.

Why. A severity-4 hazard (fire, explosion, swelling; shock, sparks, short
circuit) was a safety warning from one high-confidence report: the tier table
gives High for severity 4 however rare. On 12 October 2026 the owner chose to
ask for two separate reports (alert.single_report_tier in
hazard_parameters_v1.json); one report is now a caution. The parameter file
has always required the verdict changes at 1 and 3 reports to be reported.
This reports them, and what a rule with no count at all would do.

How. The 59 baseline listings as scanned from the review feed
(research/data/reference_scans/feed/). Each is read once by the app's checker
and its verdict is then worked out again by the app's own aggregator under
each rule, with nothing else changed:

  one_report     the rule until 12 Oct 2026 (single_report_tier "High")
  two_reports    the rule in use
  three_reports  three separate reports for the warning
  rate_only      no number of reports warns by itself: a severity-4 hazard
                 warns only when it is also common

A safety flag does not use the sentiment model, so the quick sentiment model
is used here and the run takes minutes, not half an hour. That is checked,
not assumed: the run stops if any listing's safety-flagged reviews differ from
the served checker's (research/verdict_check_feed.json), or if working a
verdict out again under the rule in use does not give the verdict the checker
gave.

    python research/safety_alert_sensitivity.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["DEFAKETIVE_SENTIMENT_MODEL"] = "tfidf"

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

SCANS = ROOT / "research" / "data" / "reference_scans" / "feed"
SERVED = ROOT / "research" / "verdict_check_feed.json"
OUT = ROOT / "research" / "safety_alert_sensitivity.json"
NO_COUNT_WARNS = 10**6
RULES = {
    "one_report": {"single_report_tier": "High"},
    "two_reports": {},
    "three_reports": {"independent_high_confidence_reports": 3},
    "rate_only": {"independent_high_confidence_reports": NO_COUNT_WARNS},
}
SAFETY_VERDICTS = ("safety_warning", "possible_safety_issue")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import registry
    from defaketive_sentiment.analyze_reviews import analyze_product_json
    from defaketive_sentiment.product_badges import ProductBadgeAggregator, load_parameters

    model = registry.load_model(registry.HAZARD_BADGES)
    shipped = load_parameters()
    aggregators = {}
    for name, settings in RULES.items():
        parameters = copy.deepcopy(shipped)
        parameters["alert"].update({key: {"value": value} for key, value in settings.items()})
        aggregators[name] = ProductBadgeAggregator(parameters=parameters, baselines=model.aggregator.baselines)

    served = {row["link"]: row for row in json.loads(SERVED.read_bytes().decode("utf-8"))["rows"]}
    listings = []
    for path in sorted(SCANS.glob("lazada_i*.json")):
        product = json.loads(path.read_bytes().decode("utf-8"))[0]
        analysed = analyze_product_json([copy.deepcopy(product)], model)[0]
        summary = analysed["sentiment_summary"]
        safety = summary["safety"]
        if safety["flagged_reviews"] != served[product["link"]]["safety_reviews"]:
            print(f"{path.name}: {safety['flagged_reviews']} safety-flagged reviews here, "
                  f"{served[product['link']]['safety_reviews']} with the served checker. Stopping.")
            return 1
        analyses = [
            comment["sentiment_analysis"] for comment in analysed.get("comments") or []
            if isinstance(comment, dict) and comment.get("included_in_analysis") and comment.get("sentiment_analysis")
        ]
        coverage = summary["coverage"]
        verdicts = {
            name: aggregator.summarize(
                analyses, reviews_available=coverage.get("reviews_listed"),
                product_type=product.get("product_type") or product.get("category"), coverage=coverage,
            )["verdict"]["state"]
            for name, aggregator in aggregators.items()
        }
        if verdicts["two_reports"] != summary["verdict"]["state"]:
            print(f"{path.name}: worked out again, the verdict is {verdicts['two_reports']}, "
                  f"but the checker gave {summary['verdict']['state']}. Stopping.")
            return 1
        listings.append({
            "link": product["link"], "ratings": product.get("total_rating"), "reviews_read": coverage["reviews_analyzed"],
            "safety_flagged_reviews": safety["flagged_reviews"],
            "separate_severity_4_reports": safety["independent_high_confidence_reports"],
            "worst_severity": safety["severity"], "how_common": safety["likelihood_bin"],
            "verdicts": verdicts,
        })

    table = {}
    for name in RULES:
        counts = Counter(listing["verdicts"][name] for listing in listings)
        table[name] = {
            "safety_warning": counts["safety_warning"],
            "possible_safety_issue": counts["possible_safety_issue"],
            "no_safety_verdict": sum(number for state, number in counts.items() if state not in SAFETY_VERDICTS),
        }
        print(f"{name:14} warning {table[name]['safety_warning']:2} | caution {table[name]['possible_safety_issue']:2}"
              f" | no safety verdict {table[name]['no_safety_verdict']:2}")
    warned_before = [listing for listing in listings if listing["verdicts"]["one_report"] == "safety_warning"]
    by_reports = Counter(listing["separate_severity_4_reports"] for listing in warned_before)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "listings": len(listings),
        "parameters_version": shipped["version"],
        "rules": {name: settings or "as shipped" for name, settings in RULES.items()},
        "verdicts": table,
        "warned_under_one_report_by_separate_severity_4_reports": dict(sorted(by_reports.items())),
        "moved_from_warning_to_caution_by_the_rule_in_use": sum(
            1 for listing in warned_before if listing["verdicts"]["two_reports"] == "possible_safety_issue"
        ),
        "rows": listings,
    }
    OUT.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print(f"warned under one report, by separate severity-4 reports: {report['warned_under_one_report_by_separate_severity_4_reports']}")
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
