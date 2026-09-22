"""Create a focused human audit of the AI risk-validation first pass."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "risk_development_round_1" / "risk_validation_ai_first_pass.csv"
OUTPUT = ROOT / "outputs" / "risk_development_round_1" / "risk_validation_human_audit.csv"

AMBIGUOUS = {
    "lazada-r1-0104": "Platform complaint without a product claim.",
    "lazada-r1-0118": "Very vague app-performance complaint.",
    "lazada-r1-0122": "Several missing, construction, and operational issues overlap.",
    "lazada-r1-0214": "Return dispute may reflect buyer compatibility rather than product failure.",
    "lazada-r1-0309": "Installer availability may be external to the product.",
    "lazada-r1-0379": "Delivery wording is unclear.",
    "lazada-r1-0504": "Handedness mismatch may be buyer selection or wrong variation.",
    "lazada-r1-0610": "Placeholder text should normally be no-risk but is low-information.",
    "lazada-r1-0707": "Reviewer is unsure whether mobile connection is unavailable or undiscovered.",
    "lazada-r1-0771": "Initial setup issue appears resolved.",
    "lazada-r1-0811": "Fit limitation affects only the smaller laptop.",
    "lazada-r1-0835": "Pairing took three hours but ultimately succeeded.",
    "lazada-r1-0904": "Poor packaging is reported without actual damage.",
}

OUTPUT_FIELDS = [
    "review_id",
    "product_id",
    "rating",
    "review_text",
    "audit_type",
    "audit_reason",
    "first_pass_risk_detected",
    "first_pass_risk_category",
    "first_pass_risk_severity",
    "first_pass_reason",
    "final_risk_detected",
    "final_risk_category",
    "final_risk_severity",
    "final_reason",
    "status",
]


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["review_id"]: row for row in rows}
    if not set(AMBIGUOUS).issubset(by_id):
        raise ValueError("An ambiguous audit ID is missing from the first-pass file.")

    rng = random.Random(42)
    remaining = [row for row in rows if row["review_id"] not in AMBIGUOUS]
    strata = {
        "no": [row for row in remaining if row["human_risk_detected"] == "no"],
        "severity_1": [row for row in remaining if row["human_risk_severity"] == "1"],
        "severity_2": [row for row in remaining if row["human_risk_severity"] == "2"],
        "severity_3": [row for row in remaining if row["human_risk_severity"] == "3"],
    }
    sample_sizes = {"no": 5, "severity_1": 3, "severity_2": 4, "severity_3": 3}
    sampled: dict[str, str] = {}
    for stratum, candidates in strata.items():
        for row in rng.sample(candidates, sample_sizes[stratum]):
            sampled[row["review_id"]] = f"Random audit sample from {stratum}."

    selected = []
    for row in rows:
        review_id = row["review_id"]
        if review_id in AMBIGUOUS:
            audit_type = "ambiguous"
            audit_reason = AMBIGUOUS[review_id]
        elif review_id in sampled:
            audit_type = "random"
            audit_reason = sampled[review_id]
        else:
            continue
        selected.append(
            {
                "review_id": review_id,
                "product_id": row["product_id"],
                "rating": row["rating"],
                "review_text": row["review_text"],
                "audit_type": audit_type,
                "audit_reason": audit_reason,
                "first_pass_risk_detected": row["human_risk_detected"],
                "first_pass_risk_category": row["human_risk_category"],
                "first_pass_risk_severity": row["human_risk_severity"],
                "first_pass_reason": row["decision_reason"],
                "final_risk_detected": "",
                "final_risk_category": "",
                "final_risk_severity": "",
                "final_reason": "",
                "status": "",
            }
        )

    if len(selected) != 28:
        raise ValueError(f"Expected 28 audit rows, found {len(selected)}")
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(selected)

    print(
        json.dumps(
            {
                "rows": len(selected),
                "audit_types": dict(Counter(row["audit_type"] for row in selected)),
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
