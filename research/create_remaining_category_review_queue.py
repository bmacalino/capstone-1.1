"""Create the remaining human category/severity review queue after the audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIRST_PASS = ROOT / "outputs" / "risk_development_round_1" / "risk_validation_ai_first_pass.csv"
COMPLETED_AUDIT = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "risk_validation_human_audit_completed.csv"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "risk_validation_remaining_category_review.csv"
)

FIELDS = [
    "review_id",
    "product_id",
    "product_name",
    "product_url",
    "category",
    "rating",
    "review_text",
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
    with FIRST_PASS.open(encoding="utf-8-sig", newline="") as handle:
        first_pass = list(csv.DictReader(handle))
    with COMPLETED_AUDIT.open(encoding="utf-8-sig", newline="") as handle:
        audit = list(csv.DictReader(handle))
    if len(audit) != 28 or any(row["status"].strip() != "Ready" for row in audit):
        raise ValueError("Completed audit must contain 28 Ready rows.")

    audited_ids = {row["review_id"] for row in audit}
    selected = [
        row
        for row in first_pass
        if row["human_risk_detected"] == "yes" and row["review_id"] not in audited_ids
    ]
    if len(selected) != 57:
        raise ValueError(f"Expected 57 remaining risk-positive rows, found {len(selected)}")

    output_rows = [
        {
            "review_id": row["review_id"],
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "product_url": row["product_url"],
            "category": row["category"],
            "rating": row["rating"],
            "review_text": row["review_text"],
            "first_pass_risk_detected": row["human_risk_detected"],
            "first_pass_risk_category": row["human_risk_category"],
            "first_pass_risk_severity": row["human_risk_severity"],
            "first_pass_reason": row["decision_reason"],
            "final_risk_detected": row["human_risk_detected"],
            "final_risk_category": row["human_risk_category"],
            "final_risk_severity": row["human_risk_severity"],
            "final_reason": row["decision_reason"],
            "status": "Needs_Review",
        }
        for row in selected
    ]
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "products": len({row["product_id"] for row in output_rows}),
                "status": "Needs_Review",
                "output": str(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
