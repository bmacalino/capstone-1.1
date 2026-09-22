from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "risk_validation_remaining_category_review.csv"
)

UNCERTAIN_REVIEW_IDS: set[str] = set()

CORRECTIONS = {
    "lazada-r1-0017": {
        "final_risk_detected": "yes",
        "final_risk_category": "quality",
        "final_risk_severity": "2",
        "final_reason": (
            "The product was damaged after only one week, indicating an early "
            "durability or material-quality failure."
        ),
    },
    "lazada-r1-0062": {
        "final_risk_detected": "yes",
        "final_risk_category": "defect|performance",
        "final_risk_severity": "2",
        "final_reason": (
            "The battery drains quickly and the charging port or power-bank "
            "component does not work properly."
        ),
    },
    "lazada-r1-0144": {
        "final_risk_category": "performance",
        "final_risk_severity": "2",
        "final_reason": (
            "The monitor and main unit power on but cannot pair, so the system's "
            "connection function fails."
        ),
    },
    "lazada-r1-0239": {
        "final_risk_category": "performance|quality",
        "final_risk_severity": "2",
        "final_reason": (
            "Device cannot be added after following the manual, and the doorbell "
            "button is physically difficult to press."
        ),
    },
    "lazada-r1-0662": {
        "final_risk_category": "wrong_item|performance",
        "final_risk_severity": "1",
        "final_reason": (
            "The expected manual is missing, which makes setup difficult."
        ),
    },
    "lazada-r1-0949": {
        "final_risk_detected": "no",
        "final_risk_category": "",
        "final_risk_severity": "",
        "final_reason": (
            "Negative fit feedback only; the review does not establish a product "
            "failure or other concrete risk."
        ),
    },
    "lazada-r1-0976": {
        "final_risk_category": "performance",
        "final_risk_severity": "2",
        "final_reason": (
            "Audio is weak and the charging case drains quickly; the packaging "
            "comment does not establish a delivery failure."
        ),
    },
}


def main() -> None:
    with QUEUE_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError("Queue has no header.")
    if len(rows) != 57:
        raise ValueError(f"Expected 57 rows, found {len(rows)}.")

    review_ids = {row["review_id"] for row in rows}
    missing_uncertain = UNCERTAIN_REVIEW_IDS - review_ids
    missing_corrections = set(CORRECTIONS) - review_ids
    if missing_uncertain or missing_corrections:
        raise ValueError(
            f"Missing IDs: uncertain={sorted(missing_uncertain)}, "
            f"corrections={sorted(missing_corrections)}"
        )

    for row in rows:
        review_id = row["review_id"]
        if review_id in CORRECTIONS:
            row.update(CORRECTIONS[review_id])
        row["status"] = (
            "Needs_Review" if review_id in UNCERTAIN_REVIEW_IDS else "Ready"
        )

    with QUEUE_PATH.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ready = sum(row["status"] == "Ready" for row in rows)
    needs_review = sum(row["status"] == "Needs_Review" for row in rows)
    print(
        {
            "rows": len(rows),
            "ready": ready,
            "needs_review": needs_review,
            "corrections": sorted(CORRECTIONS),
            "uncertain": sorted(UNCERTAIN_REVIEW_IDS),
        }
    )


if __name__ == "__main__":
    main()
