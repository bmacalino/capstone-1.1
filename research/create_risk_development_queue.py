"""Create blind risk-only annotation queues from non-test marketplace splits."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "lazada_gold_test_v1" / "lazada_round1_all_adjudicated.csv"
OUTPUT_DIR = ROOT / "outputs" / "risk_development_round_1"
QUEUE = OUTPUT_DIR / "risk_validation_blind_queue.csv"
MANIFEST = OUTPUT_DIR / "manifest.json"

OUTPUT_FIELDS = [
    "review_id",
    "product_id",
    "product_name",
    "product_url",
    "category",
    "rating",
    "review_text",
    "source_split",
    "human_risk_detected",
    "human_risk_category",
    "human_risk_severity",
    "decision_reason",
    "status",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    validation_rows = [
        row for row in source_rows if row.get("split", "").strip().casefold() == "validation"
    ]
    if len(validation_rows) != 138:
        raise ValueError(f"Expected 138 locked validation rows, found {len(validation_rows)}")
    if len({row["review_id"] for row in validation_rows}) != len(validation_rows):
        raise ValueError("Validation queue contains duplicate review IDs.")

    queue_rows = [
        {
            "review_id": row["review_id"],
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "product_url": row["product_url"],
            "category": row["category"],
            "rating": row["rating"],
            "review_text": row["review_text"],
            "source_split": "validation",
            "human_risk_detected": "",
            "human_risk_category": "",
            "human_risk_severity": "",
            "decision_reason": "",
            "status": "",
        }
        for row in validation_rows
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(queue_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Blind correction of risk labels used for model selection and threshold tuning.",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "queue": str(QUEUE.relative_to(ROOT)),
        "queue_sha256": sha256(QUEUE),
        "rows": len(queue_rows),
        "products": len({row["product_id"] for row in queue_rows}),
        "test_rows_included": 0,
        "labels_hidden": [
            "adjudicated_label",
            "risk_detected",
            "risk_category",
            "risk_severity",
            "annotation_notes",
            "model_predictions",
        ],
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
