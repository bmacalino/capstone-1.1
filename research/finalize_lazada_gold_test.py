"""Merge the completed blind adjudication into the locked Lazada gold test set."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "lazada_gold_test_v1"
ALL_PATH = OUTPUT_DIR / "lazada_round1_all_adjudicated.csv"
TEST_PATH = OUTPUT_DIR / "lazada_round1_test_adjudicated.csv"
GOLD_PATH = OUTPUT_DIR / "lazada_round1_gold_test_v1.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
BLIND_PATH = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "lazada_round1_blind_review_adjudicated.csv"
)
COMPLETED_PATH = (
    ROOT / "outputs" / "annotation_quality_round_1" / "Completed_annotate.csv"
)
SOURCE_ALL_PATH = (
    ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_all.csv"
)
SOURCE_TEST_PATH = (
    ROOT / "research" / "data" / "processed" / "lazada_round1" / "lazada_round1_test.csv"
)
BLIND_QUEUE_PATH = OUTPUT_DIR / "lazada_round1_remaining_blind_review.csv"

ALLOWED_SENTIMENTS = {"negative", "positive", "uncertain"}
ALLOWED_RISK = {"yes", "no"}
ALLOWED_CATEGORIES = {
    "damage",
    "defect",
    "delivery",
    "other",
    "performance",
    "quality",
    "seller_service",
    "wrong_item",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_blind_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        review_id = row["review_id"].strip()
        if not review_id or review_id in by_id:
            raise ValueError(f"Missing or duplicate blind-review ID: {review_id!r}")
        if row["status"].strip() != "Ready":
            raise ValueError(f"Blind-review row is not Ready: {review_id}")

        sentiment = row["final_sentiment"].strip().casefold()
        risk = row["final_risk_detected"].strip().casefold()
        categories = {
            value.strip().casefold()
            for value in row["final_risk_category"].replace(",", "|").split("|")
            if value.strip()
        }
        severity = row["final_risk_severity"].strip()
        reason = row["final_decision_reason"].strip()

        if sentiment not in ALLOWED_SENTIMENTS:
            raise ValueError(f"Unsupported sentiment for {review_id}: {sentiment}")
        if risk not in ALLOWED_RISK:
            raise ValueError(f"Unsupported risk label for {review_id}: {risk}")
        if categories.difference(ALLOWED_CATEGORIES):
            raise ValueError(f"Unsupported risk category for {review_id}: {categories}")
        if not reason:
            raise ValueError(f"Missing final decision reason for {review_id}")
        if risk == "yes":
            if not categories or not severity or int(float(severity)) not in {1, 2, 3}:
                raise ValueError(f"Incomplete active-risk fields for {review_id}")
        elif categories or severity:
            raise ValueError(f"No-risk row has active-risk fields: {review_id}")

        by_id[review_id] = row

    if len(by_id) != 55:
        raise ValueError(f"Expected 55 blind-review rows, found {len(by_id)}")
    return by_id


def merge_rows(
    rows: list[dict[str, str]], blind_by_id: dict[str, dict[str, str]]
) -> set[str]:
    merged: set[str] = set()
    for row in rows:
        review_id = row["review_id"].strip()
        blind = blind_by_id.get(review_id)
        if blind is None:
            continue

        risk = blind["final_risk_detected"].strip().casefold()
        row["adjudicated_label"] = blind["final_sentiment"].strip().casefold()
        row["risk_detected"] = risk
        row["risk_category"] = (
            blind["final_risk_category"].strip().casefold() if risk == "yes" else ""
        )
        row["risk_severity"] = (
            str(int(float(blind["final_risk_severity"]))) if risk == "yes" else ""
        )
        row["annotator_id"] = "blind_adjudicated_round1"
        row["annotation_notes"] = blind["final_decision_reason"].strip()
        row["split"] = "test"
        merged.add(review_id)
    return merged


def validate_locked_test(rows: list[dict[str, str]], expected_ids: set[str]) -> None:
    ids = [row["review_id"].strip() for row in rows]
    if len(rows) != 133 or len(set(ids)) != 133:
        raise ValueError("Locked test must contain 133 unique reviews.")
    if not expected_ids.issubset(ids):
        raise ValueError("Locked test is missing blind-adjudicated reviews.")
    if any(row["split"].strip().casefold() != "test" for row in rows):
        raise ValueError("Every locked-test row must remain in the test split.")
    if any(not row["annotation_notes"].strip() for row in rows):
        raise ValueError("Every locked-test row must include an adjudication note.")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_entry(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256(path)}


def main() -> None:
    blind_fields, blind_rows = read_csv(BLIND_PATH)
    required_blind = {
        "review_id",
        "status",
        "final_sentiment",
        "final_risk_detected",
        "final_risk_category",
        "final_risk_severity",
        "final_decision_reason",
    }
    if required_blind.difference(blind_fields):
        raise ValueError("Blind adjudication file is missing required final fields.")
    blind_by_id = validate_blind_rows(blind_rows)

    all_fields, all_rows = read_csv(ALL_PATH)
    test_fields, test_rows = read_csv(TEST_PATH)
    if all_fields != test_fields:
        raise ValueError("Combined and test datasets do not share the same schema.")

    all_merged = merge_rows(all_rows, blind_by_id)
    test_merged = merge_rows(test_rows, blind_by_id)
    expected_ids = set(blind_by_id)
    if all_merged != expected_ids or test_merged != expected_ids:
        raise ValueError("Not every blind adjudication matched both target datasets.")
    validate_locked_test(test_rows, expected_ids)

    write_csv(ALL_PATH, all_fields, all_rows)
    write_csv(TEST_PATH, test_fields, test_rows)
    write_csv(GOLD_PATH, test_fields, test_rows)

    sentiments = Counter(row["adjudicated_label"].strip().casefold() for row in test_rows)
    risks = Counter(row["risk_detected"].strip().casefold() for row in test_rows)
    categories = Counter()
    severities = Counter()
    for row in test_rows:
        if row["risk_detected"].strip().casefold() != "yes":
            continue
        categories.update(
            value.strip().casefold()
            for value in row["risk_category"].replace(",", "|").split("|")
            if value.strip()
        )
        severities[str(int(float(row["risk_severity"].strip())))] += 1

    prior_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = {
        "version": "lazada_gold_test_v1",
        "status": "final_locked",
        "created_at": prior_manifest.get("created_at"),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "originals_preserved": True,
            "merge_key": "review_id",
            "locked_test_rows": 133,
            "completed_adjudication_rows": 78,
            "blind_adjudication_rows": 55,
            "remaining_adjudication_rows": 0,
            "training_exclusion": (
                "All 133 gold reviews remain test-only and must not be used for model fitting "
                "or threshold selection."
            ),
        },
        "counts": {
            "all_rows": len(all_rows),
            "test_rows": len(test_rows),
            "test_products": len({row["product_id"].strip() for row in test_rows}),
            "test_risk": dict(sorted(risks.items())),
            "test_sentiment": dict(sorted(sentiments.items())),
            "test_risk_categories": dict(sorted(categories.items())),
            "test_risk_severity": dict(sorted(severities.items())),
        },
        "files": {
            "source_all": file_entry(SOURCE_ALL_PATH),
            "source_test": file_entry(SOURCE_TEST_PATH),
            "completed_annotations": file_entry(COMPLETED_PATH),
            "blind_review_queue": file_entry(BLIND_QUEUE_PATH),
            "blind_adjudication": file_entry(BLIND_PATH),
            "adjudicated_all": file_entry(ALL_PATH),
            "adjudicated_test": file_entry(TEST_PATH),
            "canonical_gold_test": file_entry(GOLD_PATH),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "all_rows": len(all_rows),
                "test_rows": len(test_rows),
                "blind_rows_merged": len(test_merged),
                "sentiment": dict(sorted(sentiments.items())),
                "risk": dict(sorted(risks.items())),
                "gold_sha256": sha256(GOLD_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
