"""Merge reviewed risk decisions into a corrected 138-row validation dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_VALIDATION = (
    ROOT
    / "research"
    / "data"
    / "processed"
    / "lazada_round1"
    / "lazada_round1_validation.csv"
)
FIRST_PASS = (
    ROOT
    / "outputs"
    / "risk_development_round_1"
    / "risk_validation_ai_first_pass.csv"
)
HUMAN_AUDIT = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "risk_validation_human_audit_completed.csv"
)
CATEGORY_REVIEW = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "risk_validation_remaining_category_review.csv"
)
OUTPUT = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "lazada_round1_validation_adjudicated.csv"
)
REPORT = (
    ROOT
    / "outputs"
    / "annotation_quality_round_1"
    / "lazada_round1_validation_adjudicated_summary.json"
)

VALID_CATEGORIES = {
    "defect",
    "performance",
    "quality",
    "damage",
    "delivery",
    "wrong_item",
    "seller_service",
    "other",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def by_id(rows: list[dict[str, str]], source: str) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        review_id = row["review_id"].strip()
        if not review_id or review_id in indexed:
            raise ValueError(f"{source} contains a blank or duplicate review_id: {review_id!r}")
        indexed[review_id] = row
    return indexed


def normalize_decision(
    risk: str, category: str, severity: str, reason: str, review_id: str
) -> tuple[str, str, str, str]:
    risk = risk.strip().casefold()
    category = category.strip().casefold().replace(",", "|")
    severity = severity.strip()
    reason = reason.strip()
    if risk not in {"yes", "no"}:
        raise ValueError(f"{review_id}: invalid risk decision {risk!r}")
    if not reason:
        raise ValueError(f"{review_id}: decision reason is blank")
    if risk == "no":
        if category or severity:
            raise ValueError(f"{review_id}: no-risk row must have blank category/severity")
        return risk, "", "", reason
    categories = [value.strip() for value in category.split("|") if value.strip()]
    if not categories or set(categories) - VALID_CATEGORIES:
        raise ValueError(f"{review_id}: invalid risk categories {category!r}")
    if severity:
        try:
            severity_number = float(severity)
        except ValueError as exc:
            raise ValueError(f"{review_id}: invalid severity {severity!r}") from exc
        if not severity_number.is_integer():
            raise ValueError(f"{review_id}: invalid severity {severity!r}")
        severity = str(int(severity_number))
    if severity not in {"1", "2", "3"}:
        raise ValueError(f"{review_id}: invalid severity {severity!r}")
    return risk, "|".join(dict.fromkeys(categories)), severity, reason


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base_rows = read_rows(BASE_VALIDATION)
    first_pass = by_id(read_rows(FIRST_PASS), "first pass")
    audit = by_id(read_rows(HUMAN_AUDIT), "human audit")
    category_review = by_id(read_rows(CATEGORY_REVIEW), "category review")
    base = by_id(base_rows, "base validation")

    if len(base_rows) != 138 or len(first_pass) != 138:
        raise ValueError(
            f"Expected 138 base and first-pass rows; found {len(base_rows)} and {len(first_pass)}"
        )
    if len(audit) != 28 or len(category_review) != 57:
        raise ValueError(
            f"Expected 28 audit and 57 category rows; found {len(audit)} and {len(category_review)}"
        )
    if set(audit) & set(category_review):
        raise ValueError("Audit and category-review IDs overlap.")
    if set(base) != set(first_pass):
        raise ValueError("Base validation and first-pass IDs do not match.")
    if (set(audit) | set(category_review)) - set(base):
        raise ValueError("Reviewed files contain IDs outside the validation set.")
    if any(row.get("status", "").strip() != "Ready" for row in audit.values()):
        raise ValueError("Every human-audit row must be Ready.")
    if any(row.get("status", "").strip() != "Ready" for row in category_review.values()):
        raise ValueError("Every category-review row must be Ready.")

    output_rows: list[dict[str, str]] = []
    sources: Counter[str] = Counter()
    original_changes: Counter[str] = Counter()
    for original in base_rows:
        review_id = original["review_id"].strip()
        if review_id in audit:
            decision = audit[review_id]
            risk, category, severity, reason = normalize_decision(
                decision["final_risk_detected"],
                decision["final_risk_category"],
                decision["final_risk_severity"],
                decision["final_reason"],
                review_id,
            )
            source = "human_audit"
        elif review_id in category_review:
            decision = category_review[review_id]
            risk, category, severity, reason = normalize_decision(
                decision["final_risk_detected"],
                decision["final_risk_category"],
                decision["final_risk_severity"],
                decision["final_reason"],
                review_id,
            )
            source = "reviewed_risk_positive"
        else:
            decision = first_pass[review_id]
            risk, category, severity, reason = normalize_decision(
                decision["human_risk_detected"],
                decision["human_risk_category"],
                decision["human_risk_severity"],
                decision["decision_reason"],
                review_id,
            )
            if risk != "no":
                raise ValueError(
                    f"{review_id}: an unaudited first-pass row is unexpectedly risk-positive"
                )
            source = "first_pass_no_risk"

        updated = dict(original)
        updated["risk_detected"] = risk
        updated["risk_category"] = category
        updated["risk_severity"] = severity
        updated["annotator_id"] = source
        updated["annotation_notes"] = reason
        output_rows.append(updated)
        sources[source] += 1

        if original.get("risk_detected", "").strip().casefold() != risk:
            original_changes["risk_detected"] += 1
        if original.get("risk_category", "").strip().casefold() != category:
            original_changes["risk_category"] += 1
        if original.get("risk_severity", "").strip() != severity:
            original_changes["risk_severity"] += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    risk_counts = Counter(row["risk_detected"] for row in output_rows)
    severity_counts = Counter(
        row["risk_severity"] for row in output_rows if row["risk_detected"] == "yes"
    )
    category_counts: Counter[str] = Counter()
    for row in output_rows:
        if row["risk_detected"] == "yes":
            category_counts.update(row["risk_category"].split("|"))

    report = {
        "rows": len(output_rows),
        "unique_review_ids": len({row["review_id"] for row in output_rows}),
        "label_sources": dict(sorted(sources.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "changes_from_original_validation": dict(sorted(original_changes.items())),
        "input_sha256": {
            "base_validation": sha256(BASE_VALIDATION),
            "first_pass": sha256(FIRST_PASS),
            "human_audit": sha256(HUMAN_AUDIT),
            "category_review": sha256(CATEGORY_REVIEW),
        },
        "output_sha256": sha256(OUTPUT),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
