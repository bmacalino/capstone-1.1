"""Build the resolved FiReCS risk round-two and combined development datasets.

Round two contains Codex product-only annotations. Seven ambiguous records were
resolved directly by the user. These labels are development data, not an
independent human gold test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROUND2_DIR = ROOT / "outputs/firecs_risk_annotation_round_2_targeted"
DEFAULT_SELECTION = ROUND2_DIR / "_work/firecs_round2_selection.json"
DEFAULT_ANNOTATIONS = ROUND2_DIR / "_work/firecs_round2_annotator_2.json"
DEFAULT_EXISTING = (
    ROOT / "outputs/firecs_risk_annotation_final/firecs_product_only_final.json"
)
DEFAULT_ROUND2 = ROUND2_DIR / "firecs_product_only_round2_resolved.json"
DEFAULT_COMBINED = ROUND2_DIR / "firecs_product_only_combined_development.json"
DEFAULT_MANIFEST = ROUND2_DIR / "firecs_product_only_combined_manifest.json"

USER_RESOLVED_IDS = {
    "firecs-risk-r2-0018",
    "firecs-risk-r2-0022",
    "firecs-risk-r2-0034",
    "firecs-risk-r2-0155",
    "firecs-risk-r2-0251",
    "firecs-risk-r2-0308",
    "firecs-risk-r2-0347",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_review(text: object) -> str:
    return " ".join(str(text or "").casefold().split())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_datasets(
    selection_path: Path,
    annotations_path: Path,
    existing_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selection_payload = load_json(selection_path)
    annotation_payload = load_json(annotations_path)
    existing = load_json(existing_path)

    selected = selection_payload.get("records", [])
    annotations = annotation_payload.get("annotations", [])
    if len(selected) != 400 or len(annotations) != 400:
        raise ValueError("Round two must contain exactly 400 selections and annotations.")

    selected_by_id = {str(row["review_id"]): row for row in selected}
    annotation_by_id = {str(row["review_id"]): row for row in annotations}
    if len(selected_by_id) != 400 or len(annotation_by_id) != 400:
        raise ValueError("Round two contains duplicate review IDs.")
    if set(selected_by_id) != set(annotation_by_id):
        raise ValueError("Selection and annotation review IDs do not match.")

    round2: list[dict[str, Any]] = []
    for selected_row in selected:
        review_id = str(selected_row["review_id"])
        annotation = annotation_by_id[review_id]
        risk = str(annotation.get("risk") or "").strip().casefold()
        if risk not in {"yes", "no"}:
            raise ValueError(f"Unresolved risk label for {review_id}: {risk!r}")
        round2.append(
            {
                "review_id": review_id,
                "source_row": selected_row.get("source_row"),
                "source_split": str(selected_row.get("source_split") or ""),
                "review_text": str(selected_row["review_text"]).strip(),
                "firecs_sentiment": str(
                    selected_row.get("firecs_sentiment") or ""
                ).strip().casefold(),
                "risk_detected": risk,
                "annotation_source": (
                    "user_resolved" if review_id in USER_RESOLVED_IDS
                    else "codex_product_only_review"
                ),
            }
        )

    if not isinstance(existing, list) or len(existing) != 1000:
        raise ValueError("Expected the existing adjudicated FiReCS set to contain 1,000 rows.")
    combined = [*existing, *round2]
    ids = [str(row["review_id"]) for row in combined]
    normalized = [normalized_review(row["review_text"]) for row in combined]
    if len(set(ids)) != len(ids):
        raise ValueError("Combined FiReCS dataset contains duplicate review IDs.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Combined FiReCS dataset contains duplicate normalized reviews.")

    label_counts = Counter(str(row["risk_detected"]) for row in combined)
    round2_label_counts = Counter(row["risk_detected"] for row in round2)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_labels_not_independent_gold_test",
        "limitation": (
            "Round-two labels were produced by Codex under the product-only rule. "
            "The user resolved seven ambiguous rows, but did not independently label "
            "all 400 records. Use for development and error analysis, not final accuracy claims."
        ),
        "existing_rows": len(existing),
        "round2_rows": len(round2),
        "combined_rows": len(combined),
        "combined_label_counts": dict(sorted(label_counts.items())),
        "round2_label_counts": dict(sorted(round2_label_counts.items())),
        "round2_annotation_sources": dict(
            sorted(Counter(row["annotation_source"] for row in round2).items())
        ),
        "user_resolved_review_ids": sorted(USER_RESOLVED_IDS),
        "source_files": [
            {"path": str(existing_path.relative_to(ROOT)), "sha256": sha256(existing_path)},
            {"path": str(selection_path.relative_to(ROOT)), "sha256": sha256(selection_path)},
            {"path": str(annotations_path.relative_to(ROOT)), "sha256": sha256(annotations_path)},
        ],
        "integrity": {
            "unique_review_ids": len(set(ids)),
            "unique_normalized_reviews": len(set(normalized)),
            "round2_source_splits": sorted({row["source_split"] for row in round2}),
            "unresolved_labels": 0,
        },
    }
    return round2, combined, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--round2-output", type=Path, default=DEFAULT_ROUND2)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    round2, combined, manifest = build_datasets(
        args.selection.resolve(), args.annotations.resolve(), args.existing.resolve()
    )
    for output_path, payload in (
        (args.round2_output, round2),
        (args.combined_output, combined),
        (args.manifest, manifest),
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
