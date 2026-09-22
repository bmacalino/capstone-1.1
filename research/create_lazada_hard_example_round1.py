"""Create a blind, product-disjoint Lazada risk annotation queue.

The queue targets examples that are useful for reducing false positives while
retaining difficult possible risks. Model probabilities are written only to an
internal audit file and never to the annotator-facing queue.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from defaketive_sentiment.model import DefaketiveSentimentModel  # noqa: E402
from defaketive_sentiment.trained import feature_text  # noqa: E402
from research.train_risk_scoring_v1 import positive_probability  # noqa: E402


SOURCE_LABELS = ROOT / "outputs/lazada_gold_test_v1/lazada_round1_all_adjudicated.csv"
MODEL = SENTIMENT_ROOT / "models/domain_adapted_product_risk_round2_candidate.joblib"
OUTPUT_DIR = ROOT / "outputs/lazada_hard_example_annotation_round_1"
QUEUE_JSON = OUTPUT_DIR / "lazada_hard_example_round1_queue.json"
AUDIT_JSON = OUTPUT_DIR / "lazada_hard_example_round1_selection_audit.json"
MANIFEST_JSON = OUTPUT_DIR / "manifest.json"

TARGET_PER_PRODUCT = 25
SEED = 42

TARGET_PRODUCTS: dict[str, dict[str, str]] = {
    "1120920366": {
        "category": "computer_peripherals",
        "source_file": "lazada_computer_peripherals_direct_1120920366.json",
    },
    "2531734184": {
        "category": "computer_peripherals",
        "source_file": "lazada_computer_peripherals_direct_2531734184.json",
    },
    "2313989346": {
        "category": "power",
        "source_file": "lazada_power_direct_batch_1.json",
    },
    "4693560169": {
        "category": "power",
        "source_file": "lazada_power.json",
    },
    "3169275025": {
        "category": "audio",
        "source_file": "lazada_audio_speakers.json",
    },
    "3431056306": {
        "category": "audio",
        "source_file": "lazada_audio_high_volume.json",
    },
    "3626751806": {
        "category": "smart_home",
        "source_file": "lazada_smart_home_direct_3626751806.json",
    },
    "4547888367": {
        "category": "smart_home",
        "source_file": "lazada_smart_home_direct_4547888367.json",
    },
    "4034782758": {
        "category": "mobile_accessories",
        "source_file": "lazada_mobile_accessories_direct_batch_2.json",
    },
    "4578043495": {
        "category": "mobile_accessories",
        "source_file": "lazada_mobile_accessories_direct_4578043495.json",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_id_from_url(url: object) -> str:
    match = re.search(r"(?:pdp-)?i(\d+)", str(url))
    if not match:
        raise ValueError(f"Cannot find Lazada product ID in URL: {url}")
    return match.group(1)


def normalized_text(text: object) -> str:
    return feature_text(DefaketiveSentimentModel.normalize(text))


def load_used_products_and_texts() -> tuple[set[str], set[str]]:
    with SOURCE_LABELS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        {str(row["product_id"]) for row in rows},
        {normalized_text(row["review_text"]) for row in rows},
    )


def load_product(product_id: str, source_file: str) -> dict[str, Any]:
    path = ROOT / "research/data" / source_file
    payload = json.loads(path.read_text(encoding="utf-8"))
    products = payload if isinstance(payload, list) else [payload]
    matches = [
        product
        for product in products
        if product.get("link") and product_id_from_url(product["link"]) == product_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one product {product_id} in {source_file}, found {len(matches)}"
        )
    return matches[0]


def candidate_rows(
    product_id: str,
    config: dict[str, str],
    used_texts: set[str],
) -> list[dict[str, Any]]:
    product = load_product(product_id, config["source_file"])
    unique: dict[str, dict[str, Any]] = {}
    for source_index, review in enumerate(product.get("comments", []), start=1):
        review_text = str(review.get("content", "")).strip()
        normalized = normalized_text(review_text)
        token_count = len(normalized.split())
        if (
            not normalized
            or normalized in used_texts
            or normalized in unique
            or token_count < 3
            or len(review_text) > 700
        ):
            continue
        try:
            rating = int(float(review.get("rating", 0)))
        except (TypeError, ValueError):
            continue
        if rating not in {1, 2, 3, 4, 5}:
            continue
        unique[normalized] = {
            "product_id": product_id,
            "product_name": str(product.get("name", "")).strip(),
            "product_url": str(product.get("link", "")).strip(),
            "category": config["category"],
            "rating": rating,
            "review_text": review_text,
            "normalized_review": normalized,
            "source_file": config["source_file"],
            "source_index": source_index,
        }
    return list(unique.values())


def select_product_rows(
    rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    for row in rows:
        probability = float(row["model_probability"])
        prediction = int(probability >= threshold)
        row["model_prediction"] = prediction
        row["distance_to_threshold"] = abs(probability - threshold)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()

    def take(candidates: list[dict[str, Any]], count: int, bucket: str) -> None:
        for row in candidates:
            key = (row["source_file"], row["source_index"])
            if key in selected_keys:
                continue
            picked = dict(row)
            picked["selection_bucket"] = bucket
            selected.append(picked)
            selected_keys.add(key)
            if sum(item["selection_bucket"] == bucket for item in selected) >= count:
                break

    high_star_risk = sorted(
        [row for row in rows if row["model_prediction"] == 1 and row["rating"] >= 4],
        key=lambda row: (row["distance_to_threshold"], -row["rating"]),
    )
    low_star_no_risk = sorted(
        [row for row in rows if row["model_prediction"] == 0 and row["rating"] <= 3],
        key=lambda row: (row["distance_to_threshold"], row["rating"]),
    )
    take(high_star_risk, 7, "high_star_predicted_risk")
    take(low_star_no_risk, 7, "low_star_predicted_no_risk")

    remaining = sorted(
        rows,
        key=lambda row: (
            row["distance_to_threshold"],
            hashlib.sha256(row["normalized_review"].encode("utf-8")).hexdigest(),
        ),
    )
    for row in remaining:
        if len(selected) >= TARGET_PER_PRODUCT:
            break
        key = (row["source_file"], row["source_index"])
        if key in selected_keys:
            continue
        counts = Counter(item["model_prediction"] for item in selected)
        desired = 0 if counts[0] < counts[1] else 1
        same_class = [
            candidate
            for candidate in remaining
            if candidate["model_prediction"] == desired
            and (candidate["source_file"], candidate["source_index"])
            not in selected_keys
        ]
        picked_source = same_class[0] if same_class else row
        picked_key = (picked_source["source_file"], picked_source["source_index"])
        if picked_key in selected_keys:
            continue
        picked = dict(picked_source)
        picked["selection_bucket"] = "near_threshold_class_balance"
        selected.append(picked)
        selected_keys.add(picked_key)

    if len(selected) != TARGET_PER_PRODUCT:
        raise ValueError(
            f"Could select only {len(selected)} of {TARGET_PER_PRODUCT} rows for "
            f"product {rows[0]['product_id'] if rows else 'unknown'}"
        )
    return selected


def main() -> None:
    used_products, used_texts = load_used_products_and_texts()
    overlap = used_products & set(TARGET_PRODUCTS)
    if overlap:
        raise ValueError(f"Target products overlap existing data: {sorted(overlap)}")

    artifact = joblib.load(MODEL)
    risk_model = artifact["risk_detection_pipeline"]
    threshold = float(artifact["risk_detection_threshold"])

    all_selected: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    available_counts: dict[str, int] = {}
    for product_id, config in TARGET_PRODUCTS.items():
        rows = candidate_rows(product_id, config, used_texts)
        available_counts[product_id] = len(rows)
        if len(rows) < TARGET_PER_PRODUCT:
            raise ValueError(
                f"Product {product_id} has only {len(rows)} eligible unique reviews."
            )
        probabilities = positive_probability(
            risk_model, [row["normalized_review"] for row in rows]
        )
        for row, probability in zip(rows, probabilities):
            row["model_probability"] = float(probability)
        all_selected.extend(select_product_rows(rows, threshold))
        source_path = ROOT / "research/data" / config["source_file"]
        source_hashes[config["source_file"]] = sha256(source_path)

    random.Random(SEED).shuffle(all_selected)
    queue_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index, row in enumerate(all_selected, start=1):
        review_id = f"lazada-hard-r1-{index:04d}"
        queue_rows.append(
            {
                "review_id": review_id,
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "product_url": row["product_url"],
                "category": row["category"],
                "rating": row["rating"],
                "review_text": row["review_text"],
                "human_risk_detected": "",
                "decision_reason": "",
            }
        )
        audit_rows.append({"review_id": review_id, **row})

    if len(queue_rows) != 250:
        raise ValueError(f"Expected 250 selected rows, found {len(queue_rows)}")
    if len({row["review_id"] for row in queue_rows}) != len(queue_rows):
        raise ValueError("Duplicate review IDs in annotation queue.")
    if len({normalized_text(row["review_text"]) for row in queue_rows}) != len(
        queue_rows
    ):
        raise ValueError("Duplicate normalized review texts in annotation queue.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_JSON.write_text(
        json.dumps(queue_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    AUDIT_JSON.write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Blind product-only risk annotation focused on likely false positives, "
            "false negatives, and near-threshold Lazada reviews."
        ),
        "risk_policy": "binary product-only yes/no with unsure for adjudication",
        "rows": len(queue_rows),
        "products": len(TARGET_PRODUCTS),
        "reviews_per_product": dict(
            sorted(Counter(row["product_id"] for row in queue_rows).items())
        ),
        "categories": dict(sorted(Counter(row["category"] for row in queue_rows).items())),
        "ratings": dict(
            sorted(Counter(str(row["rating"]) for row in queue_rows).items())
        ),
        "candidate_model": str(MODEL.relative_to(ROOT)),
        "candidate_model_sha256": sha256(MODEL),
        "selection_threshold": threshold,
        "selection_predictions": dict(
            sorted(
                Counter(
                    "risk" if row["model_prediction"] else "no_risk"
                    for row in audit_rows
                ).items()
            )
        ),
        "selection_buckets": dict(
            sorted(Counter(row["selection_bucket"] for row in audit_rows).items())
        ),
        "source_labels": str(SOURCE_LABELS.relative_to(ROOT)),
        "source_labels_sha256": sha256(SOURCE_LABELS),
        "existing_product_overlap": 0,
        "existing_normalized_review_overlap": 0,
        "available_eligible_reviews": available_counts,
        "source_files": [
            {"path": f"research/data/{name}", "sha256": digest}
            for name, digest in sorted(source_hashes.items())
        ],
        "annotator_facing_queue": str(QUEUE_JSON.relative_to(ROOT)),
        "internal_selection_audit": str(AUDIT_JSON.relative_to(ROOT)),
        "blind_fields_omitted_from_workbook": [
            "model_probability",
            "model_prediction",
            "distance_to_threshold",
            "selection_bucket",
        ],
    }
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
