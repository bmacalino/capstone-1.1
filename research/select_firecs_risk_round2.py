"""Select a targeted second-round FiReCS product-risk annotation sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import joblib


ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for path in (ROOT, SENTIMENT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research.cross_validate_firecs_product_risk import (  # noqa: E402
    DEFAULT_DATASET as ANNOTATED_DATASET,
    load_records as load_annotated_records,
)
from research.train_domain_adapted_product_risk import (  # noqa: E402
    DEFAULT_ARTIFACT,
)
from research.train_firecs_product_risk import normalized_text  # noqa: E402
from research.train_risk_scoring_v1 import positive_probability  # noqa: E402


DEFAULT_SOURCE = SENTIMENT_ROOT / "data/firecs/FiReCS_train_set.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/firecs_risk_annotation_round_2_targeted/_work/firecs_round2_selection.json"
)
LABEL_MAP = {
    "0": "negative",
    "0.0": "negative",
    "1": "neutral",
    "1.0": "neutral",
    "2": "positive",
    "2.0": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}
FILIPINO_SIGNAL = re.compile(
    r"\b(?:ako|ang|ano|ay|ba|bakit|di|dili|din|dito|daw|eh|ganito|ganda|"
    r"hindi|ito|kasi|ko|lang|mabilis|maganda|malakas|mahina|may|mga|mo|"
    r"naman|ng|nga|pero|po|pwede|salamat|siya|sulit|talaga|walang|yung)\b",
    re.IGNORECASE,
)
ELECTRONICS_SIGNAL = re.compile(
    r"\b(?:adapter|audio|battery|bluetooth|bulb|cable|camera|card reader|"
    r"charger|charging|cctv|computer|controller|cooler|device|earbuds?|"
    r"earphones?|electric|electronic|gamepad|gadget|headphones?|headsets?|"
    r"internet|keyboard|lamp|laptop|led|lowbat|malowbat|microphone|monitor|"
    r"mouse|phone|power ?bank|printer|rechargeable|remote(?: control)?|"
    r"ring ?light|router|screen|sim(?: card)?|speaker|stylus|tablet|"
    r"television|usb|volume|wifi|wi-fi|wire|wireless)\b",
    re.IGNORECASE,
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_candidates(source: Path, annotated_path: Path) -> list[dict[str, Any]]:
    annotated = load_annotated_records(annotated_path)
    annotated_texts = {
        normalized_text(record["review_text"]) for record in annotated
    }
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for source_row, row in enumerate(rows, start=2):
        review = str(row.get("review") or "").strip()
        label = LABEL_MAP.get(str(row.get("label") or "").strip().casefold())
        text = normalized_text(review)
        if not review or label is None or text in seen or text in annotated_texts:
            continue
        seen.add(text)
        candidates.append(
            {
                "source_split": "train",
                "source_row": source_row,
                "review_text": review,
                "normalized_text": text,
                "firecs_sentiment": label,
                "language_hint": (
                    "Filipino/Taglish signal"
                    if FILIPINO_SIGNAL.search(review)
                    else "English/other"
                ),
                "normalized_sha256": sha256_text(text),
            }
        )
    return candidates


def select_partition(
    available: list[dict[str, Any]],
    *,
    count: int,
    sort_key: Callable[[dict[str, Any]], Any],
    reverse: bool = False,
) -> list[dict[str, Any]]:
    ordered = sorted(available, key=sort_key, reverse=reverse)
    selected = ordered[:count]
    if len(selected) != count:
        raise ValueError("Not enough reviews for a requested sampling partition.")
    return selected


def select_sample(
    candidates: list[dict[str, Any]], artifact_path: Path, *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = joblib.load(artifact_path)
    pipeline = artifact["risk_detection_pipeline"]
    threshold = float(
        artifact.get(
            "risk_detection_threshold", getattr(pipeline, "threshold", 0.5)
        )
    )
    electronics_candidates = [
        row for row in candidates if ELECTRONICS_SIGNAL.search(row["review_text"])
    ]
    probabilities = positive_probability(
        pipeline, [row["normalized_text"] for row in electronics_candidates]
    )
    for row, probability in zip(electronics_candidates, probabilities):
        row["model_risk_probability"] = float(probability)
        row["absolute_threshold_margin"] = abs(float(probability) - threshold)

    selected: list[dict[str, Any]] = []
    quotas = {
        "negative": {"decision_boundary": 75, "likely_risk": 38, "likely_no_risk": 37},
        "neutral": {"decision_boundary": 85, "likely_risk": 43, "likely_no_risk": 42},
        "positive": {"decision_boundary": 40, "likely_risk": 20, "likely_no_risk": 20},
    }
    for sentiment in ("negative", "neutral", "positive"):
        pool = [
            row
            for row in electronics_candidates
            if row["firecs_sentiment"] == sentiment
        ]
        chosen_ids: set[str] = set()

        partitions = [
            (
                "decision_boundary",
                quotas[sentiment]["decision_boundary"],
                lambda row: row["absolute_threshold_margin"],
                False,
            ),
            (
                "likely_risk",
                quotas[sentiment]["likely_risk"],
                lambda row: row["model_risk_probability"],
                True,
            ),
            (
                "likely_no_risk",
                quotas[sentiment]["likely_no_risk"],
                lambda row: row["model_risk_probability"],
                False,
            ),
        ]
        for bucket, count, key, reverse in partitions:
            available = [
                row for row in pool if row["normalized_sha256"] not in chosen_ids
            ]
            picked = select_partition(
                available,
                count=count,
                sort_key=key,
                reverse=reverse,
            )
            for row in picked:
                row["sampling_bucket"] = bucket
                chosen_ids.add(row["normalized_sha256"])
            selected.extend(picked)

    random.Random(seed).shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["review_id"] = f"firecs-risk-r2-{index:04d}"

    summary = {
        "seed": seed,
        "model_version": artifact["metadata"]["model_version"],
        "model_artifact": str(artifact_path.relative_to(ROOT)),
        "risk_threshold": threshold,
        "source": str(DEFAULT_SOURCE.relative_to(ROOT)),
        "annotated_exclusion_source": str(ANNOTATED_DATASET.relative_to(ROOT)),
        "candidate_rows": len(candidates),
        "electronics_candidate_rows": len(electronics_candidates),
        "selected_rows": len(selected),
        "sentiment_counts": dict(
            sorted(Counter(row["firecs_sentiment"] for row in selected).items())
        ),
        "bucket_counts": dict(
            sorted(Counter(row["sampling_bucket"] for row in selected).items())
        ),
        "language_counts": dict(
            sorted(Counter(row["language_hint"] for row in selected).items())
        ),
        "source_split_counts": dict(
            sorted(Counter(row["source_split"] for row in selected).items())
        ),
        "selection_policy": (
            "Electronics/device reviews only. Within each sentiment stratum, half "
            "are closest to the model threshold and the remainder cover the highest "
            "and lowest predicted risk probabilities."
        ),
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--annotated", type=Path, default=ANNOTATED_DATASET)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    candidates = load_candidates(args.source.resolve(), args.annotated.resolve())
    selected, summary = select_sample(
        candidates, args.artifact.resolve(), seed=args.seed
    )
    payload = {"summary": summary, "records": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
