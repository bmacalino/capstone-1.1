"""Audit FiReCS classification errors by language and review characteristics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PATTERNS = {
    "contrast": r"\b(?:but|pero|kaso|subalit|ngunit|however)\b",
    "negation": r"\b(?:not|no|never|hindi|di|ndi|wala|walang|ayaw|wag|huwag)\b",
    "delivery": r"\b(?:delivery|deliver|shipping|ship|dumating|nadeliver|courier)\b",
    "functionality": r"\b(?:work|works|working|charge|charging|gumana|gumagana|sira|defect|defective)\b",
    "order_accuracy": r"\b(?:color|colour|kulay|size|mali|wrong|kulang|missing|complete)\b",
    "packaging": r"\b(?:package|packaging|box|balot|parcel)\b",
    "risk_language": r"\b(?:fake|counterfeit|damage|damaged|basag|leak|spill|overheat|sunog)\b",
}


def _bucket(length: int) -> str:
    if length <= 5:
        return "1-5 words"
    if length <= 15:
        return "6-15 words"
    if length <= 40:
        return "16-40 words"
    return "41+ words"


def classify_error(review: str) -> dict[str, Any]:
    normalized = review.casefold()
    words = re.findall(r"[\w']+", normalized, flags=re.UNICODE)
    flags = {
        name: bool(re.search(pattern, normalized, flags=re.IGNORECASE))
        for name, pattern in PATTERNS.items()
    }
    flags["likely_taglish"] = bool(
        re.search(r"\b(?:ang|ng|po|yan|yung|sana|naman|kasi|kaya|pero|hindi|maganda|sira)\b", normalized)
    )
    flags["emoji_or_symbol"] = not bool(re.search(r"\w", normalized, flags=re.UNICODE))
    return {
        "word_count": len(words),
        "length_bucket": _bucket(len(words)),
        "flags": flags,
    }


def audit(path: Path, example_limit: int = 3) -> dict[str, Any]:
    errors = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            review = str(row.get("review") or "").strip()
            if not review:
                continue
            details = classify_error(review)
            errors.append({**row, **details})

    by_confusion: Counter[str] = Counter()
    by_feature: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in errors:
        confusion = f"{row['actual_label']} -> {row['predicted_label']}"
        by_confusion[confusion] += 1
        for name, present in row["flags"].items():
            if present:
                by_feature[name][confusion] += 1
                if len(examples[name]) < example_limit:
                    examples[name].append(
                        {
                            "actual": row["actual_label"],
                            "predicted": row["predicted_label"],
                            "review": row["review"],
                        }
                    )

    return {
        "source": path.name,
        "error_count": len(errors),
        "confusions": dict(by_confusion.most_common()),
        "length_buckets": dict(
            Counter(row["length_bucket"] for row in errors).most_common()
        ),
        "features": {
            name: {
                "error_count": sum(counts.values()),
                "confusions": dict(counts.most_common()),
                "examples": examples[name],
            }
            for name, counts in sorted(by_feature.items(), key=lambda item: -sum(item[1].values()))
        },
        "recommendation": (
            "Prioritize contrast/aspect-aware labeling and training if those features "
            "dominate the error groups; otherwise inspect the largest confusion pair first."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("errors", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-limit", type=int, default=3)
    args = parser.parse_args()
    result = audit(args.errors, args.example_limit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()