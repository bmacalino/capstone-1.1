"""Write Codex annotations with user resolutions for Lazada hard-example round 1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "outputs/lazada_hard_example_annotation_round_1/"
    "lazada_hard_example_round1_queue.json"
)
OUTPUT = (
    ROOT
    / "outputs/lazada_hard_example_annotation_round_1/"
    "lazada_hard_example_round1_codex_annotations.json"
)


YES_NUMBERS = {
    5,
    9,
    10,
    15,
    24,
    25,
    26,
    34,
    41,
    43,
    44,
    47,
    52,
    53,
    57,
    58,
    62,
    65,
    68,
    76,
    83,
    86,
    88,
    91,
    93,
    100,
    102,
    103,
    106,
    108,
    109,
    111,
    116,
    123,
    132,
    141,
    143,
    145,
    148,
    149,
    150,
    152,
    158,
    159,
    161,
    164,
    168,
    175,
    177,
    180,
    183,
    184,
    189,
    193,
    203,
    204,
    205,
    208,
    209,
    218,
    219,
    222,
    226,
    227,
    228,
    237,
    238,
    240,
    241,
    243,
    244,
    249,
}

UNSURE_REASONS = {
    98: (
        "The review says the actual item differs from the advertised picture, "
        "but it does not identify the difference or a functional problem."
    ),
    107: (
        "The earbuds did not light, but the reviewer had not established whether "
        "they were merely discharged or defective."
    ),
    155: (
        "The product initially did not work, but the review ends after the charging "
        "attempt and does not confirm the outcome."
    ),
    229: (
        "The item is described as far from the picture, but the difference and its "
        "effect on product function are not stated."
    ),
}

USER_RESOLUTIONS = {15: "no", 98: "no", 107: "yes", 154: "yes", 155: "no", 229: "no"}
USER_CORRECTION_REASONS = {
    15: "User approved: no. Charging the phone for about 50% does not establish a failure or depleted capacity.",
    154: "User approved: yes. The review explicitly reports unclear camera video, in addition to the wrong brand.",
}


def review_number(review_id: str) -> int:
    return int(review_id.rsplit("-", 1)[1])


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    annotations = []
    for row in rows:
        number = review_number(row["review_id"])
        source = "codex_manual_review"
        if number in USER_RESOLUTIONS:
            label = USER_RESOLUTIONS[number]
            reason = (
                USER_CORRECTION_REASONS[number]
                if number in USER_CORRECTION_REASONS
                else f"User resolved: {label}. Initial ambiguity: {UNSURE_REASONS[number]}"
            )
            source = "user_resolved"
        elif number in YES_NUMBERS:
            label = "yes"
            reason = "Reports an actual product defect or performance problem."
        elif number in UNSURE_REASONS:
            label = "unsure"
            reason = UNSURE_REASONS[number]
        else:
            label = "no"
            reason = "No confirmed problem with the actual product under the product-only policy."
        annotations.append(
            {
                "review_id": row["review_id"],
                "human_risk_detected": label,
                "decision_reason": reason,
                "annotation_source": source,
            }
        )

    if len(annotations) != 250:
        raise ValueError(f"Expected 250 annotations, found {len(annotations)}")
    if len({row["review_id"] for row in annotations}) != len(annotations):
        raise ValueError("Duplicate review IDs in annotations.")
    counts = Counter(row["human_risk_detected"] for row in annotations)
    expected = {"yes": 73, "no": 177}
    if dict(counts) != expected:
        raise ValueError(f"Unexpected label counts: {dict(counts)}")

    OUTPUT.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(annotations), "labels": counts}, indent=2))


if __name__ == "__main__":
    main()
