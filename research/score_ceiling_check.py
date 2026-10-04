"""Estimate how much of the sentiment model's error is actually fixable.

The model makes 594 mistakes on the FiReCS test set, but a mistake only counts
as fixable if the dataset's label is one a careful reader would agree with.
Where the reader sides with the model instead, the dataset label is arguable
and no amount of modelling will "fix" it; where the reader can defend neither,
the example is genuinely ambiguous.

A human reads a blind sample of the disagreements, having seen neither the
dataset label nor the model's prediction, and this script compares their calls
against both. The result is an estimate of the ceiling: the share of current
errors that better modelling could realistically remove.

Sample built by the blind worksheet in research/neutral_ceiling_check.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_utils import normalize_sentiment_label

DEFAULT_WORKSHEET = ROOT / "research" / "neutral_ceiling_check.csv"
DEFAULT_KEY = ROOT / "research" / "neutral_ceiling_check_key.csv"
DEFAULT_OUTPUT = ROOT / "research" / "neutral_ceiling_check_result.json"
VALID_LABELS = ("negative", "neutral", "positive")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def classify(human: str, dataset: str, model: str) -> str:
    """Decide what one disagreement tells us about the ceiling."""
    if human == dataset:
        # The dataset was right and the model was wrong: a real, fixable error.
        return "fixable_model_error"
    if human == model:
        # The model's answer was defensible; the dataset label is the odd one out.
        return "arguable_dataset_label"
    return "ambiguous"


def score(args: argparse.Namespace) -> dict[str, Any]:
    key = {row["id"]: row for row in read_csv(args.key)}
    worksheet = read_csv(args.worksheet)

    verdicts: Counter[str] = Counter()
    unlabelled = 0
    invalid: list[str] = []
    rows: list[dict[str, str]] = []

    for row in worksheet:
        row_id = str(row.get("id") or "").strip()
        raw = str(row.get("your_label") or "").strip()
        if not raw:
            unlabelled += 1
            continue
        human = normalize_sentiment_label(raw)
        if human not in VALID_LABELS:
            invalid.append(f"{row_id}: {raw!r}")
            continue
        answer = key.get(row_id)
        if answer is None:
            invalid.append(f"{row_id}: not present in the key")
            continue
        dataset = normalize_sentiment_label(answer["dataset_label"])
        model = normalize_sentiment_label(answer["model_predicted"])
        verdict = classify(human, dataset, model)
        verdicts[verdict] += 1
        rows.append(
            {
                "id": row_id,
                "your_label": human,
                "dataset_label": dataset,
                "model_predicted": model,
                "verdict": verdict,
                "notes": str(row.get("notes") or "").strip(),
            }
        )

    if invalid:
        raise SystemExit(
            "Some rows could not be read. Use exactly negative, neutral or positive.\n  "
            + "\n  ".join(invalid)
        )
    if not verdicts:
        raise SystemExit(
            f"No labels found in {args.worksheet.name}. Fill in the your_label column "
            "with negative, neutral or positive for each row, then re-run."
        )

    scored = sum(verdicts.values())
    fixable_share = verdicts["fixable_model_error"] / scored
    total_errors = args.total_errors

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample": {
            "rows_scored": scored,
            "rows_left_blank": unlabelled,
            "sampling": "stratified across the four neutral-boundary error types, seed 42",
            "blind": True,
        },
        "verdicts": {
            "fixable_model_error": verdicts["fixable_model_error"],
            "arguable_dataset_label": verdicts["arguable_dataset_label"],
            "ambiguous": verdicts["ambiguous"],
        },
        "estimated_fixable_share": round(fixable_share, 4),
        "projection": {
            "total_test_errors": total_errors,
            "estimated_fixable_errors": round(total_errors * fixable_share, 1),
            "estimated_unfixable_errors": round(total_errors * (1 - fixable_share), 1),
            "note": (
                "Projected from a small sample; indicative of the ceiling's rough "
                "position, not a precise figure."
            ),
        },
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheet", type=Path, default=DEFAULT_WORKSHEET)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-errors", type=int, default=594)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = score(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    verdicts = report["verdicts"]
    scored = report["sample"]["rows_scored"]
    print(f"Scored {scored} blind judgements:")
    print(f"  model genuinely wrong      {verdicts['fixable_model_error']:>3}")
    print(f"  dataset label arguable     {verdicts['arguable_dataset_label']:>3}")
    print(f"  genuinely ambiguous        {verdicts['ambiguous']:>3}")
    projection = report["projection"]
    print(
        f"\nRoughly {report['estimated_fixable_share']:.0%} of the "
        f"{projection['total_test_errors']} test errors look fixable "
        f"(about {projection['estimated_fixable_errors']:.0f}); the rest are label "
        "disagreement or genuine ambiguity."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
