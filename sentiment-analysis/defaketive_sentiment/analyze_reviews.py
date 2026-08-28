"""Apply the DeFaketive model to scraped JSON or review CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .model import DefaketiveSentimentModel, ModelSetupError


TEXT_FIELDS = ("content", "comment", "review", "text")


def _review_text(review: dict[str, Any], preferred: str | None = None) -> str:
    if preferred and preferred in review:
        return str(review.get(preferred) or "")
    for field in TEXT_FIELDS:
        if field in review:
            return str(review.get(field) or "")
    return ""


def analyze_product_json(data: Any, model: DefaketiveSentimentModel) -> Any:
    products = data if isinstance(data, list) else data.get("products", [])
    if not isinstance(products, list):
        raise ValueError("JSON must be a product list or an object containing a 'products' list")

    for product in products:
        if not isinstance(product, dict):
            continue
        comments = product.get("comments", [])
        if not isinstance(comments, list):
            continue

        seen: dict[str, int] = {}
        unique_analyses = []
        for index, comment in enumerate(comments):
            if not isinstance(comment, dict):
                continue
            text = _review_text(comment)
            fingerprint = model.fingerprint(text)
            duplicate_of = seen.get(fingerprint) if fingerprint else None
            if fingerprint and duplicate_of is None:
                seen[fingerprint] = index

            analysis = model.analyze(text, rating=comment.get("rating"))
            comment["sentiment_analysis"] = analysis
            comment["is_duplicate"] = duplicate_of is not None
            comment["duplicate_of"] = duplicate_of
            if duplicate_of is None and fingerprint:
                unique_analyses.append(analysis)

        product["sentiment_summary"] = model.summarize(unique_analyses)
        product["sentiment_summary"]["duplicate_review_count"] = sum(
            1 for comment in comments if isinstance(comment, dict) and comment.get("is_duplicate")
        )
    return data


def analyze_csv_rows(
    rows: list[dict[str, str]], model: DefaketiveSentimentModel, text_column: str | None
) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    output = []
    for index, row in enumerate(rows):
        text = _review_text(row, text_column)
        analysis = model.analyze(text, rating=row.get("rating") or row.get("stars"))
        fingerprint = model.fingerprint(text)
        duplicate_of = seen.get(fingerprint) if fingerprint else None
        if fingerprint and duplicate_of is None:
            seen[fingerprint] = index

        enriched: dict[str, Any] = dict(row)
        enriched.update(
            {
                "sentiment_label": analysis["label"],
                "sentiment_positive": analysis["scores"]["pos"],
                "sentiment_neutral": analysis["scores"]["neu"],
                "sentiment_negative": analysis["scores"]["neg"],
                "sentiment_compound": analysis["scores"]["compound"],
                "risk_detected": analysis["risk"]["detected"],
                "risk_score": analysis["risk"]["score"],
                "risk_categories": ",".join(analysis["risk"]["categories"]),
                "sentiment_evidence": json.dumps(
                    analysis["sentiment_evidence"], ensure_ascii=False
                ),
                "risk_evidence": json.dumps(analysis["risk"]["evidence"], ensure_ascii=False),
                "is_duplicate": duplicate_of is not None,
                "duplicate_of": "" if duplicate_of is None else duplicate_of,
                "model_version": analysis["model_version"],
            }
        )
        output.append(enriched)
    return output


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_analyzed{input_path.suffix.lower()}")


def run_file(input_path: Path, output_path: Path, text_column: str | None = None) -> None:
    model = DefaketiveSentimentModel()
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        analyzed = analyze_product_json(data, model)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(analyzed, file, ensure_ascii=False, indent=2)
        return
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        analyzed_rows = analyze_csv_rows(rows, model, text_column)
        if not analyzed_rows:
            raise ValueError("CSV contains no data rows")
        with output_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(analyzed_rows[0]))
            writer.writeheader()
            writer.writerows(analyzed_rows)
        return
    raise ValueError("Input must be a .json or .csv file")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Shopee or Lazada English, Filipino, and Taglish reviews locally"
        )
    )
    parser.add_argument(
        "input", nargs="?", type=Path, help="Shopee/Lazada JSON or review CSV"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON/CSV path")
    parser.add_argument("--text-column", help="CSV column containing the review text")
    parser.add_argument("--text", help="Analyze one review and print JSON")
    args = parser.parse_args()

    try:
        if args.text is not None:
            result = DefaketiveSentimentModel().analyze(args.text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.input is None:
            parser.error("provide an input file or --text")
        output = args.output or _default_output(args.input)
        if args.input.resolve() == output.resolve():
            parser.error("output must be different from input to preserve the original data")
        run_file(args.input, output, args.text_column)
        print(f"Analysis saved to {output}")
        return 0
    except (OSError, ValueError, ModelSetupError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
