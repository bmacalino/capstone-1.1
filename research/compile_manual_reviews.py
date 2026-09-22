"""Compile deduplicated local marketplace reviews into an annotation worksheet."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def products(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("products"), list):
        return [item for item in payload["products"] if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def platform_for(path: Path, product: dict[str, Any]) -> str:
    return str(
        product.get("platform")
        or product.get("market")
        or next(
            (name for name in ("lazada", "shopee", "temu") if name in path.name.casefold()),
            "unknown",
        )
    ).casefold()


def review_id(index: int, path: Path, text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:32]
    return f"local-{index:04d}-{path.stem[:16]}-{slug}"


def compile_reviews(root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.json")):
        if path.name.endswith("_analyzed.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for product_index, product in enumerate(products(payload)):
            platform = platform_for(path, product)
            product_id = str(product.get("link") or product.get("id") or f"{path.stem}:{product_index}")
            for review in product.get("comments") or []:
                if not isinstance(review, dict):
                    continue
                text = str(review.get("content") or review.get("comment") or "").strip()
                key = " ".join(text.casefold().split())
                if not text or key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "review_id": review_id(len(rows) + 1, path, text),
                        "source_file": path.name,
                        "platform": platform,
                        "product_id": product_id,
                        "rating": review.get("rating") or "",
                        "review_text": text,
                        "model_prediction": "",
                        "sentiment_label": "",
                        "risk_detected": "",
                        "risk_category": "",
                        "risk_severity": "",
                        "negated": "",
                        "annotator_id": "",
                        "annotation_notes": "",
                        "adjudicated_label": "",
                    }
                )
                if limit and len(rows) >= limit:
                    return rows
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = compile_reviews(args.root, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(rows[0]) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Compiled {len(rows)} reviews into {args.output}")


if __name__ == "__main__":
    main()