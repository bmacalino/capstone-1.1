"""Freeze a weakly-labelled Lazada sentiment set from the project's own scrapes.

FiReCS -- the only human-labelled sentiment corpus the project trains on --
covers products *and* services, and models trained on it lose most of their
accuracy on real marketplace reviews (research/marketplace_shift.json). The
scraped Lazada reviews are exactly the missing domain, but they carry no
sentiment annotation, only the reviewer's own star rating.

Star ratings are used as weak (distant) labels: 1-2 stars -> negative,
4-5 stars -> positive. 3-star reviews are DROPPED. On the 470 human-labelled
marketplace reviews, 1-2 stars matched the human label 92.9% of the time and
4-5 stars 95.8%, but 3 stars matched "neutral" only 10.5%; the neutral class is
therefore learned from FiReCS's human labels alone. That rule was chosen after
seeing those agreement figures, which come from the marketplace test set; this
is disclosed in the manifest. These are human-originated signals, not LLM
labels.

Leakage exclusions, all mandatory and counted in the manifest:
  * any review whose text appears in the marketplace test set or FiReCS test;
  * every review of a product that appears in the marketplace test set, matched
    on the marketplace item id (the test CSV stores full URLs, the scrapes store
    links or bare ids, so both are reduced to the id before comparing);
  * duplicates across scrape files, and anything already in FiReCS training.

The CSV contains scraped review text and is gitignored, like the scrapes it is
built from. The manifest -- sources, checksums, counts -- is what gets versioned;
rerunning this script on the same scrapes reproduces the CSV byte for byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_firecs import audit_key  # noqa: E402  - the project's leakage key

FIRECS_DIR = ROOT / "sentiment-analysis" / "data" / "firecs"
DEFAULT_MARKETPLACE_TEST = ROOT / "complete_manual_marketplace_reviews (1).csv"
DEFAULT_OUTPUT_DIR = ROOT / "sentiment-analysis" / "data" / "lazada_star_labels"
OUTPUT_NAME = "lazada_star_labels_v1.csv"
MANIFEST_NAME = "lazada_star_labels_v1.manifest.json"
VERSION = "lazada-star-labels-1.0.0"
# Scraper side files: product discovery, quality reports, metadata, and copies
# already carrying an analysis. None holds reviews that the plain files lack.
SKIPPED_FILE = re.compile(r"(discovery|quality|metadata|analyzed)", re.I)
COLUMNS = ("text", "label", "rating", "product_id", "platform", "source_file")
PRODUCT_ID_PATTERNS = (
    r"-i(\d{6,})",  # Lazada: .../products/pdp-i4644812363-s27518600306.html
    r"goods_id=(\d+)",  # Temu link
    r"-g-(\d+)",  # Temu slug link
    r"i\.\d+\.(\d+)",  # Shopee: ...-i.<shop>.<item>
    r"^(\d{6,})$",  # a bare id
)


def product_id(value: Any) -> str:
    """Reduce a marketplace link or id to the item id, or "" if none is found."""
    text = str(value or "").strip()
    for pattern in PRODUCT_ID_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def star_label(rating: int) -> str | None:
    """1-2 stars negative, 4-5 positive, 3 stars dropped (returns None)."""
    if rating in (1, 2):
        return "negative"
    if rating in (4, 5):
        return "positive"
    return None


def parse_rating(value: Any) -> int | None:
    try:
        rating = int(float(value))
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 5 else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def platform_of(path: Path) -> str:
    name = path.name.casefold()
    return "temu" if "temu" in name else "shopee" if "shopee" in name else "lazada"


def scrape_files(roots: Iterable[Path]) -> list[Path]:
    files = []
    for root in roots:
        files += [path for path in root.glob("*.json") if not SKIPPED_FILE.search(path.name)]
    return sorted(set(files), key=lambda path: str(path).casefold())


def iter_reviews(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the reviews in one scrape file, whatever the scraper's layout."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    products = data if isinstance(data, list) else data.get("products") or [data]
    for item in products:
        if not isinstance(item, dict):
            continue
        product = product_id(item.get("goods_id") or item.get("item_id") or item.get("link"))
        for comment in item.get("comments") or []:
            if isinstance(comment, dict):
                yield {
                    "text": str(comment.get("content") or comment.get("comment") or "").strip(),
                    "rating": parse_rating(comment.get("rating") or comment.get("stars")),
                    "product_id": product,
                }


def build(
    files: list[Path],
    marketplace_test: Path,
    firecs_dir: Path = FIRECS_DIR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the frozen rows and a manifest describing how they were made."""
    market_rows = read_csv(marketplace_test)
    test_texts = {audit_key(row["review_text"]) for row in market_rows}
    test_products = {product_id(row.get("product_id")) for row in market_rows} - {""}
    firecs_test = {audit_key(row["review"]) for row in read_csv(firecs_dir / "FiReCS_test_set.csv")}
    firecs_train = {audit_key(row["review"]) for row in read_csv(firecs_dir / "FiReCS_train_set.csv")}

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    sources = []
    for path in files:
        reviews = list(iter_reviews(path))
        sources.append({"file": path.name, "sha256": sha256(path), "reviews": len(reviews)})
        for review in reviews:
            key = audit_key(review["text"])
            if not key or review["rating"] is None:
                removed["no_text_or_rating"] += 1
                continue
            if key in seen:
                removed["duplicate"] += 1
                continue
            seen.add(key)
            # Leakage checks come before the 3-star rule so every overlap is counted.
            if key in test_texts:
                removed["marketplace_test_text"] += 1
                continue
            if review["product_id"] and review["product_id"] in test_products:
                removed["marketplace_test_product"] += 1
                continue
            if key in firecs_test:
                removed["firecs_test_text"] += 1
                continue
            if key in firecs_train:
                removed["already_in_firecs_train"] += 1
                continue
            label = star_label(review["rating"])
            if label is None:
                removed["three_star_dropped"] += 1
                continue
            rows.append(
                {
                    "text": review["text"],
                    "label": label,
                    "rating": review["rating"],
                    "product_id": review["product_id"],
                    "platform": platform_of(path),
                    "source_file": path.name,
                }
            )

    manifest = {
        "version": VERSION,
        "label_rule": {"1-2 stars": "negative", "4-5 stars": "positive", "3 stars": "dropped"},
        "label_type": "weak supervision: the reviewer's own star rating, not a sentiment annotation",
        "star_agreement_with_human_labels": star_agreement(market_rows),
        "rule_selection_disclosure": (
            "Dropping 3-star reviews was decided after seeing the agreement figures, "
            "which are computed on the marketplace test set."
        ),
        "exclusions": dict(sorted(removed.items())),
        "marketplace_test_products_excluded": len(test_products),
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "platform_counts": dict(sorted(Counter(row["platform"] for row in rows).items())),
        "products": len({row["product_id"] for row in rows if row["product_id"]}),
        "rows_without_product_id": sum(1 for row in rows if not row["product_id"]),
        "sources": sources,
    }
    return rows, manifest


def star_agreement(market_rows: list[dict[str, str]]) -> dict[str, Any]:
    """How often each star band matches the human label on the marketplace set."""
    bands = {"1-2": "negative", "3": "neutral", "4-5": "positive"}
    counts: dict[str, Counter[str]] = {band: Counter() for band in bands}
    for row in market_rows:
        rating = parse_rating(row.get("rating"))
        human = str(row.get("adjudicated_label") or "").strip().casefold()
        if rating is None or human not in {"negative", "neutral", "positive"}:
            continue
        band = "1-2" if rating <= 2 else "3" if rating == 3 else "4-5"
        counts[band][human] += 1
    result = {}
    for band, expected in bands.items():
        total = sum(counts[band].values())
        result[band] = {
            "treated_as": expected,
            "reviews": total,
            "human_agrees": counts[band][expected],
            "agreement": round(counts[band][expected] / total, 4) if total else None,
            "human_labels": dict(sorted(counts[band].items())),
        }
    return result


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrape-dirs", nargs="+", type=Path, default=[ROOT, ROOT / "research" / "data"])
    parser.add_argument("--marketplace-test", type=Path, default=DEFAULT_MARKETPLACE_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows, manifest = build(scrape_files(args.scrape_dirs), args.marketplace_test)
    payload = render_csv(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / OUTPUT_NAME).write_bytes(payload)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": OUTPUT_NAME,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        **manifest,
    }
    (args.output_dir / MANIFEST_NAME).write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    print(f"{manifest['rows']} rows {manifest['label_counts']} from {manifest['products']} products")
    print(f"excluded: {manifest['exclusions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
