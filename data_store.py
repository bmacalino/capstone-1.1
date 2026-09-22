"""Small centralized SQLite store for analyzed data and research results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB = Path(__file__).resolve().parent / "defaketive.db"

# Columns added when the single risk score was replaced by badges. Applied by
# _migrate() to databases created before the change, so an existing
# defaketive.db keeps working without being rebuilt.
PRODUCT_BADGE_COLUMNS = {
    "verdict_state": "TEXT",
    "safety_tier": "TEXT",
    "safety_severity": "INTEGER",
    "safety_alert": "INTEGER",
    "counterfeit_state": "TEXT",
    "defect_state": "TEXT",
    "reviews_analyzed": "INTEGER",
    "reviews_listed": "INTEGER",
    "scrape_completeness": "TEXT",
    "summary_schema_version": "TEXT",
    "analyzed_at": "TEXT",
}

REVIEW_BADGE_COLUMNS = {
    "flag_categories": "TEXT",
    "max_severity": "INTEGER",
    "high_confidence": "INTEGER",
    "sentence_tags_json": "TEXT",
}


def _migrate(connection: sqlite3.Connection) -> list[str]:
    """Add any missing columns. Idempotent, and safe on a populated database.

    SQLite has no ALTER TABLE ... IF NOT EXISTS, so the existing columns are
    read first. `risk_score` and `reliability_score` are deliberately NOT
    dropped: DROP COLUMN support depends on the SQLite version, the columns are
    harmless, and leaving them makes `WHERE risk_score IS NOT NULL` a clean way
    to find rows still holding a pre-badge score.
    """
    applied: list[str] = []
    connection.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY, applied_at TEXT NOT NULL
        )"""
    )
    for table, columns in (
        ("products", PRODUCT_BADGE_COLUMNS),
        ("reviews", REVIEW_BADGE_COLUMNS),
    ):
        existing = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for column, declared_type in columns.items():
            if column in existing:
                continue
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declared_type}")
            applied.append(f"{table}.{column}")
    if applied:
        connection.execute(
            "INSERT OR REPLACE INTO schema_migrations(name, applied_at) VALUES (?, ?)",
            ("badges_v2", datetime.now(timezone.utc).isoformat()),
        )
    return applied


@contextmanager
def _connect(path: str | Path = DEFAULT_DB):
    connection = sqlite3.connect(Path(path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
        CREATE TABLE IF NOT EXISTS products (
            link TEXT PRIMARY KEY, name TEXT NOT NULL, platform TEXT NOT NULL,
            category TEXT, price TEXT, rating REAL, review_count INTEGER NOT NULL,
            risk_score REAL, reliability_score REAL, source TEXT, raw_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviews (
            review_key TEXT PRIMARY KEY, product_link TEXT NOT NULL,
            author TEXT, rating INTEGER, reviewed_at TEXT, content TEXT,
            sentiment TEXT, risk_detected INTEGER NOT NULL, raw_json TEXT NOT NULL,
            FOREIGN KEY(product_link) REFERENCES products(link) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS survey_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, submitted_at TEXT NOT NULL,
            shopping_frequency TEXT, sus_score REAL NOT NULL, umux_score REAL NOT NULL,
            responses_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_at TEXT NOT NULL,
            source_name TEXT, sample_count INTEGER NOT NULL, accuracy REAL NOT NULL,
            precision REAL NOT NULL, recall REAL NOT NULL, f1 REAL NOT NULL,
            labels_json TEXT NOT NULL, matrix_json TEXT NOT NULL
        );
            """
        )
        _migrate(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _insert(connection: sqlite3.Connection, table: str, values: dict[str, Any], key: str) -> None:
    """Insert or update one row, naming every column explicitly.

    The previous version used positional `INSERT INTO products VALUES (?, ...)`,
    which silently depends on the column order and breaks as soon as a column is
    added. Naming the columns is what makes the badge migration safe.
    """
    columns = list(values)
    placeholders = ", ".join("?" for _ in columns)
    assignments = ", ".join(f"{name}=excluded.{name}" for name in columns if name != key)
    connection.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({key}) DO UPDATE SET {assignments}",
        tuple(values.values()),
    )


def _product_badge_values(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten the badge summary into queryable columns.

    The whole summary already lives in `raw_json`; these columns exist so the
    admin views and exports can filter without parsing every payload.
    """
    safety = summary.get("safety") or {}
    flags = summary.get("flags") or {}
    coverage = summary.get("coverage") or {}
    return {
        "verdict_state": (summary.get("verdict") or {}).get("state"),
        "safety_tier": safety.get("tier"),
        "safety_severity": safety.get("severity"),
        "safety_alert": int(bool(safety.get("alert"))) if safety else None,
        "counterfeit_state": (flags.get("counterfeit") or {}).get("state"),
        "defect_state": (flags.get("defect") or {}).get("state"),
        "reviews_analyzed": coverage.get("reviews_analyzed"),
        "reviews_listed": coverage.get("reviews_listed"),
        "scrape_completeness": coverage.get("scrape_completeness"),
        "summary_schema_version": summary.get("schema_version"),
        "analyzed_at": summary.get("analyzed_at"),
    }


def _review_badge_values(analysis: dict[str, Any]) -> dict[str, Any]:
    tags = analysis.get("review_tags")
    if tags is None:
        return {
            "flag_categories": None,
            "max_severity": None,
            "high_confidence": None,
            "sentence_tags_json": None,
        }
    sentences = analysis.get("sentences") or []
    return {
        "flag_categories": ",".join(sorted(tags)) if tags else "",
        "max_severity": analysis.get("max_severity"),
        "high_confidence": int(bool(analysis.get("review_high_confidence"))),
        "sentence_tags_json": json.dumps(
            [
                {
                    "index": sentence.get("index"),
                    "tags": sentence.get("tags"),
                    "hazard_subtype": sentence.get("hazard_subtype"),
                }
                for sentence in sentences
                if sentence.get("tags")
            ],
            ensure_ascii=False,
        ),
    }


def save_products(products: list[dict[str, Any]], source: str = "", path: str | Path = DEFAULT_DB) -> None:
    from dashboard_utils import platform_name, product_reliability

    now = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        for product in products:
            link = str(product.get("link") or "").strip()
            if not link:
                continue
            summary = product.get("sentiment_summary") or {}
            comments = [item for item in product.get("comments") or [] if isinstance(item, dict)]
            badges = _product_badge_values(summary)
            row = {
                "link": link,
                "name": str(product.get("name") or "Unnamed product"),
                "platform": platform_name(product),
                "category": str(product.get("category") or ""),
                "price": str(product.get("price") or ""),
                "rating": product.get("rating"),
                "review_count": len(comments),
                # Written only by the legacy weighted-sum models. A badge
                # summary leaves both NULL, so `WHERE risk_score IS NOT NULL`
                # finds rows still carrying a pre-badge score.
                "risk_score": summary.get("risk_score"),
                "reliability_score": (
                    product_reliability(product) if "risk_score" in summary else None
                ),
                "source": source,
                "raw_json": json.dumps(product, ensure_ascii=False),
                "updated_at": now,
                **badges,
            }
            _insert(connection, "products", row, key="link")

            for review in comments:
                identity = "\x1f".join((link, str(review.get("author") or ""), str(review.get("time") or ""), str(review.get("content") or "")))
                key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                analysis = review.get("sentiment_analysis") or {}
                _insert(
                    connection,
                    "reviews",
                    {
                        "review_key": key,
                        "product_link": link,
                        "author": review.get("author"),
                        "rating": review.get("rating"),
                        "reviewed_at": review.get("time"),
                        "content": review.get("content"),
                        "sentiment": analysis.get("label")
                        or (analysis.get("sentiment") or {}).get("label"),
                        "risk_detected": int(
                            bool(
                                (analysis.get("risk") or {}).get("detected")
                                or analysis.get("review_tags")
                            )
                        ),
                        "raw_json": json.dumps(review, ensure_ascii=False),
                        **_review_badge_values(analysis),
                    },
                    key="review_key",
                )


def database_counts(path: str | Path = DEFAULT_DB) -> dict[str, int]:
    with _connect(path) as connection:
        return {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("products", "reviews", "survey_responses", "evaluation_runs")
        }


def load_saved_products(path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    """Load the latest analyzed product payloads for local recommendations."""
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT raw_json FROM products ORDER BY updated_at DESC"
        ).fetchall()
    products = []
    for (payload,) in rows:
        try:
            product = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(product, dict):
            products.append(product)
    return products


def save_survey_response(frequency: str, sus: list[int], umux: list[int], sus_score: float,
                         umux_score: float, path: str | Path = DEFAULT_DB) -> None:
    payload = json.dumps({"sus": sus, "umux": umux}, ensure_ascii=False)
    with _connect(path) as connection:
        connection.execute(
            "INSERT INTO survey_responses(submitted_at, shopping_frequency, sus_score, umux_score, responses_json) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), frequency, sus_score, umux_score, payload),
        )


def save_evaluation_run(source_name: str, result: dict, path: str | Path = DEFAULT_DB) -> None:
    with _connect(path) as connection:
        connection.execute(
            """INSERT INTO evaluation_runs(run_at, source_name, sample_count, accuracy, precision,
            recall, f1, labels_json, matrix_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), source_name, result["sample_count"],
             result["accuracy"], result["precision"], result["recall"], result["f1"],
             json.dumps(result["labels"]), json.dumps(result["matrix"])),
        )
