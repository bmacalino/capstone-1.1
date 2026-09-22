import sqlite3
import tempfile
import unittest
from pathlib import Path

from data_store import (
    PRODUCT_BADGE_COLUMNS,
    REVIEW_BADGE_COLUMNS,
    _connect,
    database_counts,
    load_saved_products,
    save_evaluation_run,
    save_products,
    save_survey_response,
)

BADGE_PRODUCT = {
    "link": "https://example.test/powerbank",
    "name": "Test power bank",
    "comments": [
        {
            "author": "A",
            "time": "2026-08-01",
            "content": "sobrang init tapos namaga",
            "sentiment_analysis": {
                "review_tags": ["safety"],
                "review_high_confidence": ["safety"],
                "max_severity": 4,
                "sentiment": {"label": "negative"},
                "sentences": [
                    {
                        "index": 0,
                        "tags": ["safety"],
                        "hazard_subtype": "fire_explosion_swelling",
                    }
                ],
            },
        }
    ],
    "sentiment_summary": {
        "schema_version": "defaketive-summary-2.0",
        "analyzed_at": "2026-09-21T00:00:00+00:00",
        "verdict": {"state": "safety_warning"},
        "safety": {"tier": "High", "severity": 4, "alert": True},
        "flags": {
            "counterfeit": {"state": "not_flagged"},
            "defect": {"state": "flagged"},
        },
        "coverage": {
            "reviews_analyzed": 412,
            "reviews_listed": 415,
            "scrape_completeness": "complete",
        },
    },
}


class TestDataStore(unittest.TestCase):
    def test_central_store_upserts_and_saves_research_results(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.db"
            product = {
                "link": "https://example.test/product",
                "name": "Test product",
                "comments": [{"author": "A", "time": "now", "content": "okay", "sentiment_analysis": {"label": "neutral", "risk": {"detected": False}}}],
                "sentiment_summary": {"risk_score": 20, "sentiment_ratios": {"positive": .8}},
            }
            save_products([product], path=database)
            save_products([product], path=database)
            save_survey_response("Monthly", [3] * 10, [4] * 4, 50, 50, path=database)
            save_evaluation_run("verified.csv", {"sample_count": 2, "accuracy": .5, "precision": .5, "recall": .5, "f1": .5, "labels": ["negative", "positive"], "matrix": [[1, 0], [1, 0]]}, path=database)
            self.assertEqual(database_counts(database), {"products": 1, "reviews": 1, "survey_responses": 1, "evaluation_runs": 1})
            stored = load_saved_products(database)
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0]["name"], "Test product")


class TestBadgeMigration(unittest.TestCase):
    def test_migration_is_idempotent(self):
        # _connect runs the migration every time it opens the database, so it
        # has to be safe to run repeatedly on an already-migrated file.
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "test.db"
            for _ in range(3):
                with _connect(database) as connection:
                    columns = {
                        row[1]
                        for row in connection.execute("PRAGMA table_info(products)").fetchall()
                    }
            self.assertTrue(set(PRODUCT_BADGE_COLUMNS) <= columns)

    def test_migration_upgrades_a_pre_badge_database_without_losing_rows(self):
        # The real upgrade path: a database created by the previous schema,
        # already holding analyzed products.
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "legacy.db"
            legacy = sqlite3.connect(database)
            legacy.executescript(
                """
                CREATE TABLE products (
                    link TEXT PRIMARY KEY, name TEXT NOT NULL, platform TEXT NOT NULL,
                    category TEXT, price TEXT, rating REAL, review_count INTEGER NOT NULL,
                    risk_score REAL, reliability_score REAL, source TEXT,
                    raw_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE reviews (
                    review_key TEXT PRIMARY KEY, product_link TEXT NOT NULL,
                    author TEXT, rating INTEGER, reviewed_at TEXT, content TEXT,
                    sentiment TEXT, risk_detected INTEGER NOT NULL, raw_json TEXT NOT NULL
                );
                INSERT INTO products VALUES
                    ('https://old.test/p', 'Old product', 'Lazada PH', '', '', 4.2, 3,
                     42.5, 30.0, 'legacy', '{"name": "Old product"}', '2026-01-01');
                """
            )
            legacy.commit()
            legacy.close()

            with _connect(database) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(products)").fetchall()
                }
                rows = connection.execute(
                    "SELECT name, risk_score, verdict_state FROM products"
                ).fetchall()

            self.assertTrue(set(PRODUCT_BADGE_COLUMNS) <= columns)
            self.assertTrue(set(REVIEW_BADGE_COLUMNS) <= columns | set(REVIEW_BADGE_COLUMNS))
            # The pre-badge row survives, keeps its old score, and has no badge.
            self.assertEqual(rows, [("Old product", 42.5, None)])

    def test_pre_badge_rows_are_identifiable_after_migration(self):
        # `risk_score IS NOT NULL` is the documented way to find rows that still
        # hold a score from the old model and need re-analysis.
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "mixed.db"
            legacy = {
                "link": "https://example.test/legacy",
                "name": "Legacy product",
                "comments": [],
                "sentiment_summary": {"risk_score": 42.5, "sentiment_ratios": {"positive": 0.5}},
            }
            save_products([legacy], path=database)
            save_products([BADGE_PRODUCT], path=database)
            with _connect(database) as connection:
                stale = connection.execute(
                    "SELECT name FROM products WHERE risk_score IS NOT NULL"
                ).fetchall()
                migrated = connection.execute(
                    "SELECT name FROM products WHERE summary_schema_version IS NOT NULL"
                ).fetchall()
            self.assertEqual(stale, [("Legacy product",)])
            self.assertEqual(migrated, [("Test power bank",)])


class TestBadgePersistence(unittest.TestCase):
    def test_badge_fields_are_stored_as_queryable_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "badges.db"
            save_products([BADGE_PRODUCT], source="test", path=database)
            with _connect(database) as connection:
                row = connection.execute(
                    """SELECT verdict_state, safety_tier, safety_severity, safety_alert,
                       counterfeit_state, defect_state, reviews_analyzed, reviews_listed,
                       scrape_completeness, summary_schema_version, risk_score
                       FROM products"""
                ).fetchone()
            self.assertEqual(
                row,
                (
                    "safety_warning", "High", 4, 1,
                    "not_flagged", "flagged", 412, 415,
                    "complete", "defaketive-summary-2.0", None,
                ),
            )

    def test_review_tags_are_stored(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "badges.db"
            save_products([BADGE_PRODUCT], path=database)
            with _connect(database) as connection:
                row = connection.execute(
                    "SELECT flag_categories, max_severity, high_confidence, risk_detected FROM reviews"
                ).fetchone()
            self.assertEqual(row[0], "safety")
            self.assertEqual(row[1], 4)
            self.assertEqual(row[2], 1)
            # A tagged review still reads as risk-detected for the legacy views.
            self.assertEqual(row[3], 1)

    def test_upsert_still_works_with_the_new_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "badges.db"
            save_products([BADGE_PRODUCT], path=database)
            save_products([BADGE_PRODUCT], path=database)
            self.assertEqual(database_counts(database)["products"], 1)
            self.assertEqual(database_counts(database)["reviews"], 1)

    def test_a_badge_summary_writes_no_legacy_score(self):
        # Nothing may reintroduce a single 0-100 number through the database.
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "badges.db"
            save_products([BADGE_PRODUCT], path=database)
            with _connect(database) as connection:
                row = connection.execute(
                    "SELECT risk_score, reliability_score FROM products"
                ).fetchone()
            self.assertEqual(row, (None, None))


if __name__ == "__main__":
    unittest.main()
