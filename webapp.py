"""DeFaketive's HTML/CSS/JavaScript web application.

Run locally with ``python webapp.py`` and open http://127.0.0.1:5000.
The original Streamlit dashboard remains available in app.py during migration.
"""

from __future__ import annotations

import csv
import hashlib
import os
import secrets
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, session

from dashboard_utils import (
    ROOT,
    analysis_is_current,
    analyze_products,
    cached_product_matches,
    csv_bytes,
    detect_marketplace,
    has_usable_products,
    json_bytes,
    load_products,
    merge_product_catalogs,
    output_path,
    product_risk_level,
    product_rows,
    rank_recommendations,
    result_files,
    review_rows,
)
from data_store import database_counts, load_saved_products, save_evaluation_run, save_products, save_survey_response
from research_utils import evaluate_labels, score_sus, score_umux


MANUAL_VERIFICATION_TIMEOUT = 900
SCRAPER_PROCESS_TIMEOUT = MANUAL_VERIFICATION_TIMEOUT + 300
MAX_PRODUCTS = 30
MAX_REVIEWS = 200

_catalogs: dict[str, list[dict[str, Any]]] = {}
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()

NETWORK_ACCESS_DENIED_MESSAGE = (
    "Chrome was denied network access (ERR_NETWORK_ACCESS_DENIED). "
    "Close this server, start run_capstone.cmd from File Explorer or a normal "
    "PowerShell window, and retry the live scan. Do not start the server from "
    "a restricted coding-agent terminal."
)


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, minimum), maximum)


def _catalog_key() -> str:
    key = session.get("catalog_key")
    if not key:
        key = secrets.token_urlsafe(18)
        session["catalog_key"] = key
    return str(key)


def _prepare(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if products and not analysis_is_current(products):
        products = analyze_products(products)
    return products


def _load_initial_catalog() -> list[dict[str, Any]]:
    for path in result_files():
        try:
            products = _prepare(load_products(path))
            if has_usable_products(products):
                save_products(products, source=str(path))
                return products
        except Exception:
            continue
    try:
        return _prepare(load_saved_products())
    except Exception:
        return []


def _products() -> list[dict[str, Any]]:
    key = _catalog_key()
    with _lock:
        if key not in _catalogs:
            _catalogs[key] = _load_initial_catalog()
        return _catalogs[key]


def _set_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = _prepare(products)
    with _lock:
        _catalogs[_catalog_key()] = prepared
    return prepared


def _product_for(link: str, products: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    products = products if products is not None else _products()
    return next((product for product in products if str(product.get("link") or "") == link), None)


def _command_for_scrape(keyword: str, platform: str, product_count: int, review_count: int) -> tuple[list[str], Path]:
    parsed = urlsplit(keyword)
    direct_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    direct_id = hashlib.sha256(keyword.strip().encode("utf-8")).hexdigest()[:12]
    target = output_path(f"direct_{direct_id}" if direct_url else keyword, platform)
    if platform == "shopee":
        command = [sys.executable, str(ROOT / "src" / "retriv.py"), "-n", str(product_count), "-r", str(review_count), "--output", str(target), "--no-prompt", "--verification-timeout", str(MANUAL_VERIFICATION_TIMEOUT)]
        command.extend(["--product-url", keyword] if direct_url else ["-k", keyword])
    elif platform == "lazada":
        command = [sys.executable, str(ROOT / "lazada-scraper" / "src" / "lazada_scraper.py"), "-n", str(product_count), "-r", str(review_count), "--output", str(target), "--no-verification-pause", "--verification-timeout", "600"]
        command.extend(["--product-url", keyword] if direct_url else [keyword])
    else:
        profile = ROOT / ".browser_profiles" / platform
        profile.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "temu-scraper" / "src" / "temu_scraper.py"), "-n", str(product_count), "-r", str(review_count), "--output", str(target), "--no-verification-pause", "--verification-timeout", "600", "--chrome-user-data-dir", str(profile)]
        command.extend(["--product-url", keyword] if direct_url else [keyword])
    return command, target


def _user_facing_failure(output: Any) -> str:
    """Replace known infrastructure errors with concise recovery guidance."""
    output = str(output or "").strip()
    if "ERR_NETWORK_ACCESS_DENIED" in output.upper():
        return NETWORK_ACCESS_DENIED_MESSAGE
    return output[-1800:] if output else "Marketplace collection failed."


def _scraper_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    """Turn scraper output into a useful, browser-sized error message."""
    output = "\n".join(part for part in (result.stderr, result.stdout) if part)
    return _user_facing_failure(output)


def _start_job(catalog_key: str, keyword: str, platform: str, product_count: int, review_count: int) -> dict[str, Any]:
    job_id = secrets.token_urlsafe(12)
    job = {
        "id": job_id,
        "state": "running",
        "stage": 1,
        "stage_label": "Collect reviews",
        "progress": 18,
        "message": f"Opening {platform.title()} and collecting public reviews…",
        "platform": platform,
        "query": keyword,
        "product_limit": product_count,
        "review_limit": review_count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        _jobs[job_id] = job

    def run() -> None:
        try:
            command, target = _command_for_scrape(keyword, platform, product_count, review_count)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=SCRAPER_PROCESS_TIMEOUT)
            if result.returncode:
                raise RuntimeError(_scraper_failure_message(result))
            with _lock:
                job.update(
                    stage=2,
                    stage_label="Analyze reviews",
                    progress=68,
                    message="Analyzing English, Filipino, and Taglish review sentiment…",
                )
            products = load_products(target)
            if not has_usable_products(products):
                raise RuntimeError("The marketplace did not return a usable product.")
            analyzed = analyze_products(products)
            with _lock:
                job.update(
                    stage=3,
                    stage_label="Prepare report",
                    progress=92,
                    message="Preparing the risk report and saving analyzed results…",
                )
            target.write_bytes(json_bytes(analyzed))
            save_products(analyzed, source=str(target))
            with _lock:
                _catalogs[catalog_key] = analyzed
                job.update(
                    state="complete",
                    stage=3,
                    stage_label="Complete",
                    progress=100,
                    message=f"Analysis complete — {len(analyzed)} product(s) found.",
                    product_count=len(analyzed),
                )
        except subprocess.TimeoutExpired:
            with _lock:
                job.update(
                    state="error",
                    stage_label="Stopped",
                    message="Marketplace verification timed out. Start a new scan when you are ready.",
                )
        except Exception as exc:
            with _lock:
                job.update(
                    state="error",
                    stage_label="Stopped",
                    message=_user_facing_failure(exc),
                )

    threading.Thread(target=run, daemon=True, name=f"defaketive-scrape-{job_id}").start()
    return job


def _lexicon_rows(filename: str, columns: list[str]) -> list[dict[str, str]]:
    path = ROOT / "sentiment-analysis" / "defaketive_sentiment" / "lexicons" / filename
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as source:
        for record in csv.reader((line for line in source if not line.startswith("#")), delimiter="\t"):
            if record:
                rows.append(dict(zip(columns, record)))
    return rows


def _main_product_issue(product: dict[str, Any]) -> str:
    """Return the most frequently detected risk category for a product."""
    issues: Counter[str] = Counter()
    comments = product.get("comments")
    if not isinstance(comments, list):
        return "No dominant issue"

    for comment in comments:
        if not isinstance(comment, dict) or comment.get("is_duplicate"):
            continue
        risk = (comment.get("sentiment_analysis") or {}).get("risk") or {}
        evidence = risk.get("evidence") or []
        categories = {
            str(item.get("category") or "").strip().lower()
            for item in evidence
            if isinstance(item, dict) and not item.get("negated")
        }
        if not categories:
            categories = {
                str(category).strip().lower()
                for category in (risk.get("categories") or [])
            }
        issues.update(category for category in categories if category)

    if not issues:
        return "No dominant issue"
    return issues.most_common(1)[0][0].replace("_", " ").title()


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(SECRET_KEY=os.environ.get("DEFAKETIVE_SECRET_KEY") or secrets.token_urlsafe(32), MAX_CONTENT_LENGTH=10 * 1024 * 1024)
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/products")
    def products_api() -> Response:
        return jsonify({"products": _products(), "count": len(_products())})

    @app.post("/api/search")
    def search_api() -> Response:
        payload = request.get_json(silent=True) or {}
        query = str(payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Enter a product name or supported marketplace link."}), 400
        platform = str(payload.get("platform") or "shopee").casefold()
        try:
            platform = detect_marketplace(query, platform)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if bool(payload.get("live")):
            job = _start_job(_catalog_key(), query, platform, _integer(payload.get("product_count"), 6, 1, MAX_PRODUCTS), _integer(payload.get("review_count"), 30, 1, MAX_REVIEWS))
            return jsonify({"job": job}), 202
        matches = cached_product_matches(_products(), query, limit=MAX_PRODUCTS)
        return jsonify({"products": matches, "count": len(matches), "mode": "saved"})

    @app.get("/api/jobs/<job_id>")
    def job_api(job_id: str) -> Response:
        with _lock:
            job = _jobs.get(job_id)
            if not job:
                return jsonify({"error": "That collection job no longer exists."}), 404
            return jsonify(job)

    @app.post("/api/upload")
    def upload_api() -> Response:
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Choose a scraper JSON file first."}), 400
        try:
            products = load_products(uploaded.read())
            if not has_usable_products(products):
                raise ValueError("The JSON does not contain usable product data.")
            products = _set_products(products)
            save_products(products, source=uploaded.filename)
        except Exception as exc:
            return jsonify({"error": f"Could not load file: {exc}"}), 400
        return jsonify({"products": products, "count": len(products)})

    @app.get("/api/products/detail")
    def product_detail_api() -> Response:
        link = str(request.args.get("link") or "")
        product = _product_for(link)
        if not product:
            return jsonify({"error": "Product not found in the current catalog."}), 404
        catalog = merge_product_catalogs(_products(), load_saved_products())
        return jsonify({"product": product, "recommendations": rank_recommendations(product, catalog, limit=3)})

    @app.get("/api/export/products.csv")
    def export_products() -> Response:
        return Response(csv_bytes(product_rows(_products())), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=defaketive_products.csv"})

    @app.get("/api/export/reviews.csv")
    def export_reviews() -> Response:
        return Response(csv_bytes(review_rows(_products())), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=defaketive_reviews.csv"})

    @app.get("/api/admin/overview")
    def overview_api() -> Response:
        products = _products()
        distribution = {"Low": 0, "Moderate": 0, "High": 0}
        for product in products:
            level = product_risk_level(product)
            if level in distribution:
                distribution[level] += 1
        rows = product_rows(products)
        for product, row in zip(products, rows):
            row["Main issue"] = _main_product_issue(product)
        flagged = [row for row in rows if str(row.get("Risk level")) == "High"][:10]
        sampled_reviews = sum(int(row.get("Reviews") or 0) for row in rows)
        average_reliability = (
            sum(float(row.get("Reliability score") or 0) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        marketplaces = {
            str(row.get("Platform") or "").strip()
            for row in rows
            if str(row.get("Platform") or "").strip()
        }
        summary = {
            "products": len(rows),
            "reviews": sampled_reviews,
            "average_reliability": round(average_reliability, 1),
            "high_risk": distribution["High"],
            "marketplaces": len(marketplaces),
        }
        return jsonify({"summary": summary, "risk_distribution": distribution, "flagged": flagged})

    @app.get("/api/admin/database")
    def database_api() -> Response:
        return jsonify({"counts": database_counts(), "products": product_rows(_products())})

    @app.get("/api/admin/lexicons")
    def lexicons_api() -> Response:
        return jsonify({"sentiment": _lexicon_rows("sentiment_lexicon.tsv", ["term", "weight", "language", "category"]), "risk": _lexicon_rows("risk_lexicon.tsv", ["term", "category", "severity", "language"])})

    @app.get("/api/admin/logs")
    def logs_api() -> Response:
        logs = sorted((ROOT / "logs").glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True) if (ROOT / "logs").exists() else []
        text = logs[0].read_text(encoding="utf-8", errors="replace")[-6000:] if logs else "No scraper logs yet."
        return jsonify({"log": text, "filename": logs[0].name if logs else None})

    @app.post("/api/survey")
    def survey_api() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            frequency = str(payload.get("frequency") or "")
            sus = [int(value) for value in payload.get("sus", [])]
            umux = [int(value) for value in payload.get("umux", [])]
            sus_score, umux_score = score_sus(sus), score_umux(umux)
            save_survey_response(frequency, sus, umux, sus_score, umux_score)
            return jsonify({"sus_score": sus_score, "umux_score": umux_score})
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/evaluation")
    def evaluation_api() -> Response:
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Choose a verified CSV file first."}), 400
        try:
            frame = pd.read_csv(uploaded)
            actual = next((name for name in ("actual_label", "true_label", "label") if name in frame), None)
            predicted = next((name for name in ("predicted_label", "prediction") if name in frame), None)
            text_column = next((name for name in ("review_text", "review", "text", "content") if name in frame), None)
            if not actual or not (predicted or text_column):
                raise ValueError("CSV needs actual_label and either predicted_label or review_text columns.")
            if predicted:
                predictions = frame[predicted].astype(str).tolist()
            else:
                sentiment_root = ROOT / "sentiment-analysis"
                if str(sentiment_root) not in sys.path:
                    sys.path.insert(0, str(sentiment_root))
                from defaketive_sentiment.model import DefaketiveSentimentModel
                model = DefaketiveSentimentModel()
                predictions = [model.analyze(text)["label"] for text in frame[text_column].fillna("")]
            result = evaluate_labels(frame[actual].astype(str), predictions)
            save_evaluation_run(uploaded.filename, result)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": f"Evaluation could not run: {exc}"}), 400

    @app.errorhandler(413)
    def too_large(_: Any) -> tuple[Response, int]:
        return jsonify({"error": "That file is larger than the 10 MB limit."}), 413

    return app


app = create_app()

if __name__ == "__main__":
    # Scrapes run in background subprocesses and their job state lives in this
    # process.  Flask's development reloader would restart the process when the
    # scraper imports monitored files, losing the job and closing its pipes.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
