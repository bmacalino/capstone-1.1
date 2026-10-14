"""DeFaketive's HTML/CSS/JavaScript web application.

Run locally with ``python webapp.py`` and open http://127.0.0.1:5000.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import os
import secrets
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd
from flask import Flask, Response, current_app, jsonify, render_template, request, session

import accounts
from dashboard_utils import (
    ROOT,
    analysis_is_current,
    analyze_products,
    badge_view,
    cached_product_matches,
    csv_bytes,
    detect_marketplace,
    has_usable_products,
    json_bytes,
    load_products,
    merge_product_catalogs,
    model_unavailable_error,
    output_path,
    product_rows,
    rank_recommendations,
    require_model,
    sentiment_model_version,
    result_files,
    review_rows,
    with_badge_view,
)
from data_store import database_counts, load_saved_products, save_evaluation_run, save_products, save_survey_response
from research_utils import evaluate_labels, score_sus, score_umux


MANUAL_VERIFICATION_TIMEOUT = 900
SCRAPER_PROCESS_TIMEOUT = MANUAL_VERIFICATION_TIMEOUT + 300
MAX_PRODUCTS = 30
MAX_REVIEWS = 100
# Lazada scans read up to this many written reviews from each star level; the
# analysis weights each level back to the listing's star mix.
LAZADA_REVIEWS_PER_STAR = 60
# Lazada scans run with no Chrome window; the scraper shows one by itself when
# Lazada asks for a login or a verification. Reviews are read from Lazada's
# review feed, which carries whole reviews where the page reader keeps the
# first part of a review written in several parts. The checker flags more in
# the whole text, so the "what is typical" baseline was measured again on feed
# scans (research/RESULTS_TABLES.md, Table 12d). Reading the page instead
# ("page") gives rates that baseline does not fit.
LAZADA_WINDOW_MODE = os.environ.get("DEFAKETIVE_LAZADA_WINDOW") or "headless"
LAZADA_REVIEWS_FROM = os.environ.get("DEFAKETIVE_LAZADA_REVIEWS") or "feed"
# What the progress panel says when the Lazada scraper has put its Chrome
# window on screen (_scan_progress). The scraper never answers either itself.
LAZADA_LOGIN_MESSAGE = (
    "Lazada needs you to log in. Use the Chrome window that just opened. "
    "The scan continues by itself once you are logged in."
)
LAZADA_CHECK_MESSAGE = (
    "Lazada is asking for a login or a check. Complete it in the Chrome window that just opened. "
    "The scan continues by itself afterwards."
)
LAZADA_REVIEWS_HELD_MESSAGE = (
    "Lazada is not showing this product's reviews yet. A Chrome window has opened in case it is "
    "asking for a check: if you see one, complete it there."
)
# The scraper reads the star levels in this order (RATING_COLLECTION_ORDER).
LAZADA_STAR_PROGRESS = {"3": 38, "2": 44, "4": 50, "1": 56, "5": 62}
# What the progress panel says when the Temu scraper waits for the person to
# complete a check or a login in its Chrome window (_temu_scan_progress). That
# window is always on screen; the scraper never answers either itself.
TEMU_CHECK_MESSAGE = (
    "Temu is asking for a check, usually a puzzle, or a login. Complete it in the Temu Chrome window. "
    "The scan continues by itself afterwards."
)
TEMU_CHECK_TIMEOUT_MESSAGE = (
    "Temu's check was not completed in time, so this scan stopped without a new report. "
    "Start it again when you can do the check in the Chrome window."
)
TEMU_NOT_SHOWN_MESSAGE = (
    "Temu did not show the product. If a check or a login appeared in the Chrome window, "
    "complete it and start the scan again."
)
MARKETPLACE_NAMES = ("Shopee PH", "Lazada PH", "Temu PH")

_catalogs: dict[str, list[dict[str, Any]]] = {}
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()

ModelUnavailableError = model_unavailable_error()
MODEL_UNAVAILABLE_MESSAGE = (
    "Review analysis is unavailable: the model is not installed on this server, "
    "so no report can be produced. Nothing was analyzed with a substitute model."
)

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


# Failed sign-ins per email, kept in memory: five wrong passwords in fifteen
# minutes pause sign-in for that email.
LOGIN_FAILURE_LIMIT = 5
LOGIN_FAILURE_WINDOW = 15 * 60
_login_failures: dict[str, list[float]] = {}
GUEST_LIMIT_MESSAGE = (
    "You've used your 3 free scans today. Sign in or create a free account for unlimited scans."
)


def _persistent_secret_key(folder: Path) -> str:
    """Keep one signing key across restarts, so sign-ins survive a restart."""
    path = folder / "secret_key"
    try:
        stored = path.read_text(encoding="utf-8").strip()
        if stored:
            return stored
    except OSError:
        pass
    key = secrets.token_urlsafe(48)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    return key


def _accounts_db() -> str:
    return str(current_app.config["ACCOUNTS_DB"])


def _current_user() -> dict[str, Any] | None:
    user_id = session.get("user_id")
    if user_id is None:
        return None
    user = accounts.get_user(_accounts_db(), user_id)
    if user is None:
        # The account was deleted, perhaps in another browser.
        session.pop("user_id", None)
    return user


def _guest_status() -> dict[str, int]:
    used = accounts.guest_scans_used(_accounts_db(), _catalog_key(), accounts.manila_day())
    return {"limit": accounts.GUEST_DAILY_SCANS, "left": max(0, accounts.GUEST_DAILY_SCANS - used)}


def _me() -> dict[str, Any]:
    user = _current_user()
    shown = {"name": user["name"], "email": user["email"]} if user else None
    return {"user": shown, "guest": None if user else _guest_status()}


def _json_body() -> dict[str, Any] | None:
    """The request's JSON object. Requiring JSON keeps other websites from
    submitting these forms: a cross-site JSON request needs CORS approval."""
    if not request.is_json:
        return None
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _recent_failures(email: str) -> list[float]:
    cutoff = time.monotonic() - LOGIN_FAILURE_WINDOW
    with _lock:
        failures = [moment for moment in _login_failures.get(email, []) if moment > cutoff]
        _login_failures[email] = failures
        return failures


def _merged_catalog() -> list[dict[str, Any]]:
    return merge_product_catalogs(_products(), load_saved_products())


def _card(product: dict[str, Any]) -> dict[str, Any]:
    """A product for a list card: its verdict and how many reviews were
    collected, without the full review list."""
    comments = product.get("comments")
    card = {key: value for key, value in with_badge_view(product).items() if key != "comments"}
    card["reviews_collected"] = len(comments) if isinstance(comments, list) else 0
    return card


def _cards(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Products for a list. A product's reviews are sent only with its own
    report (/api/products/detail): one product's reviews are ~150 KB."""
    return [_card(product) for product in products]


def _prepare(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if products and not analysis_is_current(products):
        products = analyze_products(products)
    return products


# Analyses made with the current model, keyed by their source (a result file,
# or the saved products) and the model version. Every new browser session loads
# the catalog; without this the transformer re-analysed the same reviews for
# each one (~13 s per 100 reviews). They are kept in memory for this run and on
# disk for the next: a restart used to repeat the analysis. The result files
# themselves are not rewritten: some are training-set sources whose SHA-256 is
# recorded in a manifest. Deleting the cache folder forces a fresh analysis.
_analysed_files: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def _digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _analysis_cache_file(key: tuple[Any, ...]) -> Path:
    folder = Path(os.environ.get("DEFAKETIVE_ANALYSIS_CACHE") or ROOT / "instance" / "analysis_cache")
    # The model version leads the name, so another version's files are easy to find.
    return folder / f"{_digest(key[-1])[:12]}-{_digest(key)[:32]}.json"


def _stored_analysis(key: tuple[Any, ...]) -> list[dict[str, Any]] | None:
    """The analysis kept on disk for this source and model version, if usable."""
    try:
        products = load_products(_analysis_cache_file(key))
    except (OSError, ValueError):
        return None
    return products if analysis_is_current(products) else None


def _store_analysis(key: tuple[Any, ...], products: list[dict[str, Any]]) -> None:
    """Keep an analysis for the next run. The cache only saves time, so
    failing to write it is not an error."""
    target = _analysis_cache_file(key)
    this_version = target.name.split("-", 1)[0]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        for other in target.parent.glob("*.json"):
            if other.name.split("-", 1)[0] != this_version:
                other.unlink()
        # Written whole, then renamed: a half-written file is never read.
        partial = target.with_suffix(".tmp")
        partial.write_bytes(json_bytes(products))
        partial.replace(target)
    except OSError:
        pass


def _analysed(key: tuple[Any, ...], load: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """`load()`'s products, analysed at most once per source and model version."""
    with _lock:
        cached = _analysed_files.get(key)
        if cached is None:
            products = load()
            if products and not analysis_is_current(products):
                stored = _stored_analysis(key)
                if stored is None:
                    stored = analyze_products(products)
                    _store_analysis(key, stored)
                products = stored
            cached = _analysed_files[key] = products
        return copy.deepcopy(cached)


def _prepare_file(path: Path) -> list[dict[str, Any]]:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, sentiment_model_version())
    return _analysed(key, lambda: load_products(path))


def _prepare_saved() -> list[dict[str, Any]]:
    """The saved products, for when no result file is usable. They used to be
    re-analysed for every browser session (108 s for 27 products)."""
    saved = load_saved_products()
    key = ("saved products", _digest(json_bytes(saved)), sentiment_model_version())
    return _analysed(key, lambda: saved)


def _load_initial_catalog() -> list[dict[str, Any]]:
    for path in result_files():
        try:
            products = _prepare_file(Path(path))
            if has_usable_products(products):
                save_products(products, source=str(path))
                return products
        except ModelUnavailableError:
            # Not a problem with this file: every file would fail the same way,
            # and an empty catalog would hide the reason from the user.
            raise
        except Exception:
            continue
    try:
        return _prepare_saved()
    except ModelUnavailableError:
        raise
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
        command = [sys.executable, str(ROOT / "lazada-scraper" / "src" / "lazada_scraper.py"), "-n", str(product_count), "-r", str(review_count), "--all-star-types", "--star-limit-per-type", str(LAZADA_REVIEWS_PER_STAR), "--output", str(target), "--no-verification-pause", "--verification-timeout", "600", "--window", LAZADA_WINDOW_MODE, "--reviews-from", LAZADA_REVIEWS_FROM]
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
    if "SHOPEE_VERIFICATION_BLOCKED" in output:
        return "Shopee could not complete browser verification and asked you to try again later. Collection stopped without a new report. Wait before retrying."
    if "SHOPEE_BROWSER_CLOSED" in output or "invalid session id" in output.lower():
        return "Chrome closed or disconnected before collection completed. Start a fresh scan and keep the scraper window open through login and collection."
    if "SHOPEE_COLLECTION_FAILED" in output:
        return "Shopee did not return usable product data. Complete login or verification in the scraper window and try again."
    if "Temu verification was not completed within the timeout" in output:
        return TEMU_CHECK_TIMEOUT_MESSAGE
    if "No Temu product pages were extracted" in output:
        return TEMU_NOT_SHOWN_MESSAGE
    return output[-1800:] if output else "Marketplace collection failed."


def _scraper_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    """Turn scraper output into a useful, browser-sized error message."""
    output = "\n".join(part for part in (result.stderr, result.stdout) if part)
    return _user_facing_failure(output)


def _scan_progress(line: str) -> dict[str, Any] | None:
    """What the progress panel should say for one line the Lazada scraper printed.

    `needs_you` marks the moments a Chrome window has opened for the user to
    log in or complete a check; the next ordinary step clears it.
    """
    if "Lazada is signed out" in line or "for Lazada login" in line:
        return {"needs_you": True, "message": LAZADA_LOGIN_MESSAGE}
    if "for Lazada verification" in line:
        return {"needs_you": True, "message": LAZADA_CHECK_MESSAGE}
    if "Step: " not in line:
        return None
    step = line.split("Step: ", 1)[1].strip()
    if step.startswith("showing the Chrome window"):
        if "needs a login" in step:
            return {"needs_you": True, "message": LAZADA_LOGIN_MESSAGE}
        if "verification" in step:
            return {"needs_you": True, "message": LAZADA_CHECK_MESSAGE}
        return {"needs_you": True, "message": LAZADA_REVIEWS_HELD_MESSAGE}
    if step.startswith("Lazada home page opened"):
        return {"needs_you": False, "progress": 22, "message": "Lazada is open. Checking the saved login…"}
    if step.startswith("session ready"):
        return {"needs_you": False, "progress": 26, "message": "Opening the product page…"}
    if step.startswith("product page opened"):
        return {"needs_you": False, "progress": 30, "message": "Reading the product page…"}
    if step.startswith("ratings per star"):
        counts = [int(part) for part in step.split(":", 1)[-1].replace(",", " ").split() if part.isdigit()]
        return {
            "needs_you": False,
            "progress": 34,
            "message": f"This listing has {sum(counts):,} ratings. Collecting reviews from every star level…",
        }
    if step.startswith("collecting ") and step[11:12] in LAZADA_STAR_PROGRESS:
        star = step[11:12]
        return {"needs_you": False, "progress": LAZADA_STAR_PROGRESS[star], "message": f"Collecting {star}-star reviews…"}
    return None


def _temu_scan_progress(line: str) -> dict[str, Any] | None:
    """What the progress panel should say for one line the Temu scraper printed.

    Temu's Chrome window is always on screen. `needs_you` marks the moments the
    scraper waits for the person to complete a check or a login in it; the
    scraper's own "check cleared" line, or the next ordinary step, clears it.
    """
    if "Waiting up to" in line and "for Temu verification" in line:
        return {"needs_you": True, "message": TEMU_CHECK_MESSAGE}
    if "Temu verification was not completed within the timeout" in line:
        return {"needs_you": False, "message": "Temu's check was not completed in time."}
    if "Step: " not in line:
        return None
    step = line.split("Step: ", 1)[1].strip()
    if step.startswith("waiting for you to complete Temu's check"):
        return {"needs_you": True, "message": TEMU_CHECK_MESSAGE}
    if step.startswith("Temu check cleared"):
        return {"needs_you": False, "message": "Thank you. The scan is carrying on…"}
    if step.startswith("Temu home page opened"):
        return {"needs_you": False, "progress": 22, "message": "Temu is open. Checking the saved session…"}
    if step.startswith("Temu session ready"):
        return {"needs_you": False, "progress": 26, "message": "Opening the product page…"}
    if step.startswith("product page opened"):
        return {"needs_you": False, "progress": 30, "message": "Reading the product page…"}
    if step.startswith("reading reviews"):
        return {"needs_you": False, "progress": 34, "message": "Collecting reviews…"}
    if step.startswith("reviews read: "):
        read, _, wanted = step[len("reviews read: "):].partition(" of ")
        if not read.strip().isdigit():
            return None
        count = int(read)
        share = min(1.0, count / int(wanted)) if wanted.strip().isdigit() and int(wanted) else 0.2
        return {
            "needs_you": False,
            # Short of the next stage (68) however many are read.
            "progress": 34 + round(28 * share),
            "message": f"Collecting reviews… {count} read so far.",
        }
    return None


# Which scrapers report their steps as they go, and how each one's lines are read.
SCAN_PROGRESS_READERS: dict[str, Callable[[str], dict[str, Any] | None]] = {
    "lazada": _scan_progress,
    "temu": _temu_scan_progress,
}


def _run_scraper(command: list[str], on_line: Callable[[str], None]) -> subprocess.CompletedProcess[str]:
    """Run a scraper to its end, passing on each line it prints as it prints it."""
    timed_out = threading.Event()
    lines: list[str] = []
    with subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    ) as process:

        def stop() -> None:
            timed_out.set()
            process.kill()

        timer = threading.Timer(SCRAPER_PROCESS_TIMEOUT, stop)
        timer.daemon = True
        timer.start()
        try:
            for line in process.stdout or ():
                lines.append(line)
                on_line(line)
        finally:
            timer.cancel()
    if timed_out.is_set():
        raise subprocess.TimeoutExpired(command, SCRAPER_PROCESS_TIMEOUT)
    return subprocess.CompletedProcess(command, process.returncode, stdout="".join(lines), stderr="")


def _start_job(catalog_key: str, keyword: str, platform: str, product_count: int, review_count: int) -> dict[str, Any]:
    job_id = secrets.token_urlsafe(12)
    job = {
        "id": job_id,
        "state": "running",
        "stage": 1,
        "stage_label": "Collect reviews",
        "progress": 18,
        "message": f"Opening {platform.title()} and collecting public reviews…",
        "needs_you": False,
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
            read_progress = SCAN_PROGRESS_READERS.get(platform)

            def follow(line: str) -> None:
                update = read_progress(line) if read_progress else None
                if update:
                    with _lock:
                        # A scan of several products repeats its steps; the bar never moves back.
                        update["progress"] = max(job["progress"], update.get("progress", 0))
                        job.update(update)

            result = _run_scraper(command, follow)
            if result.returncode:
                raise RuntimeError(_scraper_failure_message(result))
            with _lock:
                job.update(
                    stage=2,
                    stage_label="Analyze reviews",
                    progress=68,
                    message="Analyzing English, Filipino, and Taglish review sentiment…",
                    needs_you=False,
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
                    needs_you=False,
                )
        except ModelUnavailableError as exc:
            with _lock:
                job.update(
                    state="error", stage_label="Stopped", message=f"{MODEL_UNAVAILABLE_MESSAGE} {exc}", needs_you=False
                )
        except Exception as exc:
            with _lock:
                job.update(
                    state="error",
                    stage_label="Stopped",
                    message=_user_facing_failure(exc),
                    needs_you=False,
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
    """Name the category reviewers reported most, or say there is none."""
    view = badge_view(product)
    reported = [badge for badge in view["badges"] if badge["mentions"]]
    if not reported:
        return "No dominant issue"
    return max(reported, key=lambda badge: badge["mentions"])["label"]


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    instance = Path(app.instance_path)
    app.config.update(
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=30),
        ACCOUNTS_DB=os.environ.get("DEFAKETIVE_ACCOUNTS_DB") or str(instance / "accounts.db"),
    )
    if test_config:
        app.config.update(test_config)
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = os.environ.get("DEFAKETIVE_SECRET_KEY") or _persistent_secret_key(instance)

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/products")
    def products_api() -> Response:
        products = _products()
        return jsonify({"products": _cards(products), "count": len(products)})

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
            require_model()
            guest = _current_user() is None
            day = accounts.manila_day()
            # Only collecting new reviews counts as a scan; looking up a
            # product that was already checked is free for everyone.
            if guest and not accounts.reserve_guest_scan(_accounts_db(), _catalog_key(), day):
                return jsonify({"error": GUEST_LIMIT_MESSAGE, "limit_reached": True, "guest": _guest_status()}), 429
            try:
                job = _start_job(_catalog_key(), query, platform, _integer(payload.get("product_count"), 6, 1, MAX_PRODUCTS), _integer(payload.get("review_count"), 100, 1, MAX_REVIEWS))
            except Exception:
                if guest:
                    accounts.release_guest_scan(_accounts_db(), _catalog_key(), day)
                raise
            return jsonify({"job": job, "guest": _guest_status() if guest else None}), 202
        matches = cached_product_matches(_products(), query, limit=MAX_PRODUCTS)
        return jsonify({"products": _cards(matches), "count": len(matches), "mode": "saved"})

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
        except ModelUnavailableError:
            raise
        except Exception as exc:
            return jsonify({"error": f"Could not load file: {exc}"}), 400
        return jsonify({"products": _cards(products), "count": len(products)})

    @app.get("/api/products/detail")
    def product_detail_api() -> Response:
        link = str(request.args.get("link") or "")
        product = _product_for(link)
        if not product:
            return jsonify({"error": "Product not found in the current catalog."}), 404
        catalog = _merged_catalog()
        recommendations = rank_recommendations(product, catalog, limit=3)
        return jsonify(
            {"product": with_badge_view(product), "recommendations": _cards(recommendations)}
        )

    @app.get("/api/export/products.csv")
    def export_products() -> Response:
        return Response(csv_bytes(product_rows(_products())), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=defaketive_products.csv"})

    @app.get("/api/export/reviews.csv")
    def export_reviews() -> Response:
        return Response(csv_bytes(review_rows(_products())), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=defaketive_reviews.csv"})

    @app.get("/api/admin/overview")
    def overview_api() -> Response:
        products = _products()
        views = [badge_view(product) for product in products]
        rows = product_rows(products)
        verdicts: Counter[str] = Counter(view["state"] for view in views)
        flagged = []
        for product, view, row in zip(products, views, rows):
            if view["tone"] in {"high", "moderate"}:
                flagged.append({**row, "Main issue": _main_product_issue(product)})
        # Safety warnings first, then by how many safety reports they carry.
        flagged.sort(
            key=lambda row: (row["Verdict"] != "Reviewers report safety problems", -row["Safety reports"])
        )
        marketplaces = {
            str(row.get("Platform") or "").strip()
            for row in rows
            if str(row.get("Platform") or "").strip()
        }
        summary = {
            "products": len(rows),
            "reviews": sum(int(row.get("Reviews") or 0) for row in rows),
            "safety_warnings": verdicts.get("safety_warning", 0),
            "with_safety_reports": sum(1 for view in views if view["counts"]["safety"]),
            "not_analyzed": verdicts.get("not_analyzed", 0),
            "marketplaces": len(marketplaces),
        }
        return jsonify(
            {"summary": summary, "verdicts": dict(sorted(verdicts.items())), "flagged": flagged[:10]}
        )

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

    # --- Accounts -----------------------------------------------------------

    def json_required() -> tuple[Response, int]:
        return jsonify({"error": "Send this request as JSON."}), 415

    def field_error(error: accounts.AccountError, status: int = 400) -> tuple[Response, int]:
        return jsonify({"error": error.message, "fields": {error.field: error.message}}), status

    def sign_in(user: dict[str, Any]) -> None:
        catalog_key = session.get("catalog_key")
        session.clear()
        if catalog_key:
            session["catalog_key"] = catalog_key
        session["user_id"] = user["id"]
        session.permanent = True

    @app.get("/api/auth/me")
    def me_api() -> Response:
        return jsonify(_me())

    @app.post("/api/auth/signup")
    def signup_api() -> Response | tuple[Response, int]:
        payload = _json_body()
        if payload is None:
            return json_required()
        errors = accounts.signup_errors(
            payload.get("full_name"), payload.get("email"), payload.get("password"),
            payload.get("confirm"), payload.get("agree"),
        )
        if errors:
            return jsonify({"error": next(iter(errors.values())), "fields": errors}), 400
        try:
            user = accounts.create_user(_accounts_db(), payload["full_name"], payload["email"], payload["password"])
        except accounts.EmailTaken as error:
            return field_error(error, 409)
        sign_in(user)
        return jsonify(_me()), 201

    @app.post("/api/auth/login")
    def login_api() -> Response | tuple[Response, int]:
        payload = _json_body()
        if payload is None:
            return json_required()
        email = str(payload.get("email") or "").strip().lower()
        if len(_recent_failures(email)) >= LOGIN_FAILURE_LIMIT:
            return jsonify({"error": "Too many tries. Wait a few minutes, then try again."}), 429
        user = accounts.authenticate(_accounts_db(), email, payload.get("password"))
        if user is None:
            with _lock:
                _login_failures.setdefault(email, []).append(time.monotonic())
            return jsonify({"error": "Email or password is incorrect."}), 401
        with _lock:
            _login_failures.pop(email, None)
        sign_in(user)
        return jsonify(_me())

    @app.post("/api/auth/logout")
    def logout_api() -> Response:
        session.pop("user_id", None)
        return jsonify(_me())

    @app.patch("/api/auth/account")
    def update_account_api() -> Response | tuple[Response, int]:
        payload = _json_body()
        if payload is None:
            return json_required()
        user = _current_user()
        if user is None:
            return jsonify({"error": "Sign in first."}), 401
        try:
            accounts.update_user(
                _accounts_db(), user["id"], payload.get("current_password"),
                full_name=payload.get("full_name"), email=payload.get("email"),
                new_password=payload.get("new_password"),
            )
        except accounts.EmailTaken as error:
            return field_error(error, 409)
        except accounts.WrongPassword as error:
            return field_error(error, 403)
        except accounts.AccountError as error:
            return field_error(error)
        return jsonify(_me())

    @app.delete("/api/auth/account")
    def delete_account_api() -> Response | tuple[Response, int]:
        payload = _json_body()
        if payload is None:
            return json_required()
        user = _current_user()
        if user is None:
            return jsonify({"error": "Sign in first."}), 401
        try:
            accounts.delete_user(_accounts_db(), user["id"], payload.get("current_password"))
        except accounts.WrongPassword as error:
            return field_error(error, 403)
        session.pop("user_id", None)
        return jsonify(_me())

    # --- A signed-in user's history and wishlist ---------------------------

    def signed_in_payload() -> tuple[dict[str, Any] | None, dict[str, Any] | None, tuple[Response, int] | None]:
        payload = _json_body() if request.method != "GET" else {}
        if payload is None:
            return None, None, json_required()
        user = _current_user()
        if user is None:
            return None, None, (jsonify({"error": "Sign in to save to your account."}), 401)
        return user, payload, None

    @app.get("/api/me/library")
    def library_api() -> Response | tuple[Response, int]:
        user, _, problem = signed_in_payload()
        if problem:
            return problem
        return jsonify(accounts.library(_accounts_db(), user["id"]))

    @app.put("/api/me/activity")
    def activity_api() -> Response | tuple[Response, int]:
        user, payload, problem = signed_in_payload()
        if problem:
            return problem
        try:
            accounts.record_activity(_accounts_db(), user["id"], payload.get("link"), payload.get("product"), payload.get("openedAt"))
        except accounts.AccountError as error:
            return field_error(error)
        return jsonify({"saved": True})

    @app.delete("/api/me/activity")
    def clear_activity_api() -> Response | tuple[Response, int]:
        user, _, problem = signed_in_payload()
        if problem:
            return problem
        accounts.clear_activity(_accounts_db(), user["id"], (_ or {}).get("clearedAt"))
        return jsonify(accounts.library(_accounts_db(), user["id"]))

    @app.route("/api/me/wishlist", methods=["PUT", "DELETE"])
    def wishlist_api() -> Response | tuple[Response, int]:
        user, payload, problem = signed_in_payload()
        if problem:
            return problem
        try:
            accounts.set_wishlist(_accounts_db(), user["id"], payload.get("link"), request.method == "PUT")
        except accounts.AccountError as error:
            return field_error(error)
        return jsonify({"saved": request.method == "PUT"})

    @app.post("/api/me/import")
    def import_api() -> Response | tuple[Response, int]:
        user, payload, problem = signed_in_payload()
        if problem:
            return problem
        imported = accounts.import_library(_accounts_db(), user["id"], payload.get("activity"), payload.get("wishlist"))
        return jsonify({"imported": imported, **accounts.library(_accounts_db(), user["id"])})

    # --- Public numbers for the sign-in page and the Wishlist page ----------

    @app.get("/api/stats/public")
    def public_stats_api() -> Response:
        counts = database_counts()
        return jsonify({"products": counts.get("products", 0), "reviews": counts.get("reviews", 0), "marketplaces": len(MARKETPLACE_NAMES)})

    @app.get("/api/picks")
    def picks_api() -> Response:
        clean = [product for product in _merged_catalog() if badge_view(product).get("state") == "no_warning_signs"]
        clean.sort(key=lambda product: -int(badge_view(product).get("reviews_analyzed") or 0))
        return jsonify({"products": [_card(product) for product in clean[:12]]})

    @app.get("/api/wishlist/popular")
    def popular_api() -> Response:
        by_link = {str(product.get("link") or ""): product for product in _merged_catalog()}
        popular = [
            {"product": _card(by_link[link]), "saves": saves}
            for link, saves in accounts.popular_wishlist(_accounts_db())
            if link in by_link
        ]
        return jsonify({"products": popular})

    @app.errorhandler(ModelUnavailableError)
    def model_unavailable(error: Exception) -> tuple[Response, int]:
        return jsonify({"error": MODEL_UNAVAILABLE_MESSAGE, "detail": str(error), "model_unavailable": True}), 503

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
