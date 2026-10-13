"""Scan the baseline listings again, the way the app scans with the review feed.

Why. What is typical (Table 12c) was measured on saved scans read off the
product page. The page reader keeps only the first part of a review written
in several parts; the review feed has the whole text, and with the whole text
the checker flags more (lazada-scraper/README.md, 12 Oct 2026). A listing read
from the feed must therefore be compared with a line measured on listings
read from the feed.

Written before any listing was scanned:

  * Which listings: the ones the served baseline was measured on, that is
    every usable row of research/verdict_check_weighted.json, one per Lazada
    item (the row with more reviews when an item appears under two links).
  * How: the command the web app runs for a pasted link
    (webapp._command_for_scrape) with the feed as the review source: no Chrome
    window, all five star levels, up to 60 written reviews per level, per-star
    counts recorded. Two differences from the app. The result goes to
    research/data/reference_scans/feed/, never the project root, whose scan
    files are training sources. And a login or verification is not waited
    for.
  * What counts: a scan is kept only if its reviews came from the feed, it
    recorded the ratings per star level, and every star level was collected
    to its quota or to its end. A scan that fell back to reading the page is
    not kept: it would be the old reader again.
  * When to stop: two listings in a row that cannot be kept. Nothing is
    retried and nothing is worked around.

Changed on 12 Oct 2026, after six listings had been scanned and before any
rate was computed. The sixth (41,681 ratings) could not be kept: the feed
lists only the first 300 entries of a list, its 1- and 2-star list holds
2,111, and the scraper took the feed's silence past entry 300 for a failure
and read the page instead. The scraper now records that end as `feed_limit`,
and a star level cut there counts as collected if it holds any review: it is
the first of that level's reviews in the feed's order, as a quota is, only
fewer (58 two-star reviews instead of 60 on that listing). That listing is
scanned again under this rule; the five kept before it are unaffected,
because none of their lists reaches 300.

    python research/scan_baseline_listings.py            # scan what is missing
    python research/scan_baseline_listings.py --list     # show the listings only
    python research/scan_baseline_listings.py --limit 2  # the first two missing
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research" / "verdict_check_weighted.json"
OUT = ROOT / "research" / "data" / "reference_scans" / "feed"
LOG = OUT / "scan_log.json"
SCRAPER = ROOT / "lazada-scraper" / "src" / "lazada_scraper.py"
REVIEWS = 100          # webapp's default review count
PER_STAR = 60          # webapp.LAZADA_REVIEWS_PER_STAR
FEED_SOURCE = "lazada_review_feed"
TIME_BUDGET = 480      # hazard_parameters_v1.json: per_product_time_budget_seconds
PAUSE_SECONDS = 5
STOP_AFTER_FAILURES = 2


def item_id(link: str) -> str:
    match = re.search(r"-i(\d{5,})", link)
    return match.group(1) if match else link


def selection() -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in json.loads(SOURCE.read_bytes().decode("utf-8"))["rows"]:
        if not row.get("usable"):
            continue
        key = item_id(row["link"])
        if key not in best or (row.get("reviews_analysed") or 0) > (best[key].get("reviews_analysed") or 0):
            best[key] = row
    return [{"item_id": key, "link": row["link"], "name": row["name"]} for key, row in best.items()]


def target_for(listing: dict[str, Any]) -> Path:
    return OUT / f"lazada_i{listing['item_id']}.json"


def kept(path: Path) -> tuple[int, str]:
    """Written reviews in a scan that counts, or 0 and why it does not."""
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return 0, "no scan file"
    product = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
    if product is None:
        return 0, "no product in the scan file"
    if product.get("review_source") != FEED_SOURCE:
        return 0, "the reviews were not read from the feed"
    if not product.get("review_star_distribution"):
        return 0, "no ratings per star level"
    if product.get("review_filter_warning"):
        return 0, str(product["review_filter_warning"])
    levels = product.get("star_collection") or {}
    unfinished = [star for star in "12345" if not (levels.get(star) or {}).get("complete")]
    if unfinished:
        return 0, f"star level {', '.join(unfinished)} was not finished"
    written = sum(1 for comment in product.get("comments") or [] if str(comment.get("content") or "").strip())
    return written, "" if written else "no written reviews"


def scan(listing: dict[str, Any]) -> dict[str, Any]:
    target = target_for(listing)
    command = [
        sys.executable, str(SCRAPER), "-n", "1", "-r", str(REVIEWS), "--all-star-types",
        "--star-limit-per-type", str(PER_STAR), "--output", str(target), "--no-verification-pause",
        "--verification-timeout", "0", "--window", "headless", "--reviews-from", "feed",
        "--product-url", listing["link"],
    ]
    started = time.time()
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIME_BUDGET
        )
        code: Any = result.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    reviews, why_not = kept(target)
    if not reviews and target.is_file():
        # Not a baseline scan: keep it out of the folder the measurement reads.
        target.replace(target.with_name(f"not_kept_{target.name}"))
    return {
        "status": "ok" if reviews else "not kept", "reviews": reviews, "why_not": why_not, "exit": code,
        "seconds": round(time.time() - started),
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save(log: dict[str, Any]) -> None:
    LOG.write_bytes((json.dumps(log, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    listings = selection()
    if "--list" in sys.argv:
        for listing in listings:
            print(f"{listing['item_id']:12} {listing['name'][:70]}")
        print(len(listings), "listings")
        return 0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(listings)
    OUT.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_bytes().decode("utf-8")) if LOG.is_file() else {"rule": __doc__.split("\n\n")[2], "listings": {}}
    done = failures = 0
    for listing in listings:
        have, _ = kept(target_for(listing))
        if have:
            log["listings"].setdefault(listing["item_id"], {**listing, "status": "ok", "reviews": have})
            continue
        if done == limit:
            break
        if done:
            time.sleep(PAUSE_SECONDS)
        done += 1
        outcome = scan(listing)
        log["listings"][listing["item_id"]] = {**listing, **outcome}
        save(log)
        print(f"{done:3} {outcome['status']:8} {outcome['reviews']:4} reviews {outcome['seconds']:4}s"
              f" | {listing['name'][:50]}{' | ' + outcome['why_not'] if outcome['why_not'] else ''}", flush=True)
        failures = 0 if outcome["status"] == "ok" else failures + 1
        if failures >= STOP_AFTER_FAILURES:
            print(f"{STOP_AFTER_FAILURES} listings in a row could not be kept. Stopping.")
            break
    save(log)
    ok = sum(1 for entry in log["listings"].values() if entry.get("status") == "ok")
    print(f"{ok} of {len(listings)} listings scanned and kept; log in {LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
