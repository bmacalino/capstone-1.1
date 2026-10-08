"""Read how many ratings each star level has, for listings already scanned.

Why. A product verdict compares a listing's share of fault reports with what
is typical. Typical has to be measured on real listings. 65 of the 68 saved
listings were collected one star level at a time with no per-star totals, so
low-star reviews are over-represented and their rates cannot be weighted
back to the listing (research/RESULTS_TABLES.md, Table 12b). The missing
piece is five numbers per listing, not a new scan.

How. Lazada's product page loads a review feed whose `ratings.scores` holds
those five numbers. The app's own scan reads it the same way
(LazadaScraper._read_star_distribution); this calls that method for each
saved listing and collects no review. One browser, one listing at a time
with a pause. A listing whose feed gives no usable numbers is recorded as
skipped; a verification page stops the run. Nothing is worked around and
nothing is written to the project root.

    python research/fetch_star_totals.py            # read what is missing
    python research/fetch_star_totals.py --list     # show the listings only
    python research/fetch_star_totals.py --limit 3  # the first three missing
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAVED_SCANS = ROOT / "research" / "verdict_check_saved_scans.json"
OUT = ROOT / "research" / "data" / "reference_scans" / "star_totals.json"
PAUSE_SECONDS = 4.0
STOP_AFTER_FAILURES = 5


def saved_listings() -> list[dict[str, str]]:
    rows = json.loads(SAVED_SCANS.read_bytes().decode("utf-8"))["rows"]
    return [{"link": row["link"], "name": row["name"]} for row in rows if "lazada.com.ph" in row["link"]]


def load() -> dict[str, Any]:
    if OUT.is_file():
        return json.loads(OUT.read_bytes().decode("utf-8"))
    return {"what": __doc__.split("\n\n")[0], "listings": {}}


def save(state: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(state, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    state = load()
    todo = [item for item in saved_listings() if state["listings"].get(item["link"], {}).get("status") != "read"]
    if "--limit" in sys.argv:
        todo = todo[: int(sys.argv[sys.argv.index("--limit") + 1])]
    if "--list" in sys.argv:
        for item in todo:
            print(item["link"], "|", item["name"][:60])
        print(f"{len(todo)} listings to read")
        return 0
    if not todo:
        print("nothing to read")
        return 0

    import os

    os.chdir(ROOT)  # the scraper looks for its session cookie file here
    sys.path.insert(0, str(ROOT / "lazada-scraper" / "src"))
    import lazada_scraper as ls

    scratch = OUT.parent / "_fetch_star_totals_unused.json"
    scraper = ls.LazadaScraper(product_urls=[todo[0]["link"]], output=str(scratch), pause_for_verification=False)
    major = scraper._detect_chrome_major()
    kwargs: dict[str, Any] = {"options": scraper.options, "enable_cdp_events": False, "headless": False}
    if major:
        kwargs["version_main"] = major
        cached = scraper._cached_driver_for_major(major)
        if cached:
            kwargs["driver_executable_path"] = cached
    scraper.driver = ls.SafeChrome(**kwargs)
    failures = 0
    try:
        scraper._safe_get(scraper.BASE_URL, check_verification=False)
        scraper._load_cookies()
        scraper.driver.refresh()
        time.sleep(3)
        # The feed is fetched from a product page, as the app's scan does.
        scraper._safe_get(todo[0]["link"], check_verification=False)
        time.sleep(3)
        if scraper._verification_detected():
            print("Lazada shows a verification page. Stopping; nothing was read.")
            return 2
        for number, item in enumerate(todo, start=1):
            product: dict[str, Any] = {"link": item["link"]}
            distribution = scraper._read_star_distribution(product)
            entry: dict[str, Any] = {"name": item["name"], "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if distribution:
                entry.update(status="read", ratings_by_star=distribution, ratings_total=sum(distribution.values()),
                             source=product.get("review_star_distribution_source"))
                failures = 0
            else:
                entry["status"] = "skipped"
                failures += 1
            state["listings"][item["link"]] = entry
            save(state)
            print(f"{number:3}/{len(todo)} {entry['status']:8} {entry.get('ratings_by_star', '')} | {item['name'][:50]}", flush=True)
            if failures >= STOP_AFTER_FAILURES:
                print(f"{STOP_AFTER_FAILURES} listings in a row gave nothing. Stopping.")
                break
            if scraper._verification_detected():
                print("Lazada shows a verification page. Stopping.")
                break
            time.sleep(PAUSE_SECONDS)
    finally:
        try:
            scraper.driver.quit()
        except Exception:
            pass
        scratch.unlink(missing_ok=True)
    done = sum(1 for entry in state["listings"].values() if entry.get("status") == "read")
    print(f"star totals on file for {done} listings; wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
