"""Can the Temu scan run with no Chrome window showing, as the Lazada scan does? A measurement.

Why. Every Temu scan opens a visible Chrome window. The Lazada scan runs headless and shows its
window only when Lazada asks for a login or a check (research/check_hidden_lazada.py). Temu asks
for a check (a puzzle) far more often: on 14 Oct 2026 in every scan, as the product page opened.
Whether Temu serves a headless Chrome, or a window parked outside the screen, has to be measured.

How. One run is one window mode. It opens Temu the way a scan does (home page, saved session),
then for three saved Temu listings: the product page, the review section, and two scrolls of the
review list, recording whether Temu asked for a check and whether the review list kept loading.
Nothing is read from Temu that the page does not load itself, and no review is kept.

  headless   A check is recorded and ends the run: a headless Chrome has no window for the
             person, and the scan would have to start Chrome again in a visible window.
  offscreen  A check brings the window on screen, the person completes it, and the window is
             parked outside the screen again for the next listing.

Rule fixed before the first run: headless is used only if, in two runs, Temu showed all three
product pages and loaded their review lists with no check at all. Otherwise the off-screen window
is used, if in one run it loaded all three review lists (checks allowed: the window comes on
screen for them, as a visible window does now). If neither, Temu scans stay visible.

    python research/check_hidden_temu.py --mode headless
    python research/check_hidden_temu.py --mode offscreen
    python research/check_hidden_temu.py --summary
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "temu-scraper"))
from src.temu_scraper import TemuScraper  # noqa: E402

OUT = ROOT / "research" / "temu_hidden_check.json"
DATABASE = ROOT / "defaketive.db"
PROFILE = ROOT / ".browser_profiles" / "temu"
COOKIES = ROOT / "cookies_temu_ph.dat"
LISTINGS = 3
SCROLLS = 2
PAUSE_SECONDS = 2.5


def pick_listings() -> list[dict[str, Any]]:
    """The saved Temu listings with the most ratings. Read only."""
    with closing(sqlite3.connect(f"file:{DATABASE.as_posix()}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT link, name, review_count FROM products WHERE platform = 'Temu PH' ORDER BY review_count DESC"
        ).fetchall()
    return [{"link": link, "name": name or "", "ratings": count} for link, name, count in rows[:LISTINGS]]


def load() -> dict[str, Any]:
    if OUT.is_file():
        return json.loads(OUT.read_bytes().decode("utf-8"))
    rule = next(part for part in __doc__.split("\n\n") if part.startswith("Rule fixed"))
    return {"what": __doc__.split("\n\n")[0], "rule": " ".join(rule.split()), "runs": []}


def save(state: dict[str, Any]) -> None:
    OUT.write_bytes((json.dumps(state, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def run(mode: str) -> int:
    state = load()
    entry: dict[str, Any] = {
        "mode": mode,
        "run": 1 + sum(1 for item in state["runs"] if item["mode"] == mode),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": [],
        "listings": [],
    }
    state["runs"].append(entry)
    where = {"step": "home page"}

    with TemporaryDirectory() as scratch:
        scraper = TemuScraper(
            product_urls=[item["link"] for item in pick_listings()],
            chrome_user_data_dir=str(PROFILE),
            cookies_file=str(COOKIES),
            output=str(Path(scratch) / "not-kept.json"),
            pause_for_verification=False,
            verification_timeout=600,
            window_mode=mode,
        )
        waiting = scraper._wait_for_verification_clear

        def check() -> bool:
            began = time.time()
            if mode == "headless":
                # Recorded, not waited for: the run ends here.
                entry["checks"].append({"during": where["step"], "cleared": False, "seconds": 0})
                return False
            cleared = waiting()
            entry["checks"].append({"during": where["step"], "cleared": cleared, "seconds": round(time.time() - began, 1)})
            if cleared:
                # Parked outside the screen again, so the next listing is measured off screen too.
                scraper.driver.set_window_position(*TemuScraper.OFFSCREEN_POSITION)
                scraper.window_mode = "offscreen"
            return cleared

        scraper._wait_for_verification_clear = check
        scraper.driver = scraper._start_browser()
        driver = scraper.driver
        driver.set_script_timeout(30)
        try:
            ready = scraper._prepare_session()
            entry["home"] = {"ready": ready}
            entry["window"] = driver.execute_script(
                "return {screen_x: window.screenX, visibility: document.visibilityState,"
                " says_headless: /headless/i.test(navigator.userAgent), webdriver_flag: navigator.webdriver};"
            )
            save(state)
            print(f"{mode}: home page {'ready' if ready else 'NOT READY (check)'}", flush=True)
            if not ready:
                entry["stopped"] = "check on the home page"
                return finish(state, entry)
            for number, product in enumerate(scraper.product_urls, start=1):
                item = scraper._product_from_url(product)
                where["step"] = f"listing {number}: product page"
                scraper._reset_product_network_capture(item)
                started = time.time()
                blocked = scraper._safe_get(scraper.product_navigation_urls.get(product, product))
                row: dict[str, Any] = {"goods_id": item["goods_id"], "product_page": not blocked}
                if blocked:
                    entry["listings"].append(row)
                    print(f"{mode} {number}: product page held by a check", flush=True)
                    if mode == "headless":
                        entry["stopped"] = "check on a product page"
                        return finish(state, entry)
                    continue
                time.sleep(PAUSE_SECONDS)
                where["step"] = f"listing {number}: review section"
                root = scraper._open_review_section()
                row["review_section"] = root is not None
                if root is None:
                    row["check_showing"] = scraper._verification_detected()
                    entry["listings"].append(row)
                    print(f"{mode} {number}: review section not opened", flush=True)
                    if mode == "headless" and entry["checks"]:
                        entry["stopped"] = "check while opening the reviews"
                        return finish(state, entry)
                    continue
                time.sleep(PAUSE_SECONDS)
                scraper._capture_review_network(item)
                pages = [len(scraper._review_list_pages.get(item["goods_id"]) or ())]
                where["step"] = f"listing {number}: review list"
                for _ in range(SCROLLS):
                    if scraper._advance_reviews_by_list(root, item) is None:
                        break
                    pages.append(len(scraper._review_list_pages.get(item["goods_id"]) or ()))
                row.update(
                    {
                        "list_pages_loaded": pages[-1],
                        "pages_after_each_scroll": pages,
                        "written_reviews_seen": len(scraper._network_reviews_by_product.get(item["goods_id"]) or ()),
                        "seconds": round(time.time() - started, 1),
                        "passed": pages[-1] >= 1 + SCROLLS,
                    }
                )
                entry["listings"].append(row)
                save(state)
                print(
                    f"{mode} {number}: review list pages {pages} | written reviews seen {row['written_reviews_seen']}",
                    flush=True,
                )
                time.sleep(PAUSE_SECONDS)
        except Exception as exc:
            entry["stopped"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            try:
                scraper._save_cookies()
                scraper.driver.quit()
            except Exception:
                pass
    return finish(state, entry)


def finish(state: dict[str, Any], entry: dict[str, Any]) -> int:
    listings = entry["listings"]
    entry["all_listings_read"] = len(listings) == LISTINGS and all(row.get("passed") for row in listings)
    entry["checks_count"] = len(entry["checks"])
    entry["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save(state)
    print(
        f"{entry['mode']} run {entry['run']}: all {LISTINGS} review lists read: {entry['all_listings_read']}; "
        f"checks: {entry['checks_count']}" + (f"; stopped: {entry['stopped']}" if entry.get("stopped") else "")
    )
    return 0


def summary() -> int:
    state = load()
    for entry in state["runs"]:
        print(
            f"{entry['mode']:9} run {entry['run']}: read all: {entry.get('all_listings_read')} | "
            f"checks {entry.get('checks_count')} {[check['during'] for check in entry['checks']]} | "
            f"stopped: {entry.get('stopped', '-')}"
        )
    headless_runs = [entry for entry in state["runs"] if entry["mode"] == "headless"]
    offscreen_runs = [entry for entry in state["runs"] if entry["mode"] == "offscreen"]
    headless_ok = len(headless_runs) >= 2 and all(
        entry.get("all_listings_read") and not entry["checks"] for entry in headless_runs[:2]
    )
    offscreen_ok = any(entry.get("all_listings_read") for entry in offscreen_runs)
    choice = "headless" if headless_ok else "offscreen" if offscreen_ok else "visible"
    print(f"By the rule: {choice}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("headless", "offscreen", "visible"))
    parser.add_argument("--summary", action="store_true")
    arguments = parser.parse_args()
    if arguments.summary:
        return summary()
    if not arguments.mode:
        parser.error("say --mode or --summary")
    return run(arguments.mode)


if __name__ == "__main__":
    raise SystemExit(main())
