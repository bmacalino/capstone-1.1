"""Can the Lazada scan run with no Chrome window showing? A measurement.

Why. Every Lazada scan opens a visible Chrome window. Two ways to hide it
exist: true headless (nothing on screen at all) and the normal window parked
outside the screen. Which of them Lazada accepts has to be measured, and so
does whether the review feed the page itself loads can replace reading the
reviews off the page (the page's review section only loads when it is
scrolled into view, which is what stalled covered scans).

How. One run is one window mode and one browser. It opens Lazada the way the
app's scan does (home page, saved session, product page) for five saved
listings, and for each records whether the page loaded, whether a
verification page showed, and whether the feed returned reviews. On the first
listing it also asks the feed what its `filter` values mean, how many reviews
one request may carry, and whether its reviews are the ones the page shows.
A verification page stops the run. Nothing is worked around, no review is
kept, and nothing is written to the project root.

Rule fixed before the first run: true headless is used only if all five
listings pass with no verification in two separate runs; otherwise the
off-screen window; if both fail, neither.

    python research/check_hidden_lazada.py --mode visible
    python research/check_hidden_lazada.py --mode offscreen
    python research/check_hidden_lazada.py --mode headless
    python research/check_hidden_lazada.py --summary
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STAR_TOTALS = ROOT / "research" / "data" / "reference_scans" / "star_totals.json"
OUT = ROOT / "research" / "data" / "reference_scans" / "hidden_mode_check.json"
MODES = ("visible", "offscreen", "headless")
LISTINGS = 5
PAUSE_SECONDS = 1.5
OFFSCREEN_POSITION = (-20000, 0)
FEED_URL = (
    "https://my.lazada.com.ph/pdp/review/getReviewList"
    "?itemId={item_id}&pageSize={page_size}&filter={filter}&sort={sort}&pageNo={page}"
)
TITLE_SELECTORS = ("h1.pdp-mod-product-badge-title", 'h1[data-qa-locator="product-title"]', "h1")
VERIFICATION_MARKERS = ("_____tmd_____", '"action":"captcha"', "rgv587_flag", "punish")
FETCH_SCRIPT = r"""
const done = arguments[arguments.length - 1];
fetch(arguments[0], {credentials: 'include'})
  .then(response => response.text().then(
    text => done(JSON.stringify({status: response.status, text: text}))))
  .catch(error => done(JSON.stringify({error: String(error)})));
"""


def pick_listings() -> list[dict[str, Any]]:
    """Five saved listings; the first has ratings at every star level."""
    listings = json.loads(STAR_TOTALS.read_bytes().decode("utf-8"))["listings"]
    read = [
        {"link": link, "name": entry["name"], "ratings_by_star": entry["ratings_by_star"]}
        for link, entry in listings.items()
        if entry.get("status") == "read"
    ]
    read.sort(key=lambda item: min(item["ratings_by_star"].values()) < 6)
    return read[:LISTINGS]


def load() -> dict[str, Any]:
    if OUT.is_file():
        return json.loads(OUT.read_bytes().decode("utf-8"))
    return {"what": __doc__.split("\n\n")[0], "runs": []}


def save(state: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes((json.dumps(state, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def feed(driver, item_id: str, filter: int = 0, page: int = 1, page_size: int = 5, sort: int = 0) -> dict[str, Any]:
    """One request to the review feed, made by the page itself."""
    url = FEED_URL.format(item_id=item_id, page_size=page_size, filter=filter, sort=sort, page=page)
    started = time.perf_counter()
    try:
        reply = json.loads(driver.execute_async_script(FETCH_SCRIPT, url) or "{}")
    except Exception as exc:
        return {"kind": "error", "detail": str(exc)[:200], "seconds": round(time.perf_counter() - started, 2)}
    result: dict[str, Any] = {"seconds": round(time.perf_counter() - started, 2)}
    time.sleep(PAUSE_SECONDS)
    if "error" in reply:
        return {**result, "kind": "error", "detail": reply["error"][:200]}
    text = reply.get("text") or ""
    try:
        body = json.loads(text)
    except ValueError:
        blocked = any(marker in text for marker in VERIFICATION_MARKERS)
        return {**result, "kind": "verification" if blocked else "other", "detail": text[:200]}
    model = body.get("model") if isinstance(body, dict) else None
    if isinstance(model, dict) and isinstance(model.get("items"), list):
        return {**result, "kind": "reviews", "model": model}
    blocked = any(marker in text for marker in VERIFICATION_MARKERS)
    return {**result, "kind": "verification" if blocked else "other", "detail": text[:200]}


def brief(reply: dict[str, Any]) -> dict[str, Any]:
    """What a feed reply held, without the reviews themselves."""
    if reply["kind"] != "reviews":
        return {key: reply[key] for key in ("kind", "seconds", "detail") if key in reply}
    model = reply["model"]
    paging = model.get("paging") or {}
    return {
        "kind": "reviews",
        "seconds": reply["seconds"],
        "items": len(model["items"]),
        "ratings": [item.get("rating") for item in model["items"]],
        "written": sum(1 for item in model["items"] if (item.get("reviewContent") or "").strip()),
        "total_items": paging.get("totalItems"),
        "total_pages": paging.get("totalPages"),
        "scores": (model.get("ratings") or {}).get("scores"),
    }


def shape(value: Any) -> Any:
    """The type of a value, and for containers what they hold. No content."""
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def feed_questions(scraper, item_id: str, first: dict[str, Any]) -> dict[str, Any]:
    """What the feed's settings mean, measured on one listing."""
    driver = scraper.driver
    answers: dict[str, Any] = {}
    items = first["model"]["items"]
    answers["review_fields"] = shape(items[0]) if items else {}
    # Dates and variations are needed for the report; names are not recorded.
    answers["examples"] = [
        {key: item.get(key) for key in ("reviewTime", "boughtDate", "skuInfo", "likeCount", "isPurchased", "rating")}
        for item in items[:2]
    ]
    replies = next((item["replies"] for item in items if item.get("replies")), None)
    answers["seller_reply_fields"] = shape(replies) if replies else "none on this page"
    answers["filter"] = {str(value): brief(feed(driver, item_id, filter=value)) for value in range(1, 6)}
    answers["page_size"] = {str(size): brief(feed(driver, item_id, page_size=size)) for size in (10, 20, 50)}
    second = feed(driver, item_id, page=2)
    if second["kind"] == "reviews":
        ids = lambda reply: [item.get("reviewRateId") for item in reply["model"]["items"]]
        answers["page_2_differs_from_page_1"] = not set(ids(first)) & set(ids(second))
    return answers


def page_against_feed(scraper, item_id: str, star_filter: dict[int, int]) -> dict[str, Any]:
    """Are the reviews the page shows the ones the feed returns, in order?"""
    driver = scraper.driver
    texts = lambda reply: [
        scraper._clean_text(item.get("reviewContent") or "")
        for item in reply["model"]["items"]
        if (item.get("reviewContent") or "").strip()
    ]
    started = time.perf_counter()
    loaded = scraper._scroll_to_reviews()
    result: dict[str, Any] = {"review_section_loaded": loaded, "seconds": round(time.perf_counter() - started, 1)}
    if not loaded:
        return result

    def compare(filter_value: int) -> dict[str, Any]:
        on_page = [review["content"] for review in scraper._reviews_from_current_page()]
        reply = feed(driver, item_id, filter=filter_value)
        if reply["kind"] != "reviews":
            return {"feed": reply["kind"]}
        from_feed = texts(reply)
        return {
            "on_page": len(on_page),
            "from_feed": len(from_feed),
            "shared": len(set(on_page) & set(from_feed)),
            "same_order": on_page == from_feed,
        }

    result["all_stars"] = compare(0)
    if 1 in star_filter and scraper._click_rating_filter(1):
        time.sleep(2)
        result["one_star"] = compare(star_filter[1])
    return result


def run(mode: str) -> int:
    os.chdir(ROOT)  # the scraper looks for its session cookie file here
    sys.path.insert(0, str(ROOT / "lazada-scraper" / "src"))
    import lazada_scraper as ls

    listings = pick_listings()
    state = load()
    entry: dict[str, Any] = {
        "mode": mode,
        "run": 1 + sum(1 for item in state["runs"] if item["mode"] == mode),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "listings": [],
    }
    state["runs"].append(entry)

    scratch = OUT.parent / "_check_hidden_unused.json"
    scraper = ls.LazadaScraper(product_urls=[listings[0]["link"]], output=str(scratch), pause_for_verification=False)
    if mode == "offscreen":
        scraper.options.add_argument("--window-position={},{}".format(*OFFSCREEN_POSITION))
    major = scraper._detect_chrome_major()
    kwargs: dict[str, Any] = {"options": scraper.options, "enable_cdp_events": False, "headless": mode == "headless"}
    if major:
        kwargs["version_main"] = major
        cached = scraper._cached_driver_for_major(major)
        if cached:
            kwargs["driver_executable_path"] = cached
    entry["chrome_major"] = major
    scraper.driver = driver = ls.SafeChrome(**kwargs)
    try:
        driver.set_script_timeout(30)
        window: dict[str, Any] = {"at_start": driver.get_window_rect()}
        if mode == "visible":
            ls._position_scraper_window(driver)
        elif mode == "offscreen":
            driver.set_window_position(*OFFSCREEN_POSITION)
        window["after_placing"] = driver.get_window_rect()
        entry["window"] = window

        started = time.perf_counter()
        scraper._safe_get(scraper.BASE_URL, check_verification=False)
        restored = scraper._load_cookies()
        driver.refresh()
        time.sleep(3)
        entry["home"] = {
            "seconds": round(time.perf_counter() - started, 1),
            "session_restored": bool(restored),
            "logged_in": scraper._has_session(),
            "verification_page": scraper._verification_detected(),
        }
        window.update(driver.execute_script(
            "return {screen_x: window.screenX, screen_y: window.screenY,"
            " screen_width: window.screen.width, visibility: document.visibilityState,"
            " says_headless: /headless/i.test(navigator.userAgent), webdriver_flag: navigator.webdriver};"
        ))
        save(state)
        if entry["home"]["verification_page"]:
            entry["stopped"] = "verification page on the home page"
            print("Lazada shows a verification page. Stopping; nothing was read.")
            return 2

        for number, item in enumerate(listings, start=1):
            item_id = scraper._item_id(item["link"])
            started = time.perf_counter()
            scraper._safe_get(item["link"], check_verification=False)
            time.sleep(2)
            row: dict[str, Any] = {
                "link": item["link"],
                "page_seconds": round(time.perf_counter() - started, 1),
                "title_found": bool(scraper._first_text(driver, TITLE_SELECTORS)),
                "verification_page": scraper._verification_detected(),
            }
            reply = feed(driver, item_id) if not row["verification_page"] else {"kind": "not asked", "seconds": 0}
            row["feed"] = brief(reply)
            row["passed"] = bool(row["title_found"] and not row["verification_page"] and reply["kind"] == "reviews")
            entry["listings"].append(row)
            print(f"{mode} {number}/{len(listings)} page {'ok' if row['title_found'] else 'NOT LOADED'}"
                  f" | feed {reply['kind']} | {item['name'][:45]}", flush=True)
            if number == 1 and reply["kind"] == "reviews":
                answers = feed_questions(scraper, item_id, reply)
                star_filter = {
                    ratings[0]: int(value)
                    for value, found in answers["filter"].items()
                    if (ratings := [r for r in found.get("ratings", []) if r]) and len(set(ratings)) == 1
                }
                answers["page_against_feed"] = page_against_feed(scraper, item_id, star_filter)
                entry["feed_questions"] = answers
            save(state)
            if row["verification_page"] or reply["kind"] == "verification":
                entry["stopped"] = "verification"
                print("Lazada asked for verification. Stopping.")
                break
            time.sleep(PAUSE_SECONDS)
        entry["passed"] = sum(1 for row in entry["listings"] if row["passed"])
        entry["all_passed"] = entry["passed"] == len(listings) and "stopped" not in entry
    finally:
        save(state)
        try:
            driver.quit()
        except Exception:
            pass
        scratch.unlink(missing_ok=True)
    print(f"{mode} run {entry['run']}: {entry.get('passed', 0)} of {len(listings)} listings passed; wrote {OUT.name}")
    return 0


def summary() -> int:
    state = load()
    for entry in state["runs"]:
        print(f"{entry['mode']:10} run {entry['run']}: {entry.get('passed', 0)} of {len(entry['listings']) or LISTINGS}"
              f" passed{' | stopped: ' + entry['stopped'] if 'stopped' in entry else ''}")
    clean = {mode: sum(1 for entry in state["runs"] if entry["mode"] == mode and entry.get("all_passed")) for mode in MODES}
    if clean["headless"] >= 2:
        choice = "headless"
    elif clean["offscreen"] >= 2:
        choice = "offscreen"
    else:
        choice = "neither yet (two clean runs of a hidden mode are needed)"
    print(f"clean runs: {clean}\nrule says: {choice}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--summary" in sys.argv:
        return summary()
    if "--mode" not in sys.argv or sys.argv[sys.argv.index("--mode") + 1] not in MODES:
        print(__doc__)
        return 1
    return run(sys.argv[sys.argv.index("--mode") + 1])


if __name__ == "__main__":
    raise SystemExit(main())
