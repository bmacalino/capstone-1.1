"""Can Temu's own review list be read page by page, the way the Lazada scan reads Lazada's review feed? A measurement.

Why. Temu scans read few reviews (6 to 29 for the 7 saved Temu products; the last scan stopped after 8), always
lowest rating first, and Temu often shows a check (a puzzle). The scanner clicks and scrolls its way through the
review section and reads what the page itself loads. Before anything is built, three things are measured once:

  1. What the page itself asks for when it shows a product's reviews: the address, the request's fields, the names
     (never the values) of the headers it sends, and the shape of the answer: reviews per page, the total, and how
     the next page is asked for. Sorting by lowest rating and one star filter are tried too, to see their fields.
  2. When the review list is scrolled as a person would, how many pages the page loads by itself and how many
     reviews they hold: what the scanner can read without asking for anything itself.
  3. Whether the same review list answers ONE plain request for the next page, sent from inside the page with the
     session's own cookies and nothing else. No anti-bot token is copied or made: if Temu wants one, the answer is
     "no", and that is the result.

How. A Chrome window opens on Temu with the app's own Temu session (.browser_profiles/temu and cookies_temu_ph.dat),
as a scan does, on one saved product. If Temu shows a check or a login, the person does it in that window; this
script never answers one (the scanner waits up to 10 minutes, as in the app). Run it only while no Temu scan runs.

What is kept: research/temu_feed_check.json, with the review list's path and the names of its query fields, the
request's field names, safe field values (numbers and short words, never anything named like a token, a session
or a signature), header names, counts, timings and status codes. No review text and no reviewer names.

    python research/check_temu_review_feed.py
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "temu-scraper"))
from selenium.webdriver.common.by import By  # noqa: E402
from src.temu_scraper import SafeChrome, TemuScraper, _position_scraper_window  # noqa: E402

OUT = ROOT / "research" / "temu_feed_check.json"
# A saved Temu product with 377 ratings, of which the last scan read 15.
PRODUCT = "https://www.temu.com/goods.html?goods_id=601100344333196"
PROFILE = ROOT / ".browser_profiles" / "temu"
COOKIES = ROOT / "cookies_temu_ph.dat"
REVIEW_LIST = re.compile(r"review[^?]*/list|reviews?/list|comment[^?]*/list", re.I)
PAUSE_SECONDS = 2.5
SCROLLS = 3
# Field and header values are never kept for names like these.
SECRET_NAME = re.compile(r"token|anti|sign|verif|auth|cookie|session|secret|key|nonce|finger|fp", re.I)
# Fields with names like these are left out of the one plain request: it never carries an anti-bot token.
ANTI_BOT_NAME = re.compile(r"anti|token|sign|verif|nonce|finger|^fp|risk|crypt|captcha|auth", re.I)
PAGING_NAME = re.compile(r"page|total|count|more|cursor|offset|next|size|num|list_?id|has", re.I)
FETCH_SCRIPT = r"""
const [url, method, contentType, body] = arguments;
const done = arguments[arguments.length - 1];
const options = {method: method, credentials: 'include', headers: {}};
if (contentType) options.headers['content-type'] = contentType;
if (method !== 'GET' && method !== 'HEAD') options.body = body;
fetch(url, options)
  .then(response => response.text().then(text => done(JSON.stringify({status: response.status, text: text}))))
  .catch(error => done(JSON.stringify({error: String(error)})));
"""


def safe_value(name: str, value: Any) -> Any:
    """A field's value if it is safe and useful to keep, otherwise only its type and size."""
    if SECRET_NAME.search(str(name)):
        return f"<{type(value).__name__}, not kept>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and len(value) <= 24 and re.fullmatch(r"[\w .:,-]*", value):
        return value
    return f"<{type(value).__name__}, {len(json.dumps(value, ensure_ascii=False))} chars>"


def parse_body(raw: str) -> dict[str, Any] | None:
    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def walk(value: Any, path: tuple[str, ...] = (), depth: int = 4):
    """Every (path, value) in an answer, a few levels deep; of a list of records only the first is looked into."""
    if depth < 0:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield path + (str(key),), item
            yield from walk(item, path + (str(key),), depth - 1)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        yield from walk(value[0], path + ("[0]",), depth - 1)


def answer_shape(payload: Any, goods_id: str) -> dict[str, Any]:
    """What a review-list answer holds, without its reviews' text or names."""
    reviews = TemuScraper._reviews_from_network_payload(payload, expected_goods_id=goods_id)
    shape: dict[str, Any] = {
        "reviews": len(reviews),
        "stars": sorted({review["rating"] for review in reviews}),
        "answer_keys": sorted(payload)[:40] if isinstance(payload, dict) else type(payload).__name__,
        "paging": {},
    }
    for path, value in walk(payload):
        if "[0]" in path or isinstance(value, (dict, list)) or not PAGING_NAME.search(path[-1]):
            continue
        shape["paging"][".".join(path)] = safe_value(path[-1], value)
    if isinstance(payload, dict):
        for name in ("success", "error_code", "errorCode", "error_msg", "errorMsg", "code", "msg", "message"):
            if name in payload and not isinstance(payload[name], (dict, list)):
                shape.setdefault("status_fields", {})[name] = str(payload[name])[:120]
    return shape


class Recorder:
    """Every review-list request the page makes, by step, read from Chrome's own network log."""

    def __init__(self, scraper: TemuScraper, goods_id: str):
        self.scraper = scraper
        self.goods_id = goods_id
        self.requests: dict[str, dict[str, Any]] = {}
        self.extra_headers: dict[str, list[str]] = {}

    def collect(self, step: str) -> list[dict[str, Any]]:
        new: list[str] = []
        for entry in self.scraper._drain_performance_logs():
            try:
                message = json.loads(entry.get("message", "{}"))["message"]
            except (KeyError, TypeError, ValueError):
                continue
            method = message.get("method")
            params = message.get("params", {})
            request_id = params.get("requestId", "")
            if method == "Network.requestWillBeSent":
                request = params.get("request", {})
                if REVIEW_LIST.search(str(request.get("url") or "")):
                    self.requests[request_id] = {
                        "id": request_id,
                        "step": step,
                        "url": request.get("url", ""),
                        "method": request.get("method", ""),
                        "post": request.get("postData") or "",
                        "headers": request.get("headers", {}),
                    }
                    new.append(request_id)
            elif method == "Network.requestWillBeSentExtraInfo":
                self.extra_headers[request_id] = sorted(params.get("headers", {}))
            elif method == "Network.responseReceived" and request_id in self.requests:
                self.requests[request_id]["status"] = params.get("response", {}).get("status")
        for request_id in new:
            record = self.requests[request_id]
            try:
                body = self.scraper.driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
                text = body.get("body", "")
                if body.get("base64Encoded"):
                    text = base64.b64decode(text).decode("utf-8", errors="replace")
                record["answer"] = json.loads(text)
            except Exception as exc:  # the answer can be gone already, or not be JSON
                record["answer_error"] = type(exc).__name__
        return [self.requests[request_id] for request_id in new]

    def kept(self, record: dict[str, Any]) -> dict[str, Any]:
        """The part of one request that is written down."""
        parts = urlsplit(record["url"])
        body = parse_body(record["post"])
        kept = {
            "step": record["step"],
            "path": parts.path,
            "query_fields": {name: safe_value(name, value) for name, value in parse_qsl(parts.query)},
            "method": record["method"],
            "body_is_json": body is not None,
            "body_fields": {name: safe_value(name, value) for name, value in (body or {}).items()},
            "header_names": sorted(record["headers"]),
            "all_header_names": self.extra_headers.get(record["id"], []),
            "status": record.get("status"),
        }
        if "answer" in record:
            kept.update(answer_shape(record["answer"], self.goods_id))
        elif "answer_error" in record:
            kept["answer_error"] = record["answer_error"]
        return kept


def request_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {**dict(parse_qsl(urlsplit(record["url"]).query)), **(parse_body(record["post"]) or {})}


def how_pages_advance(pages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How each request field that moves from one page to the next moves: by a count, or to a value the answer gave.

    Taken from the first two pages in a row the page itself asked for. A field that moves in neither way is
    "unclear" (a time, or something the page works out for itself).
    """
    for earlier, later in zip(pages, pages[1:]):
        first, second = request_fields(earlier), request_fields(later)
        changed = sorted(name for name in second if first.get(name) != second.get(name))
        if not changed:
            continue
        rules: dict[str, dict[str, Any]] = {}
        for name in changed:
            try:
                step = int(second[name]) - int(first.get(name))
            except (TypeError, ValueError):
                step = 0
            if step > 0:
                rules[name] = {"kind": "count", "step": step}
                continue
            path = next(
                (list(path) for path, value in walk(earlier.get("answer")) if value == second[name] and "[0]" not in path),
                None,
            )
            rules[name] = {"kind": "from_answer", "answer_path": path} if path else {"kind": "unclear"}
        return {"fields": rules}
    return None


def next_request(last: dict[str, Any], advance: dict[str, Any]) -> tuple[str, str, list[str]] | None:
    """The address and body of the page after `last`, and the fields left out of it.

    Every field that moves is moved; any field named like an anti-bot token, a signature or a check is left out,
    in the body and in the address alike. None when the next page cannot be told.
    """
    rules = advance.get("fields") or {}
    if not any(rule["kind"] != "unclear" for rule in rules.values()):
        return None

    def moved(name: str, current: Any) -> Any:
        rule = rules.get(name)
        if not rule or rule["kind"] == "unclear":
            return current
        if rule["kind"] == "count":
            return int(current) + rule["step"]
        found = last.get("answer")
        for part in rule["answer_path"]:
            found = found.get(part) if isinstance(found, dict) else None
        return current if found is None else found

    parts = urlsplit(last["url"])
    query = dict(parse_qsl(parts.query))
    body = parse_body(last["post"])
    left_out = sorted(name for name in [*query, *(body or {})] if ANTI_BOT_NAME.search(name))
    query = {name: str(moved(name, value)) for name, value in query.items() if name not in left_out}
    url = urlunsplit(parts._replace(query=urlencode(query)))
    if body is None:
        return url, "" if last["post"] else last["post"], left_out
    body = {name: moved(name, value) for name, value in body.items() if name not in left_out}
    return url, json.dumps(body, separators=(",", ":")), left_out


def click_one_star_filter(scraper: TemuScraper, root: Any) -> str:
    """Click the review list's "1 star" filter once, if it shows one."""
    pattern = re.compile(r"(^|\D)1\s*stars?\b", re.I)
    for scope in (root, scraper.driver):
        try:
            candidates = scope.find_elements(
                By.CSS_SELECTOR, 'button, [role="button"], [role="option"], [role="tab"], label'
            )
        except Exception:
            continue
        for candidate in candidates:
            try:
                label = " ".join(filter(None, (candidate.text, candidate.get_attribute("aria-label"))))
                if candidate.is_displayed() and len(label) <= 40 and pattern.search(label):
                    scraper.driver.execute_script("arguments[0].scrollIntoView({block:'center'})", candidate)
                    scraper.driver.execute_script("arguments[0].click()", candidate)
                    return label.strip()
            except Exception:
                continue
    return ""


def plain_request(scraper: TemuScraper, last: dict[str, Any], advance: dict[str, Any], goods_id: str) -> dict[str, Any]:
    """ONE request for the next page from inside the page: the session's cookies and the content type, nothing else."""
    built = next_request(last, advance)
    if not built:
        return {"sent": False, "why": "the next page's request could not be told from the pages the page loaded"}
    url, body, left_out = built
    content_type = next((value for name, value in last["headers"].items() if name.casefold() == "content-type"), "")
    answer = json.loads(scraper.driver.execute_async_script(FETCH_SCRIPT, url, last["method"], content_type, body))
    result: dict[str, Any] = {
        "sent": True,
        "left_out_fields": left_out,
        "status": answer.get("status"),
        "error": answer.get("error"),
    }
    try:
        payload = json.loads(answer.get("text") or "")
    except ValueError:
        result["answer_is_json"] = False
        result["answer_start"] = re.sub(r"\s+", " ", (answer.get("text") or "")[:160])
    else:
        result.update(answer_shape(payload, goods_id))
    time.sleep(PAUSE_SECONDS)
    result["check_showing_after"] = scraper._verification_detected()
    return result


def measure(scraper: TemuScraper, recorder: Recorder, product: dict[str, Any], run: dict[str, Any], note) -> None:
    """The steps, in the order a scan takes them. A step that finds Temu not showing what it needs ends the run."""
    ready = scraper._prepare_session()
    note("home page and saved session", ready=ready)
    if not ready:
        return
    scraper._reset_product_network_capture(product)
    blocked = scraper._safe_get(scraper.product_navigation_urls.get(product["link"], product["link"]))
    note("product page", blocked=blocked, path=urlsplit(scraper.driver.current_url).path)
    if blocked:
        return
    time.sleep(PAUSE_SECONDS)
    root = scraper._open_review_section()
    if root is None:
        note("review section", found=False, check_showing=scraper._verification_detected())
        return
    time.sleep(PAUSE_SECONDS)
    first = recorder.collect("first page")
    note("review section", found=True, list_requests=len(first), dom_reviews=len(scraper._reviews_from_current_view(root)))

    pages: list[dict[str, Any]] = []
    sorted_lowest = scraper._click_lowest_rating_sort(root)
    time.sleep(PAUSE_SECONDS)
    root = scraper._review_root() or root
    lowest = recorder.collect("lowest rating")
    pages.extend(lowest)
    note(
        "sort: lowest rating", clicked=sorted_lowest, list_requests=len(lowest),
        dom_reviews=len(scraper._reviews_from_current_view(root)),
    )
    for number in range(1, SCROLLS + 1):
        scraper.driver.execute_script(scraper._review_scroll_javascript(), root)
        time.sleep(PAUSE_SECONDS)
        root = scraper._review_root() or root
        more = recorder.collect(f"scroll {number}")
        pages.extend(more)
        note(
            f"scroll {number}",
            list_requests=len(more),
            dom_reviews=len(scraper._reviews_from_current_view(root)),
            dom_signature=bool(scraper._review_signature(root)),
            check_showing=scraper._verification_detected(),
        )

    label = click_one_star_filter(scraper, root)
    time.sleep(PAUSE_SECONDS)
    filtered = recorder.collect("1-star filter")
    note("star filter", clicked=label or False, list_requests=len(filtered))

    advance = how_pages_advance(pages)
    run["advance"] = advance
    note("how the next page is asked for", advance=advance)
    if not pages or not advance:
        run["plain_request"] = {"sent": False, "why": "the page loaded no second page to learn from"}
        return
    run["plain_request"] = plain_request(scraper, pages[-1], advance, product["goods_id"])
    note(
        "one plain request for the next page",
        **{key: run["plain_request"].get(key) for key in ("sent", "status", "reviews", "check_showing_after")},
    )


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    run: dict[str, Any] = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": PRODUCT,
        "steps": [],
        "checks": [],
    }
    current = {"step": "start"}
    started = time.time()

    def note(step: str, **facts: Any) -> None:
        current["step"] = step
        run["steps"].append({"step": step, "seconds": round(time.time() - started, 1), **facts})
        print(f"[{time.time() - started:6.1f} s] {step} {json.dumps(facts, ensure_ascii=False)[:300]}", flush=True)

    with TemporaryDirectory() as scratch:
        scraper = TemuScraper(
            product_urls=[PRODUCT],
            review_limit=100,
            chrome_user_data_dir=str(PROFILE),
            cookies_file=str(COOKIES),
            output=str(Path(scratch) / "not-kept.json"),
            pause_for_verification=False,
            verification_timeout=600,
        )
        waiting = scraper._wait_for_verification_clear

        def timed_wait() -> bool:
            began = time.time()
            cleared = waiting()
            run["checks"].append({"during": current["step"], "cleared": cleared, "seconds": round(time.time() - began, 1)})
            return cleared

        scraper._wait_for_verification_clear = timed_wait
        major = scraper._detect_chrome_major()
        options: dict[str, Any] = {"options": scraper.options, "enable_cdp_events": False, "headless": False}
        if major:
            options["version_main"] = major
            cached = scraper._cached_driver_for_major(major)
            if cached:
                options["driver_executable_path"] = cached
        scraper.driver = SafeChrome(**options)
        scraper.driver.execute_cdp_cmd("Network.enable", {})
        scraper.driver.set_script_timeout(30)
        _position_scraper_window(scraper.driver)
        product = scraper._product_from_url(PRODUCT)
        recorder = Recorder(scraper, product["goods_id"])
        try:
            measure(scraper, recorder, product, run, note)
        except Exception as exc:
            note("stopped", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        finally:
            run["list_requests"] = [recorder.kept(record) for record in recorder.requests.values()]
            try:
                scraper._save_cookies()
                scraper.driver.quit()
            except Exception:
                pass
    state = json.loads(OUT.read_bytes().decode("utf-8")) if OUT.is_file() else {"what": __doc__.split("\n\n")[0], "runs": []}
    state["runs"].append(run)
    OUT.write_bytes((json.dumps(state, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print(f"Written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
