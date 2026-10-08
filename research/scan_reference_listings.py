"""Fresh scans of a fixed set of gadget listings: the reference set for the
verdict check (research/verdict_check.py) and for what is typical
(research/build_reference_baselines.py).

Which listings, written before any was scanned: from each of the seven
Lazada discovery files in research/data, the four listings with the most
reported reviews whose name passes the gadget word test
(gadget_scope.mentions_gadget); ties go to the link that sorts first, and a
listing is taken once. Up to 28.

Each is scanned the way the web app scans a pasted link
(webapp._command_for_scrape): all five star levels, up to 60 written reviews
per level, per-star counts recorded, so a scan can be weighted back to the
listing. Two differences from the app. The result goes to
research/data/reference_scans/, never the project root, whose scan files are
training sources. And a verification page is not waited for: that listing is
skipped and reported, not retried.

    python research/scan_reference_listings.py            # scan what is missing
    python research/scan_reference_listings.py --list     # show the selection only
    python research/scan_reference_listings.py --limit 1  # the first missing listing only
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
for candidate in (ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

DISCOVERY = ROOT / "research" / "data"
OUT = ROOT / "research" / "data" / "reference_scans"
LOG = OUT / "scan_log.json"
SCRAPER = ROOT / "lazada-scraper" / "src" / "lazada_scraper.py"
PER_FILE = 4
REVIEWS = 100          # webapp's default review count
PER_STAR = 60          # webapp.LAZADA_REVIEWS_PER_STAR
VERIFICATION_WAIT = 45
TIME_BUDGET = 480      # hazard_parameters_v1.json: per_product_time_budget_seconds


def selection() -> list[dict[str, Any]]:
    from gadget_scope import mentions_gadget

    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(DISCOVERY.glob("lazada_*_discovery.json")):
        rows = []
        for product in json.loads(path.read_text(encoding="utf-8")).get("products", []):
            link, name = str(product.get("link") or ""), str(product.get("name") or "")
            try:
                reported = int(float(product.get("reported_review_count") or 0))
            except (TypeError, ValueError):
                reported = 0
            if link and mentions_gadget(name):
                rows.append((-reported, link, name))
        taken = 0
        for negative, link, name in sorted(rows):
            if link in seen or taken == PER_FILE:
                continue
            seen.add(link)
            taken += 1
            item = re.search(r"i(\d+)", link.rsplit("/", 1)[-1])
            chosen.append({
                "type": path.stem.removeprefix("lazada_").removesuffix("_discovery"),
                "item_id": item.group(1) if item else str(abs(hash(link))),
                "link": link, "name": name, "reported_reviews": -negative,
            })
    return chosen


def target_for(listing: dict[str, Any]) -> Path:
    return OUT / f"lazada_i{listing['item_id']}.json"


def usable(path: Path) -> int:
    """Written reviews in a finished scan file, or 0."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    products = data if isinstance(data, list) else data.get("products", []) if isinstance(data, dict) else []
    return sum(
        1 for product in products if isinstance(product, dict)
        for comment in product.get("comments") or []
        if isinstance(comment, dict) and str(comment.get("content") or "").strip()
    )


def scan(listing: dict[str, Any]) -> dict[str, Any]:
    target = target_for(listing)
    command = [
        sys.executable, str(SCRAPER), "-n", "1", "-r", str(REVIEWS), "--all-star-types",
        "--star-limit-per-type", str(PER_STAR), "--output", str(target), "--no-verification-pause",
        "--verification-timeout", str(VERIFICATION_WAIT), "--product-url", listing["link"],
    ]
    started = time.time()
    try:
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIME_BUDGET
        )
        tail = [line for line in (result.stdout + "\n" + result.stderr).splitlines() if line.strip()][-3:]
        code: Any = result.returncode
    except subprocess.TimeoutExpired:
        tail, code = [f"stopped after {TIME_BUDGET} seconds"], "timeout"
    reviews = usable(target)
    return {
        "status": "ok" if reviews else "skipped", "reviews": reviews, "exit": code,
        "seconds": round(time.time() - started), "last_lines": tail,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    listings = selection()
    if "--list" in sys.argv:
        for listing in listings:
            print(f"{listing['type']:18} {listing['reported_reviews']:6} {listing['item_id']:12} {listing['name'][:60]}")
        print(len(listings), "listings")
        return 0
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(listings)
    OUT.mkdir(parents=True, exist_ok=True)
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.is_file() else {"rule": __doc__.split("\n\n")[1], "listings": {}}
    done = 0
    for listing in listings:
        have = usable(target_for(listing))
        if have:
            log["listings"].setdefault(listing["item_id"], {**listing, "status": "ok", "reviews": have})
            continue
        if done == limit:
            break
        done += 1
        print(f"scanning {listing['item_id']} ({listing['type']}): {listing['name'][:50]}", flush=True)
        outcome = scan(listing)
        log["listings"][listing["item_id"]] = {**listing, **outcome}
        print(f"  {outcome['status']}: {outcome['reviews']} reviews in {outcome['seconds']}s | {outcome['last_lines'][-1][:110] if outcome['last_lines'] else ''}", flush=True)
        LOG.write_bytes((json.dumps(log, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    LOG.write_bytes((json.dumps(log, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    ok = sum(1 for entry in log["listings"].values() if entry.get("status") == "ok")
    print(f"{ok} of {len(listings)} listings scanned; log in {LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
