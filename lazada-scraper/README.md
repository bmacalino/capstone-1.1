# Lazada Philippines Product and Review Scraper

This project adapts the working control flow of the Shopee Philippines scraper
for `lazada.com.ph`. It drives Chrome (on screen by default, or hidden: see
"Hidden window and the review feed"), extracts real public product data and
written reviews, and saves restart-safe JSON compatible with the DeFaketive
sentiment model.

The scraper never returns generated or sample products when extraction fails.
Instead, it logs the failure and saves page HTML and a screenshot under
`debug/` so selectors can be repaired.

## Responsible use

The scraper does not bypass CAPTCHAs, login screens, rate limits, or other
access controls. Complete any verification manually, use conservative limits,
and follow Lazada's terms and applicable privacy and research requirements.

## Installation

```powershell
cd E:\school\SCRAPPERS\shopee-scraper\lazada-scraper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Close other Chrome windows before the first run if Chrome reports a profile or
driver lock.

## Keyword search

Collect up to five products and 30 written reviews per product:

```powershell
python -m src.lazada_scraper "laptop stand" --num 5 --review-limit 30
```

Scan up to three search pages while still stopping at the requested product
limit:

```powershell
python -m src.lazada_scraper "electronic gadgets" --num 20 --pages 3 --review-limit 100
```

Verify each discovered product on its detail page and scrape reviews only when
the product has at least a 4.0 rating and 100 reported reviews:

```powershell
python -m src.lazada_scraper "electronic gadgets" --num 20 --pages 3 `
  --min-rating 4.0 --min-reviews 100 --review-limit 100
```

Thresholds are inclusive, so exactly 4.0 and exactly 100 qualify. Rejected
products remain in the output with `review_status: "not_qualified"` and a
`qualification.reasons` list. The decision uses the detail-page values rather
than trusting incomplete search-card metadata. Reusing an output with different
thresholds causes rejected products to be checked again.

## Category URL discovery

Use a Lazada Philippines category page instead of a keyword search:

```powershell
python -m src.lazada_scraper `
  --category-url "https://www.lazada.com.ph/shop-electronic-accessories/" `
  --pages 3 --num 20 --min-rating 4.0 --min-reviews 100 `
  --review-limit 100 --output lazada_electronic_accessories.json
```

`--category-url` can be repeated. Existing category query filters, such as
price and sort order, are preserved while the scraper changes the `page`
parameter. Product, account, cart, help-center, and non-Philippine Lazada URLs
are rejected. Each discovered record stores `discovery_source` and
`discovery_page` for traceability.

Discovery progress is saved atomically beside the output as
`<output-name>_discovery.json`. It contains the source configuration, completed
pages, candidate product queue, and per-product status. A matching later run
restores that queue; a checkpoint created for a different keyword or category
is ignored. Override its location when needed:

```powershell
python -m src.lazada_scraper `
  --category-url "https://www.lazada.com.ph/shop-computer-accessories/" `
  --discovery-checkpoint computer_accessories_queue.json
```

If a product cap is reached partway through a category page, that page remains
incomplete in the checkpoint. Increasing `--num` later safely rescans it and
deduplicates already discovered product URLs.

## Direct product URL

```powershell
python -m src.lazada_scraper `
  --product-url "https://www.lazada.com.ph/products/example-i123-s456.html" `
  --review-limit 100 `
  --output lazada_ph_product.json
```

`--product-url` can be repeated to process several known products.

## Rating filter

```powershell
python -m src.lazada_scraper "laptop stand" --num 5 --rating 1 --review-limit 50
```

The rating control depends on the review widget Lazada serves to the browser.
If the current layout has no identifiable star filter, the log reports that
the filter was not applied; it does not silently claim filtered results.

Collect a balanced pilot sample from all five star levels:

```powershell
python -m src.lazada_scraper "laptop stand" --num 5 `
  --all-star-types --star-limit-per-type 20
```

This requests up to 20 written reviews from each star filter, or up to 100 per
product. Each comment records the `source_rating_filter` used. The scraper
visits the filters from 1-star through 5-star, so low-rated feedback is
collected first. Regular `--review-limit` mode uses the same priority and fills
the total limit from the lowest available ratings upward. The scraper
verifies that returned review ratings match each requested filter. If Lazada's
current widget does not apply a filter reliably, collection stops and the
product receives `review_filter_warning`; mismatched results are discarded.

## Hidden window and the review feed

```powershell
python -m src.lazada_scraper --product-url "https://www.lazada.com.ph/products/example-i123-s456.html" `
  --all-star-types --star-limit-per-type 60 --window headless
```

`--window` chooses how Chrome is shown:

- `visible` (default): on screen.
- `offscreen`: the normal window, parked outside the screen. A Chrome icon
  stays on the taskbar.
- `headless`: no window at all.

A hidden scan puts the window on screen by itself when Lazada asks for a login
or a verification, or when the reviews will not load, and stays visible for
the rest of that scan. A headless
scan does this by reopening in a visible Chrome with the same session. A first
run with no saved session starts visible. The scraper answers neither a login
nor a verification.

`--reviews-from` chooses where reviews are read. `page` clicks through the
review section as before. `feed` asks the review feed
(`my.lazada.com.ph/pdp/review/getReviewList`) from inside the open product
page, and is the default for a hidden window. If the feed answers with a
verification page or anything else that is not reviews, the window is shown,
the product page is reopened, the feed is asked once more, and after that the
page is read instead. A request to the feed from outside a browser gets the
verification page; the scraper does not attempt that.

Measured on 12 Oct 2026 (`research/check_hidden_lazada.py`, Chrome 154):

- Both hidden modes loaded five listings and their feed with no verification,
  two runs each. A headless Chrome names itself as headless to Lazada.
- One feed request carries up to 50 reviews. Feed `filter` 1, 3, 4 and 5
  return that star level; `filter` 2 returns 1- and 2-star reviews together,
  so reviews are kept by their own rating. Ratings left without text are
  counted in the totals but not listed, and the listed reviews are followed by
  an empty page.
- The feed lists only the first 300 entries of a list, whatever the page size
  (page 6 of 50 is served, page 7 comes back with no `model`; a page size
  above 50 returns nothing). A quota of 60 is always within reach, except for
  2 stars on a listing whose 1- and 2-star list is longer than 300: the 2-star
  reviews among its first 300 entries are all the feed gives (58 on a listing
  with 41,681 ratings). That level is recorded with stop reason `feed_limit`
  and counts as collected if it holds any review. No other feed setting was
  found that returns 2-star reviews alone.
- One listing, 60 reviews per star level: visible window and page, 136 s;
  headless and page, 71 s with the same reviews in the same order; headless
  and feed, 27 s.
- Page and feed list the same reviews in a star level small enough to be read
  whole (27, 34 and 18 on that listing), but not the same text: the page
  reader keeps the first part of a review written in several parts ("Sound
  quality: … Battery life: …"), the feed has all of it. 45 of 199 reviews
  differed that way. The feed also ranks reviews differently, so a star level
  larger than the quota is sampled differently.

- A scan started with no saved session opened visibly, as intended. Left
  signed out, its second feed request got no reply while a captcha was
  showing in the window. The scan stopped waiting and turned to the page,
  whose review section loaded after one reload about four minutes later, and
  it then finished. Whether the captcha was answered in between was not
  recorded. Every other run that day used the saved login and met no check.

- All 59 baseline listings were scanned from the feed in one run
  (`research/scan_baseline_listings.py`): 9,064 reviews, a median of 23
  seconds a listing, no login or verification asked for.

For those two reasons a feed scan and a page scan of one listing are not
interchangeable. The web app scans from the feed, and its "what is typical"
line was measured on feed scans (`research/RESULTS_TABLES.md`, Table 12d). A feed scan records `review_source: "lazada_review_feed"`,
`review_order: "review_feed_default_within_star"` and a `comment_id` on every
review; reviews of one kind are dropped before a product is continued with
the other.

## Manual verification

When Lazada shows login or verification:

1. Complete it manually in the Chrome window (a hidden scan shows it for this).
2. Return to PowerShell.
3. Press Enter when prompted.

Cookies are stored locally in `cookies_lazada_ph.dat` for later runs. Do not
commit this file. On startup, the scraper restores the cookies and refreshes
Lazada before opening products. If the session is missing or expired, it asks
you to log in once and then saves the renewed session atomically so a browser
failure cannot truncate the previous cookie file.

## Output

Results are written incrementally to `lazada_ph_<keyword>.json`. Each product
includes:

- canonical link, name, PHP price, rating, sold count and image;
- brand, seller, description and rating count when available;
- `platform: "lazada"`, `market: "PH"`, and `currency: "PHP"`;
- `reported_review_count`, the total Lazada reports on the product page;
- `written_reviews_collected`, the unique written comments actually saved;
- `comments`, each containing `rating`, `content`, time, variation, seller
  response and helpful count.

These counts are intentionally separate. A product can report 353 reviews or
ratings while exposing only 134 written comments. The legacy `total_rating`
field remains available for compatibility and mirrors `reported_review_count`.

The JSON files are saved as UTF-8. Windows PowerShell 5.1 can display UTF-8
text as garbled characters when `Get-Content` is used without an encoding.
Use the following when inspecting results from PowerShell:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$results = Get-Content .\lazada_ph_product.json -Raw -Encoding UTF8 | ConvertFrom-Json
$results[0].comments | Select-Object rating, content
```

Editors such as VS Code and applications that parse the JSON as UTF-8 do not
need this PowerShell-specific setting.

Products with no written feedback are saved with `comments: []`,
`review_status: "no_reviews"`, and checkpoint stop reason `no_reviews`. This is
treated as a valid terminal result rather than a scraper failure or an
incomplete product that should be retried forever.

If a previous run stopped before `--review-limit` was reached, running the same
command and output path retries that incomplete product. A product is skipped
after its saved written-review count reaches the requested limit or the review
widget confirms that its Next button is disabled.

Review progress is saved atomically every five pages by default. A resumed run
deduplicates saved comments and advances through already-seen pages before
continuing. Pagination uses three attempts with a 15-second timeout per attempt;
abnormal failures save HTML and a screenshot in `debug/`. Tune these safeguards
with `--checkpoint-every`, `--pagination-retries`, `--pagination-timeout`, and
`--max-review-pages`.

The important review fields match the Shopee output:

```json
{
  "comments": [
    {
      "rating": 2,
      "content": "Sira agad after two days"
    }
  ]
}
```

This allows the standalone DeFaketive analyzer in `..\sentiment-analysis` to
process Lazada output without a separate sentiment schema:

```powershell
cd ..\sentiment-analysis
.\.venv\Scripts\Activate.ps1
python -m defaketive_sentiment ..\lazada-scraper\lazada_ph_smoke_test.json
```

## Tests

```powershell
python -m unittest -v
```

Unit tests cover URL validation, PH number/rating parsing, direct URL mode,
review normalization, and empty-comment filtering. Live Lazada tests are kept
manual because the site can require verification and its rendered layout can
change.

## Attribution and license

The browser lifecycle, cookie persistence, incremental-save pattern, and
pagination control flow were adapted from the MIT-licensed
`dtungpka/shopee-scraper` project and its Shopee Philippines modification.
Lazada-specific extraction code in this repository is newly implemented.

See [LICENSE](LICENSE).
