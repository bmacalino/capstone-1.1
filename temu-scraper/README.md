# Temu Philippines Product and Review Scraper

This visible-browser scraper collects public Temu product metadata and written
reviews into the same JSON schema used by the Shopee and Lazada projects. Its
output can be analyzed directly by the standalone DeFaketive sentiment model.

Temu is JavaScript-heavy and changes generated class names frequently. The
scraper prefers canonical product IDs, JSON-LD, accessible labels, and semantic
attributes. If real products or reviews cannot be identified, it saves HTML and
a screenshot under `debug/` instead of creating sample data.

Temu may require an account before it displays search results or product pages.
When this happens, complete the sign-in manually in the opened Chrome window
and press Enter in PowerShell. A login page is never saved as a product.

## Responsible use

The scraper does not solve CAPTCHAs, automate logins, or bypass verification.
Complete any challenge manually, use conservative limits, and follow Temu's
terms and applicable privacy and research requirements.

## Installation

```powershell
cd E:\school\SCRAPPERS\shopee-scraper\temu-scraper
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Close other Chrome windows before the first run if Chrome reports a profile or
driver lock.

## Keyword search

Scrape up to five products and 100 written reviews per product:

```powershell
python -m src.temu_scraper "laptop stand" --num 5 --pages 2 --review-limit 100
```

The default review limit is 100 per product. Use `--review-limit` to request a
higher or lower value; Temu may expose fewer written reviews than requested.

Only keep products with at least 4.0 stars and 100 displayed reviews:

```powershell
python -m src.temu_scraper "electronic gadgets" --num 20 --pages 3 `
  --min-rating 4.0 --min-reviews 100 --review-limit 100
```

`--pages` controls groups of infinite-scroll loads because Temu does not always
expose conventional numbered search pages.

## Direct product URL

Both Temu URL forms are accepted and normalized to a canonical `goods_id` URL:

```powershell
python -m src.temu_scraper `
  --product-url "https://www.temu.com/product-name-g-601099588578152.html" `
  --review-limit 100 `
  --output temu_ph_product.json
```

`--product-url` can be repeated for several known products.

## Collect every review exposed by Temu

```powershell
python -m src.temu_scraper `
  --product-url "PASTE_TEMU_PRODUCT_URL" `
  --all-reviews --max-review-batches 200
```

This continues until no new unique reviews are exposed or the safety cap is
reached. Temu may display a larger total than it makes available in the current
browser session. In that case, the product receives `review_collection_warning`
instead of the scraper claiming that the collection is complete.

When a product initially shows only a review preview, the scraper opens Temu's
`See all reviews` control before loading additional batches. It selects the
review sort menu's exact `Lowest rating` option first, then scrolls the review
modal's internal scrollbar until the requested limit or Temu's exposed end is
reached. Review cards are
identified through their accessible star, avatar, date, and text markup rather
than Temu's generated CSS class names. It also merges written reviews from
Temu's own review Fetch/XHR responses when Chrome makes those responses
available, preferring the structured review ID and translated text while
deduplicating the corresponding rendered card. Rating-only cards count toward
end-of-list detection but are omitted from `comments` unless `--include-empty`
is used. A redirect away from the product during review loading is checkpointed
as retryable instead of being treated as a completed collection.

If Temu does not expose the exact `Lowest rating` sort, both limited and
all-review modes fall back to its visible rating filters from 1-star through
5-star. If neither control is identifiable, the scraper uses the normal review
feed and sorts saved comments from the lowest extracted rating to the highest.
For captured Fetch/XHR reviews, `reviewLang.translateComment` is preferred over
the original-language comment whenever Temu supplies it.

Review progress is saved atomically every five loaded batches. Run the same
command and output path after an interruption to resume with saved comments;
duplicates are ignored while the scraper advances to new reviews. Transitions
use three attempts and a 15-second timeout by default. Use
`--checkpoint-every`, `--pagination-retries`, and `--pagination-timeout` to tune
these settings. Abnormal failures save HTML and a screenshot in `debug/`.

## Output

Results are saved incrementally to `temu_ph_<keyword>.json`. Important fields
include:

```json
{
  "platform": "temu",
  "market": "PH",
  "currency": "PHP",
  "rating": 4.7,
  "total_rating": 1200,
  "comments": [
    {
      "comment_id": "2641578380257808",
      "rating": 2,
      "content": "Sira agad after two days"
    }
  ]
}
```

For data minimization, saved review objects contain only `comment_id`, `rating`,
and `content`, and records without a structured comment ID are rejected.
Reviewer names, dates, profile details, variations, seller responses, and
helpful-vote counts are not written to the result file.

Products are saved with `review_status: "no_reviews"` only when the rendered
page explicitly displays a zero-review message. A missing review widget, blank
product metadata, or an empty unrecognized layout is recorded as the retryable
`review_widget_missing` state so layout/login failures are not mistaken for
genuinely empty products. Older incomplete `no_reviews` checkpoints are
automatically retried. A temporary missing-widget response also preserves any
written comments that were saved by an earlier successful batch.

Analyze the output with:

```powershell
cd ..\sentiment-analysis
.\.venv\Scripts\Activate.ps1
python -m defaketive_sentiment ..\temu-scraper\temu_ph_laptop_stand.json
```

## Tests

```powershell
python -m unittest -v
```

Unit tests cover Temu URL canonicalization, PH search URLs, numeric parsing,
JSON-LD normalization, review normalization, filters, and compatible output.
Live extraction remains a manual integration test because Temu can require
verification and its rendered layout changes frequently.

## Attribution and license

The browser lifecycle, cookie persistence, incremental-save pattern, and
output conventions follow the MIT-licensed Shopee/Lazada projects in this
repository. Temu-specific URL handling, extraction, filters, diagnostics, and
review collection are newly implemented. See [LICENSE](LICENSE).
