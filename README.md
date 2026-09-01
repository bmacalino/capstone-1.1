
# DeFaketive Philippine E-Commerce Review Analyzer

This research prototype collects public product and review data from Shopee Philippines, with beta Lazada Philippines and Temu connectors, then produces explainable Taglish sentiment and review-risk reports in a Streamlit dashboard.

The scraper is configured specifically for `shopee.ph`. It supports current and legacy Shopee product-card layouts and English Philippines review filters such as `5 Star`, `With Comments`, and `With Media`.

## Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/FisherFemboy/Capstone-Version-Control.git
   cd Capstone-Version-Control
   ```

2. **Create a Virtual Environment (Optional But Recommended):**
   - Linux/Mac:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows PowerShell:
     ```bash
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Before running:
- Make sure all Chrome windows are fully closed.
- Prepare to log in and solve captchas manually if prompted.

### Basic Command

```powershell
python src/retriv.py -k "your_search_term" -n 10 -r 30
```

When you see the search page loaded in the browser:
1. Log in to Shopee Philippines (if needed).
2. Solve any captcha presented.
3. After continuing to the main search page, press Enter in the terminal to proceed.
4. Keep an eye on the browser; if another captcha appears at any point, solve it to continue scraping.

### Scraping Modes

1. **Review Limit Mode:**
   - Use `-r` or `--review-limit` to collect the requested number of reviews from the lowest rating upward (1-star, then 2-star, and so on):
     ```bash
     python src/retriv.py -k "laptop" -n 5 -r 10
     ```
   This intentionally prioritizes reported problems and adverse product experiences. Because it is not a representative marketplace sample, risk scores derived from this mode will tend to be higher than scores based on Shopee's normal review order.

2. **All-Star Types Mode:**
   - Combine `--all-star-types` with `--star-limit-per-type` to specify how many reviews to retrieve for each star rating:
     ```bash
     python src/retriv.py -k "laptop" -n 5 --all-star-types --star-limit-per-type 5
     ```
   This collects up to 5 reviews for each star rating in separate queries and stores them from 1-star through 5-star.

### Command-Line Arguments

- `-k`, `--keyword`: Search term (default: "Raspberry pi")
- `-n`, `--num`: Number of products to retrieve (default: 10)
- `-r`, `--review-limit`: Total reviews to collect per product (default: 30)
- `--index-only`: If set, only retrieve index data without details
- `--all-star-types`: Collect each star rating separately
- `--star-limit-per-type`: Reviews per star type (default: 10)
- `--product-url`: Direct Shopee PH product URL; repeatable
- `--output`: Custom result JSON path
- `--no-prompt`: Dashboard-safe verification polling mode
- `--verification-timeout`: Seconds to wait for login or CAPTCHA completion

Results are saved incrementally to `shopee_ph_<search_term>.json` after each product. Shopee runs use a fresh temporary Chrome session and do not create or restore a cookie file or persistent browser profile.

## Beta marketplace connectors

- [`lazada-scraper`](lazada-scraper) provides Lazada Philippines keyword, category, and direct-product collection.
- [`temu-scraper`](temu-scraper) provides Temu Philippines keyword and direct-product collection.

The dashboard automatically selects a connector for supported product URLs. Shopee remains the primary integration. TikTok Shop and automatic cross-platform product matching are not implemented.

## Sentiment analysis

Sentiment analysis is maintained as the independent
[`sentiment-analysis`](sentiment-analysis) subproject. After scraping, run its
CLI against this scraper's JSON output. This keeps data collection and review
analysis independent.

## DeFaketive web app

The responsive HTML, CSS, and JavaScript web app adds shopper search/results/
product-risk views plus administrator overview, scraper, lexicon, database, and
model-evaluation screens. A small Flask server keeps the existing Python
scrapers, SQLite storage, and explainable Taglish sentiment analysis available
through browser-friendly endpoints. It can analyze the most recent scraper JSON
automatically, accept an uploaded scraper JSON, or launch a live scrape through
the supported connectors.

```powershell
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
python webapp.py
```

On Windows, you can instead double-click `run_capstone.cmd`. This starts the app
with the project's virtual environment and opens it in your default browser.
Run this launcher from File Explorer or a normal PowerShell window so Chrome has
normal network access; restricted coding-agent terminals can cause
`ERR_NETWORK_ACCESS_DENIED` during a live marketplace scan.

Open `http://127.0.0.1:5000` in a browser. The original Streamlit prototype
remains in `app.py` during the transition and can still be started with
`streamlit run app.py`.

Live analysis shows the three processing stages (scraping, sentiment analysis,
and risk computation). Shopee may still open Chrome for login or captcha
verification. The dashboard uses the current research formula and clearly marks
unavailable evaluation metrics rather than displaying fabricated values.

## Example Command

```bash
python src/retriv.py -k "laptop" -n 5 --all-star-types --star-limit-per-type 3
```

Use the scraper responsibly, keep request volumes low, and follow Shopee's applicable terms and local law. Shopee may require a login or manual verification; the script pauses so you can complete it in the opened Chrome window.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) and
[project notice](NOTICE.md) for copyright and contributor details.

## Capstone team

- **Capstone Adviser:** Kathleen De Guzman
- **Project Leader:** Mark Rhean Caballero
- **Programmer:** Brian Macalino
- **Sub-Programmer and Designer:** Ryan Rovic Retuya
- **Designer:** Princethom Guyo

## Attribution

The scraper foundation is derived from the MIT-licensed
[`bmacalino/shopee_scraper`](https://github.com/bmacalino/shopee_scraper), which
itself builds on work by `dtungpka`. DeFaketive dashboard, research scoring,
SQLite persistence, and integration changes are maintained in this repository.
