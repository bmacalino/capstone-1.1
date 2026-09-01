# DeFaketive 1.2.0 release notes

**Status:** Draft
**Planned release date:** 2026-09-01
**Change range:** `c5abbbd..c96b47d`

DeFaketive 1.2.0 introduces a responsive Flask interface for the existing
scraping, storage, and explainable Taglish sentiment-analysis workflows. The
original Streamlit app remains available during the transition.

## Highlights

- Search saved marketplace data or start a supported live scan from the browser.
- Review product-level risk, reliability, sentiment evidence, and recommendations
  in a responsive shopper interface.
- Upload scraper JSON and export product or review data as CSV.
- Use administrator screens for catalog summaries, database records, sentiment
  lexicons, logs, and model evaluation.
- Collect anonymous SUS and UMUX-LITE usability responses using the existing
  research scoring utilities.

## Reliability and navigation fixes

- Inactive live-analysis progress no longer remains visible and obscures the
  interface.
- "How it works" links now switch back to the home view before scrolling, so the
  section is reachable from every app view.

## Upgrade and run

1. Install the updated dependencies:

   ```powershell
   pip install -r requirements.txt
   python -m nltk.downloader vader_lexicon
   ```

2. Start the Flask app:

   ```powershell
   python webapp.py
   ```

3. Open `http://127.0.0.1:5000`.

The legacy interface can still be started with `streamlit run app.py`.

## Compatibility and known limitations

- Existing scraper JSON, SQLite storage, and sentiment-analysis code remain the
  data and analysis sources behind the new interface.
- Shopee live scans may open Chrome and require manual login or CAPTCHA
  verification.
- Lazada and Temu integrations remain beta connectors.
- TikTok Shop and automatic cross-platform product matching are not included.

## Included commits

- `88aaa46` - Migrate Streamlit dashboard to web app.
- `6017a5a` - Hide inactive live analysis panel.
- `c96b47d` - Fix how-it-works navigation.
