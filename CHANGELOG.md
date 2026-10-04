# Changelog

All notable changes to DeFaketive are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- An isolated marketplace-v2 research pipeline with product-grouped evaluation,
  validation-tuned risk-category thresholds, explicit deployment gates, and a
  safe inference wrapper that rejects unqualified artifacts by default.
- A reproducible FiReCS training pipeline that audits duplicates and cross-split
  overlap, compares four supervised candidates, and records held-out metrics.
- A compact trained word/character TF-IDF sentiment artifact with probabilities
  and active-feature evidence.
- A smart product search flow with automatic marketplace and saved-report
  detection, dynamic actions, advanced collection limits, persistent errors,
  and inline scan progress.
- Local SentiTaglish and TagaSenti datasets, provenance documentation, optional
  multisource loaders, and a punctuation-insensitive leakage audit.

### Changed

- Production analysis now uses the trained sentiment and binary-risk heads for
  review decisions while retaining the manuscript's 30% negative-review plus
  70% detected-risk product score; severity is excluded from production output
  and scoring.
- Sentiment labels now use the trained FiReCS classifier for ordinary review
  text while retaining explainable VADER/Taglish overrides and product-risk
  rules for explicit cases.
- Pinned scikit-learn and joblib to the artifact's training versions for
  reproducible installation on another computer.
- Auxiliary models must now match or beat the current trained classifier before
  they qualify for integration; the initial multisource candidate was retained
  as a rejected research result rather than replacing the stronger model.
- Redesigned the shopper dashboard, analysis report, navigation, product cards,
  filters, and interface wording around clear shopping decisions.
- Live scraper browsers now open in an adaptive right-side window so the app's
  collection progress remains visible.
- Product lists (catalog, saved search, upload, and a report's similar
  products) are sent without their review lists; a product's reviews come
  only with its own report. The list for one product drops from about
  155 KB to 9 KB.
- Finished review analyses are kept in `instance/analysis_cache/` and reused
  after a restart. The first page load no longer re-analyses the newest
  result file each time the app starts, and saved products are no longer
  re-analysed for every browser session when no result file exists.

### Fixed

- Prevented a manually selected marketplace from conflicting with the
  marketplace detected from a supported product URL.
- The page-script honesty test no longer mistakes the elapsed-time
  arithmetic (`/ 1000`) for a `/ 100` score.
- A catalog test no longer saves its sample product into the real database.

### Removed

- The legacy Streamlit prototype (`app.py`), its tests, the score-era helper
  functions only it used, and the `streamlit` and `plotly` requirements. The
  Flask web app (`webapp.py`) is the only interface.

## [1.2.0] - 2026-09-01

### Added

- A responsive Flask web app for shopper search, product results, product-risk
  analysis, and administrator workflows.
- Browser endpoints for saved and live marketplace searches, scraper JSON
  uploads, product details, CSV exports, administration data, usability surveys,
  and model evaluation.
- Administrator views for the catalog overview, stored data, sentiment lexicons,
  and application logs.
- Automated coverage for the web shell, saved search, product details, and
  research-based usability scoring.
- Open Graph artwork for link previews.

### Changed

- Documented the Flask app as the primary browser interface while retaining the
  Streamlit prototype during the transition.
- Added Flask 3 as an application dependency.

### Fixed

- Kept the inactive live-analysis status panel hidden.
- Made the "How it works" navigation return to the home view before scrolling to
  its section.

[1.2.0]: https://github.com/RovicRetuya/Capstone-Version-Control/compare/c5abbbd...c96b47d
