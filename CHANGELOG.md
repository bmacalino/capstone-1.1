# Changelog

All notable changes to DeFaketive are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
