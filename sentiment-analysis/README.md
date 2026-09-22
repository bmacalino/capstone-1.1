# DeFaketive Sentiment Analysis

This is the standalone sentiment-analysis component for DeFaketive. It reads
review data produced by either the Shopee Philippines scraper or the Lazada
Philippines scraper; neither scraper needs to contain or import this model.

The production path uses trained marketplace classifiers for the review's
positive/negative/uncertain sentiment and its separate binary risk decision.
Customized NLTK VADER and editable Filipino/Taglish lexicons remain available
as supporting evidence. The product score follows the manuscript formula:
`30% * negative-review ratio + 70% * detected-risk ratio`. Severity is not part
of this score. Analysis runs locally.

The default mode is `hybrid_manuscript`. Set
`DEFAKETIVE_MODEL_MODE=manuscript` before starting the app to run the strict
VADER/lexicon baseline, or `legacy_hybrid` to reproduce the former
probability-and-severity research score.

## Setup on Windows PowerShell

```powershell
cd path\to\capstone-1.1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader vader_lexicon
```

## Analyze one review

```powershell
python -m defaketive_sentiment --text "Maganda pero mabilis masira"
```

## Analyze Shopee results

```powershell
python -m defaketive_sentiment ..\shopee_ph_laptop_stand.json
```

## Analyze Lazada results

```powershell
python -m defaketive_sentiment ..\lazada-scraper\lazada_ph_smoke_test.json
```

## Analyze Temu results

```powershell
python -m defaketive_sentiment ..\temu-scraper\temu_ph_laptop_stand.json
```

Unless `--output` is supplied, the analyzer writes a sibling file ending in
`_analyzed.json` or `_analyzed.csv` and preserves the original scrape.

For model fields, scoring, and research limitations, see
[`docs/model.md`](docs/model.md).

## Reproduce FiReCS training

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe .\train_firecs.py
```

The script audits duplicate and cross-split text, selects among five candidates
using repeated stratified validation, including weighted and unweighted logistic
regression, retrains the winner, and records the final held-out metrics. Do not
tune future changes against the official test errors.

## Reproduce the hybrid-manuscript model

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe .\research\train_hybrid_manuscript.py
```

This trains direct positive/negative/uncertain sentiment with FiReCS and all
adjudicated Lazada reviews, trains binary risk only on the Lazada reviews, and
selects the risk threshold from product-grouped out-of-fold predictions. The
artifact intentionally contains no risk-category or severity head. It remains a
development candidate until it is evaluated on newly collected, frozen product
groups.

## Reproduce the legacy risk-scoring candidate

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe .\research\train_risk_scoring_v1.py
```

This older research script trains risk detection and ordinal severity from the adjudicated Lazada
development data using five-fold product-group cross-validation. It writes the
candidate artifact to `models/risk_scoring_v1_candidate.joblib` and its metrics
to `evaluation/risk_scoring_v1_cross_validation.json`. The candidate remains
unqualified until it passes a newly frozen, product-disjoint locked test.

## Optional auxiliary datasets

The repository also contains SentiTaglish and TagaSenti under `data/`. Their
loaders are built into `train_firecs.py`; installing the Hugging Face `datasets`
package is not required after the CSV files are present.

Run the auxiliary experiment into separate candidate files:

```powershell
.\.venv\Scripts\python.exe .\train_firecs.py `
  --include-sentitaglish --include-tagasenti `
  --model-output .\sentiment-analysis\models\multisource_sentiment_candidate.joblib `
  --result-output .\sentiment-analysis\evaluation\multisource_sentiment_candidate.json `
  --errors-output .\sentiment-analysis\evaluation\multisource_sentiment_candidate_errors.csv
```

The trainer excludes SentiTaglish's mixed class from the three-class model,
uses punctuation-insensitive leakage keys, gives FiReCS labels precedence, and
removes every matching FiReCS test review before fitting. TagaSenti is recorded
as mixed-provenance auxiliary data because its CSV does not identify which rows
are human, synthetic, translated, or LLM-labeled.

The original scraper's unrelated Vietnamese labeling script is preserved under
`legacy/` for reference, but it is not imported or installed by this project.

## Run tests

```powershell
python -m unittest -v
```
