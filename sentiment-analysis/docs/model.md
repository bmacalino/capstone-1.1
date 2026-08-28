# DeFaketive sentiment model

The first model version is a local, explainable extension of NLTK VADER for
English, Filipino, and Taglish Philippine e-commerce reviews. It does not call
an external AI service.

## What the model returns

Each review receives:

- positive, neutral, negative, and compound VADER scores;
- a `positive`, `neutral`, or `negative` label;
- the matched sentiment terms and their lexicon values;
- product-risk categories and matched evidence;
- an independent severity-based risk score.

For scraped JSON and CSV data, a one- or two-star marketplace rating is used
only as a guarded tie-breaker: neutral text becomes negative, and a positive
VADER score is changed only when strong negative or active risk evidence is
also present. Clearly positive text remains positive even when its marketplace
rating is unexpectedly low. Any correction is recorded in `label_adjustment`.

Risk detection is separate from sentiment. A neutral sentence such as “The
battery stopped working after one week” still receives a performance/durability
warning even if its general emotional language is weak.

The JSON workflow also fingerprints normalized comments, marks duplicates, and
excludes duplicate copies from product-level aggregation. Unicode symbols are
retained in fingerprints, so emoji-only reviews are included in summaries and
can be deduplicated. Common positive and negative review emojis are explicit,
reviewable entries in the sentiment TSV rather than hidden scoring rules.

## Install the model resource

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

## Analyze one review

```powershell
python -m defaketive_sentiment --text "Maganda pero mabilis masira"
```

## Analyze scraped Shopee or Lazada JSON

```powershell
python -m defaketive_sentiment ..\shopee_ph_laptop.json
```

The default output is `shopee_ph_laptop_analyzed.json`. The original scrape is
not overwritten.

## Analyze a CSV

```powershell
python -m defaketive_sentiment reviews.csv --text-column content
```

The output contains flattened scores, evidence, risk categories, and duplicate
markers.

## Research status

`defaketive-vader-seed-0.1.0` is an engineering baseline, not a validated final
research model. The Filipino/Taglish entries are deliberately kept in editable
TSV files under `defaketive_sentiment/lexicons`. Before reporting model accuracy, the project
must create a human-annotated validation set, measure inter-annotator agreement,
freeze training and test splits, tune thresholds only on training/development
data, and report precision, recall, F1, confusion matrices, and error analysis.

The preliminary product risk score is:

```text
100 * (0.30 * negative_review_ratio + 0.70 * keyword_failure_rate)
```

These weights must be confirmed with the research advisers and validated before
the score is described as calibrated product reliability.
