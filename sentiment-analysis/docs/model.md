# DeFaketive sentiment model

DeFaketive uses a local manuscript-aligned sentiment and product-risk model for
English, Filipino, and Taglish Philippine e-commerce reviews. It does not call
an external AI service.

## Architecture

The deployed hybrid-manuscript model combines:

1. A trained marketplace classifier for positive, negative, or uncertain
   review sentiment.
2. A separate trained binary classifier for review-level product risk.
3. NLTK VADER and editable Filipino/Taglish lexicons as supporting evidence.
4. Duplicate exclusion and a representative maximum of 100 reviews per product.
5. A fixed weighted-sum score: 30% negative-review ratio plus 70%
   detected-risk ratio.

Each review contributes at most one count to the detected-risk numerator.
Classifier confidence and severity do not change that binary count, and
severity is absent from product scoring. The strict VADER/lexicon implementation
remains available as the `manuscript` research baseline.

## Output fields

Each review receives:

- `label`: the final positive, uncertain, or negative label;
- `model_version` and `label_source`;
- classifier probabilities plus the original VADER score fields;
- matched sentiment terms and their lexicon values;
- a binary `risk.detected` value and any matched, non-severity evidence;
- the product-level negative and defect ratios and their weighted contributions.

Risk detection is separate from sentiment. A review can be uncertain in tone while
still reporting a concrete performance or durability problem.

The JSON workflow fingerprints normalized comments, marks duplicates, and
excludes duplicate copies from product summaries. Unicode symbols remain in the
fingerprint, so emoji-only reviews can be analyzed and deduplicated.

## Installation

Install the exact scikit-learn and joblib versions recorded for the serialized
artifact, then download VADER:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader vader_lexicon
```

Only load the committed `firecs_sentiment.joblib` artifact or another artifact
you trained yourself. Joblib model files must be treated as executable trusted
artifacts, not accepted as user uploads.

## Training protocol

Training is reproducible from the repository root:

```powershell
.\.venv\Scripts\python.exe .\train_firecs.py
```

The training audit found:

- 7,340 official training rows and 3,147 official test rows;
- no empty reviews and approximately balanced labels;
- two duplicate training rows, which were removed;
- four training rows whose normalized text also appeared in the test split,
  which were removed without consulting their test labels;
- 7,334 clean rows available for fitting.

Candidate selection used a fixed, stratified 80/20 split of only the cleaned
training data. The official test labels were not used to select the model.

| Candidate | Validation macro F1 |
| --- | ---: |
| Word TF-IDF + Linear SVM | 0.8002 |
| Character TF-IDF + Linear SVM | 0.7827 |
| Word/character TF-IDF + Linear SVM | 0.7892 |
| Word/character TF-IDF + logistic regression | **0.8040** |

The winning logistic model was retrained on all 7,334 clean training rows and
saved under `sentiment-analysis/models`.

## Auxiliary dataset experiment

Two additional sources were audited without replacing the human-gold FiReCS
evaluation split:

- SentiTaglish contains 10,510 human-annotated reviews with negative, neutral,
  positive, and mixed labels. It overlaps 10,459 normalized FiReCS texts, so it
  is an upgraded/re-annotated corpus rather than an independent addition. Mixed
  rows are excluded from the current three-class candidate.
- TagaSenti contains 35,686 rows but its downloadable CSV has no row-level
  provenance. Its card describes a mixture of human, translated, synthetic,
  and LLM-labeled sources, including FiReCS and SentiTaglish.

The auxiliary trainer uses a Unicode-normalized, punctuation-insensitive key to
remove internal duplicates and all FiReCS-test overlaps. FiReCS training labels
take precedence when sources disagree. This left 32,342 training rows:

- 7,334 FiReCS human-gold rows;
- 12 non-overlapping, non-mixed SentiTaglish rows;
- 24,996 mixed-provenance TagaSenti rows.

The resulting logistic-regression candidate reached `0.7951` macro F1 on the
frozen FiReCS test set, a decrease of `0.0186` from the FiReCS-only classifier.
It therefore failed the incumbent integration gate and is stored only as a
research candidate. The result demonstrates that additional weak or
cross-domain data can reduce in-domain performance even when it increases the
number of training examples.

## Held-out FiReCS results

| Model | Accuracy | Macro F1 | Neutral F1 |
| --- | ---: | ---: | ---: |
| Original VADER/Taglish baseline | 0.5284 | 0.4818 | 0.1889 |
| Selected trained classifier | **0.8112** | **0.8137** | **0.7469** |
| Deployed hybrid with rule safeguards | 0.7436 | 0.7370 | 0.6245 |

The hybrid intentionally gives up some benchmark performance to retain explicit
behavior already required for short negations, emoji, severe defect phrases,
and rating-context safeguards. It still improves macro F1 by 0.2552 over the
original baseline.

Generated research artifacts are stored under `sentiment-analysis/evaluation`:

- `firecs_test_baseline.json`: original VADER baseline;
- `firecs_trained_model.json`: selected classifier experiment;
- `firecs_current_model.json`: deployed hybrid benchmark;
- `multisource_sentiment_candidate.json`: rejected auxiliary-data experiment;
- corresponding CSV files containing misclassified reviews.

Run the deployed hybrid benchmark with:

```powershell
.\.venv\Scripts\python.exe .\evaluate_firecs.py
```

Run the fallback baseline explicitly with:

```powershell
.\.venv\Scripts\python.exe .\evaluate_firecs.py --vader-only
```

Do not use official test errors to tune features, rules, thresholds, or model
selection. Future development should use training/validation data or a separate
development set.

## Research limitations

FiReCS is an external human-annotated benchmark, not a complete representation
of every current Shopee, Lazada, or Temu review. Before making final in-domain
claims, the project should annotate a separate local marketplace sample, report
annotator agreement, and evaluate the frozen hybrid on that sample.

Nested five-fold product-group cross-validation was also run on all 907
previously adjudicated Lazada reviews from 24 products. No product crossed a
fold. The selected direct three-class sentiment model reached 0.6116 macro-F1
and 0.4267 F1 for the unified uncertain class. The unchanged binary risk model
reached 0.8194 risk recall and 0.7647 macro-F1. Across products, the calculated
risk score had 9.51-point mean absolute error, 0.7139 Spearman rank correlation,
and 83.33% Low/Moderate/High band agreement.

The uncertain class combines mixed, neutral, and explicitly uncertain reviews.
Its component recalls were 0.4149, 0.3377, and 0.7857 respectively, so most of
the remaining sentiment error comes from distinguishing mixed and neutral text
from clearly positive or negative text. Nested threshold tuning did not improve
the direct classifier and was rejected.

The deployed development artifact contains only the direct three-class
sentiment classifier and binary risk classifier. It contains no severity or
risk-category head. These are development estimates, not independent locked-test
results, because the reviews were previously available.

The product-risk aggregation remains:

```text
100 * (0.30 * negative_review_ratio + 0.70 * detected_risk_ratio)
```

These weights are not calibrated purchase probabilities and must be confirmed
with the research advisers before being presented as validated product risk.

Reproduce the selected artifact from the repository root with:

```powershell
.\.venv\Scripts\python.exe .\research\train_hybrid_manuscript.py
```
