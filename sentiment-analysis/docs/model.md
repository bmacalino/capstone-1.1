# DeFaketive sentiment model

DeFaketive uses a local hybrid sentiment model for English, Filipino, and
Taglish Philippine e-commerce reviews. It does not call an external AI service.

## Architecture

The deployed `defaketive-firecs-tfidf-0.2.0` model combines:

1. A supervised logistic-regression classifier trained on human-labeled FiReCS
   reviews using word and character TF-IDF features.
2. NLTK VADER plus editable Filipino/Taglish lexicons for sentiment evidence.
3. Narrow explainable overrides for explicit failures, strong negation,
   emoji-only reviews, and severe product-risk phrases.
4. An independent lexicon layer for defect, durability, counterfeit, safety,
   completeness, and other product-risk evidence.
5. A guarded one- or two-star rating tie-breaker for ambiguous marketplace text.

The trained classifier supplies the default sentiment label for ordinary review
text. The rules remain authoritative only in the narrow cases above. If the
trained artifact is missing or incompatible, analysis falls back to the VADER
and Taglish rule model instead of preventing the app from starting.

## Output fields

Each review receives:

- `label`: the final positive, neutral, or negative label;
- `model_version` and `label_source`;
- `classifier`: probabilities, confidence, and strongest active linear features;
- `vader_label` and the original VADER score fields;
- matched sentiment terms and their lexicon values;
- `hybrid_adjustment` when an explicit rule overrides the classifier;
- `label_adjustment` when guarded star-rating context changes the label;
- product-risk categories, severity, and matched evidence.

Risk detection is separate from sentiment. A review can be neutral in tone while
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

The preliminary product-risk score remains:

```text
100 * (0.30 * negative_review_ratio + 0.70 * keyword_failure_rate)
```

These weights are not calibrated purchase probabilities and must be confirmed
with the research advisers before being presented as validated product risk.
