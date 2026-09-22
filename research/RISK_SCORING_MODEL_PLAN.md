# Risk scoring model plan

## Objective

Estimate how much product risk is present in a representative set of customer
reviews. The production output is a continuous product risk score from 0 to 100,
not a prediction of risk categories.

The sentiment output remains separate and uses three labels:

- `positive`
- `negative`
- `uncertain` for neutral, mixed, vague, or insufficiently clear reviews

## Simplified model

For every review, the model makes two separate decisions:

1. sentiment: `positive`, `negative`, or `uncertain`; and
2. product risk: detected or not detected.

Trained marketplace classifiers make both decisions. VADER and the editable
lexicons remain supporting evidence and do not override the trained outputs.
Risk-category classification and severity are removed from production inference
and scoring. Historical category and severity annotations remain available for
research but do not affect the deployed result.

## Review collection

For one product, compile every usable written review when 100 or fewer are
available. When more than 100 are available, compile a representative maximum
of 100:

```text
compiled reviews = min(all usable written reviews available, 100)
```

A usable review is non-empty and unique after normalization. The production
scraper must not collect low-star or 3-star reviews first. For products above the
cap, select reviews proportionally from the actual 1-5-star distribution and
sample across pages or time within each star level.

Record the total displayed review count, displayed count by star, collected
count by star, usable count, exclusions, collection time, and any sampling
weight. A targeted low-star collection is permitted only for annotation or
error analysis and must never be passed to production scoring as though it were
representative.

## Product score

Compute the product score from the weighted negative-review and detected-risk
ratios:

```text
100 * (0.30 * negative-review ratio + 0.70 * detected-risk ratio)
```

Each review contributes at most one count to each ratio. If star levels were
sampled disproportionately, use the documented per-star weight:

```text
displayed reviews in star level / sampled reviews in star level
```

Return these values to the app:

- risk score from 0 to 100;
- negative and detected-risk contribution points;
- reviews analyzed and reviews available;
- evidence status when fewer than 30 usable reviews are available or the star
  distribution is unknown or biased;
- supporting lexicon evidence and review-level classifier decisions.

## Training and validation

Retain all reviewed products for model development; the 100-review cap applies
only to one production analysis request. Train and tune with product-group
cross-validation so complete products remain together in each fold.

Validate the components and the final score:

- sentiment: macro-F1 and per-class precision, recall, and F1;
- binary risk: macro-F1, risk recall, and risk precision;
- product score: mean and root-mean-square score error, rank correlation, and
  agreement across Low, Moderate, and High score bands;
- sampling: overall results and results by product stratum, star distribution,
  review count, and distance from score-band boundaries.

Use the risk-detection gates of at least 0.80 recall and 0.75 macro-F1. Use
sentiment gates of at least 0.75 macro-F1 and 0.60 uncertain-class F1.
Cross-validation is a development estimate because these reviews were already
available during model development; it must not be described as an independent
locked test.

## Implementation order

1. Change the scraper to collect all usable reviews up to 100 and use
   star-proportional sampling above 100.
2. Remove risk-category prediction from the production inference and scoring
   path while preserving historical annotations.
3. Use trained sentiment and binary-risk heads with VADER/lexicon explanations.
4. Implement the manuscript's weighted 30/70 aggregation and coverage warnings.
5. Evaluate with nested product-group cross-validation.
6. Improve uncertain sentiment using training-fold data only, then repeat the
   unchanged cross-validation protocol.

## Implementation status (2026-09-16)

- Implemented the 100-review application cap and deterministic proportional
  selection by observed star level for imported review collections.
- Changed Lazada's default production collection from targeted 3-star-first
  filtering to the marketplace's unfiltered review order. Explicit rating-filter
  modes remain available only for research collection.
- Removed learned risk-category and severity output from production scoring.
- Deployed a clean artifact containing only a direct three-class sentiment head
  and a binary-risk head with the manuscript's fixed 30/70 aggregation. The
  artifact contains no severity or risk-category classifier.
- Completed nested five-fold product-group cross-validation on 907 previously
  adjudicated reviews from 24 products. No product crossed a fold.
- Pooled risk recall was 0.8194 and risk macro-F1 was 0.7647, passing both
  development risk gates.
- Replacing binary sentiment plus abstention with direct three-class sentiment
  improved sentiment macro-F1 from 0.5825 to 0.6116 and uncertain-class F1 from
  0.4007 to 0.4267. A nested uncertainty-threshold variant performed slightly
  worse and was rejected. The sentiment gates still did not pass.
- The remaining `uncertain` error is concentrated in its composite labels:
  recall was 0.4149 for mixed reviews, 0.3377 for neutral reviews, and 0.7857
  for reviews originally annotated uncertain.
- With the selected direct three-class candidate, product scores had 9.51-point
  mean absolute error, 0.7139 Spearman rank correlation, and 83.33%
  Low/Moderate/High band agreement across 24 products.
- Results are stored in
  `sentiment-analysis/evaluation/hybrid_manuscript_three_class_grouped_cross_validation.json`.
- The fitted development artifact is
  `sentiment-analysis/models/hybrid_manuscript_v1.joblib`; its separate training
  record is `sentiment-analysis/evaluation/hybrid_manuscript_training.json`.
- This is suitable for continued risk-scoring development, not a final accuracy
  claim. A newly collected, frozen product-disjoint test set remains required.
