# Marketplace v2 candidate model card

> Historical experiment. The category-head candidates and unused binary
> artifacts were removed after the locked test rejected that direction. The
> current simplified design is documented in `RISK_SCORING_MODEL_PLAN.md`.

## Status

`defaketive-marketplace-v2-0.1.0-candidate` is a research candidate. It is not
connected to the Flask application and must not replace the deployed model yet.
The artifact records `qualified_for_integration: false`, and the inference
wrapper requires an explicit `allow_unqualified=True` opt-in.

## Tasks and data

The candidate separately predicts three-class sentiment, binary product/order
risk, multilabel risk category, and risk severity (1–3). Sentiment training
combines FiReCS with only the marketplace training products. Risk tasks use the
manually annotated marketplace training products. Mixed sentiment is excluded
from sentiment training but retained for risk tasks. Products never cross the
train, validation, and test boundaries.

The marketplace sheet has 498 rows from 15 products. All 498 rows have a blank
`annotator_id`. Labels are highly imbalanced, and neutral sentiment appears in
only four products. These limitations prevent a production claim.

## Held-out product results

The split is selected using only group size and label balance, never model
outcomes. On the 72-row held-out product set (67 sentiment-compatible rows):

| Task | Candidate | Current model | Required gate |
| --- | ---: | ---: | ---: |
| Sentiment macro F1 | 0.6619 | 0.5490 | 0.75 |
| Neutral F1 | 0.2857 | 0.1667 | 0.60 |
| Risk macro F1 | 0.8049 | 0.4162 | 0.75 |
| Risk recall | 1.0000 | 0.2759 | 0.80 |
| Risk-category macro F1 | 0.2659 | — | 0.65 |
| Severity weighted kappa | 0.2603 | — | 0.65 |

The validation-selected category threshold was 0.15. Risk detection passes its
gates; sentiment, risk categorization, and severity do not.

## Reproduction

```powershell
.\.venv\Scripts\python.exe train_marketplace_v2.py
$env:PYTHONPATH = "sentiment-analysis"
.\.venv\Scripts\python.exe -c "from defaketive_sentiment.marketplace_v2 import MarketplaceV2Model; print(MarketplaceV2Model(allow_unqualified=True).analyze('Hindi gumagana ang charger.'))"
```

The report is `sentiment-analysis/evaluation/marketplace_v2_candidate.json` and
the artifact is `sentiment-analysis/models/marketplace_v2_candidate.joblib`.

## Required before deployment

1. Fill `annotator_id`, independently double-label a representative subset,
   resolve disagreements, and report agreement.
2. Add neutral, no-risk, minority-category, and severity-1/2 examples across
   more products—not more repeated examples from a few listings.
3. Freeze a larger product-held-out test set before the next modeling round.
4. Compare this candidate, TagaSenti, and a fine-tuned Filipino/multilingual
   transformer on exactly that frozen test set.
5. Integrate only after every recorded gate passes.
