# Pre-registration: third sentiment training run (`xlmr_firecs_lazada_norm`)

Written 27 Sept 2026, **before** training started. Nothing below may be changed
after the run's test scores are seen.

## Why this run exists

Two runs on 26 Sept, scored on the 470 human-labelled marketplace reviews
(`research/marketplace_shift_retrained.json`):

| Model | Accuracy | Neutral F1 | Macro F1 |
|---|---|---|---|
| First XLM-R (normalized training text, FiReCS only) | 0.864 | 0.299 | **0.719** |
| FiReCS only, raw text | 0.813 | 0.192 | 0.671 |
| FiReCS + Lazada star labels, raw text | **0.932** | 0.000 | 0.620 |

Star labels raise accuracy but contain no neutral class, and the model stopped
predicting neutral on marketplace text. Normalized training text did better
than raw text. This run combines the two and down-weights the star labels so
FiReCS's human neutral labels are not outvoted.

## Recipe (fixed)

- Starting checkpoint `cardiffnlp/twitter-xlm-roberta-base-sentiment`, seed 42,
  3 epochs, batch 8 x accumulation 2, lr 2e-5, max length 128, frozen
  embeddings: unchanged from the earlier runs.
- `--text normalized`: `train_firecs.normalized_text()` for FiReCS and star rows.
- `--extra-train lazada_star_labels_v1.csv --extra-weight 0.5`: each star row
  counts half a FiReCS row in the loss.
- Epoch chosen on the validation slice (10% FiReCS + held-out whole Lazada
  products), as before.
- The app feeds it the same normalized text (tested:
  `tests/test_finetune_transformer.py::TestServedTextMatchesTraining`).

No Gemini or other machine-generated labels are used.

## Decision rule (fixed)

Scored on the marketplace set with normalized input, as served:

- **Ship this model** if its marketplace macro F1 is **above 0.719** *and* its
  marketplace accuracy is **at least 0.90**.
- **Otherwise ship the first XLM-R** (`cardiffnlp__twitter-xlm-roberta-base-sentiment`,
  0.719 / 0.864), slimmed the same way.

Either choice then goes through the unchanged slimming acceptance rule and
integration gate (`research/slim_sentiment_model.py`).

This is one run, not a search: no other weights, seeds or recipes will be tried
and compared on the test set.

## Disclosed limitations

- The marketplace test set has been used to choose between models; its scores
  for the shipped model are therefore not an untouched estimate.
- Only 18 marketplace reviews are neutral, so neutral F1, and through it macro
  F1, moves by several points on a handful of reviews.
