# Manual Review Annotation Guide

Use `manual_marketplace_reviews.csv` to label each review independently. Do not
use the `model_prediction` column while making the first label. It is reserved
for later error analysis.

## Sentiment label

Label the overall experience described by the reviewer:

- `positive`: The reviewer is satisfied and describes the product or service as
  good, useful, working, or worth buying. Minor complaints are acceptable when
  the overall conclusion is clearly satisfied.
- `negative`: The reviewer is dissatisfied, reports a meaningful failure, or
  would not recommend the product or seller. A serious defect, wrong item,
  missing item, unusable product, refund dispute, or strong complaint is usually
  negative even when the review begins with praise.
- `neutral`: The review is factual, uncertain, incomplete, or contains a minor
  issue without a clear positive or negative overall judgment.
- `mixed`: Use this only when the review clearly gives substantial positive and
  negative judgments that cannot reasonably be reduced to one overall label.

Examples:

| Review | Label |
| --- | --- |
| `Maganda at gumagana nang maayos.` | positive |
| `Sira pagdating at hindi gumagana.` | negative |
| `Dumating ang order noong Lunes.` | neutral |
| `Maganda ang packaging pero hindi gumagana ang item.` | mixed |

If the worksheet must be used for the current three-class model, keep the
`mixed` label in `sentiment_label` and resolve the three-class conversion later
with an adjudicator. Do not silently convert mixed to neutral during first-pass
annotation.

## Product-risk fields

- `risk_detected`: `yes` when the review reports a possible product, order, or
  seller problem; otherwise `no`.
- `risk_category`: choose the most specific category: `defect`, `performance`,
  `quality`, `durability`, `delivery`, `wrong_item`, `wrong_color_or_size`,
  `missing_item`, `damage`, `seller_service`, or `other`.
- `risk_severity`: `1` minor inconvenience, `2` meaningful problem, `3` severe
  failure, safety concern, unusable item, or strong fraud/counterfeit claim.
- `negated`: `yes` only when the problem is explicitly denied, such as `walang
  sira` or `not fake`. A positive statement that happens to contain a risk word
  is not automatically a risk.

## Annotation procedure

1. Read the entire review, including the final clause after `pero`, `kaso`, or
   `but`.
2. Label sentiment from the reviewer’s overall judgment, not from one keyword.
3. Record product-risk evidence separately from sentiment.
4. Preserve Filipino, English, Taglish, spelling mistakes, and emojis as written.
5. Leave `adjudicated_label` blank during independent annotation.
6. Use `annotation_notes` for ambiguity, sarcasm, unclear wording, or reasons
   for choosing `mixed`.

For reliable results, use two annotators for every row. An adjudicator should
resolve disagreements without looking at the model prediction first. Report
agreement and the final adjudicated labels before using this data for training.