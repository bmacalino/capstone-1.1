# Pre-registration: gated learned defect check (v5)

Written 3 Oct 2026, after labelling round 3 was built
(`outputs/labelling_queue_round_3/`) and **before** the model below was
trained. Nothing here may change after a score is seen.

## Why

The learned defect model of checker 0.4 (Table 11d of
`research/RESULTS_TABLES.md`) passed its pre-registered test on round 2
(F1 0.646 vs 0.578 for keywords) but was not shipped: on unlabelled reviews
about a third of the flags it added were sentences saying the product works
("gumagana po cya", "fast charging"). This round tests one fix, chosen in
advance: a learned model may only add a defect flag to a sentence that is not
praise.

## The model (one design, no search)

- **Learned part:** TK — TF-IDF word 1–2 and character 3–5 grams plus the six
  keyword features, logistic regression (`C=1`, class-balanced), exactly as in
  `sentiment-analysis/defaketive_sentiment/hazard_learned.py`.
- **Training data:** rounds 1 and 2. Round 1: annotators A1 and A2 (A3's sheet
  is missing); round 2: all three. A sentence two annotators labelled is used
  when they agree and dropped when they disagree.
- **Decision for a sentence:** defect = keyword checker 0.3.0 says defect,
  **or** (the learned model's probability ≥ threshold **and** the app's
  sentiment model does not read the sentence as positive). The learned part
  can add flags but never removes a keyword flag. "Sentiment" is the shipped
  XLM-R model's label for the sentence alone, with no star rating.
- **Threshold:** from 5-fold cross-validation on rounds 1 and 2, folds grouped
  by review and stratified by the defect label, seed 42: the threshold at which
  the out-of-fold F1 of the full decision above is highest (ties: higher
  threshold). The same model without the gate is reported for comparison only.

## Test (round 3, once) and adoption

Ground truth: round-3 sentences on which the two annotators agree.

**Adopt the gated defect model if, on round 3, all three hold:**
1. its F1 is higher than keyword checker 0.3.0's;
2. its recall is not lower than keyword checker 0.3.0's;
3. its precision weighted by sampling weight is at least keyword checker
   0.3.0's weighted precision minus 0.05.

Condition 3 exists because round 3 over-samples (with recorded weights)
reviews where the old learned model and the keywords disagreed; weighted
precision estimates how often a flag is right in reviews as they occur.

Reported whatever the decision: McNemar's test against the keywords, F1
intervals, weighted recall, precision on the disagreement stratum alone, and a
read of 25 randomly chosen flags the model adds on unlabelled reviews outside
rounds 1–3 (seed 11), so the "it works" failure can be checked by eye.

Safety (checker 0.4) is also scored on round 3 as a second held-out check; it
is not changed by this round. Counterfeit is not part of this round.

## Disclosed limits

- Round 1 lacks annotator A3.
- The gate was conceived after seeing the round-2 results and the live check;
  it is tested here for the first time.
- Round 3's extra stratum was defined with the round-1 learned defect model
  (`research/models/defect_tk_round1.joblib`), not the model tested here.
