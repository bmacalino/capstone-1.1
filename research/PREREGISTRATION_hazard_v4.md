# Pre-registration: hazard checker 0.4 (learned models)

Written 3 Oct 2026, **before** any candidate below was trained or scored.
Nothing here may change after a score is seen.

## Why

Held-out round 2 (`research/RESULTS_TABLES.md`, Table 11c) put the keyword
checker 0.3.0 at defect F1 0.578 with weighted recall 0.28, safety 10 of 16
reports caught with 15 of 25 flags false, and counterfeit F1 0.56 on 12
reports. A word list cannot cover the ways people describe a failure. This
round tests whether a model that learns from labelled sentences does better,
one category at a time.

## Data

- **Training and selection: labelling round 1 only**
  (`outputs/labelling_queue_round_1/`). Annotator A3's round-1 sheet is
  missing (overwritten on 2 Oct), so each sentence carries the labels of A1
  and/or A2: a sentence both labelled is used when they agree and dropped when
  they disagree; a sentence one of them labelled is used with that label.
  If the A3 sheet is found before training starts it is added; after, it is not.
- **Test: labelling round 2**, consensus of the two annotators per sentence
  (`research/measure_hazard_tagger.py --round 2`). Round 2 is used once more,
  only for the adopt/keep decision below. No round-2 text, label or error is
  looked at while building or choosing a model.
- Only human labels. No machine-generated labels anywhere.

## Candidates, per category (safety, defect, counterfeit)

| Code | Model |
|---|---|
| K | Keyword checker 0.3.0, as shipped (the baseline) |
| T | TF-IDF (word 1–2 grams + character 3–5 grams, `min_df=2`, sublinear tf) on the hazard normalizer's text, logistic regression (`C=1`, class-balanced) |
| TK | T plus keyword features (per-category count of counted keyword matches and the three keyword tags) |
| E | Sentence embedding (mean of the last hidden layer of the shipped XLM-R sentiment model, computed with the served ONNX file) → standardised → logistic regression (`C=1`, class-balanced) |
| EK | E plus the keyword features of TK |

Hyper-parameters are fixed as above; none is tuned.

## Selection (round 1 only)

- 5-fold cross-validation, folds grouped by review (no review in both a
  training and a validation fold), stratified by the category label, seed 42.
- Each candidate's out-of-fold probabilities set its decision threshold:
  - defect and counterfeit: the threshold that maximises out-of-fold F1;
  - safety: the highest threshold whose out-of-fold recall is at least 0.80
    (the project's safety target); if none reaches 0.80, the threshold with
    the highest recall, ties broken by F1.
- The **learned** candidate (T, TK, E or EK) with the highest out-of-fold F1
  at its threshold is chosen. Candidates within 0.01 F1 of the best are
  treated as tied and the simplest wins, in the order T, TK, E, EK.
- K is reported for reference but is not a selection candidate on round 1:
  its terms and rules were tuned on round 1, so its round-1 figures are
  in-sample and would be compared unfairly with out-of-fold ones. K competes
  only on round 2 (adoption rule below). For the same reason TK's and EK's
  keyword features carry some round-1 tuning; disclosed, and settled by the
  round-2 test.
  *(Amended 3 Oct 2026, before any candidate was trained or scored: the first
  draft let K compete on round 1.)*
- The chosen model is then trained on all of round 1 with that threshold and
  frozen (files and SHA-256 recorded) before round 2 is scored.

## Adoption (round 2, once)

For each category, the chosen learned model is compared with K on round 2:
**adopt the learned model only if its round-2 F1 is higher than keyword
checker 0.3.0's and its recall is not lower.** Otherwise that category keeps
its keyword list. McNemar's test, F1 intervals and weighted recall are
reported for every category whatever the decision.

What ships is exactly what was tested (trained on round 1 only). Retraining on
rounds 1 and 2 together would need a new held-out round to be reported.

## Disclosed limits

- Round 1 lacks annotator A3, so training data is smaller and partly
  single-annotator.
- Safety has about 30 positive training sentences and round 2 only 16; its
  result will be imprecise whichever way it goes.
- Round 2 has now been looked at once (for 0.3.0). It has not been used to
  build anything, but this is its second use as a test.
