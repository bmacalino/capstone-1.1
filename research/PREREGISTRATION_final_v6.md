# Pre-registration: final retraining (v6) and labelling round 5

Written 6 October 2026, before the v6 candidates were trained and before
round 5 was drawn. Nothing below may change once round 5 has been read.

## Why

Exploratory comparisons on rounds 1-3 (RESULTS_TABLES.md, Table 11g) showed
that other model types do not beat the shipped design, that the defect model
improves as labelled rounds are added, and that the counterfeit and safety
checks are limited by how few agreed reports exist. Rounds 2 and 3 were used
for those comparisons, so they are training data from here on. Round 5 is the
only test.

## Candidates (trained on rounds 1-3, never on round 5)

Training labels: a sentence is used for a category when every annotator who
labelled it gave the same answer (as in v4 and v5). Features are the keyword
checker 0.3.0 result taken before the counterfeit rules c1, as in the app.
Model: TK (TF-IDF word 1-2 + character 3-5 + six keyword features, logistic
regression, C = 1, balanced), unchanged from v4/v5. No hyper-parameter is tuned.

- **Defect.** Same gated design as v5: keywords OR (model >= threshold AND the
  sentence is not read as positive). Threshold = the one with the best F1 on
  out-of-fold predictions, 5 folds grouped by review, seed 42; ties go to the
  higher threshold.
- **Safety.** The model is a filter on keyword flags: keywords AND (model >=
  threshold). It can remove a flag and never adds one. Threshold = best
  out-of-fold F1 of that combination; ties go to the LOWER threshold, because
  a missed hazard costs more than a false flag. The v4 rule (highest
  threshold with recall >= 0.80) is retired: on pooled rounds it selects a
  threshold near zero.
- **Counterfeit.** No candidate. A learned model added no real report in
  cross-validation (Table 11g). Keywords 0.3.0 + rules c1 are scored as they
  ship.

Frozen in `sentiment-analysis/models/hazard_classifier_v3/` by
`research/train_final_v6.py`, with SHA-256 of every file in its manifest.

## Round 5

`research/build_labelling_queue.py --round 5 --shopee-hf research/data/shopee_hf`

- Source: only `scaredmeow/shopee-reviews-tl-stars` (Hugging Face, revision
  d096f402, 15,000 reviews, MPL-2.0). No review from that set was used in any
  earlier round or unlabelled check. Reviews whose text appeared in rounds
  1-4 are excluded.
- About 600 sentences, all blind (no computer guess shown), seed 20261006.
  Strata, in order of assignment: `counterfeit_focus` 280 (a sentence either
  counterfeit check flags, or an authenticity word), `model_disagreement` 110
  (shipped and candidate differ on safety or defect), `seed_hit` 120,
  `low_star` 45, `random_all` 45.
- Two annotators per sentence under guide 1.2; the third settles
  disagreements without seeing any computer output.
- Limits known in advance: Shopee, not Lazada; no product identity, so no
  product-level figure; the disagreement stratum favours whichever model is
  right where they differ, so weighted figures are the estimate for ordinary
  reviews.

## Test and decision (once)

`research/measure_hazard_tagger.py --round 5` scores `v6_shipped`
(hazard_classifier_v2 + c1, the app today) and `v6_candidate`
(hazard_classifier_v3 + c1) on the same sentences.

Primary truth: consensus of the two annotators (comparable with Tables 11c
and 11e). Also reported: truth with the third annotator's decisions added,
and review-level scores (a review counts when any of its sentences does).

- **Defect: adopt** if F1 is higher AND recall is not lower AND weighted
  precision >= shipped weighted precision - 0.05.
- **Safety: adopt** if F1 is higher AND recall is not lower.
- **Counterfeit:** no decision; precision, recall, F1 and interval reported.
  If round 5 holds fewer than 30 agreed counterfeit reports, that is stated
  and no accuracy claim is made.

Each category is decided separately. A category that is not adopted keeps its
shipped model. McNemar, bootstrap F1 intervals and weighted figures are
reported for every category whatever the outcome.

## Amendment 1 (6 October 2026, before the queue was drawn)

The budgets first written here were 200 / 120 / 130 / 70 / 80. A dry run,
which prints counts only and no sentence, showed that 200 sentences are
filled by 72 counterfeit-focus reviews, most of whose sentences are not
about authenticity, so round 5 would probably hold fewer than 30 agreed
counterfeit reports. The budgets were changed to those above (same total)
before any round-5 sentence was drawn or read. Nothing else changed.

## Departures

Any departure from this document is listed in RESULTS_TABLES.md with the
reason.

### Departure 1 (6 October 2026, before any round-5 label existed)

The project owner chose to have the AI assistant label round 5 instead of
two human annotators per sentence. The assistant labelled all 600 sentences
without the stratum or any checker output (`research/label_round5_ai.py`),
marking each answer sure or unsure; the owner decides every unsure sentence
and blind-audits 50 sure ones. Consequences, fixed now: round 5 is an
AI-assisted check, not ground truth; its figures are reported as agreement
with AI-proposed, owner-audited labels and never as accuracy; and the
adoption rules above are NOT applied to it, so no candidate is adopted on
this evidence alone. The assistant also wrote the counterfeit rules and
trained the candidates, which is a further reason the result cannot count
as an independent test. The human worksheets stay in the folder; if the
team labels them later, the original protocol applies to those labels.

### Departure 2 (7–8 October 2026)

The owner asked that poorly performing parts of the checker be fixed,
improved or replaced without waiting for the audit of the AI labels. What the
app serves now was therefore NOT decided by the test in this document:

1. Keyword rules 0.4 and the refit `hazard_classifier_v2b`. Checked on the
   team's round-3 labels (no report lost, one false flag fewer) with a rule
   written the same day, and round 3 had already been v2's test.
2. `hazard_classifier_v5`, the app default. Its defect model is trained on
   rounds 1–3 as re-read by the assistant, on rounds 5–7 and on round 8's
   mining listings, all AI-labelled, and it needs a higher score to add a
   flag in a 4–5 star review. Round 5, this document's single test set, is in
   its training data, so the test above cannot be run on round 5 for it.
3. The candidates of this document (`hazard_classifier_v3`) were not adopted
   and are not served.

Consequences, fixed now: no figure for v5 is an accuracy; v5 is described as
the deployed development model and the team-label models (v2b) as the ones
checked on human labels; `DEFAKETIVE_HAZARD_MODEL=team_labels` serves those.
If the team labels a fresh round with two annotators per sentence, that
round, not round 5, is the test for v5. Evidence and limits:
RESULTS_TABLES.md, Table 11j.

### Departure 3 (8 October 2026)

The owner chose to proceed with the AI-proposed labels without auditing
them: rounds 5–8 (`ai_labels.csv`), the two new labels for rounds 1–4
(`ai_extra_labels.csv`) and the re-read defect and counterfeit answers of
rounds 1–3 (`ai_relabel.csv`). The review sheets (`worksheet_for_you.csv`)
stay unfilled, and the answers the assistant marked unsure stand as proposed.
Departure 1's "the owner decides every unsure sentence and blind-audits 50
sure ones" therefore did not happen.

Consequences, fixed now: these labels have one labeller and no second
opinion, so no agreement statistic exists for them; they are never called
ground truth and no figure computed on them is an accuracy; the team's
rounds 1–3 remain the only human-labelled evidence. The same day the app
began serving rules c2 (the "poor quality" and "not as advertised" labels),
after a check on the team's rounds 2–3 that it changes no safety or defect
answer and loses no agreed fake report (RESULTS_TABLES.md, Table 11i). A
fresh round labelled by two team members is still the only real test of
`hazard_classifier_v5` and of those rules.
