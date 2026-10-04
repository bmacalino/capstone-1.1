# Sentiment model comparison — results tables

All figures measured on the frozen FiReCS test split (n = 3,147; negative 1,027,
neutral 1,087, positive 1,033). Training used the cleaned FiReCS training split
(7,334 rows after removing 2 duplicates and 4 rows overlapping the test split).
Seed 42 throughout. Test labels were read only after model selection.

Sources: `research/sentiment_model_comparison.json`,
`research/finetuned_vs_incumbent.json`, `research/sentiment_tuning.json`,
`research/cardiff_xlmr_firecs.json`, `research/taglish_roberta_firecs.json`.
Protocol: `research/MODEL_COMPARISON.md`.

---

## Table 1. Overall comparison of modelling techniques

Ranked by macro F1. Significance is McNemar's paired test against the deployed
model, Holm-corrected across the six same-data comparisons. Latency is the
median for a single review on CPU, which is how the application calls the model.

| # | Model | Accuracy | Macro F1 | 95% CI | p (adj.) | Sig. | Latency | Size |
|---|---|---|---|---|---|---|---|---|
| 1 | Fine-tuned XLM-RoBERTa (multilingual) | 0.8449 | **0.8461** | [0.8334, 0.8590] | 0.000001 | yes | 93.4 ms | 1,112 MB |
| 2 | **TF-IDF + LogisticRegression (deployed)** | 0.8112 | 0.8138 | [0.8002, 0.8274] | — | ref. | **3.0 ms** | **2.5 MB** |
| 3 | TF-IDF + SVD(300) + LogisticRegression | 0.8036 | 0.8056 | [0.7919, 0.8187] | 0.371 | no | 43.3 ms | 68.7 MB |
| 4 | Fine-tuned DOST-ASTI RoBERTa-tl (Tagalog) | 0.7998 | 0.8025 | [0.7886, 0.8158] | 0.117 | no | ~90 ms | 439 MB |
| 5 | TF-IDF + LinearSVC | 0.7982 | 0.8004 | [0.7870, 0.8147] | 0.0006 | yes | 3.0 ms | 3.1 MB |
| 6 | TF-IDF + ComplementNB | 0.7881 | 0.7871 | [0.7723, 0.8016] | 0.0013 | yes | 4.1 ms | 4.9 MB |
| 7 | TF-IDF + SVD(300) + HistGradientBoosting | 0.7757 | 0.7784 | [0.7643, 0.7925] | <0.0001 | yes | 83.7 ms | 72.6 MB |
| 8 | Taglish RoBERTa, off the shelf | 0.6635 | 0.6465 | — | — | — | — | 419 MB |
| 9 | XLM-RoBERTa, off the shelf | 0.6822 | 0.6389 | — | — | — | — | 1,112 MB |
| 10 | VADER lexicon (no learning) | 0.5284 | 0.4818 | [0.4662, 0.4972] | <0.0001 | yes | 2.1 ms | — |

"Sig." indicates whether the difference from the deployed model survives
correction for multiple comparisons at α = 0.05. Rows 7–8 were evaluated
zero-shot in an earlier experiment and are shown for reference; they were not
part of the corrected comparison family.

**Reproduction control.** The shipped `.joblib` artifact was scored as its own
arm and returned accuracy 0.8112 and macro F1 0.8138, with **zero discordant
predictions** against the retrained configuration (McNemar p = 1.0). The
harness therefore feeds models the same text the training script did.

---

## Table 2. Per-class F1 — neutral is the weakest class in every model

| Model | Negative | **Neutral** | Positive |
|---|---|---|---|
| Fine-tuned XLM-RoBERTa | 0.8638 | **0.7788** | 0.8956 |
| TF-IDF + LogisticRegression (deployed) | 0.8132 | **0.7469** | 0.8815 |
| TF-IDF + SVD + LogisticRegression | 0.8107 | **0.7341** | 0.8721 |
| TF-IDF + LinearSVC | 0.8026 | **0.7295** | 0.8692 |
| Fine-tuned DOST-ASTI RoBERTa-tl | 0.8142 | **0.7270** | 0.8663 |
| TF-IDF + SVD + HistGradientBoosting | 0.7794 | **0.7057** | 0.8500 |
| TF-IDF + ComplementNB | 0.7975 | **0.6971** | 0.8667 |
| Taglish RoBERTa, off the shelf | 0.7259 | **0.4637** | 0.7500 |
| XLM-RoBERTa, off the shelf | 0.7561 | **0.3570** | 0.8037 |
| VADER lexicon | 0.6292 | **0.1889** | 0.6274 |

Neutral is the lowest-scoring class for all ten models without exception. This
identifies the neutral boundary as a property of the task rather than a
weakness of any one method.

---

## Table 2b. Multilingual versus Tagalog-specific pretraining

Both transformers were fine-tuned identically: same cleaned training rows, same
3 epochs, same hyperparameters, same frozen-embedding strategy, epoch selected
on the same validation slice.

| Model | Pretraining | Params trained | Macro F1 | vs deployed | Size |
|---|---|---|---|---|---|
| XLM-RoBERTa | 100 languages | 86.0 M | **0.8461** | +0.0323, p = 0.000001 | 1,112 MB |
| DOST-ASTI RoBERTa-tl | Tagalog | 86.0 M | 0.8025 | −0.0113, p = 0.117 | 439 MB |

**The multilingual model beat the Tagalog-specific one decisively** — McNemar
p < 0.000001 on 478 discordant rows split 310 to 168 in XLM-R's favour.

**The Tagalog-specific model did not beat TF-IDF.** At p = 0.117 its difference
from the deployed linear model is not statistically distinguishable from
chance, despite 68 minutes of GPU fine-tuning against 6 seconds of CPU fitting.

This contradicts the intuition that a language-specific encoder should suit
Filipino reviews better. Two plausible explanations, neither tested here: XLM-R
is pretrained on far more text, so its general representations may transfer
better than a smaller monolingual model's; and the reviews are Taglish, a
code-switched register that a multilingual encoder may represent more naturally
than a Tagalog-focused one. Establishing the cause would require a controlled
pretraining comparison and is outside this study's scope.

---

## Table 3. Error structure of the deployed model

Confusion matrix, rows = true label, columns = predicted.

| True \ Predicted | Negative | Neutral | Positive | Total |
|---|---|---|---|---|
| Negative | 827 | 192 | 8 | 1,027 |
| Neutral | 164 | 841 | 82 | 1,087 |
| Positive | 16 | 132 | 885 | 1,033 |

| Error type | Count | Share of all errors |
|---|---|---|
| Negative ↔ Neutral confusion | 356 | 59.9% |
| Neutral ↔ Positive confusion | 214 | 36.0% |
| **Neutral boundary, combined** | **570** | **95.9%** |
| Negative ↔ Positive confusion | 24 | 4.0% |
| **Total errors** | **594** | 100% |

The model resolves polarity reliably — it confuses negative with positive in
fewer than 1% of the 3,147 test reviews. Almost all error is boundary
placement around neutral.

---

## Table 4. Off the shelf versus fine-tuned, same checkpoint

`cardiffnlp/twitter-xlm-roberta-base-sentiment`, identical weights, the only
difference being 3 epochs of training on the cleaned FiReCS training split.

| Condition | Accuracy | Macro F1 | Neutral F1 |
|---|---|---|---|
| Off the shelf (zero-shot) | 0.6822 | 0.6389 | 0.3570 |
| Fine-tuned on FiReCS | 0.8449 | 0.8461 | 0.7788 |
| **Gain from fine-tuning** | **+0.1627** | **+0.2072** | **+0.4218** |

Validation macro F1 by epoch: 0.8262, 0.8253, 0.8361. Epoch 3 selected on the
validation slice; total training time 5,443 s on a GTX 1650.

Nearly all of the apparent weakness of transformers in the earlier comparison
was the absence of task-specific training, not a limitation of the
architecture on Taglish.

---

## Table 5. Deployment cost — why the deployed model is retained

| Criterion | TF-IDF + LogisticRegression | Fine-tuned XLM-RoBERTa | Ratio |
|---|---|---|---|
| Macro F1 | 0.8138 | 0.8461 | +0.0323 |
| Latency per review (CPU) | 3.0 ms | 93.4 ms | 31× slower |
| Latency per review (GPU) | n/a | 23.9 ms | — |
| Artifact size | 2.5 MB | 1,112 MB | 445× larger |
| 100-review product page (CPU) | **0.3 s** | **9.3 s** | 31× slower |
| Per-word explanation of a prediction | yes | no | — |
| Requires a GPU in practice | no | yes | — |

The transformer is more accurate and the difference is statistically real
(p = 0.000001, non-overlapping confidence intervals). It is not adopted because
a 0.032 macro F1 gain does not justify a 31× latency increase, a 445× size
increase, a GPU the deployment target does not have, and the loss of the
per-word evidence the interface displays.

**Superseded.** Table 8b showed the TF-IDF model does not transfer to
marketplace reviews, and Table 8d removed the size and GPU objections (95 MB,
CPU only). The TF-IDF model was replaced by the fine-tuned XLM-RoBERTa in
September 2026; it remains the documented research baseline.

---

## Table 6. Attempts to improve the deployed model — all negative

| Intervention | Train rows | Accuracy | Macro F1 | vs deployed | Verdict |
|---|---|---|---|---|---|
| Deployed configuration | 7,334 | 0.8112 | 0.8138 | — | reference |
| 40-candidate hyperparameter search | 7,334 | 0.8144 | 0.8169 | +0.0031, p = 0.399 | not significant |
| + SentiTaglish corpus | 7,346 | 0.8106 | 0.8133 | −0.0005 | no effect |
| + SentiTaglish + TagaSenti | 32,342 | 0.7935 | 0.7949 | −0.0189 | harmful |

**SentiTaglish contributed 12 usable rows.** Of the 7,113 rows it offered,
**2,138 were present in the FiReCS test split** and 4,969 more were duplicates
of existing training rows. The cross-split leakage check in
`train_firecs.audit_training_records` removed them automatically; without it,
2,138 test reviews would have entered training and produced an invalid score.

**TagaSenti contributed 24,996 rows and degraded the model.** Its loader already
records it as mixed-provenance. Corpus size does not compensate for
distribution mismatch.

---

## Table 7. Removing the neutral class — measured and rejected

| Scheme | Classes | Test rows scored | Accuracy | Macro F1 |
|---|---|---|---|---|
| Three-class (retained) | neg / neu / pos | 3,147 | 0.8112 | 0.8138 |
| Binary (rejected) | neg / pos | 2,060 | 0.9636 | 0.9636 |

The binary figure is not comparable: it is measured on 2,060 rows after
discarding the 1,087 neutral reviews rather than classifying them. Forced
through the binary model, those 1,087 reviews are assigned **693 negative and
394 positive**.

Sentiment no longer feeds any hazard badge
(`hazard_parameters_v1.json`: `sentiment/used_for_risk_decisions: false`) but
still drives the review counts shown to users and the ranking of recommended
alternatives by positive ratio. Both would be corrupted by forcing lukewarm
reviews into a polarity. The three-class scheme is retained.

---

## Table 8. Protocol summary

| Item | Value |
|---|---|
| Dataset | FiReCS, CC BY 4.0, `huggingface.co/datasets/ccosme/FiReCS` |
| Train / test | 7,340 → 7,334 cleaned / 3,147 |
| Test class balance | negative 1,027, neutral 1,087, positive 1,033 |
| Integrity | SHA-256 pinned and re-verified before this run |
| Primary metric | Macro F1 |
| Interval method | Percentile bootstrap, 2,000 resamples |
| Significance test | McNemar's paired test (exact below 25 discordant pairs, otherwise continuity-corrected chi-square) |
| Multiple-comparison correction | Holm-Bonferroni, α = 0.05, 6 comparisons |
| Seed | 42 |
| Classical arm input | `feature_text(normalize(review))` — identical to the deployed model |
| Transformer input | Raw review text (its tokenizer normalizes) |
| Production impact | None. Every artifact records `qualified_for_integration: false` |

---

## Table 8b. Domain shift — the same models on real marketplace reviews

Every figure above is measured on FiReCS. The application reads Lazada and
Temu reviews. These are the same models, unchanged, scored on the project's
human-labelled marketplace set (`complete_manual_marketplace_reviews (1).csv`;
0% machine-labelled per the provenance audit; no text overlap with FiReCS
training). 470 reviews after excluding 28 "mixed": negative 363, positive 89,
neutral 18. Source: `research/marketplace_shift.json`.

| Model | FiReCS macro F1 | Marketplace macro F1 | 95% CI | Accuracy | p vs deployed |
|---|---|---|---|---|---|
| TF-IDF + LogisticRegression (deployed) | 0.814 | **0.563** | [0.520, 0.608] | 0.730 | — |
| Fine-tuned XLM-RoBERTa | 0.846 | **0.719** | [0.670, 0.765] | 0.864 | <0.001 |
| Fine-tuned DOST-ASTI RoBERTa-tl | 0.803 | 0.507 | [0.471, 0.543] | 0.566 | <0.001 |

| Error | Deployed TF-IDF | Fine-tuned XLM-R |
|---|---|---|
| Complaints read as **positive** | 31 of 363 | 2 of 363 |
| Complaints read as neutral | 64 of 363 | 49 of 363 |
| Praise read as negative | 3 of 89 | 1 of 89 |
| Macro F1, Lazada / Temu | 0.662 / 0.500 | 0.722 / 0.713 |

**The deployed model does not transfer.** Macro F1 falls from 0.814 to 0.563
on marketplace text, and to 0.500 on Temu. The fine-tuned transformer falls
far less (0.846 to 0.719), holds evenly across both platforms, and its
advantage over the deployed model grows from +0.032 on FiReCS to +0.156 here.
The Tagalog-specific model collapses. Neutral scores are unreliable for every
model: there are only 18 neutral reviews.

The deployed model's characteristic error is reading a complaint as praise (31
cases). That error inflates the positive ratio used to rank recommended
alternatives.

**A star-rating check helps the deployed model and not the transformer.**
Showing a review as "uncertain" when its text reading contradicts its star
rating (negative text on 4–5 stars, positive on 1–2) flags 46 of the deployed
model's labels, 35 of which were wrong: 28% of its errors caught, for 10% of
reviews. For the transformer it flags 16 labels, of which 14 were *right* —
when a stronger model disagrees with the stars, it is usually the reviewer's
stars and words that disagree, not the model that is wrong.

**Correction to Table 5.** Table 5 argued that the transformer's 93 ms per review
would make a 100-review product page take 9.3 s. That framing was wrong for
this application: reviews are analysed once, inside the background scan job,
and the saved result is what every page load reads. The latency cost is
therefore about 9 s added to a scan that already takes minutes, not a delay
seen on each page. The model's size (1.1 GB, plus the PyTorch runtime) remains
a real deployment cost. Taken with Table 8b, the case for keeping the TF-IDF
model is substantially weaker than Table 5 presents.

---

## Table 8c. Replacing the deployed model — three retraining attempts

After Table 8b the TF-IDF model was replaced. Three XLM-RoBERTa variants were
trained (same checkpoint, seed and hyper-parameters as Table 4) and scored on
the same 470 marketplace reviews. Sources:
`research/marketplace_shift_retrained.json`, `research/finetuned_*.json`.

| Model | Training text | Training data | FiReCS F1 | Marketplace accuracy | Neutral F1 | Marketplace macro F1 [95% CI] |
|---|---|---|---|---|---|---|
| TF-IDF + LR (previously deployed) | normalized | FiReCS | 0.814 | 0.730 | 0.147 | 0.563 [0.520, 0.608] |
| **First XLM-R (Table 4) — shipped** | normalized | FiReCS | 0.846 | 0.864 | **0.299** | **0.719** [0.670, 0.765] |
| Run 2 | raw | FiReCS | 0.850 | 0.813 | 0.192 | 0.671 [0.631, 0.711] |
| Run 1 | raw | FiReCS + Lazada stars | 0.843 | 0.932 | 0.000 | 0.620 [0.601, 0.637] |
| Run 3 | normalized | FiReCS + Lazada stars (weight 0.5) | 0.849 | **0.934** | 0.000 | 0.622 [0.605, 0.639] |

"Lazada stars" is 5,026 of the project's own Lazada reviews labelled by the
reviewer's star rating (1–2★ negative, 4–5★ positive, 3★ dropped): weak
supervision from human-given ratings, not sentiment annotations, and no
machine-generated labels. Every review of every marketplace-test product was
excluded (`sentiment-analysis/data/lazada_star_labels/lazada_star_labels_v1.manifest.json`).
Temu reviews never appear in training.

**Star labels trade the neutral class for accuracy.** Runs 1 and 3 label 93%
of marketplace reviews correctly, and 92% on Temu, a shop they never saw. But
star ratings contain no neutral class, and both runs stopped predicting neutral
on marketplace text, even with the star rows at half weight. Because macro F1
weights the 18 neutral reviews as heavily as the 363 negative ones, their
macro F1 falls below the first model's.

**Raw training text did not help.** Training on reviews as written (run 2)
scored lower on the marketplace than the first model, which was trained on
normalized text.

**Decision.** The rule for choosing between run 3 and the first model was
written down before run 3 was trained
(`research/PREREGISTRATION_xlmr_firecs_lazada_norm.md`): ship run 3 only if its
marketplace macro F1 exceeded 0.719 and its accuracy reached 0.90. It met the
accuracy bar but not the F1 bar, so the first model ships. The marketplace set
has now been used to choose between models, so its figures for the shipped
model are not an untouched estimate; only 18 of its reviews are neutral.

## Table 8d. Slimming the shipped model for a free CPU host

`research/slim_sentiment_model.py`, report `research/slimming_xlmr_firecs_v1.json`.
Each stage scored one review at a time, as served, single-threaded, on an
otherwise idle laptop CPU. Acceptance rule fixed in advance: the smallest ONNX
stage within 0.02 marketplace macro F1 of the full model and not significantly
different from it (McNemar p ≥ 0.05).

| Stage | Size | Peak RAM | ms per review | FiReCS F1 | Marketplace F1 | McNemar p vs full |
|---|---|---|---|---|---|---|
| Full (PyTorch fp32) | 1,112 MB | 2,588 MB | 157 | 0.846 | 0.719 | — |
| Vocabulary pruned | 377 MB | 1,203 MB | 164 | 0.847 | 0.722 | 0.63 |
| ONNX fp32 | 378 MB | 469 MB | 105 | 0.847 | 0.722 | 0.63 |
| **ONNX int8 — shipped** | **95 MB** | **175 MB** | **93** | 0.833 | 0.706 | 1.00 |

- **Vocabulary pruning** keeps 10,793 of 250,002 word pieces, those found in
  non-test training text. Training text segments identically; 15% of FiReCS test
  and 20% of marketplace texts contain a word that now splits into smaller
  pieces, and 0.05% of marketplace tokens become unknown.
- **The tokenizer is pruned with the model.** The plan had kept the full
  tokenizer and remapped ids; measured, that tokenizer alone used 262 MB.
- **Each review is scored on its own.** In batches, the int8 model's labels
  depended on which reviews shared the batch (18 of 470 changed in a test on
  this checkpoint), and padded batches were slower.
- **Input text.** The shipped model was trained on normalized text but is
  served raw text, the form in which its 0.719 was measured and recorded.
  This mismatch is disclosed rather than corrected: retraining on raw text
  (run 2) scored lower.

**In the application** (`registry.load_model()` end to end, verified to
reproduce the int8 row exactly): with the existing rule that lets a 1–2★
rating turn a neutral reading negative, marketplace accuracy is **0.936**
(macro F1 0.700), against 0.730 for the TF-IDF model. The whole web app peaks
at 352 MB with the model loaded; the model loads in 0.9 s. The first catalog
load after a server start analyses 100 reviews in about 19 s; later loads,
from any browser session, take 0.2 s.

---

# Part B — Hazard badges (safety, counterfeit, defect)

Measured against labelling round 1: 900 review sentences drawn from three
strata (`seed_hit` 380, `low_star` 240, `random_all` 280), each labelled
independently by two of three human annotators under
`research/HAZARD_SENTENCE_TAGGING_GUIDE.md`. Ground truth is the agreement of
the two annotators who saw a sentence; sentences on which they disagreed are
excluded per category pending adjudication. Sources:
`research/labelling_round_1_analysis.json`,
`research/hazard_tagger_measurement.json`.

## Table 9. Inter-annotator agreement

| Category | Sentences | Raw agreement | Cohen's κ | Disagreements |
|---|---|---|---|---|
| Safety | 879 | 0.965 | **0.617** | 31 |
| Defect | 879 | 0.809 | 0.541 | 168 |
| Counterfeit | 879 | 0.910 | 0.541 | 79 |

Of 879 doubly-labelled sentences, 630 (71.7%) were agreed on all three
categories. Safety reaches substantial agreement; defect and counterfeit are
moderate. Defect is the least consistent category — across their 600
sentences each, one annotator marked 219 as defects and another 113 — which
bounds the accuracy any defect classifier can be shown to achieve on these
labels.

**Anchoring check.** 137 sentences (15%) were shown without the keyword
pre-label. Positive rates on blind and pre-labelled sentences were close for
defect (23.4% vs 24.3%) and counterfeit (7.6% vs 6.8%). Safety (1.6% vs 3.4%)
rests on two blind positives and cannot support a conclusion either way.

## Table 10. How common each problem is

Computed on the `random_all` stratum only, the one stratum sampled without
regard to content, weighted by sampling weight.

| Category | Positive sentences | Labelled | Weighted prevalence |
|---|---|---|---|
| Safety | 0 | 280 | **0.0%** |
| Defect | 16 | 254 | 6.3% |
| Counterfeit | 1 | 273 | 0.4% |

No safety report occurred in 280 randomly sampled sentences. This is
consistent with published estimates of about 0.6% (Zaman et al., 2024) but has
a direct consequence: **safety recall on the general population cannot be
estimated from this round.** All 27 safety reports behind the recall figures in
Table 11 come from the enriched strata, where safety sentences were
deliberately over-sampled.

## Table 11. The shipped safety check, before and after model 0.2.0

The live tagger is a keyword lexicon. Two changes were measured separately:
42 safety terms added — the Tagalog hazard vocabulary the shipped list lacked
(`nasunog`, `umusok`, `pumutok`, `kuryente`, `nagliyab`, `amoy sunog`) plus
`overheats` — and a filter that discards a *mild* heat word (`umiinit`,
`mainit`) when the sentence concerns clothing, weather, a venue, or a product
whose purpose is heat. Stronger evidence is never filtered.

| Safety configuration | Precision | Recall | F1 |
|---|---|---|---|
| Model 0.1.0 (15 safety terms) | 0.381 | 0.593 | 0.464 |
| + heat-context filter only | 0.500 | 0.593 | 0.542 |
| + added vocabulary only | 0.429 | 0.778 | 0.553 |
| **Model 0.2.0, shipped (both)** | **0.539** | **0.778** | **0.636** |

848 sentences with agreed safety labels; 27 true safety reports. The shipped
check catches 21 of the 27 and raises 39 flags, of which 21 are real. The
filter removed 10 false alarms and no true report. Targets from
`hazard_parameters_v1.json` are recall 0.80 and precision 0.80: **the safety
check meets neither.**

| Category | Precision | Recall | F1 | Target recall | Meets |
|---|---|---|---|---|---|
| Safety | 0.539 | 0.778 | 0.636 | 0.80 | neither |
| Defect | 0.885 | 0.440 | 0.588 | 0.70 | precision only |
| Counterfeit | 0.804 | 0.719 | 0.759 | 0.70 | both |

Defect and counterfeit were not changed in 0.2.0.

**In-sample warning.** The added terms and the filter were chosen after
inspecting round 1 errors. Every 0.2.0 figure is therefore fitted to the data
it is measured on and is optimistic. It bounds what the changes can achieve; it
is not an estimate of field performance. Confirming it requires a second
labelled round on sentences the rules have never seen.

**Known limitation (0.2.0).** Negation is detected by looking back three
words, so in "not good umusok agad" the "not" that belongs to "good" also
cancels the smoke report. Corrected in 0.3.0 (Table 11b) for the hazard
checker only; the sentiment path keeps the old rule.

## Table 11b. Model 0.3.0 — development result on round 1 (in-sample)

Reading every round-1 error showed four causes: (1) the defect list lacked
phrasings the tagging guide itself lists (battery drain, "hindi nag-charge",
"di nagana", "ayaw mag charge"); (2) the negation bug above; (3) guide rule 1
was not implemented, so missing items ("kulang") and damaged boxes counted as
defects; (4) names and idioms were counted ("burnt orange", "smoke removal",
"fake news", "fake nails"). Model 0.3.0 addresses all four
(`hazard_model.py`, `text_norm.is_negated_adjacent`, `risk_lexicon.tsv`,
`hazard_phrase_exclusions_v1.tsv`). Source: `research/hazard_tagger_measurement.json`,
configurations `v2_shipped` and `v3_shipped`, each run with its own frozen
term list, cue list and rules.

| Check | Model | Flags real | Reports caught | F1 [95% CI] | Accuracy | Never-flag accuracy | Weighted recall |
|---|---|---|---|---|---|---|---|
| Safety | 0.2.0 | 21 of 39 | 21 of 27 | 0.636 [0.500, 0.771] | 0.972 | 0.968 | 0.733 |
| Safety | **0.3.0** | **21 of 37** | 21 of 27 | **0.656** [0.509, 0.783] | 0.974 | 0.968 | 0.733 |
| Defect | 0.2.0 | 77 of 87 | 77 of 175 | 0.588 [0.519, 0.654] | 0.848 | 0.754 | 0.242 |
| Defect | **0.3.0** | **96 of 98** | **96 of 175** | **0.703** [0.637, 0.763] | 0.886 | 0.754 | **0.426** |
| Counterfeit | 0.2.0 | 41 of 51 | 41 of 57 | 0.759 [0.659, 0.839] | 0.968 | 0.929 | 0.636 |
| Counterfeit | **0.3.0** | **46 of 50** | **46 of 57** | **0.860** [0.788, 0.924] | 0.981 | 0.929 | 0.714 |

"Never-flag accuracy" is what a checker that flags nothing would score:
because problems are rare, plain accuracy is uninformative on its own.
"Weighted recall" counts each sentence by its sampling weight, undoing the
queue's over-sampling of keyword matches; it estimates recall over reviews as
they occur. The gap for defects (0.55 unweighted, 0.43 weighted) says ordinary
reviews describe defects in more varied words than the enriched strata do.

**In-sample.** Every 0.3.0 change was chosen after reading round-1 errors, so
these figures are optimistic. The new terms were also checked on 17,429
unlabelled reviews outside rounds 1 and 2, where nearly every hit was a real
complaint. Three round-1 defect reports were lost: one relied on "kulang",
which the guide excludes, and two had their damage word next to "box" or
"packaging" while the broken part was the product itself.

**Held-out test.** Labelling round 2 (600 new sentences, two annotators each,
no computer pre-labels; `outputs/labelling_queue_round_2/`,
`research/LABELLING_ROUND_2_INSTRUCTIONS.md`) was built before 0.3.0 was
finalised and shares no review with round 1. Success rule, fixed in advance:
0.3.0 is kept if on round 2 its defect F1 beats 0.2.0's and its safety recall
is not lower than 0.2.0's. Result: Table 11c.

## Table 11c. Held-out test on labelling round 2

600 new sentences (`seed_hit` 260, `low_star` 160, `random_all` 180), each
labelled by two of the three annotators with no computer pre-label. 593 were
labelled twice; 463 (78%) agreed on all three categories. Source:
`research/hazard_tagger_measurement_round2.json`,
`research/labelling_round_2_analysis.json`.

| Category | Cohen's κ, round 1 | Cohen's κ, round 2 | Disagreements (round 2) |
|---|---|---|---|
| Safety | 0.617 | 0.551 | 24 of 593 |
| Defect | 0.541 | **0.600** | 102 of 593 |
| Counterfeit | 0.541 | 0.479 | 24 of 593 |

Defect agreement rose after the guide's new worked examples (Rule 7, version
1.1); safety and counterfeit fell slightly. Again no safety report occurred in
the 180 randomly drawn sentences.

| Check | Model | Flags real | Reports caught | F1 [95% CI] | Accuracy | Never-flag accuracy | Weighted recall |
|---|---|---|---|---|---|---|---|
| Safety | 0.2.0 | 10 of 23 | 10 of 16 | 0.513 [0.294, 0.683] | 0.967 | 0.972 | 0.465 |
| Safety | **0.3.0** | 10 of 25 | 10 of 16 | 0.488 [0.263, 0.667] | 0.963 | 0.972 | 0.465 |
| Defect | 0.2.0 | 56 of 77 | 56 of 132 | 0.536 [0.448, 0.624] | 0.802 | 0.731 | 0.266 |
| Defect | **0.3.0** | **59 of 72** | **59 of 132** | **0.578** [0.487, 0.658] | 0.825 | 0.731 | 0.278 |
| Counterfeit | 0.2.0 | 6 of 11 | 6 of 12 | 0.522 [0.235, 0.727] | 0.981 | 0.979 | 0.343 |
| Counterfeit | **0.3.0** | 7 of 13 | 7 of 12 | 0.560 [0.300, 0.765] | 0.981 | 0.979 | 0.400 |

Paired comparison of 0.2.0 and 0.3.0 on the same sentences (McNemar, exact):
defect — 0.3.0 alone right on 13 sentences, 0.2.0 alone on 2, **p = 0.007**;
safety — 0.2.0 alone right on 2, 0.3.0 on none, p = 0.5; counterfeit — 1 and
1, p = 1.0.

**Decision under the pre-registered rule: 0.3.0 is kept.** Its defect F1 is
higher (0.578 vs 0.536, and the paired difference is significant) and its
safety recall is equal (10 of 16).

**What the held-out test shows.**
- The round-1 figures were optimistic, as Table 11b warned. Defect F1 is
  0.578 here against 0.703 in-sample; counterfeit 0.560 against 0.860; safety
  precision 0.40 against 0.57.
- The real gain of 0.3.0 is in defects: fewer false alarms (13 vs 21) and
  slightly more reports caught. It does not improve safety or counterfeit.
- Safety still raises more false flags than true ones (15 vs 10), so its plain
  accuracy (0.963) is below that of a checker that never flags (0.972). Its
  value is recall: it catches 10 of 16 reports, which is why the interface
  quotes each flagged sentence instead of asserting a finding.
- Both of 0.3.0's new safety false alarms come from the narrower negation:
  "should be good if its left piece not have static/grounded noise" and
  "cant even remove smoke unlike other similar products". Neither is changed
  here, because tuning on round 2 would end its use as a test.
- Weighted for how often each kind of sentence occurs, defect recall is 0.28:
  roughly three in four everyday defect reports are missed. A keyword list
  has reached its limit for defects.
- Counterfeit rests on 12 positive sentences; its intervals are too wide to
  conclude anything.

## Table 11d. Learned models (checker 0.4): selection, test and decision

Protocol fixed in advance: `research/PREREGISTRATION_hazard_v4.md` (one
amendment, made before any scoring, removed the keyword checker from round-1
selection because its round-1 figures are in-sample). Training data: round 1,
annotators A1 and A2 only (A3's sheet is missing); sentences both labelled are
kept only when they agree. Sources: `research/hazard_classifier_selection.json`,
`research/hazard_tagger_measurement_round2.json`,
`sentiment-analysis/models/hazard_classifier_v1/manifest.json`.

**Selection (round 1, 5-fold cross-validation grouped by review, out-of-fold F1).**

| Candidate | Safety (30 positives / 873) | Defect (191 / 824) | Counterfeit (81 / 857) |
|---|---|---|---|
| T — TF-IDF word + character n-grams | 0.135 | 0.655 | 0.644 |
| **TK — T + keyword features** | **0.640** | **0.671** | **0.769** |
| E — sentence embedding (XLM-R) | 0.184 | 0.564 | 0.400 |
| EK — E + keyword features | 0.649 | 0.611 | 0.658 |

TK was chosen in every category (for safety, EK was within the 0.01 tie
margin and TK is simpler). The embedding from the fine-tuned sentiment model
did worse than plain word patterns with 900 training sentences, and adds an
extra model pass per sentence, so nothing in the shipped app uses it.
Thresholds: safety 0.819 (highest with out-of-fold recall ≥ 0.80), defect
0.556, counterfeit 0.472.

**Test (round 2, once) and decision.**

| Check | Keywords 0.3.0: F1 / recall | Learned TK: F1 / recall | McNemar (sentences only learned got right vs only keywords) | Rule | Shipped |
|---|---|---|---|---|---|
| Safety | 0.488 / 0.625 | 0.500 / 0.625 | 1 vs 0, p = 1.0 | adopt | **learned** |
| Defect | 0.578 / 0.447 | **0.646 / 0.538** | 12 vs 4, p = 0.077 | adopt | **keywords (override)** |
| Counterfeit | 0.560 / 0.583 | 0.538 / 0.583 | 0 vs 1, p = 1.0 | keep | keywords |

**Safety.** Adopted as the rule requires. In practice it is the keyword checker
with a learned second opinion: on all 29,523 unlabelled sentences outside both
rounds it kept 106 of the keyword checker's 107 safety flags, removed one
(correctly — "smoke removal" as a product feature) and added none.

**Defect — overridden, and disclosed as a departure from the
pre-registration.** The rule adopted the learned defect model: on round 2 it
caught 12 more real reports for 4 more false ones, and raised weighted recall
from 0.28 to 0.38. A live check after the test showed a failure mode the
labelled rounds under-sample. On 4,000 unlabelled reviews it added 68 defect
flags and removed none; about two in three were real, but many of the rest
were sentences saying the product *works* — "gumagana po cya", "ok naman siya,
gumagana", "fast charging" — because "gumagana" and "charging" mostly occur in
complaints in the training data. On a mostly five-star charger listing it
quoted "fast charging" and "Maayos ang packaging" to shoppers as defect
reports. The user kept the keyword checker for defects until a fix (for
example, letting the learned model add a flag only when the sentiment model
reads the sentence as a complaint) is tested on a new labelling round. The
fix was conceived after seeing these results, so it cannot be tested on
round 2.

**Shipped (checker 0.4):** safety = learned TK model on top of keywords 0.3.0;
defect and counterfeit = keywords 0.3.0. Round-2 figures of exactly this
combination (`research/hazard_tagger_measurement_round2_shipped.json`):
safety 10 of 16 caught, 10 of 24 flags real (F1 0.500); defect and
counterfeit as Table 11c.

## Table 12. Design consequence — why the interface shows reviews, not a verdict score

| Property of the safety check | Value | Consequence for the interface |
|---|---|---|
| Flags that are real | 10 of 24 (0.4, held-out round 2) | A flag cannot be presented as a finding; the flagged sentence is quoted so the shopper can judge it |
| Real reports caught | 10 of 16 (held-out round 2) | Absence of flags cannot be presented as safety; the clean verdict reads "No warning signs found" and states that this is not proof |
| Prevalence | ~0% in random reviews | A single report is shown even when there are too few reviews for an overall verdict |

The previous interface reported a 0–100 risk score. After the badge model was
enabled the screens continued to read that field, which the badge model does
not produce, and defaulted it to zero: every product, including those with fire
reports, displayed as "Low risk" with 100% reliability. The interface now
renders the badge verdict and the quoted sentences directly.

A second defect was found and corrected in the verdict rules. Counterfeit and
defect warnings require the report rate to exceed a product-type baseline, and
no baselines exist yet, so neither category could ever produce a warning: a
listing with 4 of 39 reviews calling it fake read "No warning signs found".
Without a baseline, the declared absolute floor (2% counterfeit, 3% defect, on
the Wilson lower bound) is now applied on its own. **Those floor values are
marked PLACEHOLDER in the parameter file** and should be set from the reference
corpus when it exists.

---

## Outstanding

- `ldlazaro/sentiment_roberta_taglish` fine-tune — not started. It is the second
  Tagalog-specific encoder and would test whether DOST-ASTI's result generalises.
- **Round 1's A3 worksheet is missing.** On 2 Oct 2026 the round-2 sheets were
  saved under the round-1 file names and replaced round 1's A1 and A3 files.
  A1 was recovered exactly from a copy attached in a working session (it
  reproduces every recorded A1–A2 disagreement); A2 survived. A3 (Ryan's
  round-1 sheet) has not been found, so round 1's figures (Tables 9–11b)
  cannot be recomputed until it is. Round 2 is unaffected. Finished sheets
  now live in `outputs/labelling_queue_round_<n>/worksheet_<A>_done.*`.
- Defect detection: shipped keywords, held-out F1 0.578, weighted recall 0.28.
  **Gated defect model built, awaiting round 3** (3 Oct 2026):
  `research/PREREGISTRATION_defect_v5.md`. Round 3 (600 blind sentences, no
  review from rounds 1–2; 120 from a stratum where the round-1 learned model
  and the keywords disagree) is in `outputs/labelling_queue_round_3/` with
  `research/LABELLING_ROUND_3_INSTRUCTIONS.md`. Candidate: TK trained on rounds
  1 and 2 (1,322 sentences, 323 defects), which may add a defect flag only to a
  sentence the sentiment model does not read as positive; threshold 0.485 from
  grouped 5-fold cross-validation (out-of-fold F1 0.716 vs 0.710 ungated vs
  0.623 keywords, the latter partly in-sample). In cross-validation the gate
  blocked 5 additions, none a real defect. On 4,000 unlabelled reviews it adds
  278 flags; the gate blocked 27, almost all "it works" praise. Of 25 additions
  read, about two in three are defects; most of the rest are complaints that
  are not defects ("DO NOT BUY FROM THIS STORE", late delivery). Not tuned
  further. When the sheets return: `measure_hazard_tagger.py --round 3`, then
  `train_defect_gate.py --adopt`. Adjudication of the disagreements in
  both rounds (168 + 102 defect, 79 + 24 counterfeit) is still open.
- Product-type baselines for counterfeit and defect, and real values for the
  placeholder floors.
- Normalized-text ablation for the transformer, to separate architecture from
  preprocessing.
- Ceiling estimate from the blind 30-row sample
  (`research/neutral_ceiling_check.csv`), which determines whether the
  remaining 594 errors are worth pursuing.
