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

## Table 11e. Gated defect model — held-out test on labelling round 3

Protocol: `research/PREREGISTRATION_defect_v5.md`. Round 3: 600 blind
sentences, no review from rounds 1–2; 592 labelled twice, 438 agreed on all
three categories (κ: safety 0.690, defect 0.521; counterfeit had no agreed
positive, so κ is 0 and counterfeit cannot be scored). Sources:
`research/hazard_tagger_measurement_round3.json`,
`research/labelling_round_3_analysis.json`.

| Defect check | Flags real | Reports caught | F1 [95% CI] | Weighted precision | Weighted recall |
|---|---|---|---|---|---|
| Keywords 0.3.0 | 75 of 86 | 75 of 179 | 0.566 [0.486, 0.637] | 0.891 | 0.377 |
| **Gated learned model** | **126 of 142** | **126 of 179** | **0.785** [0.735, 0.831] | 0.848 | **0.527** |

McNemar: the gated model alone was right on 51 sentences, the keywords alone
on 5 (p < 0.001). In the disagreement stratum the gated model caught 50 of 64
reports with 3 false flags (keywords: 7 of 64, 1 false).

**Decision: adopted.** All three pre-registered conditions hold: F1 higher;
recall not lower; weighted precision 0.848 ≥ 0.891 − 0.05 = 0.841 (met by a
narrow margin). Shipped as `hazard_classifier_v2` (safety model unchanged).

Safety on round 3 (second held-out check, unchanged model): 14 of 19 reports
caught, 14 of 28 flags real, F1 0.596.

**Limits.** Round 3 over-samples disagreement sentences, so the unweighted F1
flatters both the gain and the model; the weighted figures are the estimate
for ordinary reviews. Precision of added flags is lower outside the test: on
a mostly five-star charger listing the model raised flagged reviews from 5 to
20, and roughly half the additions were not defects (packaging, seller
replies, neutral remarks about charging speed).

**Post-hoc rule (5 Oct 2026, after the test).** The learned model may not add
a defect flag to a sentence about the box, courier, seller or their replies
unless the sentence still reads as a defect with those words and their two
neighbours on each side removed (guide rule 1; `hazard_model.without_non_product_words`).
A first version that dropped every such sentence removed five real round-3
reports and no false one, so it was replaced. With the final rule the round-3
figures are identical to the table above (126 caught, 16 false;
`research/hazard_tagger_measurement_round3_rule1_posthoc.json`), and on the
charger listing the packaging and seller sentences are no longer quoted
(flagged reviews 20 → 16). The rule was written after round 3 was read, so
this is a check that it does no harm there, not a held-out result.

## Table 11f. Counterfeit rules "c1" — built from round 4, checked on rounds 2 and 3

**Round 4 is a development set, not a test.** 391 sentences drawn for the
counterfeit category only (`research/build_counterfeit_round4.py`, seed
20261005): all 116 sentences the 0.3.0 keywords flagged, 200 of 410 that
mention authenticity words, 75 of 10,767 from one- to three-star reviews.
Labels are **AI-assisted and owner-adjudicated**: the assistant labelled every
sentence with the tagging guide and marked 43 as unsure; the project owner
decided those 43 and blind-audited 40 random "sure" labels (15 yes, 25 no),
agreeing on 36 (90%). Thirteen owner answers that departed from the guide
(false size or capacity claims, "scam" alone, "parang original" as praise)
were set back to the guide's rule at the owner's decision
(`outputs/labelling_queue_round_4/labels_adjudicated.csv`, column
`label_source`). Because most labels are machine-proposed, **no figure from
round 4 is reported as accuracy**; it was used only to see where the keyword
check goes wrong.

What it showed: of 80 sentences flagged through the word "scam", 5 were
counterfeit reports (seller did not deliver, item missing, product broken
are the usual meaning); the other counterfeit terms together gave 31 real
matches and 6 false ones. The reports the list missed were comparisons with
the original ("not 100% original", "hindi rin original", "copy lang",
"anlayo sa original", "hindi sya kaamoy ng tunay").

Rules c1 (`hazard_model.COUNTERFEIT_RULES`), applied after the keyword result
and after the learned models, so the learned models' keyword features and
every safety and defect figure are unchanged:

1. "scam" no longer makes a counterfeit flag by itself.
2. "hindi/not legit" followed by a number ("hindi legit na 20k mAh") is a
   disputed specification, not a counterfeit report.
3. Seven comparison-with-the-original patterns add a flag.

Check on the human-labelled rounds (consensus labels, same scripts as Tables
11c and 11e; `research/hazard_tagger_measurement_round{2,3}_counterfeit_c1.json`):

| Round | Counterfeit reports | Keywords 0.3.0: caught / false flags | With c1: caught / false flags | Precision | F1 | McNemar |
|---|---:|---:|---:|---:|---:|---|
| 2 | 12 | 7 / 6 | 7 / 4 | 0.538 → 0.636 | 0.560 → 0.609 | 0 vs 2, p = 0.5 |
| 3 | 0 | 0 / 3 | 0 / 0 | n/a | n/a | 0 vs 3, p = 0.25 |

No report was lost and no new false flag appeared in either round; five
false flags were removed. Safety and defect are identical to before.

**Limits.** The rules were written after rounds 2 and 3 had been read, so this
is a check that they do no harm on human labels, not a held-out result, and
the differences are not statistically significant on 12 reports. None of the
added patterns fired in rounds 2 or 3, so their benefit is unmeasured outside
round 4, where the same rules were developed (there: 35 → 44 of 63 caught,
81 → 5 false flags; in-sample). A fair test of the counterfeit check still
needs a human-labelled round with enough counterfeit reports.

## Table 11g. Another model, or more labels? (exploratory, 6 Oct 2026)

**Exploratory, not a test.** Each figure trains on some of rounds 1–3 and
scores on a round left out. Rounds 2 and 3 are therefore used up as tests;
from here they are training data and round 5 is the only test
(`research/PREREGISTRATION_final_v6.md`). Run once with fixed, untuned
settings; scripts were ad hoc and the figures are recorded here as the reason
for the v6 design, not as results of it.

**Defect — other model types** (same rule: keywords OR model, never on
praise; threshold from grouped cross-validation on the training rounds).

| Model | Rounds 1+3 → round 2: P / R / F1 | Rounds 1+2 → round 3: P / R / F1 |
|---|---|---|
| Keywords 0.3.0 alone | 0.808 / 0.447 / 0.576 | 0.862 / 0.419 / 0.564 |
| **TF-IDF + keyword features, logistic (shipped design)** | 0.756 / 0.682 / **0.717** | 0.875 / 0.704 / **0.780** |
| Same features, linear SVM | 0.700 / 0.689 / 0.695 | 0.865 / 0.682 / 0.762 |
| Sentence embedding (shipped XLM-R) + keyword features | 0.667 / 0.697 / 0.681 | 0.759 / 0.670 / 0.712 |
| TF-IDF + embedding + keyword features | 0.679 / 0.674 / 0.677 | 0.784 / 0.670 / 0.723 |
| TF-IDF only | 0.824 / 0.636 / 0.718 | 0.837 / 0.743 / 0.787 |

No alternative beats the shipped design by more than noise; the embedding
variants are clearly worse and would add a model pass per sentence.

**Defect — more labelled rounds, same design.**

| Trained on | Defect reports in training | Scored on | F1 | Recall |
|---|---:|---|---:|---:|
| Round 1 | 191 | round 2 | 0.643 | 0.538 |
| Rounds 1 + 3 | 370 | round 2 | 0.717 | 0.682 |
| Round 1 | 191 | round 3 | 0.739 | 0.631 |
| Rounds 1 + 2 | 323 | round 3 | 0.780 | 0.704 |

Each added round raised F1 by 0.04–0.07, almost all of it recall. The curve
has not flattened, which is the case for retraining on all three rounds.

**Safety.** Keywords alone score F1 0.476 (round 2) and 0.583 (round 3).
Letting a model trained on two rounds decide, filter or add gave 0.476–0.526
and 0.583; embedding variants 0.47–0.61. Nothing separates from keywords with
65 agreed safety reports in total. The v4 threshold rule (highest threshold
with out-of-fold recall ≥ 0.80) selects 0.04–0.05 on pooled rounds and raises
168–241 false flags, so it is retired.

**Counterfeit.** A learned model trained on every agreed label (with or
without round 4) added 5 flags in cross-validation, none real, at thresholds
0.5–0.6 and none at 0.7+. The limit is the labels: across rounds 1–3 only
**25 sentences** have two annotators agreeing on a counterfeit report (13, 12
and 0), 65 are unresolved disagreements (one annotator said fake in 46 of
them), and 68 round-1 "fake" labels rest on one annotator because A3's sheet
is missing. At review level — what the badge counts — rules c1 catch 6 of the
8 agreed fake-report reviews in round 2 with 2 false flags (0.3.0: 6 and 4).

**Not tried.** A transformer fine-tuned for hazard sentences. It would need
more labelled reports than exist for safety and counterfeit, and a second
95 MB model beside the sentiment model would take the app (about 352 MB) to
the edge of a free host's memory.

**Decision.** Keep the model type; retrain on rounds 1–3; test once on a new
round. Candidates frozen in `sentiment-analysis/models/hazard_classifier_v3/`
(`research/final_v6_training.json`; cross-validated, optimistic):

| Check | Candidate | Training sentences / reports | Cross-validated: keywords → candidate (F1) |
|---|---|---|---|
| Defect | gated TK, threshold 0.460 | 1,782 / 502 | 0.603 → 0.765 |
| Safety | TK as a filter on keyword flags, threshold 0.855 | 2,033 / 65 | 0.582 → 0.607 (precision 0.48 → 0.55, recall 0.74 → 0.68) |
| Counterfeit | none; keywords + c1 scored as shipped | — | — |

The safety filter trades four real reports for sixteen false flags in
cross-validation, so under the pre-registered rule (recall must not fall) it
is likely to be rejected; it is tested anyway because the rule was fixed
before the candidate was trained.

Round 5: `outputs/labelling_queue_round_5/`, drawn only from the public
`scaredmeow/shopee-reviews-tl-stars` set (15,000 reviews, none used before),
with `research/LABELLING_ROUND_5_INSTRUCTIONS.md` and guide 1.2. The 65
counterfeit disagreements are in `outputs/adjudication_counterfeit/` for the
third annotator. A training-script fault was fixed first: after c1 was added,
`train_hazard_classifier.featurize` took keyword features after the
counterfeit rules while the app takes them before; no shipped model was
trained in that state.

## Table 11h. Rounds 5 and 6 — AI-assisted checks, and the gadget scope (6–7 Oct 2026)

**Not accuracy.** The project owner chose to have the AI assistant label
round 5 instead of the team (`PREREGISTRATION_final_v6.md`, departure 1), and
round 6 the same way. The assistant read each sentence without the stratum or
any checker output, applied guide 1.2 and marked each answer sure or unsure
(`research/label_round5_ai.py`, `label_round6_ai.py`); the owner decides the
unsure sentences and blind-audits 50 sure ones per round
(`outputs/labelling_queue_round_{5,6}/worksheet_for_you.csv`) — **that review
is still open**, so the figures below use the assistant's labels as they
stand. They are agreement with AI-proposed labels. The assistant also wrote
the counterfeit rules and trained the candidates, so neither round is an
independent test, and the v6 adoption rules are not applied. Source:
`research/score_ai_rounds.py` → `research/ai_rounds_check.json`.

- **Round 5:** 600 sentences from the public Shopee set (any product).
- **Round 6:** 600 sentences, electronic gadgets only
  (`research/gadget_scope.py`: our electronics scrapes, or text naming a
  gadget or gadget part): 182 from unused Lazada scrapes, 418 from the Shopee
  set, none from rounds 1–5. The Lazada part had been seen by earlier
  unlabelled spot checks, so round 6 is development data in any case.

**Two owner rules (7 Oct 2026), applied to both rounds.** The assistant had
first labelled slow charging as not a defect and filed heat and "grounded"
reports under safety only. The owner ruled that (1) slow charging on its own
is a defect and (2) a safety report is a defect as well
(`label_round5_ai.apply_owner_rules`; guide 1.3, Rule 9). Both match the
team's own practice in rounds 1–3, checked afterwards: all 6 slow-charging
sentences there were labelled defect, and 40 of the 54 agreed safety reports
with a defect decision were also labelled defect. The rules moved 16 defect
answers in round 5 and 19 in round 6. AI labels with the rules: round 5 —
safety 16, defect 111, counterfeit 34 (91 sentences with an unsure answer);
round 6 — safety 21, defect 164, counterfeit 25 (96 unsure).

**Why the scope matters — round 5 split by product type** (caught of
reports, false flags, F1):

| Check | Gadget reviews (153 sentences) | Other reviews (447 sentences) |
|---|---|---|
| Defect, shipped | 37 of 62, 8 false, 0.692 | 22 of 49, 12 false, 0.530 |
| Defect, v3 candidate | 44 of 62, 11 false, 0.752 | 30 of 49, 24 false, 0.583 |
| Safety, shipped | 9 of 10, 2 false, 0.857 | 6 of 6, 4 false, 0.750 |
| Counterfeit (rules c1) | 6 of 9, 6 false, 0.571 | 14 of 25, 2 false, 0.683 |

**Round 6, gadgets only.**

| Check | Shipped (v2 + c1) | v3 candidate (rounds 1–3) | On the sentences labelled "sure" |
|---|---|---|---|
| Safety | 20 of 21, 3 false, F1 0.909 | identical | 14 of 14, 2 false |
| Defect | 77 of 164, 30 false, F1 0.568 | 106 of 164, 47 false, F1 0.669 | shipped 67 of 113; candidate 85 of 113 |
| Counterfeit | 18 of 25, 4 false, F1 0.766 | same rules | 14 of 15, 1 false |

The Lazada and Shopee parts of round 6 agree closely (defect F1 0.556 and
0.573 shipped; 0.674 and 0.667 candidate).

**Defect training dry run** (no model written). Same gated design; threshold
= best out-of-fold F1; scored on round 6:

| Trained on | Caught of 164 | False flags | F1 |
|---|---:|---:|---:|
| Keywords alone | 38 | 14 | 0.352 |
| Rounds 1–3 (= v3 candidate) | 106 | 48 | 0.667 |
| Rounds 1–3 + 5 | 98 | 35 | 0.660 |
| Rounds 1–3 + 5 + 6 (cross-validated) | 120 | 44 | 0.732 |

The model with every round, scored on the team-labelled sentences of rounds
1–3 (cross-validated): F1 0.750 against 0.765 for the rounds-1–3 model
(false flags 121 against 94); on their gadget sentences 0.765 against 0.768.
So adding the AI-labelled rounds gains 0.065 on round 6, costs nothing
measurable on the team's gadget sentences, and costs 0.015 on their
non-gadget ones. Training on gadget sentences only was also tried before the
rules (rounds 1–3 and 5, scored on gadget sentences, cross-validated): F1
0.737 against 0.753 for training on everything, so the non-gadget sentences
are kept in training and the scope is applied to what is tested and claimed,
not to what is learned from.

**What this does and does not show.**
- Safety on gadgets agrees with the labels almost completely, and the v3
  safety filter changes nothing there. Keywords stay.
- Defect is where the checkers and the assistant differ most. Before the
  owner's rules the candidate's precision against the assistant's labels was
  about 0.60; with them it is 0.69 on round 6 and 0.80 on round 5's gadget
  sentences, against 0.85–0.89 on the team's rounds. Reading the round-6
  disagreements after the first labels were saved: about 37 of the
  candidate's 51 flags on sentences the assistant was sure about were plain
  errors (praise, hopes, returns, negation such as "Wala pong sira", "walang
  ka sira sira", "lowbat lang pala", "hindi rin kami naka-experience ng
  lag"); the rest were the two conventions the owner has now settled. The
  first group is a real weakness of the checker on these reviews. One
  transcription slip in the assistant's list (row 168) was found this way and
  corrected; no other label was changed after the checkers' output was seen,
  apart from the owner's two rules.
- Nothing is adopted. v2 with rules c1 still ships. The v3 folder and the
  dry-run models are not used by the app.

## Table 11i. Five labels: "poor quality" and "not as advertised" (7 Oct 2026)

**Why.** Three labels forced a yes/no on sentences that fit none. In the
team's rounds 1–3 the defect answer for weak-but-working products was 8 yes /
6 no / 6 disagreed, for dents and scratches 7 / 9 / 11, for "low quality" or
flimsy 8 / 31 / 12; the counterfeit answer for "not like the picture" was
6 / 4 / 2. Guide 1.4 (Rule 10) gives these their own labels and fixes the
shopper wording: Safety problem; Faulty or broken; Poor quality; Possible
fake; Not as advertised. One earlier answer changes: "iba sa picture" is now
"not as advertised", no longer counterfeit. Rounds 1–3 keep their labels.

**Labels.** Rounds 5 and 6 were relabelled by the assistant with all five
(`label_round5_ai.FIVE`, `label_round6_ai.FIVE`); the owner's review is still
open, so every figure below is agreement with AI labels, not accuracy.
Round 5: safety 16, defect 103, quality 37, counterfeit 32, not as described
24 (85 sentences with an unsure answer, 91 before). Round 6: 21, 139, 35, 23,
25 (70 unsure, 96 before).

**Learned models, cross-validated dry run** (`score_ai_rounds.py`,
`five_label_dry_run`; nothing saved). Round 6, caught of reports / false / F1:

| Label | Keywords or rules | Learned TK alone | Keywords OR learned |
|---|---|---|---|
| Safety | 20 of 21 / 3 / 0.909 | 20 of 21 / 3 / 0.909 | same as keywords |
| Defect | 35 of 139 / 17 / 0.366 | 108 of 139 / 57 / 0.711 | 111 of 139 / 53 / 0.733 |
| Counterfeit (c1) | 17 of 23 / 5 / 0.756 | 15 of 23 / 8 / 0.652 | same as rules |
| Quality | — | 11 of 35 / 15 / 0.361 | — |
| Not as described | — | 11 of 25 / 15 / 0.431 | — |

With 72 and 49 examples the two new labels cannot be learned yet; on the
team's rounds the all-rounds defect model scores 0.743 against 0.765 for the
rounds-1–3 model.

**Rules c2** (`hazard_model.EXTRA_RULES`; opt-in when this was written, served
since 8 Oct 2026, see the end of this table). A separate
`extra_tags` field carries the two labels; the badge categories, the keyword
features and the learned models are untouched. Under c2 a picture mismatch
or a disputed number no longer counts as a counterfeit report.

| Label, shipped checker + c2 | Round 5 (all products) | Round 6 (gadgets) |
|---|---|---|
| Quality | 16 of 37, 1 false, F1 0.593 | 16 of 35, 0 false, F1 0.627 |
| Not as described | 8 of 24, 3 false, F1 0.457 | 15 of 25, 4 false, F1 0.682 |
| Counterfeit, c1 → c2 | 20 of 32, 8 → 6 false, 0.667 → 0.690 | 17 → 16 of 23, 5 → 3 false, 0.756 → 0.762 |

On the team's rounds (`hazard_tagger_measurement_round{2,3}_counterfeit_c2.json`)
safety and defect are identical under c2; counterfeit on round 2 keeps its 7
of 12 and drops from 4 false flags to 3; round 3 is unchanged (0 and 0).

**Limits.** The c2 patterns were written by the assistant after reading
rounds 5 and 6, so their figures there are in-sample as well as
AI-labelled: they show the rules are precise on what was seen and still miss
about half, nothing more. The rules are not used by the app until the wording
and the two new report tiles are reviewed. (Done on 8 Oct 2026: see the end
of this table.)

**Round 7 (7 Oct 2026): more gadget training data.** 600 more gadget-only
sentences, none from rounds 1–6, 250 of them from reviews that mention
quality, size, capacity or the photos (`aspect_focus`, a broad word test, not
the checker's rules); labelled by the assistant with five labels
(`research/label_round7_ai.py`; 48 unsure): safety 8, defect 107, quality 80,
counterfeit 12, not as described 35. Owner review open, as for rounds 5–6.

The c2 rules were written before round 7 existed, so round 7 is the first
look at them on unseen sentences (still AI labels):

| Label, rules c2 | Round 7: caught / false / precision / recall |
|---|---|
| Quality | 24 of 80 / 2 / 0.92 / 0.30 |
| Not as described | 10 of 35 / 1 / 0.91 / 0.29 |
| Counterfeit | 7 of 12 / 2 / 0.78 / 0.58 |

Recall is well below the in-sample 0.46 and 0.60 of rounds 5–6, as warned:
the rules are precise and cover under a third of what reviewers write.

Dry run with round 7 added to training, scored on round 6 (cross-validated):
quality learned model F1 0.361 → 0.500 (18 of 35 caught, 19 false); not as
described 0.431 → 0.449; defect keywords-or-model 0.733 → 0.750 (false
flags 53 → 41) with 0.745 on the team's rounds; safety and counterfeit
unchanged, rules still ahead of a learned model. Across rounds 6–7 the
quality model reaches 0.599 (62 of 115, 30 false). Quality is now close to
learnable; not-as-described (84 examples) is not yet.

**8 Oct 2026: the five labels are on the report screens.** The owner approved
the wording step, so the app now serves rules c2
(`hazard_model.SHIPPED_COUNTERFEIT_RULES`; the class default stays c1 so the
figures above reproduce). The checker also returns the words behind each of
the two labels (`extra_terms`) for highlighting; a product summary carries a
`reports` block beside `flags`, which no verdict rule reads; and every label
is worded once, in `dashboard_utils.BADGE_COPY`: Safety problem, Faulty or
broken, Poor quality, Possible fake, Not as advertised.

Check made before the switch, with its rule written first (go only if safety
and defect are identical, no agreed fake report is lost and fake false flags
do not rise): the served folder under c1 and under c2 on the team's rounds
(`hazard_tagger_measurement_round{2,3}_served_c2.json`, configurations
`v9_served_c1` and `v9_served_c2`).

| Team labels | Safety: caught / false | Defect: caught / false | Fake: caught / false |
|---|---|---|---|
| Round 2, c1 | 10 of 16 / 14 | 108 of 132 / 19 | 7 of 12 / 4 |
| Round 2, c2 | 10 of 16 / 14 | 108 of 132 / 19 | 7 of 12 / 3 |
| Round 3, c1 | 14 of 19 / 14 | 153 of 179 / 15 | 0 of 0 / 0 |
| Round 3, c2 | 14 of 19 / 14 | 153 of 179 / 15 | 0 of 0 / 0 |

Passed. The defect columns are **not** a test: the served defect model was
trained on these rounds. They are here only to show that c2 leaves them
alone.

The two rule-based labels on round 8, ordinary reviews their patterns were
never fitted to (`live_check.json`, `rule_labels_round_8`; AI labels, 752
sentences, unweighted):

| Label | Caught / false / precision / recall |
|---|---|
| Poor quality | 21 of 38 / 6 / 0.78 / 0.55 |
| Not as advertised | 12 of 43 / 1 / 0.92 / 0.28 |

On ordinary reviews the report will show about half of the poor-quality
remarks and under a third of the not-as-advertised ones, with few wrong ones.
That is one reason the two labels only inform and never change a verdict.

One pattern changed after a look at a whole listing: "yung isa kong gamit 20W
lang" (my other charger is only 20W) was read as a disputed number. A guard
for the reviewer's other or older device (`extra_labels.OWN_DEVICE_BEFORE`)
removes it and changes no labelled sentence in rounds 1–8.

## Table 11j. Fix, improve or replace (7–8 Oct 2026)

The owner asked that any part of the hazard checker that performs poorly be
fixed, improved or replaced. Six steps followed. **Only step 1 rests on the
team's labels alone.** Steps 2–5 use labels proposed by the AI assistant that
the owner has not audited yet: their figures are agreement with those labels,
not accuracy, and the same assistant wrote the rules and trained the models
they score. Step 6 is an attempt that did not pass its rule.

**8 Oct 2026, later: the owner chose to proceed with these labels without
auditing them** (`PREREGISTRATION_final_v6.md`, departure 3). The review
sheets stay unfilled. That removes a pending step. It does not make the
labels human-checked, and every figure below stays agreement with AI labels.

**1. Keyword rules 0.4: a negation fault fixed (team labels).** The keyword
checker read "Wala pong sira" (nothing is broken), "walang ka sira sira",
"akala ko sira na … lowbat lang pala" and "pag nasira na" as defect reports.
Rules 0.4 follow a negation across polite particles and a repeated word, and
set aside a mistaken belief or a hypothetical
(`hazard_model.NEGATION_TRANSPARENT_04`, `MISTAKEN_BELIEF_WORDS`,
`HYPOTHETICAL_WORDS`). On the team's rounds 1–3 the keyword flags lose 3 false
flags and no real report. The learned models take keyword features, so both
were refitted with the unchanged recipe on the same rounds
(`train_rules04_refit.py` → `hazard_classifier_v2b`) and checked on round 3,
which neither version was trained on
(`hazard_tagger_measurement_round3_rules04.json`):

| Round 3, team labels | Safety: caught / false / F1 | Defect: caught / false / F1 |
|---|---|---|
| Shipped until then (v2, rules 0.3) | 14 of 19 / 14 / 0.596 | 126 of 179 / 16 / 0.785 |
| Refit (v2b, rules 0.4) | 14 of 19 / 14 / 0.596 | 126 of 179 / 15 / 0.788 |

Nothing lost and one false flag fewer, so v2b replaced v2 (four checks, all
true, recorded in the folder's manifest). The checks were written the same
day, and round 3 had already served as v2's test, so this is a second look at
that round, not a new test. `DEFAKETIVE_HAZARD_MODEL=team_labels` serves v2b.

**2. Earlier rounds re-read for training.** The team's sheets are untouched;
the re-read answers are separate files used for training only.

- Two new labels for rounds 1–4 (`label_previous_rounds_ai.py` →
  `ai_extra_labels.csv`): poor quality 119 / 70 / 67 / 26, not as advertised
  56 / 23 / 19 / 29, with 109 unsure answers.
- The team's defect and counterfeit answers for rounds 1–3, re-read under
  guide 1.4 (`ai_relabel.csv`): 83 defect answers became "poor quality" (weak
  but working, or cosmetic), 18 counterfeit answers became "not as
  advertised" (picture or number disputes), and 25 safety reports now also
  count as defects (owner's rule of 7 Oct).
- Owner sheet: `outputs/extra_labels_previous_rounds/worksheet_for_you.csv`
  (168 rows).

**3. Which kind of model** (`compare_label_models.py`,
`label_model_comparison.json`). Grouped 5-fold cross-validation over every
labelled round (1–7), F1:

| Label (reports) | Rules or keywords | LR, TF-IDF + keyword features | Linear SVM, same features | LR, TF-IDF only | LR, character n-grams | Naive Bayes, words | LR, sentence embedding + keyword features | Rules OR LR |
|---|---|---|---|---|---|---|---|---|
| Safety problem (110) | 0.687 | **0.716** | 0.711 | 0.639 | 0.635 | 0.164 | 0.625 | 0.687 |
| Faulty or broken (843) | 0.508 | 0.748 | 0.732 | 0.729 | 0.714 | 0.652 | 0.662 | **0.757** |
| Poor quality (437) | 0.736 | 0.591 | 0.577 | 0.587 | 0.580 | 0.366 | 0.467 | **0.753** |
| Possible fake (150) | **0.709** | 0.652 | 0.635 | 0.587 | 0.595 | 0.128 | 0.462 | 0.709 |
| Not as advertised (212) | 0.712 | 0.519 | 0.543 | 0.539 | 0.532 | 0.205 | 0.422 | **0.732** |

No other model type beats the design in use (logistic regression on TF-IDF
and keyword features, combined with the keywords), so the type is kept. For
fakes and the two new labels every learned model loses to written rules, so
those labels are rule-based: the learned models for them were dropped and the
patterns in `defaketive_sentiment/extra_labels.py` rewritten on rounds 1–6
(in-sample there). On round 7, which the patterns were not fitted to: poor
quality 49 of 80 caught with 2 false (F1 0.75; 24 of 80 before the rewrite),
not as advertised 16 of 35 with 2 false (0.60; 10 of 35 before). Adding a
learned model to those rules gains about 0.02 for 13–17 more false flags and
was not taken.

**4. The defect model retrained, and why the first retraining was replaced.**
`train_candidate_v4.py` trains the design in use on rounds 1–3 (re-read) and
5–7: 3,643 sentences, 843 reports (`hazard_classifier_v4`). Cross-validated on
rounds 5–7 it scores 0.773 against 0.630 for the team-label models, and a
copy trained without round 3 finds 132 of the team's 179 round-3 reports with
19 false flags (0.800) against 126 and 17 (0.783). (This script's loader
scores 460 round-3 sentences, the measurement script in step 1 scores 452,
hence 17 false flags here and 15 there. Compare within one table only.)

v4 was made the default and then tried on a whole listing. On 100 reviews of
an Anker charger, all rated 4–5 stars, it added 20 defect flags, about half of
them wrong ("Gumagana naman sya.", "Fast charging naman pala."). The labelled
rounds come from complaint-heavy strata and hold few such plain remarks. In
unused 4–5 star gadget sentences the keywords flag 2.0%, the team-label
models 4.2% and v4 6.8%.

**Round 8** (`ordinary_review_pool.py`, `build_ordinary_round8.py`,
`label_round8_ai.py`): ordinary gadget reviews no round had used, 50 listings
split by listing into 42 "mining" listings for training and 8 "live check"
listings no model is trained on. 752 sentences labelled by the assistant (66
unsure; owner sheet 116 rows): defect 375, poor quality 38, not as advertised
43, safety 2, counterfeit 0.

| Round 8 stratum | Sentences in the pool | Labelled |
|---|---|---|
| Mining: flagged by a model, not by a keyword, 4–5 stars | 204 | 150 |
| Mining: the same, 1–3 stars | 702 | 150 |
| Mining: flagged by neither model | 6,716 | 60 |
| Live check: flagged by either model | 282 | 282 |
| Live check: flagged by neither | 1,118 | 110 |

`train_candidate_v4.py --ordinary` builds `hazard_classifier_v5`: the same
design with the mining sentences added (4,003 sentences, 1,010 reports,
threshold 0.501), and one new rule. **In a review rated 4 or 5 stars the
model needs a higher score, 0.637, before it adds a flag the keywords did not
raise.** Keyword flags, 1–3 star reviews and reviews without a rating are
unaffected. The bar is the threshold at which right added flags outnumber
wrong ones by the most, on out-of-fold scores of the training sentences: in
4–5 star reviews 101 right and 97 wrong at the ordinary threshold, 76 and 33
at the bar. Resampling the reviews 500 times puts the bar between 0.630 and
0.750 (10th to 90th percentile, median 0.654). The same rule leaves 1–3 star
reviews at the ordinary threshold (670 right, 171 wrong there). A stricter
sentiment gate instead (negative sentences only) was tried on the mining
sentences: the same precision, but it kept a third of the right flags where
the score bar keeps two thirds, so the score bar was used.

**5. Live check** (`live_check.py`, `live_check.json`): the 8 listings no
version was trained on, each sentence tagged with its review's rating, as the
app does. 392 labelled sentences stand for 1,400 and hold about 226 defect
reports.

| Checker | All reviews: flags / real / false / missed | Precision / recall / F1 | 4–5 stars: flags / real / false | 1–3 stars: flags / real / false |
|---|---|---|---|---|
| Keywords alone (rules 0.4) | 120 / 110 / 10 / 116 | 0.917 / 0.486 / 0.635 | 22 / 19 / 3 | 98 / 91 / 7 |
| Team-label models (v2b) | 216 / 168 / 48 / 58 | 0.778 / 0.742 / 0.760 | 53 / 29 / 24 | 163 / 139 / 24 |
| First retraining (v4) | 272 / 204 / 68 / 22 | 0.750 / 0.901 / 0.819 | 70 / 33 / 37 | 202 / 171 / 31 |
| v5 without the 4–5 star bar | 249 / 196 / 53 / 30 | 0.787 / 0.866 / 0.825 | 61 / 32 / 29 | 188 / 164 / 24 |
| **v5 (served)** | 223 / 193 / 30 / 33 | 0.866 / 0.853 / 0.859 | 35 / 29 / 6 | 188 / 164 / 24 |

On the team's round-3 labels, held out for both, the v5 design trained
without round 3 finds 129 of 179 reports with 17 false flags (0.794) against
126 and 17 (0.783) for the team-label models. On the Anker listing, through
the app's own path, v5 raises 2 defect flags (both wrong, one of them
"Charging Speed:100") where v4 raised 20 and the team-label models 3.

**6. Safety model: retraining tried, not adopted**
(`check_safety_retrain.py`, `safety_retrain_check.json`). The served safety
model learned from round 1 only (30 reports), and half of its flags on round
3 are wrong. The same design was trained on more rounds and scored on the
team's round-3 safety answers, with the rule written first: replace the
served model only if a variant finds at least as many reports and raises
fewer false flags.

| Round 3, team labels (19 reports) | Reports it was trained on | Caught / false / F1 |
|---|---|---|
| Served safety model | 30 (round 1) | 14 / 15 / 0.583 |
| Keywords alone | none | 14 / 15 / 0.583 |
| Retrained, team labels of rounds 1–2 | 46 | 14 / 162 / 0.144 |
| Retrained, rounds 1–2 and the AI-labelled rounds 5–8 | 92 | 13 / 8 / 0.650 |

The last variant halves the false flags but loses one real report. The rule
puts a missed hazard first, so the served safety model stays. The third row
shows the threshold rule (highest threshold with out-of-fold recall of 0.80)
breaking down: with 46 reports it falls to 0.05. (This loader finds 15 false
flags for the served model where step 1's script finds 14, for the reason
given in step 4.) Round 8 holds two safety reports, too few to say anything
about ordinary reviews.

**7. Every label trained on every round** (`train_all_rounds.py`,
`all_rounds_training.json`; 8 Oct 2026, asked by the owner). The design in
use was trained for each of the five labels on all rounds but one, six times
(rounds 2, 3, 5, 6, 7 and round 8's live-check listings left out in turn),
alone and combined with the label's keywords or rules. Round 4's fake labels
joined the training data, two of them re-read under guide 1.4. Rule written
first: replace what is served only if the summed F1 is at least 0.02 higher
and, for safety and fakes, no real report the served checker finds in the
team's rounds 2–3 is lost. Caught / false (F1), summed over the left-out
rounds; truth is the team's labels where they exist and AI labels otherwise:

| Label | Served now | Model alone | Rules OR model | Rules AND model | Result |
|---|---|---|---|---|---|
| Safety problem | 66 / 42 (0.698) | 62 / 24 (0.743) | 67 / 43 (0.702) | 62 / 24 (0.743) | Kept: the model loses 3 real reports on the team's rounds |
| Poor quality | 190 / 11 (0.734) | 194 / 162 (0.577) | 203 / 28 (0.741) | 188 / 11 (0.729) | Kept: gain under 0.02 |
| Possible fake | 50 / 14 (0.699) | 49 / 25 (0.640) | 50 / 14 (0.699) | 50 / 13 (0.704) | Kept: gain under 0.02 |
| Not as advertised | 76 / 6 (0.652) | 72 / 63 (0.503) | 88 / 16 (0.690) | 74 / 6 (0.641) | Meets the rule: rules OR model |

Faulty or broken is not in the table because its served model is already
trained on all rounds. Left out in turn, the same design finds 131 of 179
with 18 false flags on the team's round 3, and 97 of 132 with 36 on round 2.

Reading. For three labels a model trained on everything still does not beat
what is served. For "not as advertised" the rule is met, narrowly: 12 more
real reports for 10 more false flags, so about half of what the model adds is
wrong, where the rules alone are right 76 times in 82. The rule did not ask
how often the additions are right, which is its weakness. It is not switched
on, pending the owner's decision. For safety the model would cut false flags
from 42 to 24 and miss 4 more real reports, 3 of them in the team's rounds.
The rule keeps the served model; whether fewer false alarms are worth missed
reports is the owner's call.

Two looser safety filters were tried after that result
(`train_all_rounds.py --lenient-safety`): one set to keep every real report
the keywords flag in the training rounds removes nothing (66 / 43), and one
set to keep 95 in 100 finds 64 with 30 false flags but loses 2 real reports
in the team's rounds. Neither meets the rule. On this data, fewer false
safety flags always cost real reports.

The owner then asked for the best model to be used and left the choice to
the assistant (8 Oct 2026). Outcome: nothing changes. The served safety model
stays, because no option keeps every real report. "Not as advertised" stays
rule-based although rules OR model met the F1 rule: this is a departure from
that rule in the cautious direction, made because half of the model's
additions are wrong, the label has no human-labelled sentence to check it
against, and its gain on ordinary listings is two more reports for two more
false flags.

**8. Written (synthetic) training sentences** (`synthetic_data.py`,
`train_with_synthetic.py`, `synthetic_training.json`; 8 Oct 2026). The
adviser approved AI labels and synthetic data (departure 4), so the assistant
wrote 1,368 training sentences: short reports of each problem and look-alikes
that are not reports (denials, praise, questions, figures of speech), three
sentences from each of 456 core phrases. They are training only: never
scored, never used to set a threshold, never shown by the app. Same left-out
rounds as step 7, real sentences only. Rule written first: as in step 7, and
in addition at least 70 in 100 of the flags a design raises beyond the served
checker must be right. Caught / false (F1), summed:

| Label | Served | Best design trained with the written sentences | Passes? |
|---|---|---|---|
| Safety problem | 66 / 42 (0.698) | Filter keeping 95 in 100 flagged reports: 66 / 34 (0.729) | Yes |
| Poor quality | 190 / 11 (0.734) | Rules OR model: 215 / 50 (0.739) | No: 25 of its 64 additions are right |
| Possible fake | 50 / 14 (0.699) | Rules AND model: 50 / 13 (0.704) | No: gain under 0.02 |
| Not as advertised | 76 / 6 (0.652) | Rules OR model: 87 / 16 (0.685) | No: 11 of its 21 additions are right |

Faulty or broken: with the written sentences the left-out figure is 686 / 170
against 675 / 169 without, and on the live-check listings false flags do not
fall, so no new defect model was built.

Safety, in detail. The result took three looks at the same left-out rounds,
which is a limit. The best-F1 filter kept every report in the team's rounds
and cut false flags (60 / 27 overall) but lost six reports in the AI-labelled
rounds and missed the margin. The two looser filters of step 7 were then
tried, and the one that keeps 95 in 100 of the flagged real reports passed.
One safeguard was added before serving: the filter may not remove a flag
stronger than mild heat (manifest key `filter_spares_severity_above`), so a
fire, smoke or shock report always stays. The figures above include it. On
the team's labels: round 2 finds 10 of 16 either way, false flags 15 → 12;
round 3 finds 14 of 19 either way, false flags 15 → 11. Without the written
sentences the same filter lost two real reports on those rounds (step 7).

In cross-validation over every real round the frozen filter keeps 88 of the
92 real reports the keywords flag and removes 19 of their 64 false flags. The
four it drops are all mild-heat remarks ("mabilis lang uminit", "Lakas pa mag
init ng phone"); two of them are team-labelled sentences of round 1. On 9,662
ordinary unused sentences the keywords raise 7 safety flags and the filter
keeps all 7: its effect is on heat words in harmless sentences, which
ordinary reviews rarely contain.

**9. Raising the low F1 scores: a rule revision and a fine-tuned
transformer** (8 Oct 2026; `improve_label_rules.py`,
`label_rules_revision.json`, `finetune_label_transformer.py`,
`label_transformer_experiment.json`). Poor quality, possible fake and not as
advertised have the lowest F1 because their rules are precise and miss
reports. Split fixed first: one review in three, by a hash of its id, is held
out and only scored; the other two thirds ("dev") were read. Rule fixed
first: a label's revision is kept only if held-out F1 rises and the share of
flags that are right falls by at most 0.05. Caught / false, F1:

| Label | Dev, before → after | Held out, before | Held out, after | Kept? |
|---|---|---|---|---|
| Poor quality | 187 / 9, 0.754 → 231 / 13, 0.849 | 93 of 175 / 5, 0.681 | 105 of 175 / 8, 0.729 | Yes |
| Possible fake | 93 / 21, 0.713 → 108 / 15, 0.800 | 46 of 64 / 7, 0.786 | 46 of 64 / 6, 0.793 | Yes |
| Not as advertised | 86 / 4, 0.667 → 127 / 6, 0.844 | 47 of 87 / 4, 0.681 | 47 of 87 / 5, 0.676 | No |

The not-as-advertised revision is the clearest overfit in this file: 41 more
reports caught where the patterns were written, none where they were not. It
was removed, and that label's patterns are as before. The poor-quality
revision (noise, lag, stiffness, looks used, marks and stains, with two
guards: a condition is not a complaint, and slow charging is a defect)
generalises about half as well as it fits. The fake revision (more ways of
calling something a copy, a brand denial, and guards for "fake reviews",
"fake seller" and questions) changes one held-out sentence. The served
version string gains `r2`, so saved reports are redone once.

A fine-tuned XLM-RoBERTa was then scored on the same held-out third: the
sentiment model's encoder, one model with five yes/no outputs, trained on
four fifths of the dev part plus the synthetic sentences, thresholds from the
remaining fifth, 32 minutes on the GPU.

| Label | Rules | Transformer alone | Rules OR transformer |
|---|---|---|---|
| Poor quality | 105 / 8, 0.729 | 123 / 115, 0.596 | No addition met the 80-in-100 bar |
| Possible fake | 46 / 6, 0.793 | 47 / 23, 0.702 | 51 / 21, 0.750 |
| Not as advertised | 47 / 4, 0.681 | 49 / 27, 0.601 | 55 / 12, 0.714 (8 additions right, 8 wrong) |
| Faulty or broken (keywords as the rules) | 149 / 26, 0.488 | 353 / 97, 0.797 | 297 / 47, 0.761 |
| Safety problem (keywords as the rules) | 33 / 21, 0.725 | 23 / 14, 0.622 | No addition |

It beats the rules on none of the three labels, and for faulty-or-broken it
only matches the far smaller TF-IDF design. Nothing was saved or served.

Four ways of finding more of these reports have now been tried: a model on
all rounds, written sentences, new patterns and a transformer. Only new
patterns helped, and only for poor quality. The limit is the data: a few
hundred examples per label, worded in many ways, from one labeller. More real
examples of these reports would be the next thing to try, not another model.

**10. More data for the three weak labels** (8 Oct 2026, night;
`build_round9.py`, `label_round9_ai.py`, `train_with_more_data.py`,
`more_data_training.json`, `synthetic_data.py --v2`, `rule_revision_3/`,
`label_self_check/`). The owner asked for exactly that: more data. Fixed
before anything was built:

- The yardstick is the held-out third of step 9 and does not move: 1,748
  sentences of rounds 1–8. Later rounds and written sentences are training
  material only.
- Something is switched on only if held-out F1 is at least 0.02 above the
  served rules and at least 70 in 100 of the flags it adds are right.
- Real sentences are labelled in batches of about 1,000, up to three, and
  labelling stops early if a batch adds less than 0.01 F1 to every label.

*Round 9: 989 more real sentences.* Pool: 15,935 gadget sentences no round
had used (the training listings of our own scans and the public Shopee set;
the live-check listings stay untouched). A pointer model (the design in use,
trained on the dev part and the written sentences) chose what to read; the
assistant then labelled every sentence without seeing why it was picked.
Result: poor quality 208, not as advertised 177, possible fake 35, faulty or
broken 161, safety 0; 189 sentences carry an unsure answer.

| Why the sentence was picked | Read | Real reports of that label |
|---|---|---|
| Model scores it high for poor quality, rules silent | 200 | 92 |
| Model scores it high for not as advertised, rules silent | 250 | 85 |
| Model scores it high for possible fake, rules silent | 150 | 10 |
| Rules flag poor quality | 60 | 51 |
| Rules flag not as advertised | 60 | 56 |
| Rules flag possible fake | 19 | 17 |

So on text nobody had seen, the rules' flags were right about 9 times in 10,
and the reports they miss do exist in quantity. Fakes are the exception: the
unused gadget reviews hold few.

*Written sentences, second set.* 1,146 sentences
(`research/synthetic/synthetic_sentences_v2.csv`), each written whole as a
variation on a real wording or, for the look-alikes, on a sentence a trained
model wrongly scored high: 247 poor-quality, 234 fake and 234
not-as-advertised reports, and 431 look-alikes. Fifteen lines were left out
because they matched a real sentence word for word.

*As training data.* The design in use, as an add-on to the rules: it may add
a flag only above the score at which 80 in 100 of its additions are right on
a validation fifth. Held-out third, caught / false, F1; in brackets the F1 of
the model on its own.

| Trained on | Poor quality | Possible fake | Not as advertised |
|---|---|---|---|
| (rules alone) | 105 / 8, 0.729 | 46 / 6, 0.793 | 47 / 4, 0.681 |
| Rounds 1–8 | 105 / 8, 0.729 (0.497) | 46 / 6, 0.793 (0.650) | 48 / 9, 0.667 (0.388) |
| + round 9 | 106 / 8, 0.734 (0.501) | 46 / 6, 0.793 (0.689) | 47 / 4, 0.681 (0.521) |
| Rounds 1–8 + both written sets | 105 / 8, 0.729 (0.543) | 46 / 6, 0.793 (0.652) | 50 / 8, 0.690 (0.503) |
| + round 9 + written set 1 | 109 / 9, 0.744 (0.534) | 46 / 9, 0.773 (0.651) | 48 / 5, 0.686 (0.536) |
| + round 9 + both written sets | 110 / 10, 0.746 (0.558) | 46 / 8, 0.780 (0.667) | 48 / 5, 0.686 (0.535) |

More data does make the model better on its own, most of all for not as
advertised (0.388 to 0.535). It stays far below the rules, and what it adds
to them is small: at best 5 more poor-quality reports for 2 more false flags.
Nothing reaches the 0.02 bar. Poor quality comes closest (0.746 against the
0.749 needed), on seven sentences.

*As material for new patterns* (`rule_revision_3/`). Step 9 showed that
patterns fit the sentences they are written from. So each pattern was checked
on sentences it was not written from, and kept only if it found at least one
report there and was right at least 3 times in 4: the 22 not-as-advertised
patterns dropped in step 9 (written from dev) were checked on round 9; 31
not-as-advertised and 26 poor-quality patterns written from the round 9
misses were checked on dev. Two sets were then scored on the held-out third,
both fixed first: A, the patterns that passed (1 + the shared number and
feature pieces of the dropped revision, 4 and 2); B, A plus every round-9
pattern that was right at least twice where written and wrong nowhere.

| | Where written (round 9) | Held-out third | Held-out, gadgets only |
|---|---|---|---|
| Not as advertised, served | 65 / 4 | 47 / 4, 0.681 | 29 / 2, 0.682 |
| Not as advertised, A | 85 / 4 | 47 / 5, 0.676 | 29 / 3, 0.674 |
| Not as advertised, B | 112 / 4 | 48 / 5, 0.686 | 30 / 3, 0.690 |
| Poor quality, served | 69 / 10 | 105 / 8, 0.729 | 59 / 6, 0.690 |
| Poor quality, A | 77 / 11 | 105 / 10, 0.724 | 59 / 7, 0.686 |
| Poor quality, B | 155 / 11 | 110 / 12, 0.741 | 62 / 8, 0.705 |

The widest set catches 86 more poor-quality and 47 more not-as-advertised
reports in the sentences it was written from, and 5 and 1 more in the
held-out third. Of 57 patterns written from round 9, 10 found any report in
the dev part and 6 passed the check. Neither set reaches the bar and the served rules are unchanged
(`extra_labels.EXTRA_RULES_REVISION` is still `r2`). The 16 fake reports the
rules missed in round 9 are each worded their own way; no pattern was tried.

*How consistent is the answer key?* (`label_self_check/`) 315 dev sentences
(up to 50 labelled yes per label, 15 that the rules flag and the labels call
nothing, 150 others at random) were labelled a second time by the assistant
with the first labels hidden.

| Label | Yes both times | Yes first time only | Yes second time only | Agreement (F1), these sentences |
|---|---|---|---|---|
| Poor quality | 56 | 5 | 15 | 0.848 |
| Not as advertised | 54 | 1 | 8 | 0.923 |
| Possible fake | 49 | 11 | 6 | 0.852 |

Weighted back to all dev sentences the agreement is roughly 0.79, 0.87 and
0.79, with wide margins: a single random sentence stands for 16. Two things
follow. The same labeller reading the same sentence twice agrees with itself
at about 0.8 to 0.9, so no checker can be expected to score higher than that
against these labels; the rules are at 0.73, 0.68 and 0.79. And some early
fake labels are wrong on their face: rounds 1–2 carry over the team's "fake"
answer on sentences such as "ok lng" and "ok naman cya" (9 of the 60
fake-labelled sentences in the sample), which no sentence checker can find.
Of the rules' poor-quality flags in the sample, 41 of 50 are right by the
first labels and 48 of 50 by the second reading.

*The fine-tuned transformer again, with round 9*
(`finetune_label_transformer.py --tag round9`,
`label_transformer_experiment_round9.json`; same settings and seed as step 9,
trained on the dev part, round 9 and the first written set; 42 minutes).
Held-out third, caught / false, F1:

| Label | Rules | Transformer alone, step 9 → with round 9 | Rules OR transformer, step 9 → with round 9 |
|---|---|---|---|
| Poor quality | 105 / 8, 0.729 | 0.596 → 124 / 91, 0.636 | no addition met the 80-in-100 bar, either time |
| Possible fake | 46 / 6, 0.793 | 0.702 → 51 / 28, 0.713 | 0.750 → 49 / 12, 0.784 (3 additions right, 6 wrong) |
| Not as advertised | 47 / 4, 0.681 | 0.601 → 51 / 18, 0.654 | 0.714 → 54 / 7, 0.730 (7 additions right, 3 wrong) |

The transformer gains more from the new sentences than the TF-IDF design
does, and for not as advertised "rules OR transformer" meets the rule: F1
0.049 above the rules, 7 of 10 additions right. It meets it at the edge, on
ten sentences and one seed; a repeat with another seed and a run with both
written sets were started to see whether it holds
(`--tag round9_seed7 --seed 7`, `--tag round9_written2 --written v1,v2`).

The owner's answer (8 Oct, 22:25, before either repeat had finished an
epoch): go with the recommendation, which was to serve it only if the result
holds. "Holds" was fixed then: the repeat with seed 7 must meet the same rule
for not as advertised (F1 at least 0.02 above the rules, at least 70 in 100
additions right). The second written set joins the training data only if its
run meets the rule too and scores at least as high. The model that is saved
for the app is scored once more on the held-out third and must meet the rule
itself. If the repeat fails, nothing is served and no more batches are
labelled.

The repeat (`label_transformer_experiment_round9_seed7.json`), rules OR
transformer on the held-out third:

| Label | Rules | Seed 42 | Seed 7 |
|---|---|---|---|
| Poor quality | 105 / 8, 0.729 | no addition allowed | 111 / 9, 0.752 (6 additions right, 1 wrong) |
| Possible fake | 46 / 6, 0.793 | 49 / 12, 0.784 (3 right, 6 wrong) | 46 / 10, 0.767 (0 right, 4 wrong) |
| Not as advertised | 47 / 4, 0.681 | 54 / 7, 0.730 (7 right, 3 wrong) | 57 / 11, 0.736 (10 right, 7 wrong) |

It does not hold. For not as advertised the F1 gain repeats (about 0.05 both
times) but the second run's additions are right 10 times in 17, under the 70
in 100 required; over both runs, 17 in 27. Which label passes depends on the
seed: poor quality fails with one and passes with the other. That is what a
result resting on ten or twenty sentences looks like.

The run with both written sets
(`label_transformer_experiment_round9_written2.json`) fails as well: not as
advertised 54 / 11, 0.711, 7 additions right and 7 wrong; poor quality 109 /
10, 0.742; possible fake 51 / 17, 0.773.

What serving it would have cost, measured on the development PC with the
served int8 sentiment model standing in for a slimmed label model: about 127
ms more per review on top of 111 ms (a review has 1.8 sentences on average
and each would go through the second model), and about 175 MB more memory,
taking the app from about 352 MB to about 530 MB.

*Outcome.* Nothing is switched on, and batches two and three were not
labelled. With the TF-IDF design and with new patterns nothing reached the
bar: what the data shows is a long tail. These complaints are worded
differently almost every time, so a thousand more labelled sentences teach
the rules and a word-counting model the wordings in those sentences and few
of the wordings in the next thousand. The transformer, which does not depend
on exact wording, is the one design that more data visibly helps (not as
advertised alone: 0.601, then 0.654 and 0.663 with round 9), and its F1 as an
add-on is above the rules both times. But about 4 in 10 of the flags it adds
are wrong, it would double the analysis time, and the label it helps only
informs the shopper. Round 9, its labels and both written sets stay in the
repository as training material.

**11. Two false "faulty or broken" flags found in the app** (11 Oct 2026;
`check_keyword_guards.py`, `keyword_guards_check.json`). The owner found two
four-star reviews flagged for the word "sira" where the product was not at
fault:

- "sira ,punit at wasak2 na ung box": damage to the box. The packaging rule
  looks three words either side of the damage word and the box is the sixth.
- "yung una kong order tumagal ng 2 years bago na sira ang turbo": the
  earlier unit lasted two years.

Revision `g1` of keyword rules 0.4 (`hazard_model.KEYWORD_GUARDS_REVISION`)
adds two readings of a damage word: a run of damage and joining words that
reaches a packaging word is about the packaging; a damage word after "N
years … bago / before / until" is wear after long use. Years only: "3 months
pa lang sira na" stays a report. Once the keyword is read this way the
learned model's score for both sentences falls from about 0.77 to under 0.2,
so it does not put the flag back.

Rule fixed before the check: keep the guards only if they add no flag and
remove at most one right flag for every three wrong ones.

| Where | Sentences or reviews | Flags changed |
|---|---|---|
| Every labelled sentence with a faulty-or-broken label | 5,384 | 0 |
| Every review of the 68 saved listings | 7,112 (553 with a damage word) | 2, the two above |

So the guards cost nothing that can be measured and fix exactly what was
reported. They are narrow by design, and this is the limit of the method: a
keyword flags "sira" whatever is broken, and each harmless use has to be
written down before it is recognised. Others will turn up. The typical
rates of Table 12c were measured again with the revised checker, as the
version test requires.

**Decision.** `hazard_classifier_v6` is the app default: the defect model of
`hazard_classifier_v5`, unchanged, and the safety filter of step 8. Like v5
it is served on the owner's instruction and not by a pre-registered test
(`PREREGISTRATION_final_v6.md`, departures 2 and 4). `DEFAKETIVE_HAZARD_MODEL` selects the others: `team_labels`
(v2b), `keywords` (no learned model); `five_labels` is now the same as the
default.

**Limits.**

- Steps 2–5 are AI-labelled, and v4 and v5 learned from the same assistant's
  labels they are scored against, which favours them over the team-label
  models. The round-3 line is the only human-labelled comparison, and there
  the difference is small: three more reports, the same false flags.
- The live-check listings are where the 4–5 star problem was first seen (the
  split for v2b and v4 was read before the bar was designed), and they were
  scored more than once. The bar's value was set on other listings, but these
  eight are not a clean test of the idea.
- Eight listings, one sample. The "missed" column leans on 2 reports found
  among 110 sampled unflagged sentences, each standing for about 10.
- The bar uses the star rating, so a fault described mildly in a 5-star
  review is less likely to be found: in the live check 3 of 32 such reports
  were lost for 23 fewer false flags.
- Plain remarks can still be flagged: "Gumagana naman sya." is flagged in a
  1–3 star or unrated review, and "Mabilis ma charge, matagal malowbatt."
  (praise) scores 0.94.

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

## Table 12b. The verdict each saved listing gets (8 Oct 2026)

All work since Table 11c was on single sentences. `research/verdict_check.py
--saved` puts every saved listing with 30 or more written reviews through the
app's own path with the served checker (68 listings, 7,192 reviews;
`verdict_check_saved_scans.json`).

| Verdict | As the app shows it today | As if each sample were a complete listing |
|---|---|---|
| Not enough reviews to judge | 62 | 3 |
| No warning signs found | 2 | 7 |
| Reviewers report fakes or defects | 0 | 35 |
| Some reviewers mention safety concerns | 0 | 8 |
| Reviewers report safety problems | 4 | 15 |

- 65 of the 68 scans were collected one star level at a time without the
  listing's per-star counts, so the app rightly marks them incomplete. The
  saved catalogue therefore says almost nothing at verdict level; only fresh
  scans exercise the verdict rules.
- The right-hand column is too harsh by construction, because low-star
  reviews are over-represented, and it judges no listing. It shows that,
  unweighted, the "faulty" share has a median of 21% and its lower bound
  clears the 3% floor for 57 of 68 listings. The fake floor (2%) is cleared
  by 2.
- The two listings that carry per-star counts are weighted back to the
  listing. There the faulty rate is 6.0% and 3.9%, with lower bounds of 1.9%
  and 1.3%, and both read "No warning signs found". Weighting, not the
  floor, is what keeps the warning rare. Two listings are too few to say
  more.
- Fresh scans of a fixed set of 28 listings
  (`research/scan_reference_listings.py`, selection rule written first) are
  needed to measure this properly and to build the typical rates that the
  "Caution" verdict is meant to compare against. Two trial scans on 8 Oct
  2026 collected no reviews: the product page opened but the review section
  never loaded for the automated browser. Not retried; open.

## Table 12c. What is typical, measured (8–9 Oct 2026)

The fresh scans turned out not to be needed. The saved listings have real
reviews from every star level (about 40 five-star reviews each); what they
lacked was the number of ratings per star level, five numbers a listing.
`research/fetch_star_totals.py` read those from Lazada's review feed, the way
the app's own scan does, for all 67 saved Lazada listings without collecting
a review (`research/data/reference_scans/star_totals.json`). Lazada did not
block it.

`research/verdict_check.py --star-totals` then weights every review by how
many ratings its star level stands for, with the app's own weights
(`analyze_reviews._stratum_weights`). Rule fixed first: a listing is used
only if the sampled star levels hold 90 in 100 of its ratings and its rating
count moved by no more than a quarter since the scan. 61 of 68 listings are
usable (6 lack a five-star sample, 1 is not on Lazada); 59 after counting an
item scanned under two links once.

`research/build_reference_baselines.py` (`reference_baselines_report.json`),
choices fixed before the leave-one-out counts: typical is the median
listing's rate; the app's rule is unchanged (warn when the lower bound of a
listing's rate is above the larger of typical and the floor); each listing
is judged against the median of the others.

| | Faulty or broken | Possible fake |
|---|---|---|
| Typical listing (median), share of reviews flagged | 4.7% (95% interval 3.3%–6.0%) | 0.0% |
| Mean | 6.4% | 0.2% |
| 75th / 90th percentile | 8.2% / 14.5% | 0.03% / 0.3% |
| Listings with none at all | 0 of 59 | 43 of 59 |
| Placeholder line | 3% | 2% |
| Listings warned under the placeholder | 15 of 59 | 0 |
| Listings warned under the measured line, leave-one-out | 10 of 59 | 0 |

The placeholder for faulty was below what is normal: one listing in four
would have been warned for an ordinary rate. With at least 10 usable
listings (the rule set beforehand), the measured line replaces it:
`sentiment-analysis/models/reference_baselines_v1.json`, type `gadget`, which
`ProductBadgeAggregator` now uses for a listing that names no product type
(`default_product_type`). For fakes typical is zero, so the 2% floor remains
the line. The baseline's version joins the checker's version string, so
saved reports are redone once, and a test fails if the checker changes
without the rates being measured again.

Verdicts for the 61 usable listings, weighted (two are the same item twice):

| Verdict | Placeholder line | Measured line |
|---|---|---|
| No warning signs found | 29 | 31 |
| Reviewers report fakes or defects | 8 | 6 |
| Some reviewers mention safety concerns | 9 | 9 |
| Reviewers report safety problems | 12 | 12 |
| Not enough reviews to judge | 3 | 3 |

Limits.

- The rates are the checker's flags, not verified faults, so the line is in
  the checker's own units. That is the right unit for the comparison and the
  wrong one for a claim such as "4.7% of gadgets are faulty".
- Star totals were read on 8 October and the reviews were collected in
  September. A listing's five-star level is about 40 reviews standing for
  thousands of ratings, so one listing's rate is rough; the median over 59
  is not.
- The listings were chosen for labelling, not at random, and lean to audio,
  chargers and power banks.
- **Safety is the open part.** 21 of 61 listings carry a safety message. The
  alert counts reports and does not use a rate, and these samples read up to
  40 one-star reviews of listings with thousands of ratings, so one or two
  burn reports are easy to find. Most of the evidence read is real ("smell
  burning", "umusok", "nasunog agad", "sumabog"). Some is not: "grounded"
  said of a buzzing sound, "kuryente" where no shock is meant, and a report
  about the product this one replaced. Whether two reports should carry the
  same message on a listing of 40,000 ratings as on one of 200 is a design
  question that was not decided here.

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
- ~~Product-type baselines for counterfeit and defect.~~ Measured for gadgets
  as one type (Table 12c). Still open: separate types (59 listings are too
  few to split), and the safety alert on large listings (Table 12c, limits).
- Normalized-text ablation for the transformer, to separate architecture from
  preprocessing.
- Ceiling estimate from the blind 30-row sample
  (`research/neutral_ceiling_check.csv`), which determines whether the
  remaining 594 errors are worth pursuing.
- **AI labels accepted unaudited** (owner, 8 Oct 2026; departure 3 of
  `PREREGISTRATION_final_v6.md`). The `worksheet_for_you.csv` files of rounds
  5–8 and of `outputs/extra_labels_previous_rounds/` stay unfilled, so every
  figure in Tables 11h–11j, except step 1 of 11j and the lines scored on the
  team's labels, is agreement with one labeller's AI-proposed labels, with no
  second opinion. A fresh round labelled by two team members would be the
  first real test of `hazard_classifier_v5` and of the two rule-based labels.
- The third annotator's sheet for the 65 disputed fake sentences
  (`outputs/adjudication_counterfeit/adjudication_sheet.csv`) is still open.
  Team-label fake scores use only sentences both annotators agreed on.
- The landing page still says "3 separate checks". Three labels drive
  warnings; "poor quality" and "not as advertised" appear in the report only.
- **The early fake labels need a second reading.** Rounds 1–2 carry the
  team's "fake" answer on sentences that say nothing about fakes ("ok lng";
  Table 11j, step 10). Re-reading them would change the held-out third that
  steps 9 and 10 are scored on, so it was not done there. Such labels count
  as misses for any checker that reads one sentence at a time, so the fake
  scores in this file are somewhat lower than the checker deserves.
