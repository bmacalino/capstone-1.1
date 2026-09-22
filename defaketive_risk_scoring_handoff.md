# DeFaketive – Risk Scoring Handoff (chat memory)

Written for another AI assistant continuing this work. Conversation date: 20 Sept 2026.
**Status: planning only. Nothing has been built, run, or validated.**
Updated after decisions on scraping scope (full scrape, star filter), the primary ML technique (smoke-term scoring), and the expected app output.

Provenance labels used below:
- **[PUBLISHED]**: comes from a paper or official source I read or saw in search results
- **[MY DESIGN]**: proposed by the previous assistant, unvalidated, needs justification with data
- **[UNVERIFIED]**: could not be confirmed. Check before relying on it

---

## 1. Project and user

- **Project:** DEFAKETIVE, a capstone proposal (Capstone Project 1, BSIT Data Analytics, Pangasinan State University – Urdaneta City Campus, May 2026; file revision dated 6-9-2026). Full title: "A Counterfeit and Defective Product Detection Using Sentiment Analysis with Recommendation System."
- **Team:** 5 proponents listed in the proposal. The user said their group of **3** members would do the labeling and rating work.
- **Domain:** electronic gadgets (chargers, power banks, earphones, phones). Reviews in **Tagalog/Taglish**. Platforms named: Shopee, Lazada, TikTok Shop, Temu PH (the user first mentioned Lazada; I suggested starting with one or two).
- **User's current proposal design:** lexicon-based sentiment (NLTK, custom Taglish dictionary), keyword-based failure detection, weighted-sum risk score, recommender, Playwright + BeautifulSoup scraping, Flask backend, MySQL/Firebase, SUS/UMUX evaluation.
- **Web app flow the user wants:** user pastes a product link → live scraper runs → app shows whether the product is safe to buy. **Main concern: how the app computes that verdict.**
- User asked to stay in **plan mode**. Do not start coding unless asked.

## 2. Techniques discussed

- Baseline: TF-IDF (char n-grams) + logistic regression. Better: fine-tuned XLM-R or another multilingual/Tagalog transformer. LLM zero/few-shot for bootstrapping labels.
- Sentiment alone is a poor detector of product safety hazards. Aspect-based or category-specific detection is better.
- **FiReCS** (Hugging Face: `ccosme/FiReCS`; also `ccosme/SentiTaglishProductsAndServices`): Taglish product/service review corpus, over 10k reviews **[PUBLISHED]**. The paper reports XLM-R best (about 0.84 accuracy/F1) and lexicon tools built for single languages performing poorly on code-switched text **[PUBLISHED]**. The newer dataset card lists **four classes (Positive, Negative, Neutral, Mixed)**, 10,510 reviews from Google Maps and Shopee PH. It appears to be the same corpus, but this isn't confirmed. Earlier code in the chat assumed 3 labels, so **check `num_labels`**.
- Other datasets: SEACrowd `lazada_review_filipino` (license unknown), Filipino Shopee reviews (star-labeled, weak labels).
- Data caveats: star ratings are noisy labels, heavy class imbalance, code-mixing, negation ("hindi peke" is positive), duplicate/templated reviews, split by product not review.

## 3. Key literature (what it does and doesn't give us)

**Zaman, Goldberg, Gruss & Abrahams (2024), "A semiautomated risk assessment method for consumer products," *Risk Analysis* 44(3):705–723, doi 10.1111/risa.14180.** Open access (CC BY). The user uploaded the PDF. **[PUBLISHED]**
- Amazon reviews, 17 categories. 124,289 reviews yielded 740 verified hazard reviews (~0.6%), 5,499 hand-tagged sentences.
- Smoke terms: n-grams scored with the correlation coefficient (CC) from Fan et al. (2005); scored sentences are standardized per product, false positives removed manually, then summed. Their threshold of 10,000 is described as arbitrary.
- Likelihood = standardized smoke score ÷ total review count (review count as a sales proxy).
- Severity = w_s × w_c. w_s by number of hazard sentences: 0→0, 1→0.33, 2→0.67, 3+→1. w_c = mean expert weight per hazard.
- Expert weights (mean of 5 experts, from 54 invited): chemical 0.5, choking 1.0, mechanical 0.4, fall 0.6, fire 0.9, hitting 0.3, strangulation 0.9, suffocation 1.0, swallowing 0.6, thermal 0.4. **Electrical and biological hazards were excluded for insufficient data.**
- Severity mapped to EU levels 1–4 by quarters of a 0–1 score; likelihood × severity placed on an EU risk matrix (Low/Medium/High/Serious).
- Results: likelihood vs. experts Spearman ~0.51–0.52. Severity vs. experts Pearson −0.09/−0.02 (pre-calibration), 0.06/0.19 (post). Post-calibration expert agreement: Serious 75%, Low 46%, High 14%, Medium 0%. Authors conclude human judgment is needed for severity. 30 products, 5 experts.
- Their Python toolkit is cited as Goldberg, Gruss & Abrahams (2022), "Fumeus," *Software Impacts* 12, 100270 (not opened).

**Other sources:**
- Goldberg et al. (food-safety review mining, *Risk Analysis*): keyword methods 77–90% accuracy in top-ranked reviews vs. 11–26% for sentiment analysis **[PUBLISHED, abstract-level]**.
- Abrahams et al. (2025, *Safety and Reliability*, doi 10.1080/09617353.2025.2607196): adapts CPSC and Health Canada injury-severity coding to review text; paywalled, not read.
- Open access, related: JMIR mobility-assistive products paper (preprint 42231); arXiv 1703.00518 (Bhat & Culotta, recall leading indicators, PU learning); arXiv 1805.09772 (baby-product safety issue detection); arXiv 2503.23213 (RECALL-MM recall dataset).
- **EU RAPEX / Safety Gate:** four severity levels (reversible minor, reversible serious, irreversible major, death) and a probability × severity risk matrix. Critics note the probability estimates are subjective **[PUBLISHED]**.
- Rivara & Berger (1980), *Am J Public Health* 70(7):701–704, doi 10.2105/ajph.70.7.701: criticized the CPSC Hazard Index for arbitrary weights and proposed cost-based weighting (abstract only read). Lawrence, Spicer & Miller (2015), *Injury Prevention* 21(1):23–29, on costs of non-fatal consumer product injuries **[cited in Zaman; not read]**.
- CPSC power-bank cases used as electronics severity evidence: Casely (51 reports, six minor burns, later 28 more reports incl. one fatality), Yiisonger (79 reports of exploding/igniting/swelling, one $15M-property-damage fire), NEWDERY (nine reports, $2M-property-damage fire) **[PUBLISHED, CPSC pages]**.
- Catelli, Pelosi & Esposito (2022), *Electronics* 11(3):374: BERT generally superior; lexicons worth considering on small datasets **[PUBLISHED, abstract]**.
- OECD/EUIPO (2025): USD 467B and 2.3% of global trade for 2021 **[PUBLISHED]**. The 2021 report gives up to 2.5% for 2019.
- Philippines: DTI Bureau of Philippine Standards requires PS marks / ICC stickers for covered products. Power banks: a BPS draft technical spec reportedly released 27 Feb 2026 (comments to 26 Apr 2026); commercial sources disagree on enforcement timing **[UNVERIFIED – check with DTI-BPS]**.
- **No published severity weights exist** for electronics hazards in Taglish reviews, for shock/sparking, or for counterfeit/defect categories. Frame this as the study's gap.

## 4. Review of the user's proposal (issues found)

**Scoring (Chapter 3):** Risk = 0.3·R_negative + 0.7·F_rate; Reliability = R_positive − Risk; tiers Low ≤0.30, Moderate 0.30–0.60, High >0.60 (engine triggers at "0.61 or above").
- High Risk needs more than 43% of reviews containing failure keywords even if all reviews are negative.
- Keyword-flagged reviews are usually also negative, so they are double counted.
- No small-sample control (one negative "peke" review → risk 1.0; one positive → reliability 1.0).
- 0.3/0.7 weights are asserted. Cited support (Arrieta et al. 2020; Punetha & Jain 2023) does not obviously justify them.
- Tier boundaries overlap at 0.30 and 0.60; 0.60 vs 0.61 inconsistency.
- One number mixes counterfeit, defect, and safety; "overheating" and "not charging" count equally.
- Reliability rewards positive ratio, but the problem statement says positives can be fake. "Verified alternatives" is overclaimed. The recommender's candidate set and cross-platform matching are undefined.

**Method claims:** "lexicon is superior for Taglish" is contradicted by FiReCS findings. Objective 2 says "train" a lexicon. Negation ("hindi peke", "walang sira") not handled. "Grounded" is an ambiguous keyword.

**Evaluation:** ground truth undefined (who labels, sample size, agreement). Metrics apply to classification but the output is a product-level score. No sample size or sampling for SUS/UMUX.

**Citations:** could not find Yashaswini & Padma (2025) "Taglish sentiment classification…" or Han et al. (2026) **[UNVERIFIED, likely problematic]**. Several references (Anderson, Chen & Gupta, Choi, Ganesan & Raman, Johal, Mohamed, Wang) lack volume/pages/DOI and use "et al." in the reference list. The OECD 2.3%/$467B figures are attributed to the 2021 report but come from 2025. "Esposito (2022)" should be Catelli, Pelosi & Esposito. Several citations appear mismatched to their claims (e.g., Bhagat 2021 and Rinanda & Mustakim 2024 for scraping; Liu et al. 2024 for decision fatigue; Lim et al. 2025 for scraping security tiers) based on titles. IPOPHL ₱40.99B figure and "power bank explosions" claim need sources **[UNVERIFIED]**.

**Other:** UMUX formula does not reverse-score negatively worded items 2 and 4 (a real bug). SUS/UMUX six-band scale attributed to Bangor 2008 likely comes from Bangor et al. 2009 with seven adjectives reported as means **[from memory, verify]**. Scraping text contradicts itself (mimicking humans to evade anti-bot vs. "remaining compliant"); no mention of the Data Privacy Act (RA 10173). Duplicated paragraphs, a dangling "modeling tools will be used, such as:" with no list, tense inconsistencies, MySQL/Firebase undecided, XAMPP vs. cloud deployment mismatch.

## 5. Scoring design evolution (all **[MY DESIGN]** unless stated)

**A. Revised weighted formula (later superseded):** categories safety / counterfeit / defect / dissatisfaction (negative but unflagged); each review counts once per category; per-category rate = Wilson lower bound, scaled by an anchor rate and capped at 1; category weights sum to 1; `n_min`; half-open tier intervals; separate safety alert; Reliability uses the Wilson lower bound of positive ratio minus risk. Parameters would come from expert/rater survey, data percentiles, split-half stability, and sensitivity analysis.

**B. Two outputs instead of one number (recommended over a single combined tier):** a *safety* badge and a *quality/authenticity* indicator, because no literature says how many defect points equal one fire. Optional dashboard "worst-of" label as display only.

**C. Weight-free design (current recommendation):**
1. **Safety badge:** ordinal severity level 1–4 per hazard subtype (fire/explosion/swelling; overheating/burn; shock/sparking), RAPEX-style with written reasoning; product severity = highest level with a high-confidence report; likelihood = Wilson lower bound of flagged-review rate, binned from reference-corpus percentiles; tier from a monotone matrix (illustrative, not final); alert override: 2+ independent high-confidence reports of a severity-4 hazard → at least High (threshold 2 is a choice to test).
2. **Counterfeit and defect flags:** flag when the Wilson lower bound of the complaint rate exceeds the baseline for that product type and n ≥ `n_min`; otherwise "insufficient evidence." Known weakness: if an entire product type is bad, the baseline rises and nothing is flagged; an optional absolute floor is another parameter.
3. **Display:** two labeled badges + evidence sentences + review count and date range. No combined number.
4. **Recommender:** filter out safety alerts / Moderate-or-above safety tier / counterfeit flags, then rank by lower bound of positive ratio. Inherits fake-review inflation risk; state as limitation.
5. **Tagging boundary rule:** safety = overheating, swelling, smoke, sparks, burns, shock; defect = not charging, dead on arrival, stopped working, short battery life; counterfeit = "peke", "hindi original", "iba sa picture". Ambiguous cases like "umiinit at hindi na nag-charge" need a pre-decided winner.

**Provisional severity basis:** fire/explosion/swelling anchored to Zaman fire (0.9) plus CPSC power-bank cases; overheating/burn judgment (Zaman thermal 0.4 as a floor); shock/sparking has no published weight; counterfeit/defect/dissatisfaction ordering argued from harm potential only.

**Taglish seed keywords (candidates; native-speaker review required):** uminit, sobrang init, nag-init, napaso, nasunog, nagliyab, umuusok, sumabog, umumbok, namaga, amoy sunog, nakuryente, nag-spark, nag-short, peke, hindi original, iba sa picture, sira agad. Watch negation.

## 6. Web app verdict logic (live scrape) **[MY DESIGN]**

Evaluated in order, first match wins:
1. Fewer than `n_min` usable reviews or incomplete scrape → **Insufficient evidence**
2. Safety alert or safety tier High+ → **Safety warning**
3. Safety tier Moderate → **Caution: possible safety issue**
4. Counterfeit or defect flag → **Caution: possible counterfeit or defect**
5. Reviews look unreliable (many near-duplicates or generic reviews; optional, unvalidated) → **Unclear: reviews may be unreliable**
6. Otherwise → **No warning signs found in N reviews** (never "safe"; alternatively "Low risk based on reviews" with a limits note)

Practical points: background job queue with progress indicator; cache by platform + product ID with timestamp; **live scraping still needs a pre-scraped reference corpus per product type** for baselines and likelihood bins (a one-time cost; check that baselines stabilize as products are added); if no baseline exists, show rates without a flag. **Scraping (user confirmed): all reviews per product can be scraped, and the platform lets you filter by star rating.** Consequences: (a) rates are computed over all reviews, so no star reweighting is needed; (b) completeness check: compare scraped counts per star level with the totals the listing shows, and return Insufficient evidence or a partial-data notice if they fall short (threshold is a choice to test; it cannot catch reviews the platform hides); (c) run the tagger on every review, including 4–5 stars, since hazard reports also appear there; (d) two-pass scraping **[MY DESIGN, unvalidated]**: Pass 1 = 1–3 star reviews, so the count-based safety alert (2+ independent high-confidence severity-4 reports) can show a preliminary Safety warning early; Pass 2 = remaining reviews, since the likelihood tier and counterfeit/defect flags need the full denominator; label anything shown early as preliminary; (e) Wilson bounds guard against small counts and noise, but reviewers are self-selected, so they do not describe sampling uncertainty over all buyers (state this in the methods); (f) reference products for baselines can be scraped in full and auto-tagged (hand-labeling is only needed for validation sets), and using the same tagger for baseline and product partly cancels systematic tagger errors; (g) store review dates and timestamp every scrape. A recent-window (e.g., last 12 months) view is deferred. Evaluate on frozen snapshots, not live pages. Scraping terms and privacy still need addressing (use APIs or permission where possible, drop usernames).

Worked example (hypothetical numbers): 400 reviews, two independent severity-4 fire/swelling reports → rate 0.5%, Wilson lower bound 0.14%, low-likelihood bin gives Moderate, alert rule lifts it to High → Safety warning. 10 defect reviews (2.5%, lower bound 1.36%) vs. a 1.5% type baseline → not flagged. One review with one complaint (lower bound 20.65%) → below `n_min` → Insufficient evidence.

## 6a. Expected app output (hypothetical mockups) **[MY DESIGN]**

All numbers and review sentences are illustrative.

```
20,000mAh Power Bank (Lazada)
Reviews analyzed: 412 of 415 listed | Dec 2024 – Sep 2026
VERDICT: SAFETY WARNING
Safety        HIGH  (alert: 2 independent reports of fire/swelling; highlighted sentences shown)
Counterfeit   Not flagged  (1.2% of reviews; typical for power banks: 2.0%)
Defect        Flagged      (6.8% of reviews; typical: 3.1%)
Review quality: no unusual duplication
Based on reviews only. This cannot confirm authenticity or safety.
Alternatives: 3 products passed the filters
```
Other states: "NO WARNING SIGNS FOUND in 380 reviews" (with a note that absence of complaints does not guarantee safety), "INSUFFICIENT EVIDENCE (12 reviews; minimum 30 needed)" (30 is a placeholder for n_min), and a progress view ("PRELIMINARY: safety alert triggered. Full analysis in progress.") that the final verdict replaces.

Fields on every result: verdict state; safety badge (tier, hazard type, number of independent reports, evidence sentences); counterfeit and defect flags with the product's rate vs. the typical rate for its type; evidence line (reviews scraped vs. listed, date range, completeness); optional review-quality note; always-on limits notice; alternatives if any pass the filters; timestamp and cache status. Backend returns JSON, and the page renders it. **Not shown:** a single risk number, the word "safe" without caveats, or a "probability this is fake."

## 6b. ML technique and tagger plan **[MY DESIGN unless marked]**

**The user confirmed their adviser/panel accepts a statistical learning method (no neural or transformer model required).**

| Job | Technique | Role |
|---|---|---|
| Tag sentences as safety/counterfeit/defect | Smoke-term scoring: n-grams weighted by the CC score (Zaman et al.), learned from hand-tagged sentences | **Primary** |
| Negation | Rules (hindi, walang, not before a keyword) | Support |
| Comparison taggers | TF-IDF + class-weighted logistic regression; fine-tuned XLM-R (multi-label) | Baselines |
| Sentiment (dissatisfaction, positive ratio for the recommender) | Lexicon or XLM-R fine-tuned on FiReCS (check label count) | Secondary |
| Duplicate/review-quality flag | TF-IDF cosine similarity or MinHash | Optional, unvalidated |
| Rates, tiers, verdict | Wilson intervals, lookup matrix, rules | Statistics and logic, not ML |

Why smoke terms are primary: hazard sentences are rare (about 0.6% of reviews in Zaman et al.; classes with fewer than 20 positives were brittle), transformers need many labeled positives, and smoke terms are interpretable (which terms fired), which supports highlighted evidence. Zaman et al. cite prior studies where smoke terms matched or beat deep learning for sparse categories (not read by me). All published results are English/Amazon, so Taglish performance is unproven.

Tagger build plan:
1. **Preprocess:** normalize case, elongation, emojis; map common spelling/affix variants (init, iinit, uminit, nag-init) by hand mapping or light stemming; define a sentence-segmentation rule (punctuation, newlines, emojis) and freeze it before labeling.
2. **Label:** two annotators plus a third for adjudication; pool enriched with 1–3 star reviews (not seed keywords) plus a random all-star sample for recall.
3. **Learn terms:** split 80/20 by product; compute unigram/bigram/trigram CC scores per category; drop CC ≤ 0, common words, and brand names (as Zaman et al.); a native Taglish speaker curates; add a minimum hazard-sentence count per term (my suggestion) to avoid chance terms.
4. **Score sentences:** sum CC scores per category; separate cutoff per category (raw scores are not comparable across categories).
5. **Set cutoffs:** on the validation split, by precision at a target recall (especially severity-4 hazards); do not copy Zaman's arbitrary 10,000. High-confidence rule for the alert = no negation plus a hazard-context term; measure its precision directly (manual false-positive removal is impossible in a live app).
6. **Evaluate on the frozen holdout:** AUC, AUPRC, AUPRC-lift per category, recall at fixed precision; fivefold cross-validation as a stability check; compare with TF-IDF+LR and XLM-R; optionally test character n-grams vs. word n-grams for Taglish spelling variation; merge or drop categories with fewer than about 20 positives.

Risks: labeling volume is the bottleneck (run the pilot to count positives per 1,000 reviews); word n-grams may struggle with Taglish morphology (untested).

## 7. Validation plan

- **Labeling:** all 3 members, 300–500 reviews, two annotators plus adjudication, report Cohen's/Fleiss' kappa. Pilot ~1,000 random reviews first to measure hazard rarity (Zaman found ~0.6%), then enrich the labeling pool with 1–3 star reviews, **not with seed-keyword hits** (that would make the learned dictionary circular and hide recall on hazards phrased differently), and also label a purely random sample across all stars as the recall test set; split train/validation/test by product (about 80/20). Aim for ≥20 positives per category before scoring it separately; merge categories until then.
- **Product-level:** ~30 products spanning low to high risk, rated blind and independently by the 3 members (separate safety and quality ratings). Report rank correlation and tier agreement with bootstrap confidence intervals. Label the group a "rater panel," not an "expert panel." Add 1–2 outside raters (adviser, electronics instructor, repair technician, BPS contact) if possible.
- **Baselines:** the original 0.3/0.7 formula, sentiment-only, keywords-only.
- **Key metric:** false-negative rate, meaning how often the app says "no warning signs" for products raters or known cases judge risky.
- **Other checks:** stability across re-scrapes, stress cases (few reviews; mostly five-star with a few hazard reports; keyword-stuffed reviews; partial scrape), sensitivity (move severity levels, bins, `n_min`, alert threshold ±30%), check that no recommended product carries a safety flag.
- **Rules:** freeze all parameters before touching test products; state success criteria in advance; expect modest agreement (Zaman et al. got ~0.5 on likelihood, weak on severity); report weak results plainly.
- **`n_min`:** choose by split-half stability (smallest count where agreement is acceptable; 0.7 is a common convention).

## 8. Decisions

**Decided (user-confirmed):**
1. Live scraping when a link is submitted.
2. All reviews per product can be scraped, with star filtering available (two-pass scraping is my proposed use of it).
3. Smoke-term scoring is the primary tagger; adviser/panel accepts a statistical learning method. Lexicon and XLM-R are optional comparison baselines.
4. Stay in plan mode. Rater panel = the 3 group members (safeguards in section 7; outside raters still welcome).

**Assumed but not explicitly confirmed:** two outputs (safety badge and quality flags, no combined number); positive state worded "No warning signs found" instead of "safe" (the user originally said "safe to buy"); weight-free design with an EU-style lookup matrix for safety.

**Still open:**
1. Platforms to start with (Lazada suggested first); scraping terms and the Data Privacy Act approach; rate limits.
2. Parameter values: n_min, completeness threshold, likelihood bins, alert threshold (1, 2, or 3), matrix cells, review-quality flag thresholds.
3. Whether the recommender is in scope for the first build, and its candidate-set definition.
4. Sentence-segmentation rule and Taglish spelling normalization approach.
5. FiReCS label count (dataset card lists four classes) before any XLM-R baseline.
6. Certification signals (PS/ICC) and the power-bank rule status (check with DTI-BPS).
7. Whether outside raters can be recruited.

## 9. Suggested next steps

1. Write the sentence-tagging guide (categories, boundary rule, Taglish examples, negation cases). This was the next item offered.
2. Pilot: scrape reference products and about 1,000 random reviews to measure hazard rarity.
3. Label (1–3 star-enriched pool plus random all-star sample), split by product, kappa reported.
4. Learn and curate smoke terms; set per-category cutoffs; evaluate against TF-IDF+LR and XLM-R baselines.
5. Rater survey and product-rating sheet (blind, independent, separate safety and quality ratings).
6. Build scoring, the verdict flow, two-pass scraping, and the completeness check; freeze parameters; evaluate on held-out products, with false-negative rate as the key metric.
7. Rewrite the proposal: methodology with symbols instead of fixed 0.3/0.7, citations, UMUX scoring, duplicated paragraphs, tier boundaries, Objectives 2 and 4.
8. Optionally search Philippine incident data (DTI advisories, news on charger/power-bank fires).

## 10. Guidance for the next assistant

- Keep provenance honest. Do not present [MY DESIGN] numbers, thresholds, or matrix cells as published. Do not invent weights or citations; verify any reference before it goes into the proposal.
- Published evidence supports the *direction* (keywords beat sentiment for safety; hazard-type expert weights; ordinal severity levels; severity × likelihood), not the specific values for this project.
- Never let the app's positive verdict say "safe" without strong caveats.
- The user is a capstone student team. Prefer plain, concrete explanations, and offer drafts (tagging guide, rater survey, methodology section) when useful.
- The next task the user is likely to request is the sentence-tagging guide, then the rater survey and product-rating sheet.
- Earlier chat drafts already exist for: the revised weighted formula (Wilson bound), the weight-free decision flow, and the validation protocol. Rebuild them from this document if needed.
