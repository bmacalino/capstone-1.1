# Hazard Sentence Tagging Guide

For the three annotators labelling review sentences for DeFaketive.
Version 1.1, 29 September 2026 (1.0 frozen 21 September 2026). If a rule here
changes, the version changes and the sentences labelled under the old version
are re-checked. Version 1.1 only adds worked examples (Rule 7) for the cases
labelling round 1 split on; no rule changed, so round 1 labels stay valid.

Read this once all the way through before you label anything. Then do the
30-sentence practice set together and compare answers. Disagreements at that
stage are cheap; disagreements after 400 sentences are not.

---

## 1. What you are doing

You are looking at one **sentence** at a time (really a clause — the computer
splits at "pero", "kaso" and "but" too), and answering three yes/no questions:

| Question | Column |
|---|---|
| Does this sentence report a **safety** hazard? | `human_safety` |
| Does this sentence report a **defect**? | `human_defect` |
| Does this sentence claim the product is **counterfeit**? | `human_counterfeit` |

A sentence can be yes to more than one. A sentence can be no to all three —
most will be.

The computer has already guessed. Its guesses are in the `llm_*` columns. **Your
job is to check every guess, not to rubber-stamp it.** Set `verifier_action` to:

- `accept` — the guess is right
- `correct` — the guess is wrong; put your answer in the `human_*` columns
- `unsure` — you genuinely cannot tell; write why in `human_notes`

Some rows have the `llm_*` columns **blank on purpose**. Those are blind rows.
Label them from scratch. We use them to measure whether seeing the computer's
guess changes your answers, so please do not go looking for the guess elsewhere.

---

## 2. The three categories

### SAFETY — the product could hurt someone or damage property

Say yes when the reviewer describes heat, fire, smoke, swelling, sparks, or
electric shock coming from the product.

Real examples from our data, all **yes**:

> "Nasunog agad, meaning nagshort yan sa loob..meaning 'di maganda build."

> "Unang gamit ko nga lang noon amoy sunog na kaya bihira ko din gamitin yung fan."

> "Quality:hindi naglast yung ringlight. Second try pa lang namin gamitin umusok na sya"

> "grounded ung mic.ng headset"

> "nagkaissue sya sa loob nasunog yung wiring habang ginagamit ko"

Typical words: uminit, sobrang init, napaso, nasunog, amoy sunog, nagliyab,
umuusok, sumabog, umumbok, namaga, lumobo, nakuryente, nag-spark, nag-short,
grounded, overheat, burning smell, exploded, swollen, shock, sparks.

**Also record which kind of hazard** in `human_hazard_subtype`:

| Subtype | Use it for |
|---|---|
| `fire_explosion_swelling` | caught fire, exploded, swelled up, battery bloated |
| `shock_spark_short` | electric shock, kuryente, sparks, short circuit, grounded |
| `overheat_burn` | got very hot, burned someone's skin |
| `smoke_smell` | smoke, burning smell, but no fire described |
| `safety_unspecified` | clearly a safety worry but you cannot tell which kind |

If two apply, pick the most severe (the table is ordered most severe first).

### DEFECT — the product does not work properly

Say yes when the product is broken, dead, stopped working, or never worked.

> "Wala pa 1 month di na agad working yung fan"

> "hindi gumagana kahit long press yung off hindi na ma off"

> "im so sorry pero nung dumating siya sakin is sira yung pinaka dulo ng wire niya"

Typical words: sira, sirang, basag, hindi gumagana, ayaw gumana, dead on
arrival, hindi nag-charge, mabilis maubos ang battery, nag-hang, walang power.

### COUNTERFEIT — the reviewer says it is fake or not the real brand

> "looks like imitation. item not shinny as the original."

> "the controller is fake, it's a replica off the original"

> "don't claim original kung may peke naman sa loob!"

> "Product is not 100% genuine. it's class A product different from what is
> advertised."

Typical words: peke, mukhang peke, hindi original, di orig, class a, replica,
imitation, fake, counterfeit, not genuine, iba sa picture, hindi tugma.

---

## 3. The rules that decide the hard cases

### Rule 1 — Only the product counts, not the seller, courier or box

We are judging **the product**, not the shopping experience. All of these are
**no to all three**:

- "sobrang tagal ng delivery" — late delivery
- "ang sungit ng seller" — rude seller
- "yung box nadurog pero okay naman yung item" — the box was damaged, not the
  product
- "kulang yung padala, wala yung cable" — missing item from the order

This is a change from the old labelling, which lumped these in as "risk". If a
sentence is only about delivery, the seller, or the packaging, the answer is no.

### Rule 2 — Negation means no

If the reviewer **denies** the problem, it is not a report of that problem.

| Sentence | Answer |
|---|---|
| "walang sira" | defect: **no** |
| "hindi naman umiinit kahit matagal gamitin" | safety: **no** |
| "not fake, legit talaga" | counterfeit: **no** |
| "comfy it's not mainit sa skin" | safety: **no** |

But be careful — some phrases contain a negator and are still a complaint:

| Sentence | Answer |
|---|---|
| "hindi gumagana" | defect: **yes** — "hindi" is part of the complaint |
| "ayaw mag-charge" | defect: **yes** |
| "walang power pagdating" | defect: **yes** |

The test: is the reviewer saying *"the problem did not happen"* (no) or *"the
product does not work"* (yes)?

### Rule 3 — A sentence with both heat AND failure is BOTH

This is the boundary case the guide exists to settle. Take:

> "umiinit at hindi na nag-charge"

Do **not** choose one. Mark **safety = yes** (umiinit) **and defect = yes**
(hindi nag-charge). The two categories are answered independently. This is why
they are three separate yes/no questions instead of one dropdown.

### Rule 4 — A counterfeit claim counts even in a happy review

> "Product is not 100% genuine. it's class A product... It's fine by me since I
> expected it to be for its price."

> "Maganda at mura... kahit hindi original sulit naman."

Both are **counterfeit = yes**. The reviewer is content, but they are still
telling us the product is not authentic, and a future buyer deserves to know.
**Do not let the overall tone of the review decide your answer.** We label what
is reported, not how the reviewer feels about it. (Both of these were
mislabelled in the old data precisely because tone was allowed to drive it.)

### Rule 5 — Guessing or fearing is not reporting

| Sentence | Answer |
|---|---|
| "baka pumutok yan sa sobrang init" — actually describes heat | safety: **yes** |
| "nakakatakot gamitin, baka masunog" — only a fear, nothing happened | safety: **no**, note it |
| "mukhang peke" — reviewer's judgement of the item | counterfeit: **yes** |
| "baka peke pero di ko sure" — pure speculation | counterfeit: **no**, note it |

The line: did the reviewer describe something **about this item**, or are they
imagining what might happen? Use `unsure` and a note when it is genuinely
borderline — those go to the adjudicator, which is the system working properly.

### Rule 6 — Watch the false friends

These come up constantly in our data and the computer gets them wrong:

| Sentence | Looks like | Actually |
|---|---|---|
| "not legit dual shock 4 but it's working fine" | shock hazard | **counterfeit yes, safety no** — "Dual Shock" is a PlayStation controller |
| "Sabog ang sound. Pangit." | explosion | **no** — "sabog" here means muddy sound |
| "okay naman po siya. Sabog lang po ung sounds" | explosion | **no** |
| "fake nails, ang ganda" | counterfeit | **no** — "fake nails" is the product type |
| "comfy it's not mainit sa skin" | overheating | **no** — negated, and about a cosmetic |
| "any text you copy will be saved here" | counterfeit ("copy") | **no** — pasted keyboard text |
| "umuusok na sabaw" | smoke | **no** — that is soup |

The general lesson: **read the whole sentence before deciding.** A single
matching word is never enough.

### Rule 7 — Worked examples for the cases round 1 split on (added in 1.1)

In round 1 the two annotators on a sentence disagreed most often about defects
(168 sentences). These examples apply the rules above; none is new.

| Sentence (from round 1) | Answer | Why |
|---|---|---|
| "mdaling malowbat, d na full charge cp ko" | defect **yes** | battery that drains fast or will not fill is "mabilis maubos ang battery" |
| "pero hindi nag chacharge so ano nalang?" | defect **yes** | does not charge |
| "left side lang ang gumagana" | defect **yes** | part of the product does not work |
| "no response, the seller only sends the automatic reply" | **no** to all | Rule 1: the seller, not the product |
| "kulang ng 5 yung brick sa wallpaper" | **no** to all | Rule 1: missing from the order, not faulty |
| "the box is damaged and the controller works perfectly" | **no** to all | Rule 1: the box |
| "kaso iba sa picture yung pinadala nyo" | counterfeit **yes** | "iba sa picture" is on the counterfeit list |
| "ang hina ng init, matagal ko nang sinaksak" (a hair straightener) | defect **yes**, safety **no** | too little heat from a heating product is a failure, not a hazard |
| "okay din, fake news, di naman nag-stop" | counterfeit **no** | "fake news" means "that claim was false", not "this is a fake" |

---

## 4. Filling in the worksheet

| Column | What to put |
|---|---|
| `verifier_id` | your initials, the same every time |
| `verifier_action` | `accept`, `correct`, or `unsure` |
| `human_safety` / `human_defect` / `human_counterfeit` | `yes` or `no` |
| `human_hazard_subtype` | only when `human_safety` is `yes` |
| `human_negated` | `yes` when you answered no *because* of a negation |
| `human_notes` | anything ambiguous, sarcastic, or unclear |
| `seconds_spent` | leave it; the sheet fills this in |

Do **not** fill in the `llm_*` columns, and do not delete them. Do not touch
`adjudicated_*` — that is the third person's job, and they work without seeing
the computer's guess.

---

## 5. Practical notes

- **Keep the reviewer's text exactly as written.** Spelling mistakes, emoji,
  ALL CAPS and Taglish all stay. They are signal, not noise.
- Some sentences will look cut off. That is expected — the computer splits long
  reviews into clauses. Label the clause in front of you. The full review is in
  the `review_text_raw` column if you need the context, and you should check it
  whenever a clause is ambiguous on its own.
- If the same review appears twice, label both the same way and note it.
- Go in order. Do not skip ahead to the interesting-looking ones; that biases
  the sample.
- Stop when you get tired. Rushed labels are worse than fewer labels, and we
  measure agreement between you, so fatigue shows up as noise in the results.

## 6. When you are unsure

Use `unsure` and write a note. Do not guess to keep the sheet tidy. Every
`unsure` goes to the adjudicator, and a documented disagreement is a *result* we
report, not a failure.

Bring anything that seems to need a new rule to the group before you invent one
on the spot. If we change a rule, everyone re-checks the sentences already
labelled under the old one.
