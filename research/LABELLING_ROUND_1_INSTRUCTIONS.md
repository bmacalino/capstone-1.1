# Labelling Round 1 — how to start

For the three people doing the labelling. Read this once, then keep it open
beside you for the first hour.

This tells you how to run the sheet. It does **not** tell you how to decide
yes or no — that is [HAZARD_SENTENCE_TAGGING_GUIDE.md](HAZARD_SENTENCE_TAGGING_GUIDE.md),
and you need to have read it before you touch a worksheet.

---

## What you are about to do

There are 900 sentences waiting. Each one gets three yes/no answers: is this a
safety hazard, is this a defect, is this a counterfeit claim.

Each sentence is labelled by **two** of you, independently, without talking
about it. The third person settles the ones you disagree on. That is the whole
design — the disagreement rate is a number we report, so two people quietly
agreeing on a wrong answer is worse than two people openly disagreeing.

Your own sheet has 600 rows. At roughly 15 seconds a row that is around two and
a half hours, which is three or four sittings. Do not try to do it in one.

---

## Your files

Everything is in `outputs/labelling_queue_round_1/`.

| File | Who opens it |
|---|---|
| `practice_set.csv` | all three of you, together, first |
| `worksheet_A1.csv` | person 1 |
| `worksheet_A2.csv` | person 2 |
| `worksheet_A3.csv` | person 3 |
| `queue_master.csv` | nobody — this is the record copy |
| `practice_set_master.csv` | nobody — same, for the practice set |
| `manifest.json` | nobody — this is for the methodology chapter |

Open **only your own worksheet**. The overlap between sheets is deliberate and
it stops being a measurement the moment you compare notes.

---

## Opening the file without wrecking it

The reviews are in Taglish and full of emoji. A careless open turns `nasunog`
into `nasunogÃ¯Â¿Â½` and the row is ruined.

**Google Sheets (recommended — easiest for three people):**

1. File → Import → Upload → pick your worksheet
2. Import location: **Create new spreadsheet**
3. Separator type: **Comma**
4. Convert text to numbers, dates and formulas: **turn this OFF**

Step 4 matters. Left on, Sheets turns a review that starts with `-` into a
formula and shows `#ERROR!` where the text used to be.

**Excel:** double-clicking usually works, because the file carries a marker
that tells Excel it is UTF-8. If you see mojibake anyway, close without saving
and use Data → From Text/CSV → Encoding: **65001: Unicode (UTF-8)**.

Never save over the original with "Save as → CSV" from Excel unless you pick
UTF-8 again. Safest habit: work in Google Sheets, export at the end.

---

## Start with the practice set

All three of you label the same 30 sentences, on your own, then sit down and
compare.

It is stacked on purpose with the cases that trip people up — the ones where a
hazard word appears but there is no hazard. `not legit dual shock 4`,
`sabog ang sound`, `fake nails`. If all three of you get those right, you are
ready. If you disagree, argue it out now and write down what you decided.

Do not start your real worksheet until you have done this. Thirty sentences of
disagreement over coffee costs an hour. Six hundred sentences of undetected
disagreement costs the round.

---

## Filling in a row

Read `sentence_text`. If it does not make sense alone, read `review_text_raw`
in the next column — that is the whole review it came from.

Then:

| Column | What to put |
|---|---|
| `verifier_id` | already filled with your initials — leave it |
| `verifier_action` | `accept`, `correct`, or `unsure` |
| `human_safety` | `yes` or `no` |
| `human_defect` | `yes` or `no` |
| `human_counterfeit` | `yes` or `no` |
| `human_hazard_subtype` | only when `human_safety` is `yes` |
| `human_negated` | `yes` if you answered no *because* the reviewer denied it |
| `human_notes` | anything odd, sarcastic, or borderline |
| `seconds_spent` | **leave blank.** A plain sheet cannot time you |

Fill in the three `human_*` columns on **every** row, including the ones where
you agree with the computer. `accept` in `verifier_action` records that you
agreed; it does not fill in the answer for you.

### About the computer's guess

The `llm_*` columns hold a guess. **It is a keyword match and nothing more** —
it looks for words like `nasunog` and `peke` and has no idea what the sentence
means. It is wrong often, and confidently.

It will say yes to "charges efficiently **without overheating**", because it
saw `overheating` and cannot read the `without`. It will say yes to "mainit ang
tela" on a T-shirt.

So do not read it as a hint about the right answer. Read the sentence first,
decide, then look at the guess and mark `accept` or `correct`. If you find
yourself agreeing with it more than about four times in five, you have started
rubber-stamping — take a break.

### The blind rows

About one row in seven has the `llm_*` columns empty. Those are deliberate.
Label them from scratch and do not go looking for the guess in another file.

They exist to measure whether seeing the computer's guess changes your answers.
If it does — and it usually does, for everyone — that is a finding we report,
not a mistake you made.

---

## The rules you will reach for most

All of these are in the tagging guide with examples. The four that come up
constantly:

- **Only the product counts.** Late delivery, a rude seller, a crushed box —
  no to all three.
- **A sentence can be two things.** "umiinit at hindi na nag-charge" is safety
  **and** defect. Answer the three questions separately.
- **A counterfeit claim counts even in a happy review.** "kahit hindi original
  sulit naman" is still counterfeit = yes.
- **Fear is not a report.** "nakakatakot gamitin, baka masunog" — nothing
  happened. No, and leave a note.

---

## Things that will ruin the round

- Do not sort or reorder the sheet. Row order is part of the design.
- Do not delete or rename columns, including ones you never fill in.
- Do not fix the reviewers' spelling. Typos and emoji are data.
- Do not skip ahead to the interesting-looking rows. Go in order.
- Do not discuss specific rows with your labelling partner until adjudication.
- Do not guess to keep the sheet tidy. `unsure` plus a note is a real answer.

---

## When you are done

Export as CSV, UTF-8, keeping the filename. Put all three back in
`outputs/labelling_queue_round_1/` as `worksheet_<INITIALS>_done.csv`.

Then the third person adjudicates every row where the two answers differ,
working from the sentence alone — not from what either of you wrote.

---

## Two notes for whoever runs the script

**Set the real initials before handing these out.** The sheets currently say
`A1`, `A2`, `A3`. Regenerate with the actual people:

```
python research/build_labelling_queue.py --annotators CAB,MAC,RET
```

The filenames and the `verifier_id` column follow whatever you pass. The random
seed is fixed, so the same rows come back — only the names on them change.

**The Shopee dataset is not wired in yet.** The queue currently draws from your
own Lazada and Temu scrapes plus FiReCS. To add the Hugging Face set
(`scaredmeow/shopee-reviews-tl-stars`), download it as a CSV with `text` and
`label` columns and pass it:

```
python research/build_labelling_queue.py --shopee-hf path/to/shopee.csv
```

It will be recorded as its own cohort in the manifest, which is what the
methodology chapter needs.
