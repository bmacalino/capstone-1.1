# Labelling Round 2 — how to start

For the three people doing the labelling. Round 2 works like round 1, with
three differences. Read those, then keep [LABELLING_ROUND_1_INSTRUCTIONS.md](LABELLING_ROUND_1_INSTRUCTIONS.md)
open for everything else (opening the file safely, filling in a row, the rules
that ruin a round).

The yes/no decisions still come from
[HAZARD_SENTENCE_TAGGING_GUIDE.md](HAZARD_SENTENCE_TAGGING_GUIDE.md). It is now
**version 1.1**: read the new **Rule 7** before you start. No rule changed; it
adds worked examples for the cases you disagreed on most in round 1.

---

## Why this round exists

The safety, defect and fake-product checks were improved using your round 1
labels. Measured on those same sentences, any improvement looks better than it
really is. Round 2 is **600 new sentences nobody has seen**, and it is the test
the improved checks will be graded on. None of these reviews was in round 1.

## What is different

**1. There is no computer guess.** Every `llm_*` column is empty on every row,
and `is_blind` says `yes` everywhere. That is deliberate: the check being tested
is the thing that made the guesses, so showing them would nudge your answers
towards it. Label every row from scratch. `verifier_action` becomes simply
`labelled` or `unsure`.

**2. It is shorter.** 600 sentences, each labelled by two of you, so **400 rows
each** (round 1 was 600). At about 15 seconds a row, allow two sittings.

**3. Different files.** Everything is in `outputs/labelling_queue_round_2/`:

| File | Who opens it |
|---|---|
| `practice_set.csv` | all three of you, together, first (30 sentences) |
| `worksheet_A1.csv` | person 1 |
| `worksheet_A2.csv` | person 2 |
| `worksheet_A3.csv` | person 3 (Ryan in round 1) |
| `queue_master.csv`, `practice_set_master.csv`, `manifest.json` | nobody — record copies |

Use the same person-to-file assignment as round 1, so each of you keeps the
same initials.

## Before you start

Do the 30-sentence practice set together, as in round 1, and compare answers
against Rule 7. The practice set still shows the computer's guess — it is
warm-up, not test data.

## When you are done

Export as CSV (UTF-8), keeping the filename, and save it in
`outputs/labelling_queue_round_2/` as `worksheet_<INITIALS>_done.csv`.

Please do not look at the checker's output on these sentences, or discuss
specific rows with anyone working on the checker, until all three sheets are
back. The test only counts if the answers are yours alone.
