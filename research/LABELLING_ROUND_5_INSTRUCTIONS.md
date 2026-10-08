# Labelling Round 5 — how to start

For the three people doing the labelling. Round 5 works like rounds 2 and 3:
about 600 new sentences, each labelled by two of you, about 400 rows each, no
computer guess on any row. Follow
[LABELLING_ROUND_2_INSTRUCTIONS.md](LABELLING_ROUND_2_INSTRUCTIONS.md) and the
tagging guide [HAZARD_SENTENCE_TAGGING_GUIDE.md](HAZARD_SENTENCE_TAGGING_GUIDE.md),
**version 1.2**. Read the new Rule 8 (counterfeit examples) before you start.

## Why this round exists

Round 5 is the final, one-time test of all three checks (see
`research/PREREGISTRATION_final_v6.md`). The reviews come from a public set of
Shopee reviews that no earlier round used. Many sentences were picked because
they mention fake, original or legit, so expect more counterfeit decisions
than before. Label them exactly as the guide says.

## Your files

Everything is in `outputs/labelling_queue_round_5/`:

| File | Who opens it |
|---|---|
| `practice_set.csv` | all three of you, together, first (30 sentences) |
| `worksheet_A1.csv` | person 1 (same person as in earlier rounds) |
| `worksheet_A2.csv` | person 2 |
| `worksheet_A3.csv` | person 3 (Ryan) |
| `queue_master.csv`, `practice_set_master.csv`, `manifest.json` | nobody — record copies |

## When you are done

1. Save each finished sheet **inside `outputs/labelling_queue_round_5/`** as
   `worksheet_A1_done.csv`, `worksheet_A2_done.xlsx` and so on. Do not save it
   in the main project folder.
2. Do not discuss specific rows with anyone working on the checker until all
   three sheets are back.
3. After the three sheets are in, a short second sheet will list the sentences
   two of you disagreed on. The third person decides those, without seeing
   who answered what.

## A separate, smaller job: 65 older sentences

`outputs/adjudication_counterfeit/adjudication_sheet.csv` lists 65 sentences
from rounds 1–3 where two annotators disagreed about counterfeit. The
`adjudicator` column says who decides each row (the person who did not label
it). Type `yes` or `no` in `adjudicated_counterfeit`, using the guide and
Rule 8. Save it in the same folder as `adjudication_sheet_done.csv`. Do not
open `adjudication_master.csv`; it holds the earlier answers.
