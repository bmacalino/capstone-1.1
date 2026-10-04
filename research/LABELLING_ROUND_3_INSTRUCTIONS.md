# Labelling Round 3 — how to start

For the three people doing the labelling. Round 3 works exactly like round 2:
600 new sentences, each labelled by two of you, **400 rows each**, no computer
guess on any row. Follow [LABELLING_ROUND_2_INSTRUCTIONS.md](LABELLING_ROUND_2_INSTRUCTIONS.md)
and the tagging guide [HAZARD_SENTENCE_TAGGING_GUIDE.md](HAZARD_SENTENCE_TAGGING_GUIDE.md)
(version 1.1, unchanged since round 2).

## Why this round exists

Round 3 is the one-time test of an improved defect check (see
`research/PREREGISTRATION_defect_v5.md`). None of these reviews was in round 1
or round 2. Some sentences were picked because the old and new checks disagree
about them, so you may notice more sentences about products working or not
working than before. Label them exactly as the guide says; that is the point.

## Your files

Everything is in `outputs/labelling_queue_round_3/`:

| File | Who opens it |
|---|---|
| `practice_set.csv` | all three of you, together, first (30 sentences) |
| `worksheet_A1.csv` | person 1 (same person as in rounds 1 and 2) |
| `worksheet_A2.csv` | person 2 |
| `worksheet_A3.csv` | person 3 (Ryan) |
| `queue_master.csv`, `practice_set_master.csv`, `manifest.json` | nobody — record copies |

## When you are done — please read, this matters

Save each finished sheet **inside `outputs/labelling_queue_round_3/`** as
`worksheet_<INITIALS>_done.csv` (or `.xlsx`). **Do not save it in the main
project folder.** In round 2 the finished sheets were saved there under the
same names as round 1's, and round 1's sheets for A1 and A3 were overwritten;
A3's has not been recovered.

Please do not discuss specific rows with anyone working on the checker until
all three sheets are back.
