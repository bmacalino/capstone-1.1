"""How consistent are the AI labels with themselves?

Draws labelled sentences from the DEV part of rounds 1-8 (never the held-out
third), hides their labels and writes them shuffled, to be labelled again
without looking. score.py compares the two readings.

  50 per label that were labelled yes (poor quality, not as advertised, possible fake)
  50 labelled no to all three but flagged by a rule or scored high by the pointer model
  150 labelled no to all three, at random
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
sys.path[:0] = [ROOT, ROOT + r"\sentiment-analysis", ROOT + r"\research"]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from five_label_data import keyword_view, load_five_label_records  # noqa: E402
from improve_label_rules import YARDSTICK_ROUNDS, held_out  # noqa: E402

HERE = Path(__file__).parent
TARGETS = ("quality", "not_as_described", "counterfeit")
rng = random.Random(20261008)

records = load_five_label_records(include_live_check=False)
dev = [i for i, r in enumerate(records) if str(r["round"]) in YARDSTICK_ROUNDS and not held_out(r)
       and all(r["labels"].get(label) is not None for label in TARGETS)]
view = keyword_view(records)
chosen: dict[int, str] = {}
for label in TARGETS:
    yes = [i for i in dev if records[i]["labels"][label] and i not in chosen]
    for i in rng.sample(yes, min(50, len(yes))):
        chosen[i] = f"yes_{label}"
no = [i for i in dev if not any(records[i]["labels"][label] for label in TARGETS)]
flagged = [i for i in no if any(view["flags"][label][i] for label in TARGETS)]
for i in rng.sample(flagged, min(50, len(flagged))):
    chosen[i] = "no_flagged"
rest = [i for i in no if i not in chosen]
for i in rng.sample(rest, 150):
    chosen[i] = "no_random"

order = list(chosen)
rng.shuffle(order)
key = {}
lines = []
for number, i in enumerate(order, start=1):
    key[number] = {"index": i, "stratum": chosen[i], "labels": {label: bool(records[i]["labels"][label]) for label in TARGETS},
                   "round": str(records[i]["round"])}
    lines.append(f"{number}\t{records[i]['text']}")
(HERE / "self_check_key.json").write_text(json.dumps(key), encoding="utf-8")
(HERE / "self_check_sentences.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("sentences", len(order), "dev pool", len(dev), "no pool", len(no), "flagged no", len(flagged))
