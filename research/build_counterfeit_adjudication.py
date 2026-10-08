"""Build the sheet for settling counterfeit disagreements in rounds 1-3.

A sentence is listed when its two annotators gave different answers to "does
this sentence claim the product is counterfeit?". The third annotator decides,
seeing only the sentence and its review: no annotator's answer and no computer
output (guide, section 4). research/measure_hazard_tagger.py --adjudication
reads the finished sheet.

    python research/build_counterfeit_adjudication.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from measure_hazard_tagger import flag, read_worksheet  # noqa: E402
from train_final_v6 import ROUND_SHEETS  # noqa: E402

OUT = ROOT / "outputs" / "adjudication_counterfeit"
SEED = 20261006
CATEGORY = "counterfeit"
ANNOTATORS = ("A1", "A2", "A3")
SHEET_FIELDS = ["row_id", "adjudicator", "rating", "sentence_text", "review_text_raw", "adjudicated_counterfeit", "adjudicator_notes"]


def disagreements() -> list[dict[str, str]]:
    rows = []
    for number, sheets in ROUND_SHEETS.items():
        folder = ROOT / "outputs" / f"labelling_queue_round_{number}"
        paths = [folder / name for name in sheets] if sheets else sorted(
            path for path in folder.glob("worksheet_*_done.*") if path.suffix in {".csv", ".xlsx"}
        )
        answers: dict[str, dict[str, bool]] = {}
        for path in paths:
            for row in read_worksheet(path):
                row_id = str(row.get("row_id") or "").strip()
                answer = flag(row.get("human_" + CATEGORY))
                if row_id and answer is not None:
                    answers.setdefault(row_id, {})[str(row.get("verifier_id") or path.stem).strip()] = answer
        for master in read_worksheet(folder / "queue_master.csv"):
            given = answers.get(str(master.get("row_id") or "").strip(), {})
            if len(given) < 2 or len(set(given.values())) == 1:
                continue
            others = [name for name in ANNOTATORS if name not in given]
            rows.append({
                "row_id": master["row_id"],
                "round": str(number),
                "adjudicator": others[0] if others else "",
                "rating": str(master.get("rating") or ""),
                "sentence_text": str(master.get("sentence_text") or ""),
                "review_text_raw": str(master.get("review_text_raw") or ""),
                "annotator_answers": json.dumps({name: "yes" if value else "no" for name, value in sorted(given.items())}),
                "adjudicated_counterfeit": "",
                "adjudicator_notes": "",
            })
    return rows


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(("﻿" + buffer.getvalue()).encode("utf-8"))


def main() -> int:
    rows = disagreements()
    random.Random(SEED).shuffle(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "adjudication_sheet.csv", SHEET_FIELDS, rows)
    write(OUT / "adjudication_master.csv", SHEET_FIELDS + ["round", "annotator_answers"], rows)
    summary = {
        "category": CATEGORY, "seed": SEED, "rows": len(rows),
        "by_round": dict(Counter(row["round"] for row in rows)),
        "by_adjudicator": dict(Counter(row["adjudicator"] for row in rows)),
        "note": "The sheet hides both annotators' answers and all computer output; the master keeps them.",
    }
    (OUT / "manifest.json").write_bytes((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
