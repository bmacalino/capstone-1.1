"""Round 5 labels proposed by the AI assistant (departure from the
pre-registration, chosen by the project owner on 6 October 2026).

The assistant read each sentence without the stratum, the checker's output or
either model's answer, and applied the tagging guide. Every answer is marked
sure or unsure. The owner decides every unsure sentence and blind-audits a
random sample of the sure ones (worksheet_for_you.csv). Because the labels are
machine-proposed, round 5 is an AI-assisted check, not a reportable accuracy.

The sets are layered in the order the decisions were made:
  1. the assistant's first three-label answers (SURE_YES, UNSURE_YES, UNSURE_NO);
  2. the owner's two rules of 7 October 2026 (apply_owner_rules);
  3. the two labels added in guide 1.4, "quality" and "not_as_described",
     with the sentences they take out of defect and counterfeit
     (apply_five_labels).

    python research/label_round5_ai.py
"""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "outputs" / "labelling_queue_round_5"
SEED = 20261006
CATEGORIES = ("safety", "defect", "quality", "counterfeit", "not_as_described")
AUDIT = {"with_a_yes": 25, "all_no": 25}
Sets = dict[str, set[int]]

# Row numbers (r5-00001 -> 1). Anything not listed is a sure "no".
SURE_YES: Sets = {
    "safety": {66, 194, 401, 468, 480, 482, 529, 547},
    "defect": {
        1, 6, 15, 16, 40, 44, 45, 68, 70, 74, 76, 79, 103, 105, 109, 111, 129, 133, 145, 154, 161, 168, 172, 174,
        207, 209, 224, 228, 249, 254, 259, 262, 264, 272, 277, 279, 295, 299, 303, 311, 318, 321, 333, 348, 377,
        387, 392, 396, 405, 407, 424, 426, 469, 477, 479, 493, 499, 504, 508, 509, 511, 514, 538, 539, 548, 574,
    },
    "counterfeit": {102, 111, 119, 145, 209, 242, 244, 277, 336, 356, 369, 436, 439, 458, 503, 578, 587},
}
UNSURE_YES: Sets = {
    "safety": {94, 142, 201, 389, 439, 492, 514, 532},
    "defect": {
        7, 30, 139, 148, 150, 184, 202, 211, 219, 278, 285, 325, 331, 357, 372, 373, 395, 400, 406, 416, 449, 456,
        473, 505, 519, 582, 588, 593, 595,
    },
    "counterfeit": {48, 77, 90, 100, 107, 109, 115, 195, 216, 247, 350, 363, 377, 429, 476, 519, 549},
}
UNSURE_NO: Sets = {
    "safety": {437, 526},
    "defect": {8, 20, 87, 99, 106, 155, 170, 187, 217, 310, 366, 371, 378, 381, 440, 441, 455, 583},
    "counterfeit": {
        73, 87, 106, 120, 128, 144, 166, 170, 174, 255, 261, 268, 322, 342, 374, 390, 393, 454, 466, 539, 562, 592,
    },
}

# The owner's rules of 7 October 2026. Both match how the team labelled rounds
# 1-3 (6 of 6 slow-charging sentences, and 40 of the 54 agreed safety reports
# with a defect decision, were labelled defect):
#   1. slow charging, with nothing else reported, is a defect;
#   2. a safety report (heat, smoke, "grounded" ...) is a defect as well.
SLOW_CHARGING = {441}

# Guide 1.4 (7 October 2026): two more labels. Four answers were corrected the
# same day after reading where the revised rules and these labels disagreed
# (quality 43, 299, 480; not_as_described 441), all omissions under the
# assistant's own rule.
#   quality           works, but poorly; feels cheap; dented, scratched, flimsy
#   not_as_described  a size, capacity, feature or look that differs from the listing
FIVE: dict[str, Sets] = {
    "sure_yes": {
        "quality": {
            16, 45, 187, 195, 197, 211, 217, 219, 224, 251, 285, 299, 310, 352, 363, 371, 440, 454, 480, 503, 520, 562, 573,
            581, 582, 583, 587,
        },
        "not_as_described": {6, 48, 87, 170, 174, 254, 312, 350, 390, 441, 499, 539, 576},
    },
    "unsure_yes": {
        "quality": {43, 51, 53, 118, 155, 213, 268, 366, 378, 385, 455, 466, 483},
        "not_as_described": {7, 65, 73, 81, 128, 165, 166, 180, 216, 255, 353, 355},
    },
    # Sentences the new labels take over: a sure "no" for the old label now.
    "no_longer": {
        "defect": {7, 16, 45, 87, 170, 187, 211, 217, 219, 224, 285, 310, 371, 378, 440, 455, 582, 583},
        "counterfeit": {48, 73, 87, 128, 166, 170, 174, 322, 350, 374, 390},
    },
}


def apply_owner_rules(sure_yes: Sets, unsure_yes: Sets, unsure_no: Sets, slow_charging: set[int]) -> tuple[Sets, Sets, Sets]:
    """The label sets with both rules applied to the defect answers. A safety
    answer the assistant was unsure of gives an unsure defect answer."""
    sure = sure_yes["defect"] | slow_charging | sure_yes["safety"]
    unsure = (unsure_yes["defect"] | unsure_yes["safety"]) - sure
    return (
        {**sure_yes, "defect": sure},
        {**unsure_yes, "defect": unsure},
        {**unsure_no, "defect": unsure_no["defect"] - sure - unsure},
    )


def apply_five_labels(sure_yes: Sets, unsure_yes: Sets, unsure_no: Sets, five: dict[str, Sets]) -> tuple[Sets, Sets, Sets]:
    """Add the two guide-1.4 labels and clear the old answers they replace."""
    sure, unsure, no = dict(sure_yes), dict(unsure_yes), dict(unsure_no)
    for category, rows in five["no_longer"].items():
        sure[category] = sure[category] - rows
        unsure[category] = unsure[category] - rows
        no[category] = no[category] - rows
    for category in ("quality", "not_as_described"):
        sure[category] = set(five["sure_yes"][category])
        unsure[category] = set(five["unsure_yes"][category]) - sure[category]
        no[category] = set()
    return sure, unsure, no


def final_sets(sure_yes: Sets, unsure_yes: Sets, unsure_no: Sets, slow_charging: set[int], five: dict[str, Sets]) -> tuple[Sets, Sets, Sets]:
    return apply_five_labels(*apply_owner_rules(sure_yes, unsure_yes, unsure_no, slow_charging), five)


def dump(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(("﻿" + buffer.getvalue()).encode("utf-8"))


def build(folder: Path, sure_yes: Sets, unsure_yes: Sets, unsure_no: Sets, seed: int, out: Path | None = None) -> dict[str, object]:
    """Write ai_labels.csv, the owner's blind worksheet and the review plan for
    one round's queue. `out` exists so the result can be checked elsewhere."""
    out = out or folder
    with (folder / "queue_master.csv").open(encoding="utf-8-sig", newline="") as source:
        master = list(csv.DictReader(source))
    labelled = []
    for row in master:
        number = int(row["row_id"].split("-")[1])
        item = {key: row[key] for key in ("row_id", "review_id", "rating", "sentence_text", "review_text_raw")}
        unsure = []
        for category in CATEGORIES:
            yes = number in sure_yes[category] or number in unsure_yes[category]
            item[f"ai_{category}"] = "yes" if yes else "no"
            if number in unsure_yes[category] or number in unsure_no[category]:
                unsure.append(category)
        item["ai_unsure"] = "|".join(unsure)
        labelled.append(item)
    dump(out / "ai_labels.csv", list(labelled[0].keys()), labelled)

    rng = random.Random(seed)
    unsure_rows = [row for row in labelled if row["ai_unsure"]]
    sure = [row for row in labelled if not row["ai_unsure"]]
    with_yes = [row for row in sure if any(row[f"ai_{c}"] == "yes" for c in CATEGORIES)]
    all_no = [row for row in sure if row not in with_yes]
    audit = rng.sample(with_yes, AUDIT["with_a_yes"]) + rng.sample(all_no, AUDIT["all_no"])
    sheet = [dict(row) for row in unsure_rows + audit]
    rng.shuffle(sheet)
    answers = [f"human_{category}" for category in CATEGORIES]
    for row in sheet:
        row.update({column: "" for column in answers + ["human_notes"]})
    dump(out / "worksheet_for_you.csv", ["row_id", "rating", "sentence_text", "review_text_raw", *answers, "human_notes"], sheet)
    plan = {
        "seed": seed,
        "labels": list(CATEGORIES),
        "unsure_rows": [row["row_id"] for row in unsure_rows],
        "audit_rows": [row["row_id"] for row in audit],
        "audit_design": "25 random AI-sure rows with a yes + 25 random AI-sure all-no rows; AI answers hidden",
        "ai_yes": {c: sum(row[f"ai_{c}"] == "yes" for row in labelled) for c in CATEGORIES},
    }
    (out / "review_plan.json").write_bytes((json.dumps(plan, indent=1) + "\n").encode("utf-8"))
    return {"rows": len(labelled), "unsure": len(unsure_rows), "audit": len(audit), "sheet": len(sheet), "ai_yes": plan["ai_yes"]}


def main() -> int:
    print(json.dumps(build(FOLDER, *final_sets(SURE_YES, UNSURE_YES, UNSURE_NO, SLOW_CHARGING, FIVE), SEED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
