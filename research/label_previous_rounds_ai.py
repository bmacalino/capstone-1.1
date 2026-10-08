"""The two guide-1.4 labels for rounds 1-4, proposed by the AI assistant.

Rounds 1-3 were labelled by the team for safety, defect and counterfeit, and
round 4 for counterfeit only; none carries "quality" or "not_as_described".
The assistant read every sentence (text only, no checker output) and proposed
those two labels so the earlier rounds can be used to train them. The team's
own answers are not touched: this writes a separate ai_extra_labels.csv in
each round folder, and one blind review sheet for the owner.

It also writes ai_relabel.csv for rounds 1-3: the team's defect and
counterfeit answers re-read under guide 1.4 and the owner's rules (a weak or
scratched product is quality, not defect; a picture or number dispute is
not-as-described, not counterfeit; a safety report is also a defect). That
file is for training only. Every reported result on rounds 1-3 still uses
the team's own sheets.

    python research/label_previous_rounds_ai.py
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from label_round5_ai import dump

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REVIEW = OUTPUTS / "extra_labels_previous_rounds"
SEED = 20261009
LABELS = ("quality", "not_as_described")
AUDIT = {"with_a_yes": 30, "all_no": 30}

# Row numbers within each round. Anything not listed is a sure "no".
# Seventeen answers on rounds 1-6 were corrected on 7 October 2026 after reading
# where the revised rules and these labels disagreed; each was an omission
# under the assistant's own rule ("mabilis masira", "manipis", "fake specs").
# In this file: quality r1 20, 421, 473, 886; r2 253, 346, 366; r3 156, 353;
# r4 48; not_as_described r1 97, 836; r4 84. Round 7 was not revisited, so it
# stays a check the rules were not fitted to.
EXTRA = {
    1: {
        "quality": {
            "sure": {
                11, 12, 20, 26, 36, 55, 62, 73, 75, 100, 116, 124, 152, 160, 182, 212, 213, 222, 238, 252, 253, 257, 269,
                274, 301, 334, 336, 340, 351, 361, 363, 373, 403, 406, 409, 416, 418, 421, 447, 452, 455, 472, 473, 476, 478,
                481, 498, 509, 511, 533, 554, 561, 576, 584, 591, 600, 614, 624, 638, 641, 644, 656, 664, 675, 678,
                685, 686, 692, 697, 712, 714, 721, 737, 742, 751, 752, 775, 788, 807, 818, 821, 847, 885, 886, 896, 897,
            },
            "unsure": {
                17, 42, 48, 74, 78, 208, 228, 239, 240, 248, 258, 260, 287, 302, 305, 398, 407, 426, 433, 436, 465,
                479, 489, 492, 567, 593, 616, 717, 791, 816, 822, 888, 893,
            },
        },
        "not_as_described": {
            "sure": {
                42, 92, 97, 100, 120, 149, 161, 194, 221, 222, 267, 269, 340, 367, 373, 378, 392, 443, 462, 473, 495, 510,
                519, 534, 596, 626, 646, 658, 660, 661, 664, 675, 679, 727, 751, 758, 804, 814, 836, 840, 850, 853, 863, 884,
            },
            "unsure": {117, 141, 162, 189, 250, 328, 332, 345, 399, 681, 711, 772},
        },
    },
    2: {
        "quality": {
            "sure": {
                9, 34, 43, 50, 57, 65, 71, 74, 75, 93, 115, 130, 134, 146, 163, 166, 172, 212, 222, 236, 239, 253, 255, 291,
                326, 346, 352, 355, 356, 358, 362, 369, 380, 385, 388, 396, 410, 411, 430, 438, 459, 467, 498, 506, 510, 519,
                366, 526, 533, 535, 562, 597, 600,
            },
            "unsure": {5, 7, 21, 118, 132, 174, 192, 195, 279, 283, 290, 298, 328, 347, 373, 494, 581},
        },
        "not_as_described": {
            "sure": {5, 15, 75, 85, 98, 121, 136, 170, 203, 208, 231, 360, 405, 432, 459, 462, 474, 495},
            "unsure": {22, 37, 112, 359, 371},
        },
    },
    3: {
        "quality": {
            "sure": {
                11, 24, 27, 30, 31, 55, 61, 69, 113, 129, 133, 156, 164, 170, 173, 177, 178, 194, 195, 200, 202, 203, 210,
                234, 276, 281, 313, 332, 353, 371, 402, 411, 416, 427, 484, 486, 499, 522, 549, 558, 559, 577, 587, 589, 592,
                599,
            },
            "unsure": {
                47, 81, 104, 123, 211, 233, 236, 240, 243, 253, 262, 314, 404, 434, 445, 446, 450, 457, 463, 494, 541,
            },
        },
        "not_as_described": {
            "sure": {47, 81, 158, 187, 239, 297, 316, 350, 359, 374, 494, 508, 535, 573, 586},
            "unsure": {232, 282, 330, 334},
        },
    },
    4: {
        "quality": {
            "sure": {48, 98, 112, 114, 162, 169, 188, 225, 235, 248, 254, 284, 288, 314, 327, 334, 367, 383},
            "unsure": {89, 121, 179, 236, 245, 266, 301, 302},
        },
        "not_as_described": {
            "sure": {3, 40, 63, 67, 79, 112, 163, 169, 189, 192, 204, 207, 227, 237, 246, 249, 372, 373, 379},
            "unsure": {84, 97, 143, 158, 164, 223, 233, 245, 270, 279},
        },
    },
}

# Team answers that guide 1.4 moves to one of the new labels (RELABEL[round][old
# label] = rows whose old label becomes "no"). Rows that also report a real
# failure, or a real copy, keep the team's answer.
RELABEL = {
    1: {
        "defect": {12, 73, 75, 124, 160, 222, 238, 301, 340, 426, 478, 481, 616, 641, 685, 712, 717, 721, 737, 752, 775, 818, 885},
        "counterfeit": {42, 194, 222, 267, 269, 378, 443, 534, 596, 646, 661, 664, 814, 850},
    },
    2: {
        "defect": {
            5, 7, 34, 50, 65, 71, 93, 172, 174, 212, 236, 290, 291, 326, 355, 358, 362, 459, 467, 498, 506, 510, 533, 535,
            581, 597, 600,
        },
        "counterfeit": {37, 85, 121},
    },
    3: {
        "defect": {
            30, 31, 81, 113, 129, 133, 164, 173, 177, 178, 202, 203, 234, 276, 313, 314, 332, 371, 402, 404, 411, 446, 457,
            463, 484, 494, 522, 549, 558, 559, 587, 589, 599,
        },
        "counterfeit": {158},
    },
}


def relabel() -> dict[str, dict[str, int]]:
    """Write ai_relabel.csv for rounds 1-3 and return what changed."""
    import sys

    for folder in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    from train_final_v6 import load_records

    word = {True: "yes", False: "no", None: ""}
    by_round: dict[str, list[dict[str, str]]] = {}
    changed: dict[str, dict[str, int]] = {}
    for record in load_records():
        number, n = int(record["round"]), int(record["row_id"].split("-")[1])
        team = record["labels"]
        new = {"defect": team["defect"], "counterfeit": team["counterfeit"]}
        reasons = []
        for label in ("defect", "counterfeit"):
            if n in RELABEL[number][label] and team[label] is not False:
                new[label] = False
                reasons.append("quality_not_defect" if label == "defect" else "not_as_described_not_counterfeit")
        if team["safety"] is True and new["defect"] is not True:
            new["defect"] = True
            reasons.append("safety_is_also_defect")
        by_round.setdefault(str(number), []).append({
            "row_id": record["row_id"], "sentence_text": record["text"],
            **{f"team_{label}": word[team[label]] for label in ("safety", "defect", "counterfeit")},
            "relabel_defect": word[new["defect"]], "relabel_counterfeit": word[new["counterfeit"]],
            "reason": "|".join(reasons),
        })
        for reason in reasons:
            changed.setdefault(f"round_{number}", {}).setdefault(reason, 0)
            changed[f"round_{number}"][reason] += 1
    for number, rows in by_round.items():
        dump(OUTPUTS / f"labelling_queue_round_{number}" / "ai_relabel.csv", list(rows[0].keys()), rows)
    return changed


def main() -> int:
    everything = []
    counts: dict[str, dict[str, int]] = {}
    for number, sets in EXTRA.items():
        folder = OUTPUTS / f"labelling_queue_round_{number}"
        with (folder / "queue_master.csv").open(encoding="utf-8-sig", newline="") as source:
            master = [row for row in csv.DictReader(source) if row["row_id"].startswith(f"r{number}-")]
        rows = []
        for row in master:
            n = int(row["row_id"].split("-")[1])
            item = {key: row.get(key, "") for key in ("row_id", "rating", "sentence_text", "review_text_raw")}
            unsure = []
            for label in LABELS:
                sure, maybe = sets[label]["sure"], sets[label]["unsure"] - sets[label]["sure"]
                item[f"ai_{label}"] = "yes" if n in sure or n in maybe else "no"
                if n in maybe:
                    unsure.append(label)
            item["ai_unsure"] = "|".join(unsure)
            rows.append(item)
        dump(folder / "ai_extra_labels.csv", list(rows[0].keys()), rows)
        counts[f"round_{number}"] = {label: sum(row[f"ai_{label}"] == "yes" for row in rows) for label in LABELS} | {
            "sentences": len(rows), "unsure": sum(bool(row["ai_unsure"]) for row in rows)
        }
        everything += rows

    rng = random.Random(SEED)
    unsure_rows = [row for row in everything if row["ai_unsure"]]
    sure = [row for row in everything if not row["ai_unsure"]]
    with_yes = [row for row in sure if any(row[f"ai_{label}"] == "yes" for label in LABELS)]
    all_no = [row for row in sure if row not in with_yes]
    audit = rng.sample(with_yes, AUDIT["with_a_yes"]) + rng.sample(all_no, AUDIT["all_no"])
    sheet = [dict(row) for row in unsure_rows + audit]
    rng.shuffle(sheet)
    answers = [f"human_{label}" for label in LABELS]
    for row in sheet:
        row.update({column: "" for column in answers + ["human_notes"]})
    REVIEW.mkdir(parents=True, exist_ok=True)
    dump(REVIEW / "worksheet_for_you.csv", ["row_id", "rating", "sentence_text", "review_text_raw", *answers, "human_notes"], sheet)
    plan = {
        "seed": SEED, "labels": list(LABELS), "counts": counts,
        "unsure_rows": [row["row_id"] for row in unsure_rows], "audit_rows": [row["row_id"] for row in audit],
        "audit_design": "30 random AI-sure rows with a yes + 30 random AI-sure all-no rows; AI answers hidden",
        "note": "The team's safety, defect and counterfeit answers for these rounds are unchanged.",
    }
    (REVIEW / "review_plan.json").write_bytes((json.dumps(plan, indent=1) + "\n").encode("utf-8"))
    print(json.dumps({"counts": counts, "sheet_rows": len(sheet), "relabelled": relabel()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
