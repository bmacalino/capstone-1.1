"""Compare a second blind reading of 315 dev sentences with their first labels."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
sys.path[:0] = [ROOT, ROOT + r"\sentiment-analysis", ROOT + r"\research"]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from five_label_data import keyword_view, load_five_label_records  # noqa: E402
from improve_label_rules import YARDSTICK_ROUNDS, held_out  # noqa: E402

HERE = Path(__file__).parent
TARGETS = ("quality", "not_as_described", "counterfeit")
SECOND = {
    "quality": {
        1, 6, 10, 25, 27, 34, 36, 44, 53, 57, 61, 63, 68, 73, 74, 79, 85, 86, 87, 100, 106, 111, 114, 115, 117, 118, 120,
        126, 128, 132, 134, 136, 148, 153, 156, 159, 167, 170, 171, 175, 179, 180, 183, 184, 189, 190, 191, 205, 206, 218,
        233, 235, 239, 245, 253, 257, 260, 262, 263, 265, 271, 273, 277, 290, 294, 297, 301, 305, 306, 311, 313,
    },
    "not_as_described": {
        5, 7, 36, 40, 42, 43, 47, 63, 65, 76, 77, 82, 85, 87, 88, 91, 95, 97, 103, 104, 108, 114, 121, 124, 125, 128, 133,
        135, 141, 146, 149, 151, 155, 166, 169, 170, 185, 193, 226, 228, 237, 240, 251, 254, 259, 263, 264, 270, 271, 273,
        280, 285, 286, 287, 289, 290, 292, 300, 301, 304, 310, 311,
    },
    "counterfeit": {
        2, 5, 9, 14, 16, 20, 25, 29, 30, 39, 41, 50, 64, 66, 68, 71, 77, 84, 90, 94, 95, 99, 101, 123, 124, 129, 141, 142,
        146, 148, 149, 151, 154, 168, 179, 188, 189, 193, 194, 195, 199, 204, 210, 214, 219, 228, 233, 241, 242, 255, 256,
        294, 301, 307, 315,
    },
}


def stratum_of(labels, flagged):
    for label in TARGETS:
        if labels[label]:
            return f"yes_{label}"
    return "no_flagged" if flagged else "no_random"


key = {int(number): row for number, row in json.loads((HERE / "self_check_key.json").read_text(encoding="utf-8")).items()}
records = load_five_label_records(include_live_check=False)
view = keyword_view(records)
dev = [i for i, r in enumerate(records) if str(r["round"]) in YARDSTICK_ROUNDS and not held_out(r)
       and all(r["labels"].get(label) is not None for label in TARGETS)]


def flagged(i):
    return any(view["flags"][label][i] for label in TARGETS)


population = Counter(stratum_of({label: bool(records[i]["labels"][label]) for label in TARGETS}, flagged(i)) for i in dev)
sample = Counter(stratum_of(row["labels"], flagged(row["index"])) for row in key.values())
weight = {name: population[name] / sample[name] for name in sample}
print("population", dict(population))
print("sample    ", dict(sample))
report = {}
for label in TARGETS:
    both = first_only = second_only = 0
    w_both = w_first = w_second = 0.0
    disagreements = {"first_only": [], "second_only": []}
    rule_right_first = rule_right_second = rule_flags = 0
    for number, row in key.items():
        first, second = row["labels"][label], number in SECOND[label]
        w = weight[stratum_of(row["labels"], flagged(row["index"]))]
        if first and second:
            both += 1
            w_both += w
        elif first:
            first_only += 1
            w_first += w
            disagreements["first_only"].append(number)
        elif second:
            second_only += 1
            w_second += w
            disagreements["second_only"].append(number)
        if view["flags"][label][row["index"]]:
            rule_flags += 1
            rule_right_first += first
            rule_right_second += second
    f1 = 2 * w_both / (2 * w_both + w_first + w_second)
    report[label] = {
        "yes_both_readings": both, "yes_first_only": first_only, "yes_second_only": second_only,
        "first_yes_kept": round(both / (both + first_only), 3),
        "second_yes_that_were_yes_before": round(both / (both + second_only), 3),
        "f1_in_sample": round(2 * both / (2 * both + first_only + second_only), 3),
        "f1_weighted_to_all_dev": round(f1, 3),
        "weighted": [round(w_both), round(w_first), round(w_second)],
        "rule_flags_in_sample": rule_flags, "rule_flags_right_first": rule_right_first, "rule_flags_right_second": rule_right_second,
        "disagreements": disagreements,
    }
    print(label, json.dumps({k: v for k, v in report[label].items() if k != "disagreements"}))
    print("   first only:", disagreements["first_only"])
    print("   second only:", disagreements["second_only"])
(HERE / "self_check_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
