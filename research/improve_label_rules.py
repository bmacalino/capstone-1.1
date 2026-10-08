"""Revise the written rules for the three labels with the lowest F1, and
check the revision on sentences that were not read while writing it.

The labels: poor quality, possible fake, not as advertised. Their rules are
precise and miss many reports (research/RESULTS_TABLES.md, Table 11j). Two
attempts to add a trained model on top found more reports but about half of
the additions were wrong. This revises the rules themselves instead.

Split, fixed before anything was read: every labelled sentence belongs to
its review, and one review in three (by a hash of its id) is HELD OUT. Only
the other two thirds ("dev") are listed by --misses and read when new
patterns are written. The held-out third is scored and never listed.

Rule for keeping a revision, written first: for a label, the revised rules
are kept only if on the held-out third F1 is higher than before and the
share of flags that are right does not fall by more than 0.05.

What happened (8 October 2026; research/label_rules_revision.json). "before"
is the rules as they were; "after" is all three revisions; "final" is what
was kept. Poor quality and possible fake passed and are kept. The
not-as-advertised revision caught 41 more reports in dev and none in the
held-out third, so it was removed and that label's patterns are as before.

    python research/improve_label_rules.py --save before   # once, before editing the rules
    python research/improve_label_rules.py --misses quality
    python research/improve_label_rules.py --save after
    python research/improve_label_rules.py --save final    # the configuration that was kept
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from five_label_data import keyword_view, load_five_label_records  # noqa: E402
from train_hazard_classifier import scores  # noqa: E402

REPORT = ROOT / "research" / "label_rules_revision.json"
TARGETS = ("quality", "counterfeit", "not_as_described")
PRECISION_MAY_FALL = 0.05


# The held-out third is made of rounds 1-8 and stays as it is: rounds labelled
# later, and written sentences, are training data and never join it.
YARDSTICK_ROUNDS = frozenset(str(number) for number in range(1, 9))


def held_out(record: dict[str, Any]) -> bool:
    if str(record.get("round")) not in YARDSTICK_ROUNDS:
        return False
    return int(hashlib.sha1(str(record["review_id"]).encode("utf-8")).hexdigest(), 16) % 3 == 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_five_label_records(include_live_check=True)
    view = keyword_view(records)
    held = np.array([held_out(record) for record in records])

    if "--misses" in sys.argv:
        label = sys.argv[sys.argv.index("--misses") + 1]
        flags = view["flags"][label].astype(bool)
        dev = [i for i, record in enumerate(records) if not held[i] and record["labels"][label] is not None]
        missed = [i for i in dev if records[i]["labels"][label] and not flags[i]]
        wrong = [i for i in dev if not records[i]["labels"][label] and flags[i]]
        print(f"{label}: dev sentences {len(dev)}, missed reports {len(missed)}, wrong flags {len(wrong)}")
        print("MISSED")
        # The rules read normalized text, so that is what is listed.
        for i in missed:
            print(f"  r{records[i]['round']} | {view['text'][i][:170]}")
        print("WRONG FLAGS")
        for i in wrong:
            print(f"  r{records[i]['round']} | {view['text'][i][:170]}")
        return 0

    result: dict[str, Any] = {}
    for label in TARGETS:
        flags = view["flags"][label].astype(bool)
        rows = np.array([i for i, record in enumerate(records) if record["labels"][label] is not None])
        truth = np.array([int(records[i]["labels"][label]) for i in rows])
        result[label] = {
            name: {"sentences": int(part.sum()), **scores(truth[part], flags[rows][part].astype(int))}
            for name, part in (("dev", ~held[rows]), ("held_out", held[rows]))
        }
        print(label, json.dumps(result[label]))
    if "--save" in sys.argv:
        name = sys.argv[sys.argv.index("--save") + 1]
        report = json.loads(REPORT.read_bytes().decode("utf-8")) if REPORT.is_file() else {
            "split": "one review in three held out by a hash of its id; only the dev part was read",
            "rule": "keep a label's revised rules only if held-out F1 rises and precision falls by at most 0.05",
        }
        report[name] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "labels": result}
        if "before" in report and "after" in report:
            report["decision"] = {}
            for label in TARGETS:
                old, new = report["before"]["labels"][label]["held_out"], report["after"]["labels"][label]["held_out"]
                report["decision"][label] = {
                    "held_out_f1": [old["f1"], new["f1"]], "held_out_precision": [old["precision"], new["precision"]],
                    "keep": bool(new["f1"] > old["f1"] and new["precision"] >= old["precision"] - PRECISION_MAY_FALL),
                }
            print("decision:", json.dumps(report["decision"]))
        REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
        print(f"saved '{name}' in {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
