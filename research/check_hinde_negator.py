"""Does reading "hinde" as a negator (13 October 2026) cost real reports?

A review the owner found in the app ("ok naman boses nya hinde basag" - the
sound is fine, not broken) was flagged "Faulty or broken" because "hinde", a
common informal spelling of "hindi", was not in the shared negator list
(text_norm.FILIPINO_NEGATORS has "hindi", "di", "ndi", "wala", "walang" but
not "hinde"). extra_labels.py already treated "hinde" as a negator for the
two newer labels; rules 0.4 (hazard_model.NEGATION_EXTRA_04,
NEGATION_EXTRA_04_REVISION) now does the same for safety, defect and
counterfeit, the three categories text_norm.is_negated_adjacent governs.

This runs the served checker with and without "hinde" in the negator list on
every labelled sentence (five_label_data) and counts the safety/defect/
counterfeit flags that change, the same way research/check_keyword_guards.py
checked the two guards of 11 October.

Rule, fixed before it was run: keep the fix only if it adds no flag and
removes no more than one right flag for every three wrong flags removed. A
negator addition can only ever cancel a match that was already made (it can
never create a new one), so "added_*" is expected to be zero by construction;
this is checked, not assumed.

    python research/check_hinde_negator.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

REPORT = ROOT / "research" / "hinde_negator_check.json"
CATEGORIES = ("safety", "defect", "counterfeit")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model, registry
    from five_label_data import load_five_label_records

    model = registry.load_model(registry.HAZARD_BADGES)
    tagger = model.tagger
    records = [
        record
        for record in load_five_label_records(include_live_check=True)
        if any(record["labels"].get(category) is not None for category in CATEGORIES)
    ]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    with_hinde = hazard_model.NEGATION_EXTRA_04
    without_hinde = frozenset()
    for record in records:
        rating = record.get("rating") or None
        hazard_model.NEGATION_EXTRA_04 = with_hinde
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.NEGATION_EXTRA_04 = without_hinde
        before = tagger.tag_sentence(record["text"], rating)
        hazard_model.NEGATION_EXTRA_04 = with_hinde
        for category in CATEGORIES:
            truth = record["labels"].get(category)
            if truth is None:
                continue
            was, now = category in before["tags"], category in after["tags"]
            if was == now:
                continue
            key = ("removed_" if was else "added_") + ("right" if truth == was else "wrong")
            counts[key] += 1
            changed.append(
                {
                    "round": str(record["round"]),
                    "category": category,
                    "labelled_true": truth,
                    "change": key,
                    "text": record["text"],
                }
            )
            print(f"{key:14} {category:11} r{record['round']} | {record['text'][:150]}")
    passes = counts["added_wrong"] == 0 and counts["added_right"] == 0 and 3 * counts["removed_right"] <= counts["removed_wrong"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checker": model.checker_version,
        "labelled_sentences": len(records),
        **counts,
        "rule": "keep if no flag is added and removed_right <= removed_wrong / 3",
        "meets_rule": bool(passes),
        "changed": changed,
    }
    REPORT.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print({key: value for key, value in report.items() if key != "changed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
