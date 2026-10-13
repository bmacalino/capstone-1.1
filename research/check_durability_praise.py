"""Does the "matagal/tumagal masira" durability-praise guard cost real reports?

Found scanning freshly-scraped reviews (13 Oct 2026): "mabigat...matagal
masira...hoping na sana tumagal" (feels like it'll take a while to break,
hoping it lasts) was flagged "Faulty or broken" even though it praises
durability. hazard_model.DURABILITY_PRAISE_WORDS/_WISH add a new "excluded"
reason, "durability_praise" (DENIED_REASONS), narrowed to sentences that
also wish for the future ("sana"/"hoping"), so an already-broken complaint
("matagal na sira ito, wala pa ring refund") is not touched.

Same method and rule as the other checks in this series: toggle the guard's
own pattern between on and off (a pattern that never matches), on two bare
taggers, the same way check_keyword_guards.py toggled context_guards.

    python research/check_durability_praise.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

REPORT = ROOT / "research" / "durability_praise_check.json"
NEVER_MATCHES = re.compile(r"(?!x)x")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model
    from five_label_data import load_five_label_records

    records = [r for r in load_five_label_records(include_live_check=True) if r["labels"].get("defect") is not None]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    real_wish = hazard_model.DURABILITY_PRAISE_WISH
    tagger = hazard_model.LexiconHazardTagger(rules=hazard_model.SHIPPED_RULES, counterfeit_rules=hazard_model.SHIPPED_COUNTERFEIT_RULES)
    for record in records:
        rating = record.get("rating") or None
        hazard_model.DURABILITY_PRAISE_WISH = real_wish
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.DURABILITY_PRAISE_WISH = NEVER_MATCHES
        before = tagger.tag_sentence(record["text"], rating)
        hazard_model.DURABILITY_PRAISE_WISH = real_wish
        was, now = "defect" in before["tags"], "defect" in after["tags"]
        if was == now:
            continue
        truth = bool(record["labels"]["defect"])
        key = ("removed_" if was else "added_") + ("right" if truth else "wrong")
        counts[key] += 1
        changed.append({"round": str(record["round"]), "change": key, "labelled_faulty": truth, "text": record["text"]})
        print(f"{key:14} r{record['round']} | {record['text'][:150]}")
    passes = counts["added_wrong"] == 0 and counts["added_right"] == 0 and 3 * counts["removed_right"] <= counts["removed_wrong"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labelled_sentences": len(records),
        **counts,
        "rule": "keep if no flag is added and removed_right <= removed_wrong / 3",
        "meets_rule": bool(passes),
        "changed": changed,
    }
    REPORT.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print({k: v for k, v in report.items() if k != "changed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
