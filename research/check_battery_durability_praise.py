"""Does blocking the learned model on "matagal/tumagal + lowbat" cost real reports?

Found from the owner's own screenshots (13 Oct 2026): "Functionality:Matagal
ma lobat" (slow to lose battery - good) and "Mabilis ma charge, matagal
malowbatt." (both halves good) were flagged "Faulty or broken" by the LEARNED
model alone (no keyword match at all - bare "lowbat"/"malowbat" is not a
registered term). The sentiment reader calls these short fragments "neutral",
never "positive", so the existing praise gate never blocks them, and their
probability (0.61-0.87) clears the high-star bar (0.637). hazard_model.
BATTERY_DURABILITY_PRAISE gates the learned "or_gated" decision directly:
"matagal/tumagal" right before a lowbat-family word is always praise in the
labelled data, "mabilis/madali/bilis" before the same word is always a real
report.

Two bare taggers (one with a learned scorer, one without a sentiment reader
needed) are not enough here since this gates the LEARNED decision, not a
keyword match - uses the real registry model, toggling the module pattern.

Rule, fixed before it was run: keep only if it adds no flag and removes no
more than one right flag for every three wrong flags removed, same as every
other check in this series.

    python research/check_battery_durability_praise.py
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

REPORT = ROOT / "research" / "battery_durability_praise_check.json"
NEVER_MATCHES = re.compile(r"(?!x)x")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model, registry
    from five_label_data import load_five_label_records

    model = registry.load_model(registry.HAZARD_BADGES)
    tagger = model.tagger
    records = [r for r in load_five_label_records(include_live_check=True) if r["labels"].get("defect") is not None]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    real_pattern = hazard_model.BATTERY_DURABILITY_PRAISE
    for record in records:
        rating = record.get("rating") or None
        hazard_model.BATTERY_DURABILITY_PRAISE = real_pattern
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.BATTERY_DURABILITY_PRAISE = NEVER_MATCHES
        before = tagger.tag_sentence(record["text"], rating)
        hazard_model.BATTERY_DURABILITY_PRAISE = real_pattern
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
        "checker": model.checker_version,
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
