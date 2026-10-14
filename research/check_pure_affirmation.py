"""Does removing a learned-only "defect" flag from a pure "it works" clause
cost real reports?

Found the same night as the battery guard, reading every learned-model-only
defect flag in the labelled corpus: "Gumagana naman" (it works), "Gumagana
sa M3." are plain positive statements with no complaint of any kind, yet
score high enough (no keyword, no "positive" sentiment call on short
fragments) to clear the star bar. hazard_model.PURE_AFFIRMATION requires the
WHOLE clause to be nothing but the verb and mild fillers; ONLY_N_WORKS keeps
its hands off "isa/isang lang gumagana" (only one works), a real
partial-failure report. Checked by hand against every "gumagana/gumana/
nagana" sentence in five_label_data before being wired in: 0 matches among
201 TRUE sentences, 9 correct matches among FALSE ones.

Same method and rule as the other checks in this series.

    python research/check_pure_affirmation.py
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

REPORT = ROOT / "research" / "pure_affirmation_check.json"
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
    real_pattern = hazard_model.PURE_AFFIRMATION
    for record in records:
        rating = record.get("rating") or None
        hazard_model.PURE_AFFIRMATION = real_pattern
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.PURE_AFFIRMATION = NEVER_MATCHES
        before = tagger.tag_sentence(record["text"], rating)
        hazard_model.PURE_AFFIRMATION = real_pattern
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
