"""Do the two context guards of 11 October 2026 cost real fault reports?

Two reviews the owner found in the app were flagged "Faulty or broken" for
the word "sira" where the product was not at fault: a list of damage that
ends at the box ("sira, punit at wasak2 na ung box") and an earlier unit that
lasted years ("tumagal ng 2 years bago nasira"). Revision g1 of keyword rules
0.4 (hazard_model.KEYWORD_GUARDS_REVISION) reads both as not a report, and
does not let the learned model put the flag back.

This runs the served checker with and without the guards on every labelled
sentence (five_label_data, the assistant's and the team's labels as loaded
there) and counts the "faulty or broken" flags that change.

Rule, fixed before it was run: the guards are kept only if they remove no
more than one right flag for every three wrong flags they remove, and add
none. With nothing removed on labelled sentences they are kept on the
strength of the two reviews and the unit tests, and that is said.

    python research/check_keyword_guards.py
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

REPORT = ROOT / "research" / "keyword_guards_check.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import registry
    from five_label_data import load_five_label_records

    model = registry.load_model(registry.HAZARD_BADGES)
    tagger = model.tagger
    records = [record for record in load_five_label_records(include_live_check=True) if record["labels"].get("defect") is not None]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    for record in records:
        rating = record.get("rating") or None
        tagger.context_guards = True
        after = tagger.tag_sentence(record["text"], rating)
        tagger.context_guards = False
        before = tagger.tag_sentence(record["text"], rating)
        tagger.context_guards = True
        was, now = "defect" in before["tags"], "defect" in after["tags"]
        if was == now:
            continue
        truth = bool(record["labels"]["defect"])
        key = ("removed_" if was else "added_") + ("right" if truth == was else "wrong")
        counts[key] += 1
        reasons = sorted({m["excluded"] for m in after["terms"] if m.get("excluded")})
        changed.append({"round": str(record["round"]), "labelled_faulty": truth, "change": key, "reasons": reasons, "text": record["text"]})
        print(f"{key:14} r{record['round']} {reasons} | {record['text'][:150]}")
    passes = counts["added_wrong"] == 0 and counts["added_right"] == 0 and 3 * counts["removed_right"] <= counts["removed_wrong"]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checker": model.checker_version, "labelled_sentences": len(records), **counts,
        "rule": "keep if no flag is added and removed_right <= removed_wrong / 3",
        "meets_rule": bool(passes), "changed": changed,
    }
    REPORT.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print({key: value for key, value in report.items() if key != "changed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
