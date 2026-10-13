"""Does adding "kung" to HYPOTHETICAL_WORDS cost real defect reports now?

hazard_model.py's own 0.4 changelog says "kung" was tried once (7 Oct 2026)
and rejected: it cancelled a real report on the team's rounds ("paano ito
magagamit kung sira" - how can this even be used, given it's broken). That
was against rounds 1-3 only. The labelled corpus has since grown (rounds
4-9 + live_check), so this re-measures on everything available now, the
same way research/check_keyword_guards.py and check_hinde_negator.py did
for the last two false-flag fixes.

Rule, fixed before it was run: same as the other two checks - keep only if
it adds no flag and removes no more than one right flag for every three
wrong flags removed.

    python research/check_kung_hedge.py
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

REPORT = ROOT / "research" / "kung_hedge_check.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model, registry
    from five_label_data import load_five_label_records

    model = registry.load_model(registry.HAZARD_BADGES)
    tagger = model.tagger
    records = [r for r in load_five_label_records(include_live_check=True) if r["labels"].get("defect") is not None]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    with_kung = frozenset(hazard_model.HYPOTHETICAL_WORDS) | {"kung"}
    without_kung = hazard_model.HYPOTHETICAL_WORDS
    for record in records:
        rating = record.get("rating") or None
        hazard_model.HYPOTHETICAL_WORDS = with_kung
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.HYPOTHETICAL_WORDS = without_kung
        before = tagger.tag_sentence(record["text"], rating)
        was, now = "defect" in before["tags"], "defect" in after["tags"]
        if was == now:
            continue
        truth = bool(record["labels"]["defect"])
        key = ("removed_" if was else "added_") + ("right" if truth == was else "wrong")
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
