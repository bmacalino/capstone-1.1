"""Do the "hinde"/"d" lead-word variants on lexicon phrases ("hinde gumagana",
"d legit"...) and the counterfeit phrase builder's "hinde" change anything
wrong on the labelled corpus?

A lead-word variant can only ever ADD a match (a phrase that did not match
before now does), never remove one, so "removed_*" is expected to be zero by
construction - checked, not assumed, the same way check_hinde_negator.py
checked that an added negator could only remove matches.

Two bare LexiconHazardTagger instances (no learned model - this isolates the
keyword/phrase layer the change actually touches) are built with
LEXICON_LEAD_VARIANTS on and off, and every safety/defect/counterfeit-labelled
sentence is tagged by both.

Rule, fixed before it was run: keep only if every added flag is right, or at
least 70% are (the same bar research/improve_label_rules.py used for pattern
revisions) and nothing is removed.

    python research/check_lexicon_lead_variants.py
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

REPORT = ROOT / "research" / "lexicon_lead_variants_check.json"
CATEGORIES = ("safety", "defect", "counterfeit")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model
    from five_label_data import load_five_label_records

    records = [
        r for r in load_five_label_records(include_live_check=True)
        if any(r["labels"].get(c) is not None for c in CATEGORIES)
    ]

    saved = dict(hazard_model.LEXICON_LEAD_VARIANTS)
    hazard_model.LEXICON_LEAD_VARIANTS = {}
    before_tagger = hazard_model.LexiconHazardTagger(rules=hazard_model.SHIPPED_RULES, counterfeit_rules=hazard_model.SHIPPED_COUNTERFEIT_RULES)
    hazard_model.LEXICON_LEAD_VARIANTS = saved
    after_tagger = hazard_model.LexiconHazardTagger(rules=hazard_model.SHIPPED_RULES, counterfeit_rules=hazard_model.SHIPPED_COUNTERFEIT_RULES)

    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    for record in records:
        rating = record.get("rating") or None
        before = before_tagger.tag_sentence(record["text"], rating)
        after = after_tagger.tag_sentence(record["text"], rating)
        for category in CATEGORIES:
            truth = record["labels"].get(category)
            if truth is None:
                continue
            was, now = category in before["tags"], category in after["tags"]
            if was == now:
                continue
            # "right"/"wrong" is about the label itself (was a flag ever
            # warranted here), independent of which way it moved: adding a
            # flag is right iff truth is True, removing one is right iff
            # truth is True too (a removal is only GOOD when truth is False).
            key = ("removed_" if was else "added_") + ("right" if truth else "wrong")
            counts[key] += 1
            changed.append({"round": str(record["round"]), "category": category, "labelled_true": truth, "change": key, "text": record["text"]})
            print(f"{key:14} {category:11} r{record['round']} | {record['text'][:150]}")
    added_total = counts["added_right"] + counts["added_wrong"]
    added_precision = counts["added_right"] / added_total if added_total else 1.0
    passes = counts["removed_right"] == 0 and counts["removed_wrong"] == 0 and added_precision >= 0.70
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labelled_sentences": len(records),
        **counts,
        "added_precision": round(added_precision, 3),
        "rule": "keep if nothing is removed and at least 70% of added flags are right",
        "meets_rule": bool(passes),
        "changed": changed,
    }
    REPORT.write_bytes((json.dumps(report, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    print({k: v for k, v in report.items() if k != "changed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
