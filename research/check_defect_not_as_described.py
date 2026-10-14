"""Does removing a learned-only "defect" flag from a disputed-spec clause
cost real reports?

Found reading every learned-model-only defect flag in the labelled corpus
(13 Oct 2026, continuing the same night's investigation): "not 3000.", "TV
is only 20 inches not 24.", "hindi sya fast charging." are reviewers
disputing a claimed spec, not reporting a broken product - the same
distinction the checker already makes for COUNTERFEIT via the same
not_as_described_spans() the "iba sa picture" / disputed-number rules use.
Reuses that existing, already-measured detector rather than building a new
one: if a clause has no keyword-backed defect match AND disputes a spec,
the learned model's own "defect" addition is removed.

Isolates just this one gate by patching LexiconHazardTagger._not_as_described
to always return no spans for the "before" pass (everything else about the
tagger - rules, negation, the learned model - stays identical), the same
idea as toggling a module constant in the other checks in this series.

Rule, fixed before it was run: keep only if it adds no flag and removes no
more than one right flag for every three wrong flags removed.

    python research/check_defect_not_as_described.py
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

REPORT = ROOT / "research" / "defect_not_as_described_check.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model, registry
    from five_label_data import load_five_label_records

    model = registry.load_model(registry.HAZARD_BADGES)
    tagger = model.tagger
    records = [r for r in load_five_label_records(include_live_check=True) if r["labels"].get("defect") is not None]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    real_method = hazard_model.LexiconHazardTagger._not_as_described
    for record in records:
        rating = record.get("rating") or None
        hazard_model.LexiconHazardTagger._not_as_described = real_method
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.LexiconHazardTagger._not_as_described = lambda self, normalized: []
        before = tagger.tag_sentence(record["text"], rating)
        hazard_model.LexiconHazardTagger._not_as_described = real_method
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
