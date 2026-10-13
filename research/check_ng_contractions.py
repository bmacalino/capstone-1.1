"""Does reading "-ng"-contracted pronouns/particles (siyang, kong, bang...)
as transparent to negation cost real reports?

Found scanning freshly-scraped reviews (13 Oct 2026) for more "hinde"-shaped
gaps: "Wala naman po siyang sira" ("it has no damage") was wrongly flagged,
because the base transparent-word list only has "siya", not its "-ng"-linked
form "siyang". hazard_model.NEGATION_TRANSPARENT_CONTRACTIONS_04 adds the
contracted forms of words already in the base set, picked to avoid two
collision risks: "itong"/"tong" ("ito" + linker) can point AT the very thing
being complained about, and "akong" ("ako" + linker) repeats a pronoun this
file's own 0.4 changelog already tried and rejected.

Same method and rule as the other checks in this series: keep only if it
adds no flag and removes no more than one right flag for every three wrong
flags removed.

    python research/check_ng_contractions.py
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

REPORT = ROOT / "research" / "ng_contractions_check.json"
CATEGORIES = ("safety", "defect", "counterfeit")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from defaketive_sentiment import hazard_model, registry
    from five_label_data import load_five_label_records

    model = registry.load_model(registry.HAZARD_BADGES)
    tagger = model.tagger
    records = [
        r for r in load_five_label_records(include_live_check=True)
        if any(r["labels"].get(c) is not None for c in CATEGORIES)
    ]
    changed = []
    counts = {"removed_wrong": 0, "removed_right": 0, "added_right": 0, "added_wrong": 0}
    with_contractions = hazard_model.NEGATION_TRANSPARENT_CONTRACTIONS_04
    without_contractions = frozenset()
    for record in records:
        rating = record.get("rating") or None
        hazard_model.NEGATION_TRANSPARENT_CONTRACTIONS_04 = with_contractions
        after = tagger.tag_sentence(record["text"], rating)
        hazard_model.NEGATION_TRANSPARENT_CONTRACTIONS_04 = without_contractions
        before = tagger.tag_sentence(record["text"], rating)
        hazard_model.NEGATION_TRANSPARENT_CONTRACTIONS_04 = with_contractions
        for category in CATEGORIES:
            truth = record["labels"].get(category)
            if truth is None:
                continue
            was, now = category in before["tags"], category in after["tags"]
            if was == now:
                continue
            key = ("removed_" if was else "added_") + ("right" if truth else "wrong")
            counts[key] += 1
            changed.append({"round": str(record["round"]), "category": category, "labelled_true": truth, "change": key, "text": record["text"]})
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
    print({k: v for k, v in report.items() if k != "changed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
