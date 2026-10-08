"""How each defect checker behaves on whole listings it was never trained on.

The eight live_check listings of round 8 (research/ordinary_review_pool.py)
took no part in any training set. Every sentence there that the team-label
models or the v4 model flagged was labelled, plus a random 110 of the
sentences neither flagged; the random ones stand for the rest by weight, so a
flag a newer model raises outside the labelled set is still counted fairly.
Each sentence is tagged with its review's star rating, as the app does, so
the v5 folder's higher bar for 4-5 star reviews is in play; one extra line
shows v5 with that bar switched off.

Labels are AI-proposed (research/label_round8_ai.py): this measures agreement
with them on ordinary reviews, which is what the labelled rounds, drawn from
enriched strata, cannot show. It also records which folder the app should
serve and why.

A last block scores the two rule-based labels (poor quality, not as
advertised) on all of round 8. Their patterns were written from rounds 1-6,
so round 8 is ordinary text they were never fitted to.

    python research/live_check.py
    python research/live_check.py --serve hazard_classifier_v5
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from ordinary_review_pool import checker  # noqa: E402

FOLDER = ROOT / "outputs" / "labelling_queue_round_8"
REPORT = ROOT / "research" / "live_check.json"
MODELS = ("hazard_classifier_v2b", "hazard_classifier_v4", "hazard_classifier_v5")
NAMES = {
    "keywords": "keywords alone (rules 0.4)",
    "hazard_classifier_v2b": "team-label models",
    "hazard_classifier_v4": "retrained on rounds 1-7",
    "hazard_classifier_v5": "retrained on rounds 1-8 (mining listings), higher bar in 4-5 star reviews",
    "hazard_classifier_v5_without_the_bar": "the same model without the 4-5 star bar",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def weighted(rows: list[dict[str, Any]], flagged: dict[str, bool]) -> dict[str, Any]:
    tp = sum(row["weight"] for row in rows if row["defect"] and flagged[row["row_id"]])
    fp = sum(row["weight"] for row in rows if not row["defect"] and flagged[row["row_id"]])
    fn = sum(row["weight"] for row in rows if row["defect"] and not flagged[row["row_id"]])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "flags": round(tp + fp, 1), "real": round(tp, 1), "false": round(fp, 1), "missed": round(fn, 1),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def rule_labels_on_round_8() -> dict[str, Any]:
    """Poor quality and not as advertised, as the app serves them, against the
    AI labels of every round-8 sentence. Unweighted: the round over-samples
    sentences a defect model flagged, so these are not listing-level rates."""
    from defaketive_sentiment.hazard_model import SHIPPED_COUNTERFEIT_RULES, SHIPPED_RULES, LexiconHazardTagger

    tagger = LexiconHazardTagger(rules=SHIPPED_RULES, counterfeit_rules=SHIPPED_COUNTERFEIT_RULES)
    rows = read(FOLDER / "ai_labels.csv")
    flags = [set(tagger.tag_sentence(row["sentence_text"])["extra_tags"]) for row in rows]
    out: dict[str, Any] = {"sentences": len(rows)}
    for label in ("quality", "not_as_described"):
        truth = [row[f"ai_{label}"] == "yes" for row in rows]
        tp = sum(1 for yes, found in zip(truth, flags) if yes and label in found)
        fp = sum(1 for yes, found in zip(truth, flags) if not yes and label in found)
        fn = sum(1 for yes, found in zip(truth, flags) if yes and label not in found)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        out[label] = {
            "reports": tp + fn, "caught": tp, "false": fp,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision and recall else None,
        }
    return out


def main() -> int:
    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, MODELS_ROOT
    from defaketive_sentiment.hazard_model import SHIPPED_RULES, LexiconHazardTagger

    master = {row["row_id"]: row for row in read(FOLDER / "queue_master.csv")}
    rows = []
    for row in read(FOLDER / "ai_labels.csv"):
        info = master[row["row_id"]]
        if info["part"] != "live_check":
            continue
        rows.append({
            "row_id": row["row_id"], "text": row["sentence_text"], "defect": row["ai_defect"] == "yes",
            "rating": info["rating"],
            "high_star": info["rating"] in ("4", "5"), "weight": float(info["pool"]) / float(info["sampled"]),
        })
    taggers: dict[str, Any] = {"keywords": LexiconHazardTagger(rules=SHIPPED_RULES)}
    for name in MODELS:
        if (MODELS_ROOT / name / MANIFEST_NAME).is_file():
            taggers[name] = checker(name)
            if taggers[name].learned.high_star_threshold:
                taggers[f"{name}_without_the_bar"] = checker(name)
                taggers[f"{name}_without_the_bar"].learned.high_star_threshold = {}
    results: dict[str, Any] = {}
    for name, tagger in taggers.items():
        flagged = {row["row_id"]: "defect" in tagger.tag_sentence(row["text"], row["rating"])["tags"] for row in rows}
        results[name] = {
            "all_reviews": weighted(rows, flagged),
            "four_and_five_star": weighted([row for row in rows if row["high_star"]], flagged),
            "one_to_three_star": weighted([row for row in rows if not row["high_star"]], flagged),
        }
        print(NAMES[name])
        for part, stats in results[name].items():
            print(f"  {part:20} flags {stats['flags']:6} real {stats['real']:6} false {stats['false']:6} missed {stats['missed']:6}"
                  f" | P {stats['precision']} R {stats['recall']} F1 {stats['f1']}")
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Agreement with AI-proposed labels on listings no model was trained on. Not accuracy.",
        "sentences_labelled": len(rows),
        "sentences_represented": round(sum(row["weight"] for row in rows)),
        "defect_reports_represented": round(sum(row["weight"] for row in rows if row["defect"])),
        "results": results,
        "rule_labels_round_8": rule_labels_on_round_8(),
    }
    print(f"rule labels on round 8: {report['rule_labels_round_8']}")
    if "--serve" in sys.argv:
        serve = sys.argv[sys.argv.index("--serve") + 1]
        report["served"] = serve
        # The team-label folder keeps its own record: it was adopted on the team's labels.
        for name in MODELS[1:]:
            path = MODELS_ROOT / name / MANIFEST_NAME
            if not path.is_file():
                continue
            manifest = json.loads(path.read_bytes().decode("utf-8"))
            manifest["adoption"] = {
                **{k: v for k, v in manifest.get("adoption", {}).items() if k == "round_3_team_labels_held_out_for_both"},
                "adopted": name == serve,
                **({
                    "adopted_by": "owner instruction",
                    "instruction": "fix, improve or replace if model performs poorly (7 October 2026)",
                    "not_a_preregistered_test": "research/PREREGISTRATION_final_v6.md, departure 2",
                    "fallback": "DEFAKETIVE_HAZARD_MODEL=team_labels",
                } if name == serve else {"status": "not served; see research/live_check.json"}),
                "live_check": {key: results[key]["all_reviews"] for key in results},
            }
            path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
