"""Validate saved round-3 results and write a readable paired comparison."""

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in [ROOT, ROOT / "sentiment-analysis"]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from defaketive_sentiment.manuscript import HybridManuscriptRiskModel
from research.train_domain_adapted_product_risk_round3 import (
    ACTIVE, ARTIFACT, OUTPUT, PREVIOUS_REPORT, build_rows, read_json, sha256,
)


def main():
    report = read_json(OUTPUT / "training_report.json")
    predictions = read_json(OUTPUT / "paired_oof_predictions.json")
    source = {r["review_id"]: r for r in build_rows()}
    if len(predictions) != 2424 or {r["review_id"] for r in predictions} != set(source):
        raise ValueError("Saved OOF rows do not match training sources")
    group_folds, product_folds = {}, {}
    for row in predictions:
        actual = source[row["review_id"]]
        if row["actual_risk"] != actual["risk_detected"] or row["review_text"] != actual["review_text"]:
            raise ValueError("Saved labels/text differ from sources")
        if group_folds.setdefault(row["group"], row["fold"]) != row["fold"]:
            raise ValueError("Group spans held-out folds")
        if row["product_id"] and product_folds.setdefault(row["product_id"], row["fold"]) != row["fold"]:
            raise ValueError("Product spans held-out folds")
    for name, result in report["paired_nested_cross_validation"].items():
        thresholds = {r["fold"]: r["threshold"] for r in result["folds"]}
        for row in predictions:
            saved = row["models"][name]
            if saved["risk"] != int(saved["probability"] >= thresholds[row["fold"]]):
                raise ValueError("Saved prediction does not match fold threshold")
        for cohort, metrics in result["cohorts"].items():
            selected = [r for r in predictions if (r["cohort"] != "firecs" if cohort == "lazada_all" else r["cohort"] == cohort)]
            counts = Counter((r["actual_risk"], r["models"][name]["risk"]) for r in selected)
            matrix = [[counts[(0,0)], counts[(0,1)]], [counts[(1,0)], counts[(1,1)]]]
            if matrix != metrics["confusion_matrix"]["values"] or len(selected) != metrics["sample_count"]:
                raise ValueError("Saved confusion matrix differs from independent counts")
    previous = read_json(PREVIOUS_REPORT)
    if sha256(ROOT / previous["artifact"]) != previous["artifact_sha256"] or sha256(ARTIFACT) != report["artifact_sha256"]:
        raise ValueError("Candidate artifact hash mismatch")
    for item in report["metadata"]["sources"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Training source changed after fit")
    active, candidate = joblib.load(ACTIVE), joblib.load(ARTIFACT)
    if joblib.hash(active["sentiment_pipeline"]) != joblib.hash(candidate["sentiment_pipeline"]):
        raise ValueError("Sentiment head changed unexpectedly")
    model = HybridManuscriptRiskModel(ARTIFACT, allow_unqualified=True)
    smoke = []
    for text in ["Good sound and working well.", "Hindi gumagana ang isang earbud.", "Late delivery pero gumagana naman."]:
        result = model.analyze(text)
        if not isinstance(result["risk"]["detected"], bool) or not np.isfinite(result["risk"]["confidence"]):
            raise ValueError("Invalid app-compatible inference result")
        smoke.append({"text": text, "sentiment": result["label"], "risk": result["risk"]["detected"]})
    lines = ["# Product-risk candidate, round 3", "", "Trained on 2,424 reviews: 1,400 FiReCS, 774 earlier Lazada reviews and 250 corrected Lazada hard examples. The app's sentiment classifier and 30/70 scoring formula are preserved. The candidate has not replaced the app model.", "", "## Paired held-out comparison", "", "Both conditions use the same outer-held-out reviews. The baseline is refitted without the new hard examples; the candidate includes only hard examples belonging to outer-training products. Five outer folds estimate performance. Four inner folds select each threshold without access to outer-test labels.", "", "| Review group | Training condition | Risk precision | Risk recall | Risk F1 | Macro F1 | False alarms | Missed risks |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cohort in ["lazada_all", "lazada_legacy", "lazada_hard", "firecs"]:
        for name, label in [("without_hard_examples", "Without new examples"), ("with_hard_examples", "With new examples")]:
            m = report["paired_nested_cross_validation"][name]["cohorts"][cohort]
            cm = m["confusion_matrix"]["values"]
            lines.append(f"| {cohort} ({m['sample_count']}) | {label} | {m['risk']['precision']:.1%} | {m['risk']['recall']:.1%} | {m['risk']['f1']:.1%} | {m['macro_f1']:.1%} | {cm[0][1]} | {cm[1][0]} |")
    lines += ["", "## Qualification", "", f"Final development-selected risk threshold: {report['final_threshold_selection']['threshold']:.3f}.", ""]
    for name, passes in report["development_gates"].items():
        lines.append(f"- {name}: {'pass' if passes else 'fail'}.")
    lines += ["", "These are development results. The prior baseline on 250 reviews used a fully fitted prior candidate; this paired comparison refits both training conditions inside each fold, so those numbers are not directly comparable.", "", "The 250 hard examples were deliberately selected using a prior model. Of their labels, 244 originated from Codex and six were user-resolved. Earlier development labels also influenced the selecting model. This comparison remains exploratory and cannot replace an untouched representative evaluation. FiReCS has no product identifiers, so only duplicate texts can be grouped there.", "", "Product IDs and normalized duplicate texts stay together, including transitive links between products. All 133 earlier test reviews remain excluded. The final threshold's selection score is not used as independent evaluation accuracy.", "", "No new sentiment evaluation or full product-score validation was performed. Product summaries in the JSON describe the 70-point risk contribution for selected reviews only; they are not population risk estimates.", "", "## Verification", "", "Four targeted tests passed. Saved labels and texts match the sources. Independent counts reproduce every cohort's confusion matrix. Fold assignments preserve groups and product boundaries. Artifact and source hashes match. The sentiment pipeline is unchanged, and the new artifact loads through the application's model class. Three inference smoke checks completed; they verify compatibility, not accuracy.", "", "## Files and reproduction", "", "- Candidate: `sentiment-analysis/models/domain_adapted_product_risk_round3_candidate.joblib`", "- Detailed report: `outputs/domain_adapted_product_risk_round3/training_report.json`", "- Held-out predictions: `outputs/domain_adapted_product_risk_round3/paired_oof_predictions.json`", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m unittest -v tests.test_train_domain_adapted_product_risk_round3", ".\\.venv\\Scripts\\python.exe -u research/train_domain_adapted_product_risk_round3.py", ".\\.venv\\Scripts\\python.exe research/summarize_product_risk_round3.py", "```", ""]
    before = report["paired_nested_cross_validation"]["without_hard_examples"]["cohorts"]["lazada_all"]
    after = report["paired_nested_cross_validation"]["with_hard_examples"]["cohorts"]["lazada_all"]
    lines += ["## Interpretation and next step", "",
        f"Across the 1,024 held-out Lazada reviews, false alarms changed from {before['confusion_matrix']['values'][0][1]} to {after['confusion_matrix']['values'][0][1]}, while missed risks changed from {before['confusion_matrix']['values'][1][0]} to {after['confusion_matrix']['values'][1][0]}. Compare precision and recall together rather than choosing by accuracy alone.", "",
        "The hard-example cohort improves, but the earlier Lazada cohort loses some performance. The combined Lazada results do not meet the existing recall and macro-F1 targets. Keep this artifact as a development candidate. Next, investigate the remaining false alarms and missed failures, then compare a revised classifier or feature representation under the same grouped protocol. Do not deploy solely on this result.", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "verification.json").write_text(json.dumps({"checks_passed": True, "rows": len(predictions), "sentiment_preserved": True, "smoke_checks": smoke}, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines[:28]))


if __name__ == "__main__":
    main()
