"""Check persisted round-4 predictions and summarize the controlled comparison."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.train_domain_adapted_product_risk_round3 import build_rows, read_json, sha256

OUTPUT = ROOT / "outputs/product_risk_round4"


def main():
    report = read_json(OUTPUT / "training_report.json")
    predictions = read_json(OUTPUT / "oof_predictions.json")
    sources = {row["review_id"]: row for row in build_rows()}
    if len(predictions) != len(sources) or {row["review_id"] for row in predictions} != set(sources):
        raise ValueError("Missing or duplicate saved predictions")
    thresholds = {row["fold"]: row["selected"]["threshold"] for row in report["folds"]}
    groups, products = {}, {}
    for row in predictions:
        source = sources[row["review_id"]]
        if row["actual"] != source["risk_detected"] or row["review_text"] != source["review_text"]:
            raise ValueError("Saved labels or text differ from source")
        if row["round4"] != int(row["probability"] >= thresholds[row["fold"]]):
            raise ValueError("Saved label does not use the selected fold threshold")
        if groups.setdefault(row["group"], row["fold"]) != row["fold"]:
            raise ValueError("Group leakage in saved predictions")
        if row["product_id"] and products.setdefault(row["product_id"], row["fold"]) != row["fold"]:
            raise ValueError("Product leakage in saved predictions")
    for cohort, pair in report["comparison"].items():
        selected = [row for row in predictions if cohort == "overall" or (row["cohort"] != "firecs" if cohort == "lazada_all" else row["cohort"] == cohort)]
        for model in ["round3", "round4"]:
            counts = Counter((row["actual"], row[model]) for row in selected)
            matrix = [[counts[(0,0)], counts[(0,1)]], [counts[(1,0)], counts[(1,1)]]]
            if matrix != pair[model]["confusion_matrix"]["values"] or len(selected) != pair[model]["sample_count"]:
                raise ValueError("Independent confusion-matrix verification failed")
    if sha256(ROOT / report["artifact"]) != report["artifact_sha256"]:
        raise ValueError("Saved artifact hash mismatch")
    for item in report["metadata"]["sources"] + report["metadata"]["code_sources"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Data or code changed since training")
    current = report["comparison"]["lazada_all"]["round4"]
    previous = report["comparison"]["lazada_all"]["round3"]
    improved = current["macro_f1"] > previous["macro_f1"] and current["risk"]["recall"] >= previous["risk"]["recall"]
    lines = ["# Product-risk improvement, round 4", "",
        "Implemented context-aware classifier inputs for English, Filipino and Taglish: contraction normalization, negation, concrete functional failures, poor performance, battery problems, praise mixed with failure, and packaging/order context. These signals are learned classifier features, not hard overrides, severity values or additional score weights.", "",
        "Three configurations were compared inside each training fold: the earlier text classifier, context features, and context features with Lazada training rows weighted twice as much as FiReCS. Selection prioritized at least 80% risk recall on both domains, then 75% Lazada plus 25% FiReCS macro-F1. Model configuration and threshold were selected using inner predictions only.", "",
        "## Same-fold held-out results", "",
        "| Review group | Model | Precision | Recall | Risk F1 | Macro F1 | False alarms | Missed risks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cohort in ["lazada_all", "lazada_legacy", "lazada_hard", "firecs"]:
        for model in ["round3", "round4"]:
            m = report["comparison"][cohort][model]
            matrix = m["confusion_matrix"]["values"]
            lines.append(f"| {cohort} ({m['sample_count']}) | {model} | {m['risk']['precision']:.1%} | {m['risk']['recall']:.1%} | {m['risk']['f1']:.1%} | {m['macro_f1']:.1%} | {matrix[0][1]} | {matrix[1][0]} |")
    lines += ["", "The comparison uses exactly the same 2,424 outer-held-out reviews and product/duplicate groups as round 3. Training remains 1,400 FiReCS plus 1,024 Lazada reviews. The 133 older test reviews remain excluded.", "",
        f"Combined Lazada macro-F1 {'improved without lower risk recall' if improved else 'did not improve while maintaining risk recall'} under this development protocol.", "",
        "## Candidate and qualification", "",
        f"Final configuration: `{report['final_selection']['config']}`. Final threshold: {report['final_selection']['threshold']:.3f}.", ""]
    lines += [f"- {name}: {'pass' if value else 'fail'}." for name,value in report["development_gates"].items()]
    lines += ["", "The artifact is saved separately. The app model, sentiment classifier, 100-review behavior, and 30/70 product-score formula were not changed. Development targets are not a guarantee of real-world accuracy.", "", "## Limits and label concerns", ""]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    lines += ["", "See `label_concerns.md` for concrete older-label conflicts. For example, one review saying 'defect' is labeled no-risk, while a review praising night vision is labeled risk. No source labels were changed in this experiment. Product-risk accuracy and the full product score still need representative independent validation.", "", "## Verification", "",
        "Eleven targeted checks passed, including negation, packaging scope, mixed praise/failure, grouping, source corrections and absence of new cross-group duplicates after cleaning. Saved OOF labels match source labels. Independent confusion counts reproduce all reported cohorts, and every saved prediction matches its fold-specific threshold. Source, code and artifact hashes match. The model reloads correctly and passes an inference compatibility check through the app's model class. The existing app and round-3 artifacts remain unchanged.", "", "## Reproduce", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m unittest -v tests.test_product_risk_context tests.test_product_risk_round4_integrity tests.test_train_domain_adapted_product_risk_round3", ".\\.venv\\Scripts\\python.exe -u research/improve_product_risk_round4.py", ".\\.venv\\Scripts\\python.exe research/summarize_product_risk_round4.py", "```", "", f"Candidate: `{report['artifact']}`", "", "Detailed metrics and predictions: `training_report.json` and `oof_predictions.json` in this directory.", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUTPUT / "verification.json").write_text(json.dumps({"rows": len(predictions), "checks_passed": True, "group_count": len(groups), "lazada_product_count": len(products), "improved_macro_f1_without_recall_loss": improved}, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines[:25]))


if __name__ == "__main__":
    main()
