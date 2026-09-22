"""Reconcile adjudicated development labels and compare LR with calibrated SVM.

Original datasets and the live app are immutable. All selection happens inside
outer training folds. SVM probability calibration itself uses grouped folds.
"""
import csv
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "sentiment-analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from defaketive_sentiment.product_risk_context import context_pipeline, clean_risk_text
from defaketive_sentiment.manuscript import HybridManuscriptRiskModel
from research.train_domain_adapted_product_risk_round3 import (
    ACTIVE, SEED, build_rows, checked_splits, connected_groups, read_json, sha256,
)
from research.improve_product_risk_round4 import operating_point
from research.cross_validate_firecs_product_risk import risk_metrics
from research.evaluate_firecs_product_risk_locked_lazada import product_metrics
from research.train_risk_scoring_v1 import positive_probability

OUTPUT = ROOT / "outputs/product_risk_round5"
ADJUDICATED = ROOT / "outputs/annotation_quality_round_1/lazada_round1_validation_adjudicated.csv"
PENDING = {"lazada-r1-0076", "lazada-r1-0190", "lazada-r1-0904"}
CONFIGS = ("lr", "svm_0.1", "svm_1", "svm_4")


def reconcile(rows):
    """Restore existing decisions only, keyed by ID AND matching raw text."""
    rows = [dict(row) for row in rows]
    indexed = {row["review_id"]: row for row in rows}
    with ADJUDICATED.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    if len(source) != 138 or len({r["review_id"] for r in source}) != 138:
        raise ValueError("Unexpected adjudication source coverage")
    differences = []
    for record in source:
        row = indexed[record["review_id"]]
        if row["cohort"] != "lazada_legacy" or row["review_text"] != record["review_text"]:
            raise ValueError("Adjudication identity/text mismatch")
        if record["risk_detected"] not in {"yes", "no"}:
            raise ValueError("Invalid adjudicated binary label")
        new = int(record["risk_detected"] == "yes")
        if new == row["risk_detected"]:
            continue
        apply = row["review_id"] not in PENDING
        differences.append({"review_id": row["review_id"], "old": row["risk_detected"],
                            "adjudicated": new, "applied": apply,
                            "review_text": row["review_text"], "reason": record["annotation_notes"]})
        if apply:
            row["risk_detected"] = new
    if len(differences) != 31 or sum(r["applied"] for r in differences) != 29:
        raise ValueError("Re-audit changed adjudications before running")
    return rows, {"source": str(ADJUDICATED.relative_to(ROOT)), "sha256": sha256(ADJUDICATED),
                  "differences": differences, "restored_count": 29, "pending_ids": sorted(PENDING),
                  "policy": "Restore 29 existing adjudications; retain original labels for pending cases. No source edits."}


def calibration_splits(texts, labels, groups, indices, seed):
    # CalibratedClassifierCV receives a LOCAL subset, so remap global indices.
    positions = {int(index): position for position, index in enumerate(indices)}
    return [(np.array([positions[int(i)] for i in train]), np.array([positions[int(i)] for i in test]))
            for train, test in checked_splits(texts, labels, groups, indices, folds=3, seed=seed)]


def fit_model(config, texts, labels, groups, indices, seed):
    model = context_pipeline()
    if config.startswith("svm_"):
        model.set_params(classifier=LinearSVC(C=float(config.split("_")[1]),
                                             class_weight="balanced", max_iter=20000, random_state=SEED))
        model = CalibratedClassifierCV(model, method="sigmoid", ensemble=False,
                                      cv=calibration_splits(texts, labels, groups, indices, seed), n_jobs=1)
    elif config != "lr":
        raise ValueError(config)
    model.fit(texts[indices].tolist(), labels[indices])
    return model


def inner_select(texts, labels, groups, cohorts, train, seed):
    entries = []
    splits = list(checked_splits(texts, labels, groups, train, folds=4, seed=seed))
    for config in CONFIGS:
        probability = np.full(len(labels), np.nan)
        for number, (fit, validation) in enumerate(splits):
            model = fit_model(config, texts, labels, groups, fit, seed + 1000 + number)
            probability[validation] = positive_probability(model, texts[validation])
        if not np.isfinite(probability[train]).all():
            raise ValueError("Missing inner predictions")
        key, threshold, scores = operating_point(labels[train], probability[train], cohorts[train])
        entries.append({"config": config, "key": key, "threshold": threshold, "scores": scores})
        print(f"  {config}: inner threshold {threshold:.3f}", flush=True)
    return {"lr": entries[0], "svm": max(entries[1:], key=lambda row: row["key"])}, entries


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = build_rows()
    rows, audit = reconcile(original)
    write_json("label_audit.json", audit)
    texts = np.array([r["text"] for r in rows], dtype=object)
    labels = np.array([r["risk_detected"] for r in rows], dtype=int)
    cohorts = np.array([r["cohort"] for r in rows], dtype=object)
    groups = connected_groups(rows)
    cleaner_groups = {}
    for text, group in zip(texts, groups):
        clean = clean_risk_text(text)
        if clean in cleaner_groups and cleaner_groups[clean] != group:
            raise ValueError("Cleaning creates cross-group duplicates")
        cleaner_groups[clean] = group
    previous = read_json(ROOT / "outputs/product_risk_round4/training_report.json")
    old_predictions = {r["review_id"]: r for r in read_json(ROOT / "outputs/product_risk_round4/oof_predictions.json")}
    source_paths = [ROOT / r["path"] for r in previous["metadata"]["sources"]] + [ADJUDICATED]
    protected = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths + [ACTIVE, ROOT / previous["artifact"]]}
    old_labels = np.array([r["risk_detected"] for r in original])
    results = {name: {"pred": np.full(len(rows), -1), "prob": np.full(len(rows), np.nan), "folds": []}
               for name in ("lr", "svm")}
    folds = np.full(len(rows), -1)
    # Freeze the previous outer assignment so both algorithms have exactly the
    # same products held out, despite corrected labels. Inner folds use new labels.
    outer = list(checked_splits(texts, old_labels, groups, np.arange(len(rows)), folds=5, seed=SEED))
    for fold, (train, test) in enumerate(outer, 1):
        if any(old_predictions[rows[i]["review_id"]]["fold"] != fold for i in test):
            raise ValueError("Previous outer folds changed")
        print(f"Outer fold {fold}/5: selecting using {len(train)} training rows", flush=True)
        selected, candidates = inner_select(texts, labels, groups, cohorts, train, SEED + 100 * fold)
        folds[test] = fold
        for name, choice in selected.items():
            model = fit_model(choice["config"], texts, labels, groups, train, SEED + 2000 + fold)
            result = results[name]
            result["prob"][test] = positive_probability(model, texts[test])
            result["pred"][test] = (result["prob"][test] >= choice["threshold"]).astype(int)
            result["folds"].append({"fold": fold, "selected": choice, "inner_candidates": candidates,
                                    "train_count": len(train), "test_count": len(test)})
        write_json("progress.json", {"status": "running", "completed_outer_folds": fold,
                                     "results": {name: {"pred": r["pred"].tolist(), "folds": r["folds"]} for name,r in results.items()}})
    comparison = {}
    for domain, mask in [("overall", np.ones(len(rows), bool)), ("lazada_all", cohorts != "firecs")] + [(c, cohorts == c) for c in ("lazada_legacy", "lazada_hard", "firecs")]:
        comparison[domain] = {}
        for name, result in results.items():
            if (result["pred"] < 0).any() or not np.isfinite(result["prob"]).all():
                raise ValueError("Incomplete outer predictions")
            comparison[domain][name] = risk_metrics(labels[mask], result["pred"][mask], result["prob"][mask])
    print("Outer evaluation done. Fitting separate development candidates.", flush=True)
    all_indices = np.arange(len(rows))
    final, entries = inner_select(texts, labels, groups, cohorts, all_indices, SEED + 900)
    active = joblib.load(ACTIVE)
    artifacts = {}
    for name, choice in final.items():
        model = fit_model(choice["config"], texts, labels, groups, all_indices, SEED + 3000)
        path = ROOT / f"sentiment-analysis/models/product_risk_round5_{name}_candidate.joblib"
        metadata = {**active["metadata"], "model_version": f"product-risk-round5-{name}-candidate",
                    "risk_detection_threshold": choice["threshold"], "qualified_for_integration": False,
                    "label_overlay": audit, "source_hashes": protected,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        candidate = {"metadata": metadata, "sentiment_pipeline": active["sentiment_pipeline"],
                     "risk_detection_pipeline": model, "risk_detection_threshold": choice["threshold"]}
        joblib.dump(candidate, path)
        loaded = joblib.load(path)
        assert np.allclose(positive_probability(model, texts[:30]), positive_probability(loaded["risk_detection_pipeline"], texts[:30]))
        assert joblib.hash(active["sentiment_pipeline"]) == joblib.hash(loaded["sentiment_pipeline"])
        smoke = HybridManuscriptRiskModel(path, allow_unqualified=True).analyze("Hindi gumagana ang isang earbud.")
        assert isinstance(smoke["risk"]["detected"], bool)
        artifacts[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "selection": choice}
    for path, expected in protected.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Protected source changed: {path}")
    lazada_indices = np.where(cohorts != "firecs")[0]
    sampled_product = {name: product_metrics([rows[i] for i in lazada_indices], r["pred"][lazada_indices]) for name,r in results.items()}
    report = {"comparison": comparison, "label_audit": audit, "folds": {n:r["folds"] for n,r in results.items()},
              "final_candidates": entries, "artifacts": artifacts, "sampled_product_risk_contribution": sampled_product,
              "protected_hashes": protected, "code_sha256": sha256(Path(__file__)),
              "protocol": "Frozen 5 outer product/text-group folds; 4 inner folds select C/threshold; SVM uses a further 3 grouped calibration folds with fold-local vocabularies. Same features and labels for both algorithms.",
              "deployment_changed": False, "limitations": [
                  "Iterative development comparison, not an untouched final test.",
                  "29 existing adjudications restored; three queried cases retain original labels pending confirmation.",
                  "Other legacy product-only label conflicts remain possible. No full manual reannotation is claimed.",
                  "FiReCS has no product IDs, so only normalized duplicates can be grouped there.",
                  "Selected review samples cannot validate population-level product risk scores.",
                  "SVM has a three-value C search while LR uses the existing C=2 baseline.",
              ]}
    predictions = [{"review_id": r["review_id"], "cohort": r["cohort"], "product_id": r["product_id"],
                    "group": str(groups[i]), "fold": int(folds[i]), "actual": int(labels[i]),
                    "original_actual": int(old_labels[i]),
                    "models": {n: {"pred": int(v["pred"][i]), "prob": float(v["prob"][i])} for n,v in results.items()}}
                   for i,r in enumerate(rows)]
    write_json("training_report.json", report)
    write_json("oof_predictions.json", predictions)
    lines = ["# Product-risk classifier comparison", "", "29 existing adjudications restored in a separate overlay. Original files and app unchanged.",
             "", "| Dataset | Model | Risk precision | Risk recall | Risk F1 | Macro-F1 |", "|---|---|---:|---:|---:|---:|"]
    for domain in ("lazada_all", "lazada_hard", "firecs"):
        for name, scores in comparison[domain].items():
            lines.append(f"| {domain} | {name} | {scores['risk']['precision']:.1%} | {scores['risk']['recall']:.1%} | {scores['risk']['f1']:.1%} | {scores['macro_f1']:.1%} |")
    lines += ["", report["protocol"], "", "## Limitations", ""] + [f"- {s}" for s in report["limitations"]]
    lines += ["", "Calibration implementation: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json("progress.json", {"status": "complete"})
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
