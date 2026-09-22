"""Paired nested grouped CV before/after adding the 250 Lazada hard examples."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "sentiment-analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from research.cross_validate_firecs_product_risk import risk_metrics
from research.evaluate_firecs_product_risk_locked_lazada import product_metrics, sha256
from research.train_domain_adapted_product_risk import build_development_rows
from research.train_firecs_product_risk import normalized_text
from research.train_risk_scoring_v1 import positive_probability, select_risk_threshold
from train_marketplace_v2 import classifier_pipeline, load_marketplace

DATA = ROOT / "outputs/lazada_hard_example_annotation_round_1"
OUTPUT = ROOT / "outputs/domain_adapted_product_risk_round3"
ARTIFACT = ROOT / "sentiment-analysis/models/domain_adapted_product_risk_round3_candidate.joblib"
ACTIVE = ROOT / "sentiment-analysis/models/hybrid_manuscript_v1.joblib"
PREVIOUS_REPORT = ROOT / "sentiment-analysis/evaluation/domain_adapted_product_risk_round2_candidate_training.json"
FIRECS = ROOT / "outputs/firecs_risk_annotation_round_2_targeted/firecs_product_only_combined_development.json"
LAZADA = ROOT / "outputs/lazada_gold_test_v1/lazada_round1_all_adjudicated.csv"
SEED = 42


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows():
    previous = read_json(PREVIOUS_REPORT)
    for source in previous["metadata"]["sources"]:
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Changed previous training source: {source['path']}")
    firecs, lazada = build_development_rows(FIRECS, LAZADA)
    rows = [dict(r, cohort="firecs", domain="firecs", product_id="", text=normalized_text(r["review_text"])) for r in firecs]
    rows += [dict(r, review_text=r["review"], cohort="lazada_legacy", domain="lazada") for r in lazada]
    queue = read_json(DATA / "lazada_hard_example_round1_queue.json")
    annotations = read_json(DATA / "lazada_hard_example_round1_codex_annotations.json")
    by_id = {r["review_id"]: r for r in annotations}
    if len(queue) != 250 or len(by_id) != 250 or {r["review_id"] for r in queue} != set(by_id):
        raise ValueError("Incomplete or duplicate hard-example annotation IDs")
    if by_id["lazada-hard-r1-0154"]["human_risk_detected"] != "yes" or by_id["lazada-hard-r1-0015"]["human_risk_detected"] != "no":
        raise ValueError("Approved corrections are missing")
    prior = load_marketplace(LAZADA)
    prior_products = {r["product_id"] for r in prior}
    prior_texts = {r["text"] for r in prior} | {r["text"] for r in rows}
    for row in queue:
        annotation = by_id[row["review_id"]]
        label = annotation["human_risk_detected"]
        text = normalized_text(row["review_text"])
        if label not in {"yes", "no"} or not text:
            raise ValueError("Invalid label or empty review")
        if row["product_id"] in prior_products or text in prior_texts:
            raise ValueError(f"Hard-example overlap with prior data: {row['review_id']}")
        rows.append(dict(row, **{"risk_detected": int(label == "yes"), "text": text, "cohort": "lazada_hard", "domain": "lazada", "annotation_source": annotation["annotation_source"]}))
    if len(rows) != 2424 or len({r["review_id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected development size or duplicate IDs")
    if any(not r.get("review_text") for r in rows):
        raise ValueError("Every training record must preserve its source review text")
    excluded = [r for r in prior if r["split"] == "test"]
    excluded_products = {r["product_id"] for r in excluded}
    excluded_texts = {r["text"] for r in excluded}
    if any(r["product_id"] in excluded_products or r["text"] in excluded_texts for r in rows):
        raise ValueError("Excluded test product or text leaked into development")
    return rows


def connected_groups(rows):
    """Keep products AND normalized duplicates together, including transitive links."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        keys = [("text", row["text"])]
        if row["product_id"]:
            keys.append(("product", row["product_id"]))
        for key in keys:
            if key in owners:
                parent[find(i)] = find(owners[key])
            else:
                owners[key] = i
    return np.array([f"group:{find(i)}" for i in range(len(rows))], dtype=object)


def checked_splits(texts, labels, groups, indices, *, folds, seed):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    seen = []
    for train_pos, test_pos in splitter.split(indices, labels[indices], groups[indices]):
        train, test = indices[train_pos], indices[test_pos]
        if set(groups[train]) & set(groups[test]) or set(texts[train]) & set(texts[test]):
            raise ValueError("Product or duplicate-text leakage")
        if len(set(labels[train])) != 2 or len(set(labels[test])) != 2:
            raise ValueError("Fold missing a risk class")
        seen.extend(test.tolist())
        yield train, test
    if sorted(seen) != sorted(indices.tolist()):
        raise ValueError("Incomplete or duplicated validation coverage")


def select_threshold(texts, labels, groups, indices, *, folds, seed):
    probabilities = np.full(len(labels), np.nan)
    for train, validation in checked_splits(texts, labels, groups, indices, folds=folds, seed=seed):
        model = classifier_pipeline()
        model.fit(texts[train].tolist(), labels[train])
        probabilities[validation] = positive_probability(model, texts[validation].tolist())
    if not np.isfinite(probabilities[indices]).all():
        raise ValueError("Missing inner predictions")
    threshold, selection = select_risk_threshold(labels[indices], probabilities[indices])
    return threshold, selection


def training_indices(train, cohorts, *, augmented):
    return train if augmented else train[cohorts[train] != "lazada_hard"]


def main():
    rows = build_rows()
    texts = np.array([r["text"] for r in rows], dtype=object)
    labels = np.array([r["risk_detected"] for r in rows], dtype=int)
    cohorts = np.array([r["cohort"] for r in rows], dtype=object)
    groups = connected_groups(rows)
    all_indices = np.arange(len(rows))
    active_hash = sha256(ACTIVE)
    previous = read_json(PREVIOUS_REPORT)
    previous_path = ROOT / previous["artifact"]
    previous_hash = sha256(previous_path)
    print(json.dumps({"rows": len(rows), "cohorts": dict(Counter(cohorts)), "groups": len(set(groups)), "lazada_products": len({r['product_id'] for r in rows if r['product_id']})}), flush=True)
    results = {name: {"predictions": np.full(len(rows), -1), "probabilities": np.full(len(rows), np.nan), "folds": []} for name in ["without_hard_examples", "with_hard_examples"]}
    fold_assignments = np.full(len(rows), -1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for fold, (outer_train, test) in enumerate(checked_splits(texts, labels, groups, all_indices, folds=5, seed=SEED), start=1):
        fold_assignments[test] = fold
        for name, result in results.items():
            train = training_indices(outer_train, cohorts, augmented=name == "with_hard_examples")
            threshold, selection = select_threshold(texts, labels, groups, train, folds=4, seed=SEED + fold * 100)
            model = classifier_pipeline()
            model.fit(texts[train].tolist(), labels[train])
            probabilities = positive_probability(model, texts[test].tolist())
            predictions = (probabilities >= threshold).astype(int)
            result["predictions"][test] = predictions
            result["probabilities"][test] = probabilities
            result["folds"].append({"fold": fold, "train_rows": len(train), "test_rows": len(test), "train_cohorts": dict(Counter(cohorts[train])), "test_cohorts": dict(Counter(cohorts[test])), "threshold": threshold, "inner_selection": selection, "held_out_products": sorted({rows[i]['product_id'] for i in test if rows[i]['product_id']}), "metrics": risk_metrics(labels[test], predictions, probabilities)})
            print(f"Fold {fold}/5 {name}: {len(train)} training, {len(test)} held out, threshold={threshold:.3f}", flush=True)
            checkpoint = {"status": "in_progress", "completed_fold": fold, "completed_condition": name,
                "results": {key: {"predictions": value["predictions"].tolist(), "probabilities": [float(p) if np.isfinite(p) else None for p in value["probabilities"]], "folds": value["folds"]} for key,value in results.items()}}
            (OUTPUT / "progress.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summaries = {}
    for name, result in results.items():
        pred, prob = result["predictions"], result["probabilities"]
        if np.any(pred < 0) or not np.isfinite(prob).all():
            raise ValueError("Missing outer predictions")
        summaries[name] = {"overall": risk_metrics(labels, pred, prob), "cohorts": {}, "folds": result["folds"]}
        for cohort in ["firecs", "lazada_legacy", "lazada_hard", "lazada_all"]:
            mask = (cohorts != "firecs") if cohort == "lazada_all" else (cohorts == cohort)
            summaries[name]["cohorts"][cohort] = risk_metrics(labels[mask], pred[mask], prob[mask])
            if cohort.startswith("lazada"):
                summaries[name].setdefault("sample_product_metrics", {})[cohort] = product_metrics([rows[i] for i in np.flatnonzero(mask)], pred[mask])
    print("Selecting final threshold using development-only out-of-fold predictions...", flush=True)
    final_threshold, final_selection = select_threshold(texts, labels, groups, all_indices, folds=5, seed=SEED)
    risk_model = classifier_pipeline()
    risk_model.fit(texts.tolist(), labels)
    active = joblib.load(ACTIVE)
    sources = [FIRECS, LAZADA, DATA / "lazada_hard_example_round1_queue.json", DATA / "lazada_hard_example_round1_codex_annotations.json", DATA / "Lazada_Hard_Example_Risk_Annotation_Round_1_Adjudicated.xlsx", PREVIOUS_REPORT, ACTIVE]
    scores = summaries["with_hard_examples"]["cohorts"]
    gates = {f"{domain}_{metric}": scores[domain]["risk"]["recall"] >= .80 if metric == "risk_recall_at_least_0_80" else scores[domain]["macro_f1"] >= .75 for domain in ["firecs", "lazada_all"] for metric in ["risk_recall_at_least_0_80", "macro_f1_at_least_0_75"]}
    metadata = {
        "model_version": "defaketive-domain-adapted-product-risk-round3-candidate",
        "created_at": datetime.now(timezone.utc).isoformat(), "sklearn_version": sklearn.__version__,
        "deployment_status": "development_candidate", "qualified_for_integration": False,
        "qualified_for_new_locked_test": all(gates.values()), "training_rows": len(rows),
        "training_groups": len(set(groups)), "domain_rows": {"firecs": 1400, "lazada": 1024},
        "lazada_products": 27, "risk_distribution": dict(Counter(labels.tolist())),
        "risk_policy": "binary product-only yes/no; no categories or severity",
        "score_formula": "100 * (0.30 * negative_review_ratio + 0.70 * detected_risk_review_ratio)",
        "sentiment_policy": "preserved from app default hybrid artifact; not retrained",
        "random_state": SEED, "risk_detection_threshold": final_threshold,
        "threshold_selection": "five-fold grouped development OOF; reported evaluation uses four inner folds within five outer folds",
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources],
    }
    report = {
        "metadata": metadata, "paired_nested_cross_validation": summaries,
        "comparison_design": "Both training conditions predict exactly the same outer-held-out rows. The baseline excludes new hard examples from training and inner threshold selection. Features are fitted inside each fold.",
        "limitations": ["Development comparison, not an untouched external test.", "The 250 new reviews were selected using a prior model, and 244 labels are Codex annotations with six user decisions. Selection depended on a prior model trained on the legacy development data; results remain exploratory.", "FiReCS lacks product IDs; only normalized duplicates can be grouped there.", "Sample product metrics cover the 70-point risk contribution only. Targeted hard-review sampling cannot validate real product scores or population risk rates."],
        "integrity": {"excluded_test_rows": 133, "excluded_test_rows_used": 0, "product_and_normalized_duplicate_groups_checked_in_every_fold": True, "active_artifact_unchanged": sha256(ACTIVE) == active_hash, "previous_candidate_unchanged": sha256(previous_path) == previous_hash},
        "development_gates": gates, "passes_all_development_gates": all(gates.values()),
        "final_threshold_selection": {"threshold": final_threshold, **final_selection},
        "deployment_changed": False,
    }
    artifact = {"metadata": metadata, "sentiment_pipeline": active["sentiment_pipeline"], "risk_detection_pipeline": risk_model, "risk_detection_threshold": final_threshold}
    if not all(report["integrity"][k] for k in ["active_artifact_unchanged", "previous_candidate_unchanged"]):
        raise RuntimeError("Protected artifact changed during training")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    reloaded = joblib.load(ARTIFACT)
    if not np.allclose(positive_probability(risk_model, texts[:10]), positive_probability(reloaded["risk_detection_pipeline"], texts[:10])):
        raise RuntimeError("Saved artifact predictions differ after reload")
    report.update(artifact=str(ARTIFACT.relative_to(ROOT)), artifact_sha256=sha256(ARTIFACT))
    predictions = [{"review_id": r["review_id"], "product_id": r["product_id"], "cohort": r["cohort"], "group": groups[i], "fold": int(fold_assignments[i]), "actual_risk": int(labels[i]), "review_text": r["review_text"], "models": {name: {"risk": int(result["predictions"][i]), "probability": float(result["probabilities"][i])} for name,result in results.items()}} for i,r in enumerate(rows)]
    for filename, content in [("training_report.json", report), ("paired_oof_predictions.json", predictions)]:
        (OUTPUT / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "progress.json").write_text(json.dumps({"status": "complete", "report": "training_report.json", "predictions": "paired_oof_predictions.json"}) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": report["artifact"], "cohorts": {name: result["cohorts"] for name,result in summaries.items()}, "development_gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
