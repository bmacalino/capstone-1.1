"""Select context-aware risk classifiers inside grouped CV, preserving the app."""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT / "sentiment-analysis"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from defaketive_sentiment.product_risk_context import context_pipeline
from defaketive_sentiment.manuscript import HybridManuscriptRiskModel
from research.cross_validate_firecs_product_risk import risk_metrics
from research.train_domain_adapted_product_risk_round3 import (
    ACTIVE, SEED, build_rows, checked_splits, connected_groups, read_json, sha256,
)
from research.train_risk_scoring_v1 import positive_probability
from train_marketplace_v2 import classifier_pipeline

OUTPUT = ROOT / "outputs/product_risk_round4"
ARTIFACT = ROOT / "sentiment-analysis/models/product_risk_round4_candidate.joblib"
PREVIOUS = ROOT / "outputs/domain_adapted_product_risk_round3"
CONFIGS = ("existing", "context", "context_lazada_weight2")


def fit_model(config, texts, labels, cohorts, indices):
    model = classifier_pipeline() if config == "existing" else context_pipeline()
    weights = np.where(cohorts[indices] == "firecs", 1., 2.) if config.endswith("weight2") else np.ones(len(indices))
    model.fit(texts[indices].tolist(), labels[indices], classifier__sample_weight=weights)
    return model


def operating_point(labels, probability, cohorts):
    """Training-only operating point: satisfy both recalls, then weighted macro-F1."""
    best = None
    for threshold in np.linspace(.20, .80, 121):
        pred = (probability >= threshold).astype(int)
        scores = {}
        for name, mask in [("lazada", cohorts != "firecs"), ("firecs", cohorts == "firecs")]:
            y, p = labels[mask], pred[mask]
            tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
            fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
            recall = tp / max(1, tp + fn)
            macro = (2 * tp / max(1, 2 * tp + fp + fn) + 2 * tn / max(1, 2 * tn + fp + fn)) / 2
            scores[name] = {"recall": recall, "macro_f1": macro}
        passes = scores["lazada"]["recall"] >= .8 and scores["firecs"]["recall"] >= .8
        weighted = .75 * scores["lazada"]["macro_f1"] + .25 * scores["firecs"]["macro_f1"]
        key = (int(passes), weighted, min(scores["lazada"]["recall"], scores["firecs"]["recall"]), -abs(float(threshold) - .5))
        if best is None or key > best[0]:
            best = (key, float(threshold), scores)
    return best


def inner_select(texts, labels, groups, cohorts, train, seed):
    split = list(checked_splits(texts, labels, groups, train, folds=4, seed=seed))
    entries = []
    for config in CONFIGS:
        probability = np.full(len(labels), np.nan)
        for fit, validation in split:
            model = fit_model(config, texts, labels, cohorts, fit)
            probability[validation] = positive_probability(model, texts[validation])
        if not np.isfinite(probability[train]).all():
            raise ValueError("Missing inner predictions")
        key, threshold, scores = operating_point(labels[train], probability[train], cohorts[train])
        entries.append({"config": config, "key": key, "threshold": threshold, "scores": scores})
    selected = max(entries, key=lambda row: row["key"])
    return selected, entries


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    texts = np.array([r["text"] for r in rows], dtype=object)
    labels = np.array([r["risk_detected"] for r in rows], dtype=int)
    cohorts = np.array([r["cohort"] for r in rows], dtype=object)
    groups = connected_groups(rows)
    old = {r["review_id"]: r for r in read_json(PREVIOUS / "paired_oof_predictions.json")}
    old_report = read_json(PREVIOUS / "training_report.json")
    for source in old_report["metadata"]["sources"]:
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("Round-3 sources changed")
    protected = {p: sha256(p) for p in [ACTIVE, ROOT / old_report["artifact"]]}
    pred = np.full(len(rows), -1); prob = np.full(len(rows), np.nan); fold_ids = np.full(len(rows), -1)
    fold_reports = []
    print(f"Evaluating {len(rows)} rows, {len(set(groups))} groups, {len(CONFIGS)} inner-selected configurations", flush=True)
    for fold, (train, test) in enumerate(checked_splits(texts, labels, groups, np.arange(len(rows)), folds=5, seed=SEED), 1):
        for i in test:
            saved = old[rows[i]["review_id"]]
            if saved["fold"] != fold or saved["actual_risk"] != int(labels[i]) or saved["review_text"] != rows[i]["review_text"]:
                raise ValueError("Comparison rows/folds differ from round 3")
        selected, entries = inner_select(texts, labels, groups, cohorts, train, SEED + 100 * fold)
        model = fit_model(selected["config"], texts, labels, cohorts, train)
        prob[test] = positive_probability(model, texts[test]); pred[test] = (prob[test] >= selected["threshold"]).astype(int); fold_ids[test] = fold
        fold_reports.append({"fold": fold, "train_rows": len(train), "test_rows": len(test), "selected": selected, "inner_candidates": entries, "test_metrics": risk_metrics(labels[test], pred[test], prob[test])})
        checkpoint = {"status": "in_progress", "folds": fold_reports, "predictions": pred.tolist(), "probabilities": [float(p) if np.isfinite(p) else None for p in prob]}
        (OUTPUT / "progress.json").write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
        print(f"Fold {fold}/5: {selected['config']} threshold={selected['threshold']:.3f}", flush=True)
    if (pred < 0).any() or not np.isfinite(prob).all():
        raise ValueError("Missing held-out predictions")
    comparison = {}
    old_pred = np.array([old[r["review_id"]]["models"]["with_hard_examples"]["risk"] for r in rows])
    old_prob = np.array([old[r["review_id"]]["models"]["with_hard_examples"]["probability"] for r in rows])
    for name, mask in [("overall", np.ones(len(rows), dtype=bool)), ("lazada_all", cohorts != "firecs")] + [(c, cohorts == c) for c in ["lazada_legacy", "lazada_hard", "firecs"]]:
        comparison[name] = {"round3": risk_metrics(labels[mask], old_pred[mask], old_prob[mask]), "round4": risk_metrics(labels[mask], pred[mask], prob[mask])}
    print("Outer evaluation complete. Selecting final configuration on development OOF only...", flush=True)
    selected, final_entries = inner_select(texts, labels, groups, cohorts, np.arange(len(rows)), SEED + 900)
    final_model = fit_model(selected["config"], texts, labels, cohorts, np.arange(len(rows)))
    gate_scores = {d: comparison[d]["round4"] for d in ["lazada_all", "firecs"]}
    gates = {f"{d}_{m}": score["risk"]["recall"] >= .8 if m == "recall_at_least_0_80" else score["macro_f1"] >= .75 for d,score in gate_scores.items() for m in ["recall_at_least_0_80", "macro_f1_at_least_0_75"]}
    metadata = {**old_report["metadata"], "model_version": "defaketive-product-risk-round4-context-candidate", "created_at": datetime.now(timezone.utc).isoformat(), "risk_detection_threshold": selected["threshold"], "selected_config": selected["config"], "qualified_for_integration": False, "qualified_for_new_locked_test": all(gates.values()), "threshold_selection": "four-fold grouped development OOF; evaluation uses nested five outer/four inner grouped folds and inner-only model selection", "architecture_selection": "75% Lazada + 25% FiReCS macro-F1, prioritizing recall >=80% on both; selected only on inner predictions", "code_sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in [Path(__file__), ROOT / "sentiment-analysis/defaketive_sentiment/product_risk_context.py"]]}
    active = joblib.load(ACTIVE)
    artifact = {"metadata": metadata, "sentiment_pipeline": active["sentiment_pipeline"], "risk_detection_pipeline": final_model, "risk_detection_threshold": selected["threshold"]}
    joblib.dump(artifact, ARTIFACT)
    loaded = joblib.load(ARTIFACT)
    if not np.allclose(positive_probability(final_model, texts[:20]), positive_probability(loaded["risk_detection_pipeline"], texts[:20])):
        raise ValueError("Artifact roundtrip changed predictions")
    smoke = HybridManuscriptRiskModel(ARTIFACT, allow_unqualified=True).analyze("Hindi gumagana ang isang earbud.")
    if not isinstance(smoke["risk"]["detected"], bool) or not np.isfinite(smoke["risk"]["confidence"]):
        raise ValueError("App inference compatibility failed")
    if joblib.hash(active["sentiment_pipeline"]) != joblib.hash(loaded["sentiment_pipeline"]):
        raise ValueError("Sentiment classifier changed")
    if any(sha256(p) != expected for p,expected in protected.items()):
        raise ValueError("Existing app or candidate artifact changed")
    report = {"metadata": metadata, "comparison": comparison, "folds": fold_reports, "final_selection": selected, "final_candidates": final_entries, "development_gates": gates, "deployment_changed": False, "artifact": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": sha256(ARTIFACT), "limitations": old_report["limitations"] + ["Feature engineering followed inspection of development errors; these are iterative development estimates, not an untouched test.", "Older Lazada annotations contain apparent product-only policy conflicts; no labels were changed to improve metrics."], "integrity": {"same_held_out_rows_and_folds_as_round3": True, "all_training_and_thresholds_exclude_outer_test": True, "original_artifacts_unchanged": True, "sentiment_preserved": True, "app_inference_smoke_passed": True, "excluded_test_rows_used": 0}}
    predictions = [{"review_id": r["review_id"], "cohort": r["cohort"], "product_id": r["product_id"], "group": str(groups[i]), "fold": int(fold_ids[i]), "review_text": r["review_text"], "actual": int(labels[i]), "round3": int(old_pred[i]), "round4": int(pred[i]), "probability": float(prob[i])} for i,r in enumerate(rows)]
    for filename, value in [("training_report.json", report), ("oof_predictions.json", predictions)]:
        (OUTPUT / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "progress.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    print(json.dumps({"comparison": comparison, "gates": gates, "selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
