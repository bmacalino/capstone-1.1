"""Evaluate frozen risk detectors on the adjudicated, targeted Lazada set.

This is a hard-case diagnostic, not a representative accuracy or product-score
estimate. No fitting, threshold selection, or deployment occurs here.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "sentiment-analysis"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from research.create_lazada_hard_example_round1 import normalized_text
from research.cross_validate_firecs_product_risk import risk_metrics
from research.evaluate_firecs_product_risk_locked_lazada import (
    predict_artifact, product_metrics, sha256,
)
from train_marketplace_v2 import load_marketplace

DATA = ROOT / "outputs/lazada_hard_example_annotation_round_1"
OUTPUT = ROOT / "outputs/lazada_hard_example_baseline_round_1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(rows):
    result = {r["review_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate review IDs")
    return result


def main():
    queue = read_json(DATA / "lazada_hard_example_round1_queue.json")
    annotations = indexed(read_json(DATA / "lazada_hard_example_round1_codex_annotations.json"))
    audit = indexed(read_json(DATA / "lazada_hard_example_round1_selection_audit.json"))
    manifest = read_json(DATA / "manifest.json")
    ids = set(indexed(queue))
    if len(queue) != 250 or ids != set(annotations) or ids != set(audit):
        raise ValueError("Expected exactly 250 matching IDs in queue, annotations and audit")
    if Counter(r["human_risk_detected"] for r in annotations.values()) != {"yes": 73, "no": 177}:
        raise ValueError("Unexpected or unresolved annotation labels")
    for row in queue:
        row.update(annotations[row["review_id"]])
        row["risk_detected"] = int(row["human_risk_detected"] == "yes")
    texts = [normalized_text(r["review_text"]) for r in queue]
    if len(set(texts)) != 250:
        raise ValueError("Duplicate normalized evaluation reviews")
    labels = np.array([r["risk_detected"] for r in queue])
    candidate_path = ROOT / manifest["candidate_model"]
    if sha256(candidate_path) != manifest["candidate_model_sha256"]:
        raise ValueError("Selection candidate artifact has changed")
    training = read_json(ROOT / "sentiment-analysis/evaluation/domain_adapted_product_risk_round2_candidate_training.json")
    for source in training["metadata"]["sources"]:
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Training source has changed: {source['path']}")
    prior_lazada = load_marketplace(ROOT / manifest["source_labels"])
    firecs = read_json(ROOT / training["metadata"]["sources"][0]["path"])
    prior_texts = {r["text"] for r in prior_lazada} | {normalized_text(r["review_text"]) for r in firecs}
    overlapping_ids = [r["review_id"] for r, text in zip(queue, texts) if text in prior_texts]
    product_overlap = sorted({r["product_id"] for r in queue} & {r["product_id"] for r in prior_lazada})
    if overlapping_ids or product_overlap:
        raise ValueError(f"Prior-data overlap: reviews={overlapping_ids}, products={product_overlap}")

    report = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "targeted_hard_case_diagnostic",
            "rows": 250, "products": len({r["product_id"] for r in queue}),
            "labels": {"risk": 73, "no_risk": 177},
            "annotation_sources": dict(Counter(r["annotation_source"] for r in queue)),
            "label_review_pending": [],
            "user_approved_label_corrections": [
                {"review_id": "lazada-hard-r1-0154", "previous": "no", "approved": "yes", "reason": "Review reports unclear video as well as a wrong brand."},
                {"review_id": "lazada-hard-r1-0015", "previous": "yes", "approved": "no", "reason": "Charging a phone for about 50% does not explicitly establish a failure or depleted capacity."},
            ],
            "limitations": [
                "Reviews were selected using the round-2 candidate's predictions and uncertainty. Metrics are not representative population accuracy.",
                "244 labels are Codex annotations; six decisions were resolved or corrected by the user. This is not a fully independent human-labeled benchmark.",
                "Product metrics describe these selected reviews only and only the 70-point detected-risk contribution. No sentiment accuracy or full product score is validated.",
                "No fitting or threshold tuning performed. No app or model artifact updated.",
            ],
            "normalized_text_overlap_with_prior_lazada_and_firecs": overlapping_ids,
            "product_overlap_with_prior_lazada": product_overlap,
            "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in [
                DATA / "lazada_hard_example_round1_queue.json",
                DATA / "lazada_hard_example_round1_codex_annotations.json",
                DATA / "Lazada_Hard_Example_Risk_Annotation_Round_1_Adjudicated.xlsx",
                DATA / "manifest.json",
            ]],
        },
        "models": {},
    }
    prediction_rows = [{**r, "selection_bucket": audit[r["review_id"]]["selection_bucket"], "predictions": {}} for r in queue]
    for name, model_path in [
        ("round2_candidate", candidate_path),
        ("app_default_hybrid", ROOT / "sentiment-analysis/models/hybrid_manuscript_v1.joblib"),
    ]:
        artifact = joblib.load(model_path)
        predicted, probability, threshold = predict_artifact(artifact, texts)
        if name == "app_default_hybrid":
            # The app rounds class probabilities to six decimal places first.
            probability = np.array([round(float(p), 6) for p in probability])
            predicted = (probability >= threshold).astype(int)
        if name == "round2_candidate":
            if threshold != manifest["selection_threshold"]:
                raise ValueError("Candidate threshold differs from selection")
            if not np.array_equal(predicted, [audit[r["review_id"]]["model_prediction"] for r in queue]):
                raise ValueError("Candidate predictions do not reproduce selection audit")
            if not np.allclose(probability, [audit[r["review_id"]]["model_probability"] for r in queue], rtol=0, atol=1e-7):
                raise ValueError("Candidate probabilities do not reproduce selection audit")
        metrics = risk_metrics(labels, predicted, probability)
        grouped = {}
        for field in ["category", "selection_bucket"]:
            grouped[field] = {}
            for value in sorted({r[field] for r in prediction_rows}):
                indices = [i for i,r in enumerate(prediction_rows) if r[field] == value]
                grouped[field][value] = risk_metrics(labels[indices], predicted[indices])
        report["models"][name] = {
            "artifact": str(model_path.relative_to(ROOT)), "sha256": sha256(model_path),
            "threshold": threshold, "review_level": metrics,
            "selected_sample_product_level": product_metrics(queue, predicted),
            "by_group": grouped,
        }
        for i, row in enumerate(prediction_rows):
            row["predictions"][name] = {
                "risk": int(predicted[i]), "probability": float(probability[i]),
                "outcome": ("true_positive" if labels[i] else "false_positive") if predicted[i] else ("false_negative" if labels[i] else "true_negative"),
            }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, content in [("baseline_report.json", report), ("baseline_predictions.json", prediction_rows)]:
        (OUTPUT / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["review_level"] for name,value in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
