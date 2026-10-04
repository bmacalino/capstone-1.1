"""Train the gated learned defect model on rounds 1 and 2 (no round 3).

Implements research/PREREGISTRATION_defect_v5.md: one design (TK + complaint
gate), threshold from 5-fold cross-validation grouped by review, final model
trained on rounds 1 and 2, frozen into a candidate folder that also carries the
shipped safety model unchanged. Round 3 is not read here; it is scored once by
research/measure_hazard_tagger.py --round 3.

    python research/train_defect_gate.py
    python research/train_defect_gate.py --adopt research/hazard_tagger_measurement_round3.json
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from train_hazard_classifier import FOLDS, SEED, featurize, load_round, scores  # noqa: E402

ROUNDS = (ROOT / "outputs" / "labelling_queue_round_1", ROOT / "outputs" / "labelling_queue_round_2")
SHIPPED = SENTIMENT_ROOT / "models" / "hazard_classifier_v1"
CANDIDATE = SENTIMENT_ROOT / "models" / "hazard_classifier_v2"
REPORT = ROOT / "research" / "defect_gate_training.json"
CATEGORY = "defect"
VERSION = "hazard-learned-2.0.0"


def sentiment_reader():
    """The app's sentiment reading of one sentence (no star rating)."""
    from defaketive_sentiment.model import DefaketiveSentimentModel
    from defaketive_sentiment.transformer_sentiment import TransformerSentimentClassifier

    classifier = TransformerSentimentClassifier()
    model = DefaketiveSentimentModel(classifier=classifier)
    return (lambda text: model.analyze(text)["label"]), classifier


def decisions(keyword: np.ndarray, probability: np.ndarray, positive: np.ndarray, threshold: float, gated: bool) -> np.ndarray:
    add = probability >= threshold
    if gated:
        add &= ~positive
    return (keyword.astype(bool) | add).astype(int)


def best_threshold(y, keyword, probability, positive, gated) -> tuple[float, dict[str, Any]]:
    best = None
    for threshold in sorted(set(np.round(probability, 6))):
        stats = scores(y, decisions(keyword, probability, positive, threshold, gated))
        if best is None or (stats["f1"], threshold) >= (best[1]["f1"], best[0]):
            best = (float(threshold), stats)
    return best


PRECISION_MARGIN = 0.05


def adopt(report_path: Path) -> dict[str, Any]:
    """The pre-registered round-3 rule; records the decision in the manifest."""
    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, file_sha256

    report = json.loads(report_path.read_bytes().decode("utf-8"))
    if report.get("labelling_round") != 3 or "v5_candidate" not in report["results"]:
        raise SystemExit(f"{report_path} is not a round-3 report with a v5_candidate result.")
    manifest_path = CANDIDATE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    entry = manifest["categories"][CATEGORY]
    if file_sha256(CANDIDATE / entry["file"]) != entry["sha256"]:
        raise SystemExit("defect.joblib changed since it was frozen; refusing to adopt it.")
    k, g = report["results"]["v3_shipped"][CATEGORY], report["results"]["v5_candidate"][CATEGORY]
    checks = {
        "f1_higher": (g["f1"] or 0) > (k["f1"] or 0),
        "recall_not_lower": (g["recall"] or 0) >= (k["recall"] or 0),
        "weighted_precision_within_margin": (g["weighted"]["precision"] or 0)
        >= (k["weighted"]["precision"] or 0) - PRECISION_MARGIN,
    }
    stratum = report.get("results_by_stratum", {})
    decision = {
        "adopted": all(checks.values()),
        "checks": checks,
        "round3_keywords": {key: k[key] for key in ("precision", "recall", "f1")} | {"weighted": k["weighted"]},
        "round3_gated": {key: g[key] for key in ("precision", "recall", "f1")} | {"weighted": g["weighted"]},
        "mcnemar": report["paired_comparisons"].get(f"v3_shipped_vs_v5_candidate:{CATEGORY}"),
        "disagreement_stratum": {
            name: stratum.get(name, {}).get("defect_disagreement", {}).get(CATEGORY)
            for name in ("v3_shipped", "v5_candidate")
        },
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path),
    }
    manifest["adoption"] = {"rule": "research/PREREGISTRATION_defect_v5.md", **decision}
    manifest["qualified_for_integration"] = decision["adopted"]
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return decision


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--adopt":
        decision = adopt(Path(sys.argv[2]).resolve())
        print(json.dumps({"adopted": decision["adopted"], "checks": decision["checks"]}, indent=2))
        if decision["adopted"]:
            print("Adopted. Point hazard_learned.DEFAULT_MODEL_DIR at hazard_classifier_v2 to ship it.")
        return 0
    from sklearn.model_selection import StratifiedGroupKFold

    import joblib

    from defaketive_sentiment.hazard_learned import (
        MANIFEST_NAME,
        file_sha256,
        fit_category_model,
        keyword_file_hashes,
    )

    records = []
    for folder in ROUNDS:
        loaded = load_round(folder)
        for record in loaded:
            record["round"] = folder.name[-1]
        records += loaded
    feats = featurize(records, need_embeddings=False)
    read_sentiment, classifier = sentiment_reader()
    idx = np.array([i for i, r in enumerate(records) if r["labels"][CATEGORY] is not None])
    y = np.array([int(records[i]["labels"][CATEGORY]) for i in idx])
    groups = np.array([records[i]["review_id"] for i in idx])
    keyword = np.array([int(CATEGORY in feats["keyword_tags"][i]) for i in idx])
    labels = [read_sentiment(records[i]["text"]) for i in idx]
    positive = np.array([label == "positive" for label in labels])
    print(f"{len(idx)} labelled sentences, {int(y.sum())} defects; sentiment {dict(Counter(labels))}", flush=True)

    out_of_fold = np.zeros(len(idx))
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train, valid in folds.split(np.zeros(len(idx)), y, groups):
        model = fit_category_model(
            "TK", list(feats["normalized"][idx[train]]), feats["kw"][idx[train]], None, y[train],
            category=CATEGORY, seed=SEED,
        )
        out_of_fold[valid] = model.probability(list(feats["normalized"][idx[valid]]), feats["kw"][idx[valid]], None)

    threshold, gated_stats = best_threshold(y, keyword, out_of_fold, positive, gated=True)
    ungated_threshold, ungated_stats = best_threshold(y, keyword, out_of_fold, positive, gated=False)
    keyword_stats = scores(y, keyword)
    print(f"keywords alone        {keyword_stats}")
    print(f"ungated (reference)   thr {ungated_threshold:.4f} {ungated_stats}")
    print(f"GATED (the candidate) thr {threshold:.4f} {gated_stats}")
    blocked = int(((out_of_fold >= threshold) & ~keyword.astype(bool) & positive).sum())
    blocked_true = int(((out_of_fold >= threshold) & ~keyword.astype(bool) & positive & (y == 1)).sum())
    print(f"gate blocked {blocked} additions in positive sentences, of which {blocked_true} were real defects")

    final = fit_category_model(
        "TK", list(feats["normalized"][idx]), feats["kw"][idx], None, y, category=CATEGORY, seed=SEED
    )
    final.threshold = threshold
    if CANDIDATE.exists():
        shutil.rmtree(CANDIDATE)
    CANDIDATE.mkdir(parents=True)
    joblib.dump(final, CANDIDATE / "defect.joblib")
    shipped = json.loads((SHIPPED / MANIFEST_NAME).read_bytes().decode("utf-8"))
    shutil.copyfile(SHIPPED / shipped["categories"]["safety"]["file"], CANDIDATE / shipped["categories"]["safety"]["file"])
    manifest = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": "research/PREREGISTRATION_defect_v5.md",
        "categories": {
            "safety": {**shipped["categories"]["safety"], "combine": "decide", "from": "hazard_classifier_v1 (unchanged)"},
            CATEGORY: {
                "file": "defect.joblib",
                "sha256": file_sha256(CANDIDATE / "defect.joblib"),
                "candidate": "TK",
                "threshold": threshold,
                "combine": "or_gated",
                "gate": "sentence sentiment is not positive",
                "needs_embedding": False,
                "training_rows": int(len(idx)),
                "training_positives": int(y.sum()),
                "training_data": "labelling rounds 1 (A1, A2) and 2 (A1, A2, A3)",
            },
        },
        "keyword_files": keyword_file_hashes(),
        "embedding_model": None,
        "sentiment_model": {"model_version": classifier.model_version, "model_sha256": classifier.manifest["files"]["model.onnx"]},
        "qualified_for_integration": False,
        "adoption": {"status": "pending: tested once on labelling round 3"},
    }
    (CANDIDATE / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    report = {
        "generated_at": manifest["trained_at"],
        "preregistration": manifest["preregistration"],
        "data": {
            "sentences": int(len(idx)),
            "defects": int(y.sum()),
            "by_round": dict(Counter(records[i]["round"] for i in idx)),
            "sentence_sentiment": dict(Counter(labels)),
        },
        "cross_validation": {"folds": FOLDS, "grouped_by": "review_id", "seed": SEED},
        "keywords_alone": keyword_stats,
        "ungated_reference": {"threshold": round(ungated_threshold, 6), **ungated_stats},
        "gated_candidate": {"threshold": round(threshold, 6), **gated_stats},
        "gate_blocked_additions": {"total": blocked, "real_defects": blocked_true},
        "frozen": manifest["categories"][CATEGORY],
        "note": "Keyword features carry round-1 tuning (in-sample there); round 3 decides.",
    }
    REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(f"frozen {CANDIDATE / 'defect.joblib'} (threshold {threshold:.4f}); wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
