"""Retrain the learned hazard models on rounds 1-3 (no round 5).

Implements research/PREREGISTRATION_final_v6.md: the gated defect model and a
safety model used as a filter on keyword flags, both TK, thresholds from
5-fold cross-validation grouped by review. Frozen into a candidate folder;
round 5 is not read here and is scored once by
research/measure_hazard_tagger.py --round 5.

    python research/train_final_v6.py
    python research/train_final_v6.py --adopt research/hazard_tagger_measurement_round5.json
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

from train_defect_gate import PRECISION_MARGIN, best_threshold, sentiment_reader  # noqa: E402
from train_hazard_classifier import FOLDS, SEED, cross_validate, featurize, load_round, scores  # noqa: E402

OUTPUTS = ROOT / "outputs"
# Finished sheets per round; None = every worksheet_*_done.* in the folder.
ROUND_SHEETS: dict[int, list[str] | None] = {
    1: None,
    2: None,
    3: ["worksheet_A1_rhean_done.csv", "GUYO_worksheet_A2_round3.xlsx", "worksheet_Part3_A3_ryan.xlsx"],
}
SHIPPED = SENTIMENT_ROOT / "models" / "hazard_classifier_v2"
CANDIDATE = SENTIMENT_ROOT / "models" / "hazard_classifier_v3"
REPORT = ROOT / "research" / "final_v6_training.json"
PREREGISTRATION = "research/PREREGISTRATION_final_v6.md"
VERSION = "hazard-learned-3.0.0"
SHIPPED_CONFIG, CANDIDATE_CONFIG = "v6_shipped", "v6_candidate"


def load_records() -> list[dict[str, Any]]:
    records = []
    for number, sheets in ROUND_SHEETS.items():
        folder = OUTPUTS / f"labelling_queue_round_{number}"
        loaded = load_round(folder, [folder / name for name in sheets] if sheets else None)
        for record in loaded:
            record["round"] = str(number)
        records += loaded
    return records


def filter_threshold(y: np.ndarray, keyword: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Best out-of-fold F1 of keywords AND model; ties go to the lower threshold."""
    best = None
    for threshold in sorted(set(np.round(probability, 6)), reverse=True):
        stats = scores(y, (keyword.astype(bool) & (probability >= threshold)).astype(int))
        if best is None or stats["f1"] >= best[1]["f1"]:
            best = (float(threshold), stats)
    return best


def adopt(report_path: Path) -> dict[str, Any]:
    """The pre-registered round-5 rules, per category. A category that is not
    adopted gets its shipped model back, so the folder can ship as a whole."""
    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, file_sha256

    report = json.loads(report_path.read_bytes().decode("utf-8"))
    results = report.get("results", {})
    if report.get("labelling_round") != 5 or SHIPPED_CONFIG not in results or CANDIDATE_CONFIG not in results:
        raise SystemExit(f"{report_path} is not a round-5 report with {SHIPPED_CONFIG} and {CANDIDATE_CONFIG}.")
    manifest_path = CANDIDATE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    shipped = json.loads((SHIPPED / MANIFEST_NAME).read_bytes().decode("utf-8"))
    decisions: dict[str, Any] = {}
    for category in ("defect", "safety"):
        entry = manifest["categories"][category]
        if file_sha256(CANDIDATE / entry["file"]) != entry["sha256"]:
            raise SystemExit(f"{entry['file']} changed since it was frozen; refusing to decide on it.")
        old, new = results[SHIPPED_CONFIG][category], results[CANDIDATE_CONFIG][category]
        checks = {
            "f1_higher": (new["f1"] or 0) > (old["f1"] or 0),
            "recall_not_lower": (new["recall"] or 0) >= (old["recall"] or 0),
        }
        if category == "defect":
            checks["weighted_precision_within_margin"] = (new["weighted"]["precision"] or 0) >= (
                old["weighted"]["precision"] or 0
            ) - PRECISION_MARGIN
        decisions[category] = {
            "adopted": all(checks.values()),
            "checks": checks,
            "round5_shipped": {key: old[key] for key in ("precision", "recall", "f1")} | {"weighted": old["weighted"]},
            "round5_candidate": {key: new[key] for key in ("precision", "recall", "f1")} | {"weighted": new["weighted"]},
            "mcnemar": report["paired_comparisons"].get(f"{SHIPPED_CONFIG}_vs_{CANDIDATE_CONFIG}:{category}"),
        }
        if not decisions[category]["adopted"]:
            kept = shipped["categories"][category]
            name = f"{category}_shipped_v2.joblib"
            shutil.copyfile(SHIPPED / kept["file"], CANDIDATE / name)
            manifest["categories"][category] = {**kept, "file": name, "from": "hazard_classifier_v2 (candidate not adopted)"}
    manifest["adoption"] = {
        "rule": PREREGISTRATION,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path),
        "adopted": any(item["adopted"] for item in decisions.values()),
        "categories": decisions,
    }
    manifest["qualified_for_integration"] = manifest["adoption"]["adopted"]
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return decisions


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--adopt":
        decisions = adopt(Path(sys.argv[2]).resolve())
        print(json.dumps({name: {"adopted": item["adopted"], "checks": item["checks"]} for name, item in decisions.items()}, indent=2))
        if any(item["adopted"] for item in decisions.values()):
            print("Point hazard_learned.DEFAULT_MODEL_DIR at hazard_classifier_v3 to ship the adopted parts.")
        return 0

    import joblib

    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, file_sha256, fit_category_model, keyword_file_hashes

    records = load_records()
    feats = featurize(records, need_embeddings=False)
    read_sentiment, classifier = sentiment_reader()
    if CANDIDATE.exists():
        shutil.rmtree(CANDIDATE)
    CANDIDATE.mkdir(parents=True)
    categories: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for category, mode in (("safety", "and"), ("defect", "or_gated")):
        idx = np.array([i for i, record in enumerate(records) if record["labels"][category] is not None])
        y = np.array([int(records[i]["labels"][category]) for i in idx])
        groups = np.array([records[i]["review_id"] for i in idx])
        keyword = np.array([int(category in feats["keyword_tags"][i]) for i in idx])
        out_of_fold = cross_validate("TK", category, idx, y, groups, feats)
        keyword_stats = scores(y, keyword)
        if mode == "and":
            threshold, stats = filter_threshold(y, keyword, out_of_fold)
            extra: dict[str, Any] = {"rule": "keywords AND model; a filter, never adds a flag"}
        else:
            labels = [read_sentiment(records[i]["text"]) for i in idx]
            positive = np.array([label == "positive" for label in labels])
            threshold, stats = best_threshold(y, keyword, out_of_fold, positive, gated=True)
            extra = {"gate": "sentence sentiment is not positive", "sentence_sentiment": dict(Counter(labels))}
        print(f"{category}: {len(idx)} sentences, {int(y.sum())} reports", flush=True)
        print(f"  keywords alone {keyword_stats}")
        print(f"  candidate      thr {threshold:.4f} {stats}")
        final = fit_category_model(
            "TK", list(feats["normalized"][idx]), feats["kw"][idx], None, y, category=category, seed=SEED
        )
        final.threshold = threshold
        joblib.dump(final, CANDIDATE / f"{category}.joblib")
        categories[category] = {
            "file": f"{category}.joblib",
            "sha256": file_sha256(CANDIDATE / f"{category}.joblib"),
            "candidate": "TK",
            "threshold": threshold,
            "combine": mode,
            "needs_embedding": False,
            "training_rows": int(len(idx)),
            "training_positives": int(y.sum()),
            "training_data": "labelling rounds 1 (A1, A2), 2 and 3 (A1, A2, A3)",
            **({"gate": extra["gate"]} if "gate" in extra else {}),
        }
        report[category] = {
            "sentences": int(len(idx)),
            "reports": int(y.sum()),
            "by_round": dict(Counter(records[i]["round"] for i in idx)),
            "keywords_alone": keyword_stats,
            "candidate": {"threshold": round(threshold, 6), **stats},
            **extra,
        }
    manifest = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": PREREGISTRATION,
        "categories": categories,
        "keyword_files": keyword_file_hashes(),
        "embedding_model": None,
        "sentiment_model": {"model_version": classifier.model_version, "model_sha256": classifier.manifest["files"]["model.onnx"]},
        "qualified_for_integration": False,
        "adoption": {"status": "pending: tested once on labelling round 5"},
    }
    (CANDIDATE / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    REPORT.write_bytes(
        (
            json.dumps(
                {
                    "generated_at": manifest["trained_at"],
                    "preregistration": PREREGISTRATION,
                    "cross_validation": {"folds": FOLDS, "grouped_by": "review_id", "seed": SEED},
                    "categories": report,
                    "note": "Cross-validated figures are in-sample for the keyword features; round 5 decides.",
                },
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    )
    print(f"frozen {CANDIDATE}; wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
