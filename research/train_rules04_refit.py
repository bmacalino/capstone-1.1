"""Refit the shipped learned models under keyword rules 0.4 (a bug-fix release).

Keyword rules 0.4 stop the keyword checker flagging "Wala pong sira", "akala ko
sira, lowbat lang pala" and "pabili ulit pag nasira na" as defects. That
changes the keyword features the learned models read, so the shipped models
(hazard_classifier_v2) cannot be served with it. This refits them with the
SAME data and the SAME recipes, changing nothing else:

  safety  round 1, TK, highest threshold with out-of-fold recall >= 0.80
          (research/PREREGISTRATION_hazard_v4.md); the model decides
  defect  rounds 1 and 2, TK, best out-of-fold F1 of keywords OR (model AND
          not praise) (research/PREREGISTRATION_defect_v5.md)

Only the team's labels are used. Round 3 is not read, so it remains a held-out
check of the refit: research/measure_hazard_tagger.py --round 3 scores
v6_shipped against v8_rules04.

    python research/train_rules04_refit.py
    python research/train_rules04_refit.py --adopt research/hazard_tagger_measurement_round3_rules04.json
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from five_label_data import sentiment_labels  # noqa: E402
from train_defect_gate import ROUNDS, best_threshold  # noqa: E402
from train_hazard_classifier import SEED, choose_threshold, cross_validate, featurize, load_round, scores  # noqa: E402

RULES = "0.4"
SHIPPED = SENTIMENT_ROOT / "models" / "hazard_classifier_v2"
OUT = SENTIMENT_ROOT / "models" / "hazard_classifier_v2b"
REPORT = ROOT / "research" / "rules04_refit_training.json"
VERSION = "hazard-learned-2.1.0"


def adopt(report_path: Path) -> dict:
    """Record the round-3 comparison. The refit ships only if, on round 3
    (held out for both), defect F1 and recall are not lower and safety is not
    worse: it is a bug fix and must not cost a real report."""
    from defaketive_sentiment.hazard_learned import MANIFEST_NAME

    report = json.loads(report_path.read_bytes().decode("utf-8"))
    if report.get("labelling_round") != 3 or "v8_rules04" not in report["results"]:
        raise SystemExit(f"{report_path} is not a round-3 report with v8_rules04.")
    old, new = report["results"]["v6_shipped"], report["results"]["v8_rules04"]
    checks = {
        "defect_f1_not_lower": (new["defect"]["f1"] or 0) >= (old["defect"]["f1"] or 0),
        "defect_recall_not_lower": (new["defect"]["recall"] or 0) >= (old["defect"]["recall"] or 0),
        "safety_f1_not_lower": (new["safety"]["f1"] or 0) >= (old["safety"]["f1"] or 0),
        "safety_recall_not_lower": (new["safety"]["recall"] or 0) >= (old["safety"]["recall"] or 0),
    }
    keys = ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")
    decision = {
        "adopted": all(checks.values()),
        "checks": checks,
        "round3_shipped": {c: {k: old[c][k] for k in keys} for c in ("safety", "defect")},
        "round3_refit": {c: {k: new[c][k] for k in keys} for c in ("safety", "defect")},
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report": str(report_path.relative_to(ROOT)) if report_path.is_relative_to(ROOT) else str(report_path),
    }
    path = OUT / MANIFEST_NAME
    manifest = json.loads(path.read_bytes().decode("utf-8"))
    manifest["adoption"] = decision
    manifest["qualified_for_integration"] = decision["adopted"]
    path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return decision


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--adopt":
        decision = adopt(Path(sys.argv[2]).resolve())
        print(json.dumps({"adopted": decision["adopted"], "checks": decision["checks"]}, indent=2))
        return 0
    import joblib

    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, file_sha256, fit_category_model, keyword_file_hashes

    shipped = json.loads((SHIPPED / MANIFEST_NAME).read_bytes().decode("utf-8"))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    categories = {}
    report = {}

    # Safety: round 1 only, as in checker 0.4.
    records = load_round(ROUNDS[0])
    feats = featurize(records, need_embeddings=False, rules=RULES)
    idx = np.array([i for i, record in enumerate(records) if record["labels"]["safety"] is not None])
    y = np.array([int(records[i]["labels"]["safety"]) for i in idx])
    groups = np.array([records[i]["review_id"] for i in idx])
    out_of_fold = cross_validate("TK", "safety", idx, y, groups, feats)
    threshold = choose_threshold(y, out_of_fold, "safety")
    model = fit_category_model("TK", list(feats["normalized"][idx]), feats["kw"][idx], None, y, category="safety", seed=SEED)
    model.threshold = threshold
    joblib.dump(model, OUT / "safety.joblib")
    categories["safety"] = {
        "file": "safety.joblib", "sha256": file_sha256(OUT / "safety.joblib"), "candidate": "TK", "threshold": threshold,
        "combine": "decide", "needs_embedding": False, "training_rows": int(len(idx)), "training_positives": int(y.sum()),
        "training_data": "labelling round 1 (A1, A2)",
    }
    report["safety"] = {
        "threshold": threshold, "shipped_threshold": shipped["categories"]["safety"]["threshold"],
        "cross_validated": scores(y, (out_of_fold >= threshold).astype(int)),
    }
    print(f"safety: {len(idx)} sentences, {int(y.sum())} reports, threshold {threshold:.4f} (shipped {report['safety']['shipped_threshold']:.4f})", flush=True)

    # Defect: rounds 1 and 2, gated, as in the adopted v5 design.
    records = []
    for folder in ROUNDS:
        records += load_round(folder)
    feats = featurize(records, need_embeddings=False, rules=RULES)
    idx = np.array([i for i, record in enumerate(records) if record["labels"]["defect"] is not None])
    y = np.array([int(records[i]["labels"]["defect"]) for i in idx])
    groups = np.array([records[i]["review_id"] for i in idx])
    keyword = np.array([int("defect" in feats["keyword_tags"][i]) for i in idx])
    praise = sentiment_labels([records[i] for i in idx]) == "positive"
    out_of_fold = cross_validate("TK", "defect", idx, y, groups, feats)
    threshold, stats = best_threshold(y, keyword, out_of_fold, praise, gated=True)
    model = fit_category_model("TK", list(feats["normalized"][idx]), feats["kw"][idx], None, y, category="defect", seed=SEED)
    model.threshold = threshold
    joblib.dump(model, OUT / "defect.joblib")
    categories["defect"] = {
        "file": "defect.joblib", "sha256": file_sha256(OUT / "defect.joblib"), "candidate": "TK", "threshold": threshold,
        "combine": "or_gated", "gate": "sentence sentiment is not positive", "needs_embedding": False,
        "training_rows": int(len(idx)), "training_positives": int(y.sum()),
        "training_data": "labelling rounds 1 (A1, A2) and 2 (A1, A2, A3)",
    }
    report["defect"] = {
        "threshold": threshold, "shipped_threshold": shipped["categories"]["defect"]["threshold"],
        "cross_validated": stats, "keywords_alone": scores(y, keyword),
    }
    print(f"defect: {len(idx)} sentences, {int(y.sum())} reports, threshold {threshold:.4f} (shipped {report['defect']['shipped_threshold']:.4f}) {stats}", flush=True)

    manifest = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "refit_of": "hazard_classifier_v2: same data and recipes, keyword features taken under rules 0.4",
        "keyword_rules": RULES,
        "categories": categories,
        "keyword_files": keyword_file_hashes(),
        "embedding_model": None,
        "sentiment_model": shipped["sentiment_model"],
        "qualified_for_integration": False,
        "adoption": {"status": "pending: compared with hazard_classifier_v2 on labelling round 3 (held out for both)"},
    }
    (OUT / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    REPORT.write_bytes((json.dumps({"generated_at": manifest["trained_at"], "keyword_rules": RULES, **report}, indent=2) + "\n").encode("utf-8"))
    print(f"frozen {OUT}; wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
