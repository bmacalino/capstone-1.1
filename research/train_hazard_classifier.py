"""Choose and train the learned hazard models on labelling round 1 only.

Implements research/PREREGISTRATION_hazard_v4.md:

  1. Build the round-1 dataset (A1/A2 labels; a sentence both labelled is kept
     only when they agree).
  2. For each category, 5-fold cross-validation grouped by review: out-of-fold
     probabilities for the learned candidates T, TK, E, EK; each candidate's
     threshold set on those probabilities (safety: highest threshold with
     recall >= 0.80; others: best F1).
  3. Choose the best learned candidate per category (ties within 0.01 F1 go to
     the simpler one); the keyword checker is reported for reference only,
     because its round-1 figures are in-sample.
  4. Train the chosen models on all of round 1 and freeze them, with a manifest
     of every file their features depend on.

Round 2 is not read here. It is scored once, afterwards, by
research/measure_hazard_tagger.py --round 2. Then `--adopt` applies the
pre-registered adoption rule to that report and records the decision in the
model manifest; categories not adopted are removed from the shipped folder.

    python research/train_hazard_classifier.py
    python research/train_hazard_classifier.py --adopt research/hazard_tagger_measurement_round2.json
"""

from __future__ import annotations

import argparse
import json
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

from measure_hazard_tagger import CATEGORIES, flag, read_worksheet  # noqa: E402

ROUND_1 = ROOT / "outputs" / "labelling_queue_round_1"
DEFAULT_OUTPUT = SENTIMENT_ROOT / "models" / "hazard_classifier_v1"
DEFAULT_REPORT = ROOT / "research" / "hazard_classifier_selection.json"
SIMPLEST_FIRST = ("T", "TK", "E", "EK")
TIE = 0.01
SAFETY_RECALL = 0.80
FOLDS = 5
SEED = 42
VERSION = "hazard-learned-1.0.0"


def load_round(queue_dir: Path, worksheets: list[Path] | None = None) -> list[dict[str, Any]]:
    """One record per sentence: text, review, stratum, weight and labels.

    A category's label is None when nobody labelled it or two annotators
    disagreed; otherwise it is the (shared) answer.
    """
    master = [
        row for row in read_worksheet(queue_dir / "queue_master.csv")
        if str(row.get("row_id") or "").strip() and str(row.get("is_practice") or "no").casefold() != "yes"
    ]
    paths = worksheets or sorted(
        path for path in queue_dir.glob("worksheet_*_done.*") if path.suffix in {".csv", ".xlsx"}
    )
    if not paths:
        raise SystemExit(f"No finished worksheets in {queue_dir}")
    answers: dict[str, list[dict[str, bool | None]]] = {}
    for path in paths:
        for row in read_worksheet(path):
            row_id = str(row.get("row_id") or "").strip()
            if row_id:
                answers.setdefault(row_id, []).append(
                    {category: flag(row.get("human_" + category)) for category in CATEGORIES}
                )
    records = []
    for row in master:
        given = answers.get(row["row_id"], [])
        labels: dict[str, bool | None] = {}
        for category in CATEGORIES:
            values = [answer[category] for answer in given if answer[category] is not None]
            labels[category] = values[0] if values and all(value == values[0] for value in values) else None
        try:
            weight = float(row.get("sampling_weight") or 1.0)
        except ValueError:
            weight = 1.0
        records.append(
            {
                "row_id": row["row_id"],
                "review_id": str(row.get("review_id") or row["row_id"]),
                "text": str(row.get("sentence_text") or ""),
                "stratum": str(row.get("stratum") or ""),
                "weight": weight,
                "annotators": len(given),
                "labels": labels,
            }
        )
    return records


def featurize(records: list[dict[str, Any]], need_embeddings: bool = True) -> dict[str, Any]:
    """Normalized text, keyword features and keyword tags (and embeddings)."""
    from defaketive_sentiment import text_norm
    from defaketive_sentiment.hazard_learned import keyword_features
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    tagger = LexiconHazardTagger()
    tagged = [tagger.tag_sentence(record["text"]) for record in records]
    features = {
        "normalized": np.array([text_norm.normalize_text(record["text"]) for record in records], dtype=object),
        "kw": np.vstack([keyword_features(item) for item in tagged]),
        "keyword_tags": [set(item.get("tags") or []) for item in tagged],
        "emb": None,
        "embedding_model": None,
    }
    if need_embeddings:
        from defaketive_sentiment.transformer_sentiment import TransformerSentimentClassifier

        encoder = TransformerSentimentClassifier()
        features["emb"] = np.vstack([encoder.embed(record["text"]) for record in records])
        features["embedding_model"] = {
            "model_version": encoder.model_version,
            "model_sha256": encoder.manifest["files"]["model.onnx"],
        }
    return features


def scores(y: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def choose_threshold(y: np.ndarray, probability: np.ndarray, category: str) -> float:
    """The pre-registered threshold rule, applied to out-of-fold probabilities."""
    results = []
    for threshold in sorted(set(np.round(probability, 6))):
        stats = scores(y, (probability >= threshold).astype(int))
        results.append((threshold, stats))
    if category == "safety":
        reaching = [(t, s) for t, s in results if s["recall"] >= SAFETY_RECALL]
        if reaching:
            return float(max(t for t, _ in reaching))
        best = max(results, key=lambda item: (item[1]["recall"], item[1]["f1"]))
        return float(best[0])
    best = max(results, key=lambda item: (item[1]["f1"], item[0]))
    return float(best[0])


def cross_validate(code: str, category: str, idx: np.ndarray, y: np.ndarray, groups: np.ndarray, feats: dict) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    from defaketive_sentiment.hazard_learned import fit_category_model

    out = np.zeros(len(idx))
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train, valid in folds.split(np.zeros(len(idx)), y, groups):
        tr, va = idx[train], idx[valid]
        emb = feats["emb"]
        model = fit_category_model(
            code, list(feats["normalized"][tr]), feats["kw"][tr], None if emb is None else emb[tr], y[train],
            category=category, seed=SEED,
        )
        out[valid] = model.probability(list(feats["normalized"][va]), feats["kw"][va], None if emb is None else emb[va])
    return out


def select(records: list[dict[str, Any]], feats: dict) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for category in CATEGORIES:
        idx = np.array([i for i, record in enumerate(records) if record["labels"][category] is not None])
        y = np.array([int(records[i]["labels"][category]) for i in idx])
        groups = np.array([records[i]["review_id"] for i in idx])
        keyword = np.array([int(category in feats["keyword_tags"][i]) for i in idx])
        entry: dict[str, Any] = {
            "rows": int(len(idx)),
            "positives": int(y.sum()),
            "reviews": int(len(set(groups))),
            "keywords_in_sample": scores(y, keyword),
            "candidates": {},
        }
        for code in SIMPLEST_FIRST:
            if code.startswith("E") and feats["emb"] is None:
                continue
            probability = cross_validate(code, category, idx, y, groups, feats)
            threshold = choose_threshold(y, probability, category)
            stats = scores(y, (probability >= threshold).astype(int))
            entry["candidates"][code] = {"threshold": round(threshold, 6), **stats}
            print(f"  {category:11} {code:3} thr {threshold:.4f}  P {stats['precision']:.3f} R {stats['recall']:.3f} F1 {stats['f1']:.3f}", flush=True)
        best = max(item["f1"] for item in entry["candidates"].values())
        chosen = next(code for code in SIMPLEST_FIRST if code in entry["candidates"] and entry["candidates"][code]["f1"] >= best - TIE)
        entry["chosen"] = chosen
        report[category] = entry
    return report


def train_final(records, feats, selection, out_dir: Path) -> dict[str, Any]:
    import joblib

    from defaketive_sentiment.hazard_learned import (
        MANIFEST_NAME,
        file_sha256,
        fit_category_model,
        keyword_file_hashes,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    categories = {}
    needs_embedding = False
    for category, entry in selection.items():
        code = entry["chosen"]
        idx = np.array([i for i, record in enumerate(records) if record["labels"][category] is not None])
        y = np.array([int(records[i]["labels"][category]) for i in idx])
        emb = feats["emb"]
        model = fit_category_model(
            code, list(feats["normalized"][idx]), feats["kw"][idx], None if emb is None else emb[idx], y,
            category=category, seed=SEED,
        )
        model.threshold = entry["candidates"][code]["threshold"]
        path = out_dir / f"{category}.joblib"
        joblib.dump(model, path)
        needs_embedding |= model.needs_embedding
        categories[category] = {
            "file": path.name,
            "sha256": file_sha256(path),
            "candidate": code,
            "threshold": model.threshold,
            "needs_embedding": model.needs_embedding,
            "training_rows": int(len(idx)),
            "training_positives": int(y.sum()),
        }
    manifest = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_data": "labelling round 1 (annotators A1, A2; A3 sheet missing)",
        "preregistration": "research/PREREGISTRATION_hazard_v4.md",
        "categories": categories,
        "keyword_files": keyword_file_hashes(),
        "embedding_model": feats["embedding_model"] if needs_embedding else None,
        "qualified_for_integration": False,
        "note": "Adoption per category is decided on round 2 (research/measure_hazard_tagger.py --round 2).",
    }
    (out_dir / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def adopt(model_dir: Path, round_two_report: Path, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Apply the adoption rule: learned F1 higher than keywords 0.3.0 and
    recall not lower, on round 2. Rewrites the manifest; drops other files.

    `overrides` ({category: reason}) keeps the keyword checker for a category
    the rule would adopt. Each override is recorded with its reason, because
    it departs from the pre-registration.
    """
    overrides = overrides or {}
    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, file_sha256

    report = json.loads(round_two_report.read_bytes().decode("utf-8"))
    if report.get("labelling_round") != 2 or "v4_learned" not in report["results"]:
        raise SystemExit(f"{round_two_report} is not a round-2 report with a v4_learned result.")
    manifest_path = model_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    for category, entry in manifest["categories"].items():
        if file_sha256(model_dir / entry["file"]) != entry["sha256"]:
            raise SystemExit(f"{entry['file']} changed since it was frozen; refusing to adopt it.")
    keywords, learned = report["results"]["v3_shipped"], report["results"]["v4_learned"]
    decisions, adopted, rejected = {}, {}, {}
    previous = (manifest.get("adoption") or {}).get("decisions") or {}
    for category, entry in manifest["categories"].items():
        k, l = keywords[category], learned[category]
        by_rule = (l["f1"] or 0) > (k["f1"] or 0) and (l["recall"] or 0) >= (k["recall"] or 0)
        take = by_rule and category not in overrides
        comparison = report["paired_comparisons"].get(f"v3_shipped_vs_v4_learned:{category}", {})
        decisions[category] = {
            "adopted": take,
            "adopted_by_rule": by_rule,
            **({"override": overrides[category]} if category in overrides else {}),
            "round2_keywords": {key: k[key] for key in ("precision", "recall", "f1")},
            "round2_learned": {key: l[key] for key in ("precision", "recall", "f1")},
            "mcnemar_p": comparison.get("p_value"),
        }
        (adopted if take else rejected)[category] = entry
    for category, entry in rejected.items():
        (model_dir / entry["file"]).unlink(missing_ok=True)
    # Decisions from an earlier --adopt run for categories already removed.
    for category, decision in previous.items():
        decisions.setdefault(category, decision)
    manifest["categories"] = adopted
    manifest["not_adopted"] = {
        **(manifest.get("not_adopted") or {}),
        **{category: {"candidate": e["candidate"], "sha256": e["sha256"]} for category, e in rejected.items()},
    }
    manifest["adoption"] = {
        "rule": "adopt if round-2 F1 > keyword checker 0.3.0 and round-2 recall not lower",
        "round2_report": str(round_two_report.relative_to(ROOT)) if round_two_report.is_absolute() and round_two_report.is_relative_to(ROOT) else str(round_two_report),
        "decisions": decisions,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest["qualified_for_integration"] = bool(adopted)
    manifest["note"] = "Ships only the adopted categories; the others keep the keyword checker."
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROUND_1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--adopt", type=Path, help="Round-2 measurement report: apply the adoption rule.")
    parser.add_argument(
        "--keep-keywords", nargs=2, action="append", metavar=("CATEGORY", "REASON"), default=[],
        help="Override: keep the keyword checker for CATEGORY although the rule adopts it.",
    )
    args = parser.parse_args()
    if args.adopt:
        decisions = adopt(args.output, args.adopt.resolve(), dict(args.keep_keywords))
        print(json.dumps(decisions, indent=2))
        return 0

    records = load_round(args.queue)
    print(f"round 1: {len(records)} sentences, annotators per sentence {dict(Counter(r['annotators'] for r in records))}")
    feats = featurize(records)
    selection = select(records, feats)
    manifest = train_final(records, feats, selection, args.output)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistration": "research/PREREGISTRATION_hazard_v4.md",
        "data": {
            "sentences": len(records),
            "annotators_per_sentence": dict(Counter(r["annotators"] for r in records)),
            "worksheets": sorted(p.name for p in args.queue.glob("worksheet_*_done.*")),
        },
        "cross_validation": {"folds": FOLDS, "grouped_by": "review_id", "seed": SEED},
        "selection": selection,
        "frozen_models": manifest["categories"],
        "embedding_model": manifest["embedding_model"],
        "note": "keywords_in_sample is the keyword checker 0.3.0 on round 1, which it was tuned on; reference only.",
    }
    args.report.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print("chosen:", {c: e["chosen"] for c, e in selection.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
