"""Which model, if any, should stand behind each of the five labels?

For every label this compares, in 5-fold cross-validation grouped by review:

  rules           the keyword / rule flag alone (keyword rules 0.4, rules c2)
  lr_tk           TF-IDF word + character n-grams + keyword features, logistic
                  regression: the shipped design
  svm_tk          the same features, linear SVM
  lr_t            TF-IDF only
  lr_char         character n-grams only
  nb_word         complement naive Bayes on word n-grams
  lr_ek           sentence embedding of the shipped sentiment encoder + keyword features
  rules_or_lr_tk  the rule flag OR the logistic model (never on praise for
                  defect and quality): the hybrid

Each model's threshold is the one with the best out-of-fold F1, so every
figure is optimistic by the same amount and the comparison is fair between
rows, not a forecast. Labels are described in research/five_label_data.py:
only the team's safety answers are human. Nothing is saved but the report.

    python research/compare_label_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from five_label_data import LABELS, embeddings, keyword_view, load_five_label_records, sentiment_labels  # noqa: E402
from train_hazard_classifier import FOLDS, SEED, scores  # noqa: E402

REPORT = ROOT / "research" / "label_model_comparison.json"
GATED = ("defect", "quality")


def out_of_fold(kind: str, idx: np.ndarray, y: np.ndarray, groups: np.ndarray, view: dict[str, Any], emb: np.ndarray) -> np.ndarray:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.naive_bayes import ComplementNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    text, kw = view["text"], view["kw"]
    scores_out = np.zeros(len(idx))
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for train, valid in folds.split(np.zeros(len(idx)), y, groups):
        tr, va = idx[train], idx[valid]
        parts_train, parts_valid = [], []
        if kind in ("lr_tk", "svm_tk", "lr_t", "nb_word"):
            word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text[tr])
            parts_train.append(word.transform(text[tr]))
            parts_valid.append(word.transform(text[va]))
        if kind in ("lr_tk", "svm_tk", "lr_t", "lr_char"):
            char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit(text[tr])
            parts_train.append(char.transform(text[tr]))
            parts_valid.append(char.transform(text[va]))
        dense_train, dense_valid = [], []
        if kind in ("lr_tk", "svm_tk", "lr_ek"):
            dense_train.append(kw[tr])
            dense_valid.append(kw[va])
        if kind == "lr_ek":
            dense_train.append(emb[tr])
            dense_valid.append(emb[va])
        if dense_train:
            scaler = StandardScaler().fit(np.hstack(dense_train))
            parts_train.append(csr_matrix(scaler.transform(np.hstack(dense_train))))
            parts_valid.append(csr_matrix(scaler.transform(np.hstack(dense_valid))))
        x_train, x_valid = hstack(parts_train).tocsr(), hstack(parts_valid).tocsr()
        if kind == "svm_tk":
            scores_out[valid] = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED).fit(x_train, y[train]).decision_function(x_valid)
        elif kind == "nb_word":
            scores_out[valid] = ComplementNB().fit(x_train, y[train]).predict_proba(x_valid)[:, 1]
        else:
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)
            scores_out[valid] = model.fit(x_train, y[train]).predict_proba(x_valid)[:, 1]
    return scores_out


def best_threshold(y: np.ndarray, decide) -> tuple[float, np.ndarray]:
    """`decide(threshold)` -> predictions; the threshold with the best F1."""
    top = None
    for threshold in decide.candidates:
        f1 = scores(y, decide(threshold))["f1"]
        if top is None or f1 >= top[1]:
            top = (float(threshold), f1)
    return top[0], decide(top[0])


class Decide:
    def __init__(self, score: np.ndarray, flag: np.ndarray | None = None, gate: np.ndarray | None = None):
        self.score, self.flag, self.gate = score, flag, gate
        self.candidates = sorted(set(np.round(score, 6)))

    def __call__(self, threshold: float) -> np.ndarray:
        add = self.score >= threshold
        if self.gate is not None:
            add = add & self.gate
        return ((self.flag | add) if self.flag is not None else add).astype(int)


def main() -> int:
    records = load_five_label_records()
    view = keyword_view(records)
    praise = sentiment_labels(records) == "positive"
    emb = embeddings(records)
    rounds = np.array([record["round"] for record in records])
    gadget = np.array([record["gadget"] for record in records])
    groups_all = np.array([record["review_id"] for record in records])
    subsets = {
        "all": np.ones(len(records), dtype=bool),
        "gadget_sentences": gadget,
        "rounds_6_7": np.isin(rounds, ["6", "7"]),
        "round_7": rounds == "7",
    }
    report: dict[str, Any] = {}
    for label in LABELS:
        idx = np.array([i for i, record in enumerate(records) if record["labels"][label] is not None])
        y = np.array([int(records[i]["labels"][label]) for i in idx])
        flag = view["flags"][label][idx]
        gate = ~praise[idx] if label in GATED else None
        rows: dict[str, Any] = {"rules": {"threshold": None, "predicted": flag.astype(int)}}
        kept: dict[str, np.ndarray] = {}
        for kind in ("lr_tk", "svm_tk", "lr_t", "lr_char", "nb_word", "lr_ek"):
            kept[kind] = out_of_fold(kind, idx, y, groups_all[idx], view, emb)
            threshold, predicted = best_threshold(y, Decide(kept[kind]))
            rows[kind] = {"threshold": threshold, "predicted": predicted}
        threshold, predicted = best_threshold(y, Decide(kept["lr_tk"], flag, gate))
        rows["rules_or_lr_tk"] = {"threshold": threshold, "predicted": predicted}
        entry: dict[str, Any] = {"sentences": int(len(idx)), "reports": int(y.sum())}
        for name, row in rows.items():
            entry[name] = {"threshold": row["threshold"]} | {
                key: scores(y[mask[idx]], row["predicted"][mask[idx]]) for key, mask in subsets.items() if y[mask[idx]].sum()
            }
        report[label] = entry
        print(f"{label}: {entry['reports']} reports in {entry['sentences']} sentences", flush=True)
        for name in rows:
            cells = "  |  ".join(
                f"{key} {entry[name][key]['f1']:.3f} ({entry[name][key]['tp']}/{entry[name][key]['tp'] + entry[name][key]['fn']}, {entry[name][key]['fp']} false)"
                for key in subsets if key in entry[name]
            )
            print(f"  {name:15} {cells}", flush=True)
    REPORT.write_bytes(
        (json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "warning": "Cross-validated on AI-proposed labels (see five_label_data.py); a comparison between rows, not accuracy.",
            "cross_validation": {"folds": FOLDS, "grouped_by": "review_id", "seed": SEED},
            "labels": report,
        }, indent=2) + "\n").encode("utf-8")
    )
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
