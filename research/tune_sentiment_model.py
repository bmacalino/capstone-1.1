"""Search sentiment model hyperparameters, optionally with extra Filipino corpora.

Two gaps this closes. First, no hyperparameter search has ever been run in this
project: the deployed settings (C=4.0, word 1-2 grams, char_wb 3-5 grams) were
chosen by hand and compared as five fixed candidates. Second, the shipped model
trained on FiReCS alone, although SentiTaglish and TagaSenti sit unused in the
data directory and together hold roughly six times as many rows.

Search and selection use cross-validation on the training split only. The test
labels are read once, after the winning configuration has been refit, and are
compared against the incumbent with McNemar's test.

Two ways to run it:

  Tune on FiReCS alone (fast):
      python research/tune_sentiment_model.py --n-iter 40

  Measure what an extra corpus adds, holding the settings fixed:
      python research/tune_sentiment_model.py --n-iter 0 --include-sentitaglish

Adding a corpus multiplies the row count, so a wide search over the combined
data is slow. Prefer tuning on FiReCS, then testing corpora at fixed settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

import numpy as np
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from evaluate_firecs import load_firecs
from research_utils import evaluate_labels, mcnemar_test
from train_firecs import (
    DEFAULT_LAZADA_FILIPINO,
    DEFAULT_SENTITAGLISH,
    DEFAULT_TAGASENTI,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    audit_training_records,
    load_lazada_filipino,
    load_sentitaglish,
    load_tagasenti,
    normalized_text,
)

DEFAULT_OUTPUT = ROOT / "research" / "sentiment_tuning.json"
SEED = 42

# The deployed configuration, used as the starting point and the comparison.
INCUMBENT_PARAMS = {
    "features__word__ngram_range": (1, 2),
    "features__word__min_df": 2,
    "features__word__max_features": 40_000,
    "features__word__sublinear_tf": True,
    "features__character__ngram_range": (3, 5),
    "features__character__min_df": 2,
    "features__character__max_features": 60_000,
    "features__character__sublinear_tf": True,
    "classifier__C": 4.0,
    "classifier__class_weight": None,
}


def build_pipeline() -> Pipeline:
    """The deployed architecture, with every searchable setting named."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                max_df=0.995,
                                strip_accents=None,
                                dtype=np.float32,
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2_000, solver="lbfgs")),
        ]
    )


def search_space() -> dict[str, Any]:
    """Ranges centred on the deployed settings, wide enough to escape them."""
    return {
        "features__word__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "features__word__min_df": [1, 2, 3],
        "features__word__max_features": [20_000, 40_000, 80_000, None],
        "features__word__sublinear_tf": [True, False],
        "features__character__ngram_range": [(2, 4), (3, 5), (3, 6), (2, 5)],
        "features__character__min_df": [1, 2, 3],
        "features__character__max_features": [30_000, 60_000, 120_000, None],
        "features__character__sublinear_tf": [True, False],
        "classifier__C": loguniform(0.25, 32.0),
        "classifier__class_weight": [None, "balanced"],
    }


def collect_training_records(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    train_records = load_firecs(args.train.resolve())
    test_records = load_firecs(args.test.resolve())
    for record in train_records:
        record["source"] = "firecs_human"

    audits: list[dict[str, Any]] = []
    if args.include_sentitaglish:
        records, audit = load_sentitaglish(args.sentitaglish.resolve())
        train_records.extend(records)
        audits.append(audit)
    if args.include_tagasenti:
        records, audit = load_tagasenti(args.tagasenti.resolve())
        train_records.extend(records)
        audits.append(audit)
    if args.include_lazada_filipino:
        records, audit = load_lazada_filipino(args.lazada_filipino.resolve())
        train_records.extend(records)
        audits.append(audit)

    args.auxiliary_audits = audits
    return train_records, test_records


def run(args: argparse.Namespace) -> dict[str, Any]:
    train_records, test_records = collect_training_records(args)
    texts, labels, audit = audit_training_records(train_records, test_records)
    test_texts = [normalized_text(record["review"]) for record in test_records]
    test_labels = [record["label"] for record in test_records]
    print(f"Training rows after cleaning: {len(texts)}", flush=True)

    pipeline = build_pipeline()
    search_report: dict[str, Any] = {}
    best_params = dict(INCUMBENT_PARAMS)

    if args.n_iter > 0:
        splitter = RepeatedStratifiedKFold(
            n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=SEED
        )
        search = RandomizedSearchCV(
            pipeline,
            search_space(),
            n_iter=args.n_iter,
            scoring="f1_macro",
            cv=splitter,
            random_state=SEED,
            n_jobs=args.jobs,
            refit=False,
            verbose=2,
        )
        started = time.perf_counter()
        search.fit(texts, labels)
        search_seconds = round(time.perf_counter() - started, 1)

        # Score the deployed settings under the identical splits, so the
        # comparison is like for like rather than against a recorded number.
        from sklearn.model_selection import cross_val_score

        incumbent_scores = cross_val_score(
            build_pipeline().set_params(**INCUMBENT_PARAMS),
            texts,
            labels,
            scoring="f1_macro",
            cv=splitter,
            n_jobs=args.jobs,
        )
        best_params = search.best_params_
        search_report = {
            "candidates_evaluated": args.n_iter,
            "cv": f"RepeatedStratifiedKFold({args.cv_splits} splits x {args.cv_repeats} repeats)",
            "search_seconds": search_seconds,
            "best_cv_macro_f1": round(float(search.best_score_), 4),
            "incumbent_cv_macro_f1": round(float(incumbent_scores.mean()), 4),
            "cv_gain_over_incumbent": round(
                float(search.best_score_ - incumbent_scores.mean()), 4
            ),
            "best_params": {key: str(value) for key, value in best_params.items()},
        }
        print(
            f"Best CV macro F1 {search.best_score_:.4f} vs incumbent "
            f"{incumbent_scores.mean():.4f}",
            flush=True,
        )

    # Refit the winner on all cleaned rows, then read the test labels once.
    final = build_pipeline().set_params(**best_params)
    started = time.perf_counter()
    final.fit(texts, labels)
    fit_seconds = round(time.perf_counter() - started, 3)
    predictions = list(final.predict(test_texts))
    metrics = evaluate_labels(test_labels, predictions)

    # Only worth testing when the search actually moved off the deployed settings;
    # otherwise the two models are the same and McNemar has nothing to compare.
    if best_params != INCUMBENT_PARAMS:
        incumbent = build_pipeline().set_params(**INCUMBENT_PARAMS)
        incumbent.fit(texts, labels)
        significance = mcnemar_test(
            test_labels, list(incumbent.predict(test_texts)), predictions
        )
    else:
        significance = None

    print(f"Test accuracy={metrics['accuracy']:.4f} macro F1={metrics['f1']:.4f}", flush=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deployment_status": "comparison_only",
        "qualified_for_integration": False,
        "corpora": {
            "firecs": True,
            "sentitaglish": args.include_sentitaglish,
            "tagasenti": args.include_tagasenti,
            "lazada_filipino": args.include_lazada_filipino,
            "auxiliary_audits": getattr(args, "auxiliary_audits", []),
        },
        "training_audit": audit,
        "search": search_report,
        "final": {
            "params": {key: str(value) for key, value in best_params.items()},
            "fit_seconds": fit_seconds,
            "test": metrics,
        },
        "mcnemar_vs_incumbent_same_data": significance,
        "reference": {
            "shipped_model_test_macro_f1": 0.8138,
            "note": (
                "The shipped model trained on FiReCS alone. When extra corpora are "
                "included the incumbent comparison is refit on the same rows, so the "
                "McNemar test isolates the settings rather than the data."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--include-sentitaglish", action="store_true")
    parser.add_argument("--sentitaglish", type=Path, default=DEFAULT_SENTITAGLISH)
    parser.add_argument("--include-tagasenti", action="store_true")
    parser.add_argument("--tagasenti", type=Path, default=DEFAULT_TAGASENTI)
    parser.add_argument("--include-lazada-filipino", action="store_true")
    parser.add_argument("--lazada-filipino", type=Path, default=DEFAULT_LAZADA_FILIPINO)
    parser.add_argument("--n-iter", type=int, default=40, help="0 skips the search")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
