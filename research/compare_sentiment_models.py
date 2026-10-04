"""Compare sentiment modelling techniques on the locked FiReCS test set.

Every arm trains only on the cleaned FiReCS training split produced by
``train_firecs.audit_training_records``, so each one sees the identical rows,
the identical text normalization, and the identical duplicate and cross-split
leakage removal as the shipped model. Only the algorithm differs.

Held-out test labels are used for reporting only. No arm here is promoted into
the application; every artifact this script describes is a comparison arm.

Alongside accuracy and macro F1 this records what the existing evaluation
JSONs do not: a bootstrap confidence interval for macro F1, McNemar's paired
test against the incumbent with a Holm correction for running several
comparisons at once, single-review inference latency, and serialized model
size. The last two are the deployment argument and belong in the same table.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

from evaluate_firecs import load_firecs
from research_utils import (
    bootstrap_macro_f1_ci,
    evaluate_labels,
    holm_bonferroni,
    mcnemar_test,
)
from train_firecs import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    audit_training_records,
    combined_features,
    normalized_text,
)

DEFAULT_ARTIFACT = SENTIMENT_ROOT / "models" / "firecs_sentiment.joblib"
DEFAULT_OUTPUT = ROOT / "research" / "sentiment_model_comparison.json"
INCUMBENT_ARM = "tfidf_logistic_regression_incumbent"
LATENCY_SAMPLE = 200
SEED = 42


class LexiconArm:
    """Wrap the untrained VADER path so it shares the arm interface.

    This arm reads raw review text. It does its own normalization internally,
    and the marker tokens feature_text appends exist only to help the TF-IDF
    arms, so feeding them to a lexicon would be noise.
    """

    wants_raw_text = True

    def __init__(self) -> None:
        from defaketive_sentiment.model import DefaketiveSentimentModel

        self._model = DefaketiveSentimentModel(use_trained=False)

    def fit(self, texts: list[str], labels: list[str]) -> "LexiconArm":
        # A lexicon has nothing to learn; kept so every arm is driven the same way.
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self._model.analyze(text)["label"] for text in texts]


class ShippedArtifactArm:
    """Score the joblib artifact the application actually loads today.

    This exists to prove the harness feeds models the same text the training
    script did. If this arm does not reproduce the recorded scores exactly,
    every other number in the table is suspect.
    """

    def __init__(self, artifact_path: Path) -> None:
        payload = joblib.load(artifact_path)
        self._pipeline = payload["pipeline"]
        self.metadata = payload.get("metadata", {})
        self.artifact_path = artifact_path

    def fit(self, texts: list[str], labels: list[str]) -> "ShippedArtifactArm":
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return list(self._pipeline.predict(texts))


def arm_factories() -> dict[str, Callable[[], Any]]:
    """Return every comparison arm, keyed by the name used in the report."""
    return {
        # The floor: no learning at all, scored through the same harness.
        "vader_lexicon": LexiconArm,
        # The incumbent's exact configuration, retrained here.
        INCUMBENT_ARM: lambda: Pipeline(
            [
                ("features", combined_features()),
                (
                    "classifier",
                    LogisticRegression(
                        C=4.0,
                        class_weight=None,
                        max_iter=1_000,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        # Selected during training cross-validation but never scored on the test set.
        "tfidf_linear_svc": lambda: Pipeline(
            [
                ("features", combined_features()),
                ("classifier", LinearSVC(C=1.0, class_weight="balanced")),
            ]
        ),
        # The canonical text-classification baseline.
        "tfidf_complement_naive_bayes": lambda: Pipeline(
            [
                ("features", combined_features()),
                ("classifier", ComplementNB(alpha=0.3)),
            ]
        ),
        # Latent semantic analysis: dense, reduced features with a linear head.
        "tfidf_svd_logistic_regression": lambda: Pipeline(
            [
                ("features", combined_features()),
                ("reduce", TruncatedSVD(n_components=300, random_state=SEED)),
                ("normalize", Normalizer(copy=False)),
                (
                    "classifier",
                    LogisticRegression(C=4.0, max_iter=2_000, solver="lbfgs"),
                ),
            ]
        ),
        # Non-linear arm. Trees need dense input, so they sit on the SVD features.
        "tfidf_svd_gradient_boosting": lambda: Pipeline(
            [
                ("features", combined_features()),
                ("reduce", TruncatedSVD(n_components=300, random_state=SEED)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def serialized_size(model: Any) -> int | None:
    """Return the model's serialized size in bytes, or None if it cannot be dumped."""
    if isinstance(model, LexiconArm):
        return None
    if isinstance(model, ShippedArtifactArm):
        return model.artifact_path.stat().st_size
    buffer = io.BytesIO()
    try:
        joblib.dump(model, buffer)
    except (TypeError, ValueError):
        return None
    return buffer.getbuffer().nbytes


def measure_latency(model: Any, texts: list[str], sample: int) -> dict[str, float]:
    """Time single-review prediction, which is how the web app calls the model."""
    subset = texts[:sample]
    timings: list[float] = []
    for text in subset:
        started = time.perf_counter()
        model.predict([text])
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms_per_review": round(statistics.median(timings), 4),
        "p95_ms_per_review": round(
            statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings),
            4,
        ),
        "sampled_reviews": len(timings),
    }


def run_arm(
    name: str,
    factory: Callable[[], Any],
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    bootstrap_iterations: int,
    latency_sample: int,
    raw_test_texts: list[str],
) -> dict[str, Any]:
    model = factory()
    # Only untrained arms ask for raw text, so the training text they are handed
    # is never used. Asserted rather than assumed, so a future trainable raw-text
    # arm fails loudly instead of silently training on the wrong text.
    wants_raw = getattr(model, "wants_raw_text", False)
    if wants_raw:
        if not isinstance(model, LexiconArm):
            raise NotImplementedError(
                f"Arm {name!r} wants raw text but also trains; supply raw training text first."
            )
        test_texts = raw_test_texts
    started = time.perf_counter()
    model.fit(train_texts, train_labels)
    fit_seconds = round(time.perf_counter() - started, 3)

    predictions = list(model.predict(test_texts))
    metrics = evaluate_labels(test_labels, predictions)
    metrics.update(
        {
            "arm": name,
            "fit_seconds": fit_seconds,
            "macro_f1_ci": bootstrap_macro_f1_ci(
                test_labels, predictions, iterations=bootstrap_iterations, seed=SEED
            ),
            "latency": measure_latency(model, test_texts, latency_sample),
            "serialized_bytes": serialized_size(model),
            "text_input": "raw_review" if wants_raw else "normalized_with_feature_markers",
        }
    )
    print(
        f"{name}: accuracy={metrics['accuracy']:.4f}, macro F1={metrics['f1']:.4f} "
        f"[{metrics['macro_f1_ci']['ci_lower']:.4f}, {metrics['macro_f1_ci']['ci_upper']:.4f}], "
        f"{metrics['latency']['median_ms_per_review']:.3f} ms/review",
        flush=True,
    )
    return {"metrics": metrics, "predictions": predictions}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    train_records = load_firecs(args.train)
    test_records = load_firecs(args.test)
    train_texts, train_labels, audit = audit_training_records(train_records, test_records)
    test_texts = [normalized_text(record["review"]) for record in test_records]
    raw_test_texts = [record["review"] for record in test_records]
    test_labels = [record["label"] for record in test_records]

    factories = arm_factories()
    if args.artifact.exists():
        factories = {
            "shipped_artifact_reproduction": lambda: ShippedArtifactArm(args.artifact),
            **factories,
        }
    selected = args.arms or list(factories)
    unknown = [name for name in selected if name not in factories]
    if unknown:
        raise SystemExit(f"Unknown arm(s): {', '.join(unknown)}")

    results: dict[str, Any] = {}
    predictions: dict[str, list[str]] = {}
    for name in selected:
        outcome = run_arm(
            name,
            factories[name],
            train_texts,
            train_labels,
            test_texts,
            test_labels,
            args.bootstrap_iterations,
            args.latency_sample,
            raw_test_texts,
        )
        results[name] = outcome["metrics"]
        predictions[name] = outcome["predictions"]

    significance: dict[str, Any] = {}
    if INCUMBENT_ARM in predictions:
        raw_p_values: dict[str, float] = {}
        for name, arm_predictions in predictions.items():
            if name == INCUMBENT_ARM:
                continue
            test = mcnemar_test(test_labels, predictions[INCUMBENT_ARM], arm_predictions)
            results[name]["mcnemar_vs_incumbent"] = test
            raw_p_values[name] = test["p_value"]
        if raw_p_values:
            correction = holm_bonferroni(raw_p_values, alpha=args.alpha)
            for name, corrected in correction["results"].items():
                results[name]["mcnemar_vs_incumbent"].update(
                    {
                        "p_adjusted": corrected["p_adjusted"],
                        "significant_after_correction": corrected["significant"],
                    }
                )
            significance = {
                "reference_arm": INCUMBENT_ARM,
                "correction": "holm_bonferroni",
                "alpha": args.alpha,
                "comparisons": correction["comparisons"],
            }

    return {
        "dataset": "FiReCS",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "train_file": args.train.name,
            "test_file": args.test.name,
            "seed": SEED,
            "text_normalization": "train_firecs.normalized_text (feature_text + model normalize)",
            "fairness_note": (
                "Every arm receives identical normalized text and the identical "
                "cleaned training rows. Differences reflect the algorithm only."
            ),
            "bootstrap_iterations": args.bootstrap_iterations,
            "latency_sample": args.latency_sample,
            "deployment_status": "comparison_only",
            "qualified_for_integration": False,
        },
        "training_audit": audit,
        "significance": significance,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--latency-sample", type=int, default=LATENCY_SAMPLE)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    np.random.seed(SEED)
    report = compare(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
