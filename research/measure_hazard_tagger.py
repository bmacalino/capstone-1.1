"""Measure the hazard tagger against the round 1 human labels.

This is the first measurement of the safety/defect/counterfeit tagger against
human judgement. Until round 1 was annotated there was nothing to measure it
on: the only locked test set contains no safety or counterfeit rows and both of
its annotators were the same language-model pass.

Five configurations, each run with the rules and files of its own release
(frozen copies under research/lexicons/), so an old version is never scored
with newer code:

  v1_lexicon         the 0.1.0 term list, heat filter off
  v1_plus_filter     the 0.1.0 term list, heat filter on
  v2_lexicon         the 0.2.0 term list (+43 safety terms), heat filter off
  v2_shipped         model 0.2.0 as it shipped
  v3_shipped         model 0.3.0: negation scoped to the word it governs,
                     the guide's defect and counterfeit phrasings, missing
                     items and damaged boxes no longer defects, name phrases
                     ("burnt orange", "fake news") excluded

ROUND 2 is the held-out test (research/LABELLING_ROUND_2_INSTRUCTIONS.md):

  python research/measure_hazard_tagger.py --round 2 \
      --queue outputs/labelling_queue_round_2/queue_master.csv \
      --worksheets outputs/labelling_queue_round_2/worksheet_*_done.csv

Besides precision, recall and F1, each result reports plain accuracy next to
the accuracy of a checker that never flags anything (rare categories make
accuracy look high by default), a bootstrap 95% interval for F1, and the same
figures weighted by each row's sampling weight (the queue over-samples likely
positives; weighting undoes that).

The heat filter drops a safety tag when the only evidence is a mild heat word
and the sentence is about clothing, weather, a venue, or a product whose job is
heat (lexicons/safety_false_friends_v1.tsv).

IN-SAMPLE WARNING (round 1). The added terms, cues and rules were chosen after
inspecting round 1 errors, so every v2, v3 and filter figure on round 1 is
fitted to that data and optimistic. They bound what the changes can achieve; they do not estimate
field performance. A fresh labelled round is required to confirm the gains.

Ground truth is the consensus of the two annotators who saw each sentence.
Rows where they disagreed are excluded per category, because an adjudicated
label does not exist yet.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

CATEGORIES = ("safety", "defect", "counterfeit")
DEFAULT_QUEUE = ROOT / "outputs" / "labelling_queue_round_1" / "queue_master.csv"
FROZEN = ROOT / "research" / "lexicons"
DEFAULT_V1_LEXICON = FROZEN / "risk_lexicon_v1_frozen.tsv"
V2_LEXICON = FROZEN / "risk_lexicon_v2_frozen.tsv"
V1_FALSE_FRIENDS = FROZEN / "safety_false_friends_v1.0_frozen.tsv"
DEFAULT_OUTPUT = ROOT / "research" / "hazard_tagger_measurement.json"
BOOTSTRAP_ITERATIONS = 1000
YES = {"yes", "y", "1", "true"}
NO = {"no", "n", "0", "false"}
# Targets declared in lexicons/hazard_parameters_v1.json.
TARGET_RECALL = {"safety": 0.80, "defect": 0.70, "counterfeit": 0.70}
PRECISION_FLOOR = 0.80


def flag(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if text in YES:
        return True
    if text in NO:
        return False
    return None


def read_worksheet(path: Path) -> list[dict[str, Any]]:
    """Read a worksheet or the queue master, accepting .xlsx from Excel."""
    if path.suffix.casefold() in {".xlsx", ".xlsm"}:
        import pandas as pd

        return pd.read_excel(path, sheet_name=0, dtype=str).fillna("").to_dict("records")
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def build_truth(worksheets: list[Path]) -> dict[str, dict[str, bool]]:
    """Consensus labels: both annotators labelled the category and agreed."""
    labelled: dict[str, dict[str, dict[str, bool | None]]] = defaultdict(dict)
    for path in worksheets:
        for row in read_worksheet(path):
            row_id = str(row.get("row_id") or "").strip()
            if not row_id:
                continue
            verifier = str(row.get("verifier_id") or path.stem).strip()
            labelled[row_id][verifier] = {
                category: flag(row.get("human_" + category)) for category in CATEGORIES
            }

    truth: dict[str, dict[str, bool]] = {}
    for row_id, verifiers in labelled.items():
        if len(verifiers) < 2:
            continue
        names = sorted(verifiers)
        first, second = verifiers[names[0]], verifiers[names[1]]
        agreed = {
            category: bool(first[category])
            for category in CATEGORIES
            if first[category] is not None
            and second[category] is not None
            and first[category] == second[category]
        }
        if agreed:
            truth[row_id] = agreed
    return truth


def _f1(tp: float, fp: float, fn: float) -> float | None:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if not precision or not recall:
        return None
    return 2 * precision * recall / (precision + recall)


def bootstrap_f1_ci(
    outcomes: list[tuple[bool, bool]], iterations: int = BOOTSTRAP_ITERATIONS, seed: int = 42
) -> list[float] | None:
    """Percentile 95% interval for F1, resampling sentences with replacement."""
    if not outcomes:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [outcomes[rng.randrange(len(outcomes))] for _ in outcomes]
        tp = sum(1 for real, flagged in sample if real and flagged)
        fp = sum(1 for real, flagged in sample if not real and flagged)
        fn = sum(1 for real, flagged in sample if real and not flagged)
        values.append(_f1(tp, fp, fn) or 0.0)
    values.sort()
    return [round(values[int(0.025 * iterations)], 4), round(values[int(0.975 * iterations) - 1], 4)]


def weighted_scores(
    ids: list[str], category: str, truth: dict[str, dict[str, bool]],
    predicted: dict[str, set[str]], weights: dict[str, float],
) -> dict[str, Any]:
    """Precision, recall and F1 with each sentence counted by its sampling weight."""
    tp = sum(weights.get(r, 1.0) for r in ids if truth[r][category] and category in predicted[r])
    fp = sum(weights.get(r, 1.0) for r in ids if not truth[r][category] and category in predicted[r])
    fn = sum(weights.get(r, 1.0) for r in ids if truth[r][category] and category not in predicted[r])
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = _f1(tp, fp, fn)
    return {
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def score(
    truth: dict[str, dict[str, bool]],
    predicted: dict[str, set[str]],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for category in CATEGORIES:
        ids = [row_id for row_id in truth if category in truth[row_id]]
        true_positive = sum(1 for r in ids if truth[r][category] and category in predicted[r])
        false_positive = sum(
            1 for r in ids if not truth[r][category] and category in predicted[r]
        )
        false_negative = sum(
            1 for r in ids if truth[r][category] and category not in predicted[r]
        )
        flagged = true_positive + false_positive
        actual = true_positive + false_negative
        precision = true_positive / flagged if flagged else None
        recall = true_positive / actual if actual else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        correct = len(ids) - false_positive - false_negative
        outcomes = [(truth[r][category], category in predicted[r]) for r in ids]
        results[category] = {
            "labelled_rows": len(ids),
            "positives": actual,
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "f1_ci_95": bootstrap_f1_ci(outcomes),
            "accuracy": round(correct / len(ids), 4) if ids else None,
            # What a checker that never flags anything would score.
            "never_flag_accuracy": round((len(ids) - actual) / len(ids), 4) if ids else None,
            "weighted": weighted_scores(ids, category, truth, predicted, weights or {}),
            "meets_recall_target": (
                recall >= TARGET_RECALL[category] if recall is not None else None
            ),
            "meets_precision_floor": (
                precision >= PRECISION_FLOOR if precision is not None else None
            ),
        }
    return results


def format_row(category: str, stats: dict[str, Any]) -> str:
    def show(value: Any) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    met = []
    if stats["meets_recall_target"]:
        met.append("recall")
    if stats["meets_precision_floor"]:
        met.append("precision")
    return (
        f"{category:<13}{show(stats['precision']):>10}{show(stats['recall']):>9}"
        f"{show(stats['f1']):>8}   {'+'.join(met) if met else 'neither'}"
    )


def configurations(v1_lexicon: Path) -> dict[str, Any]:
    """Each release with its own term list, cue list and rules."""
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    old = {"rules": "0.2", "false_friends_path": V1_FALSE_FRIENDS}
    return {
        "v1_lexicon": LexiconHazardTagger(v1_lexicon, false_friends=False, **old),
        "v1_plus_filter": LexiconHazardTagger(v1_lexicon, **old),
        "v2_lexicon": LexiconHazardTagger(V2_LEXICON, false_friends=False, **old),
        "v2_shipped": LexiconHazardTagger(V2_LEXICON, **old),
        "v3_shipped": LexiconHazardTagger(),
        **learned_configuration(),
        **gated_candidate_configuration(),
    }


def gated_candidate_configuration() -> dict[str, Any]:
    """v5_candidate: the gated defect model (research/PREREGISTRATION_defect_v5.md)
    with the shipped safety model, once research/train_defect_gate.py froze it."""
    from defaketive_sentiment.hazard_learned import LearnedHazardScorer
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    folder = SENTIMENT_ROOT / "models" / "hazard_classifier_v2"
    if not (folder / "manifest.json").is_file():
        return {}
    from defaketive_sentiment.model import DefaketiveSentimentModel
    from defaketive_sentiment.transformer_sentiment import TransformerSentimentClassifier

    classifier = TransformerSentimentClassifier()
    sentiment = DefaketiveSentimentModel(classifier=classifier)
    scorer = LearnedHazardScorer(folder, sentiment_model_sha256=classifier.manifest["files"]["model.onnx"])
    return {
        "v5_candidate": LexiconHazardTagger(
            learned=scorer, sentiment_label=lambda text: sentiment.analyze(text)["label"]
        )
    }


def learned_configuration() -> dict[str, Any]:
    """v4_learned: keyword checker 0.3.0 with every frozen learned model on top.

    Present only once research/train_hazard_classifier.py has frozen the
    models. Categories are independent, so its per-category figures are those
    each category would have if its learned model were adopted.
    """
    from defaketive_sentiment.hazard_learned import DEFAULT_MODEL_DIR, LearnedHazardScorer
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    if not (DEFAULT_MODEL_DIR / "manifest.json").is_file():
        return {}
    scorer = LearnedHazardScorer(DEFAULT_MODEL_DIR)
    embedder = None
    if scorer.needs_embedding:
        from defaketive_sentiment.transformer_sentiment import TransformerSentimentClassifier

        embedder = TransformerSentimentClassifier().embed
    return {"v4_learned": LexiconHazardTagger(learned=scorer, embedder=embedder)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from defaketive_sentiment.hazard_model import MODEL_VERSION

    truth = build_truth(args.worksheets)
    master = {
        row["row_id"]: row
        for row in read_worksheet(args.queue)
        if str(row.get("row_id") or "").strip()
    }
    missing = [row_id for row_id in truth if row_id not in master]
    if missing:
        raise SystemExit(f"{len(missing)} labelled rows are not in {args.queue}, e.g. {missing[:3]}")
    weights = {}
    for row_id in truth:
        try:
            weights[row_id] = float(master[row_id].get("sampling_weight") or 1.0)
        except ValueError:
            weights[row_id] = 1.0

    results: dict[str, Any] = {}
    suppressed: dict[str, dict[str, int]] = {}
    all_predicted: dict[str, dict[str, set[str]]] = {}
    by_stratum: dict[str, Any] = {}
    for name, tagger in configurations(args.v1_lexicon).items():
        predicted: dict[str, set[str]] = {}
        counts: dict[str, int] = defaultdict(int)
        for row_id in truth:
            tagged = tagger.tag_sentence(master[row_id]["sentence_text"])
            predicted[row_id] = set(tagged.get("tags") or [])
            for cue_class in (tagged.get("suppressed") or {}).values():
                counts[cue_class] += 1
            for reasons in (tagged.get("excluded") or {}).values():
                for reason in reasons:
                    counts[f"excluded:{reason}"] += 1
        results[name] = score(truth, predicted, weights)
        strata = sorted({str(master[row_id].get("stratum") or "") for row_id in truth})
        by_stratum[name] = {
            stratum: score(
                {row_id: labels for row_id, labels in truth.items() if master[row_id].get("stratum") == stratum},
                predicted,
            )
            for stratum in strata
            if stratum
        }
        all_predicted[name] = predicted
        if counts:
            suppressed[name] = dict(sorted(counts.items()))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labelling_round": args.round,
        "shipped_model_version": MODEL_VERSION,
        "deployment_status": "measurement_only",
        "qualified_for_integration": False,
        "lexicons": {
            "v1": str(args.v1_lexicon.relative_to(ROOT)),
            "v2": str(V2_LEXICON.relative_to(ROOT)),
            "v2_filter": str(V1_FALSE_FRIENDS.relative_to(ROOT)),
            "v3": "sentiment-analysis/defaketive_sentiment/lexicons/risk_lexicon.tsv",
            "v3_filter": "sentiment-analysis/defaketive_sentiment/lexicons/safety_false_friends_v1.tsv",
            "v3_phrase_exclusions": "sentiment-analysis/defaketive_sentiment/lexicons/hazard_phrase_exclusions_v1.tsv",
        },
        "ground_truth": {
            "source": [path.name for path in args.worksheets],
            "queue": str(args.queue.relative_to(ROOT)) if args.queue.is_relative_to(ROOT) else str(args.queue),
            "rule": "consensus of the two annotators who saw each sentence",
            "rows_with_any_consensus": len(truth),
        },
        "in_sample_warning": (
            "Round 1 is the development set: the v2 and v3 terms, cues and rules "
            "were chosen after inspecting its errors, so their round 1 figures are "
            "in-sample and optimistic."
            if args.round == 1
            else "Round 2 is held out: no v3 term, cue or rule was chosen with it in view."
        ),
        "known_issue": (
            "Through 0.2.0, negation looked three words back, so 'not good umusok' "
            "negated the smoke report. 0.3.0 scopes negation to the word it governs "
            "(text_norm.is_negated_adjacent); the sentiment path keeps is_negated."
        ),
        "targets": {"recall": TARGET_RECALL, "precision_floor": PRECISION_FLOOR},
        "results": results,
        "safety_tags_suppressed_by_cue_class": suppressed,
        "paired_comparisons": paired_comparisons(truth, all_predicted),
        "results_by_stratum": by_stratum,
    }


def paired_comparisons(truth: dict[str, dict[str, bool]], predicted: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    """McNemar's test between consecutive shipped releases, per category."""
    from research_utils import mcnemar_test

    pairs = [("v2_shipped", "v3_shipped"), ("v3_shipped", "v4_learned"), ("v3_shipped", "v5_candidate")]
    out: dict[str, Any] = {}
    for first, second in pairs:
        if first not in predicted or second not in predicted:
            continue
        for category in CATEGORIES:
            ids = [row_id for row_id in truth if category in truth[row_id]]
            actual = ["yes" if truth[r][category] else "no" for r in ids]
            a = ["yes" if category in predicted[first][r] else "no" for r in ids]
            b = ["yes" if category in predicted[second][r] else "no" for r in ids]
            out[f"{first}_vs_{second}:{category}"] = mcnemar_test(actual, a, b)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worksheets", nargs="+", type=Path, required=True)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--round", type=int, default=1, help="Labelling round the worksheets belong to.")
    parser.add_argument("--v1-lexicon", type=Path, default=DEFAULT_V1_LEXICON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.v1_lexicon = args.v1_lexicon.resolve()
    args.queue = args.queue.resolve()
    report = run(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Bytes, not write_text: on Windows text mode turns every LF into CRLF.
        args.output.write_bytes(
            (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
        print(f"Wrote {args.output}\n")
    for name, stats_by_category in report["results"].items():
        print(f"--- {name} ---")
        print(f"{'category':<13}{'precision':>10}{'recall':>9}{'F1':>8}   targets met")
        for category, stats in stats_by_category.items():
            print(format_row(category, stats))
        print()
    print("safety tags suppressed:", report["safety_tags_suppressed_by_cue_class"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
