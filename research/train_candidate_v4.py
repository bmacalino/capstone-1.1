"""Train hazard_classifier_v4 and v5: the defect model on every labelled round.

What both are:
  keyword rules 0.4
  defect   the adopted gated design (TF-IDF + keyword features, logistic
           regression; keywords OR (model AND not praise)), trained on rounds
           1-3 as re-read under guide 1.4 and on rounds 5-7
  safety   the model of hazard_classifier_v2b, unchanged (team labels only)
  counterfeit, quality, not_as_described   rules only: no learned model beat
           them (research/label_model_comparison.json)

Most of the defect model's labels were proposed by the AI assistant and were
accepted by the owner unaudited (8 Oct 2026), so its figures are agreement with those
labels, not accuracy. One figure is not: a copy trained without round 3 is
scored on the team's own round-3 labels, where the team-label models were
also untrained.

v4 (no flag) over-flagged ordinary reviews: on whole listings about half of
what it added in 4-5 star reviews was not a defect ("Gumagana naman sya").
--ordinary builds v5, which differs in two ways:
  * it also learns from round 8's mining listings: sentences the models flag
    in ordinary unused reviews, many of them wrongly;
  * in a review rated 4 or 5 stars it needs a higher score before it adds a
    flag the keywords did not raise (high_star_bar below). Keyword flags and
    reviews without a rating are untouched.
research/live_check.py compares the folders on listings none was trained on
and records which one the app serves. Neither is a pre-registered adoption
(PREREGISTRATION_final_v6.md, departure 2); DEFAKETIVE_HAZARD_MODEL=team_labels
serves hazard_classifier_v2b, whose models learned from the team's labels only.

    python research/train_candidate_v4.py
    python research/train_candidate_v4.py --ordinary
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

from five_label_data import KEYWORD_RULES, keyword_view, load_five_label_records, sentiment_labels  # noqa: E402
from train_defect_gate import best_threshold, decisions  # noqa: E402
from train_hazard_classifier import FOLDS, SEED, cross_validate, scores  # noqa: E402

ORDINARY = "--ordinary" in sys.argv
OUT = SENTIMENT_ROOT / "models" / ("hazard_classifier_v5" if ORDINARY else "hazard_classifier_v4")
TEAM_LABELS = SENTIMENT_ROOT / "models" / "hazard_classifier_v2b"
REPORT = ROOT / "research" / ("candidate_v5_training.json" if ORDINARY else "candidate_v4_training.json")
VERSION = "hazard-learned-5.0.0" if ORDINARY else "hazard-learned-4.0.0"
AI_ROUNDS = ("5", "6", "7")
RESAMPLES = 500
HIGH_STAR_RULE = (
    "Among sentences of 4-5 star reviews that the keywords did not flag and that are not praise, the threshold - "
    "never below the ordinary one - at which the added flags that are right outnumber those that are wrong by the "
    "most (out-of-fold scores; round 8 rows count for the pool sentences they stand for; ties go to the lower "
    "threshold). Below it an added flag is more often wrong than right."
)


def served_defect_flags(records: list[dict[str, Any]]) -> np.ndarray:
    """What the team-label models (hazard_classifier_v2b) say about each sentence."""
    from defaketive_sentiment.hazard_learned import LearnedHazardScorer
    from defaketive_sentiment.hazard_model import LexiconHazardTagger
    from train_defect_gate import sentiment_reader

    read, classifier = sentiment_reader()
    known: dict[str, str] = {}

    def label(text: str) -> str:
        if text not in known:
            known[text] = read(text)
        return known[text]

    scorer = LearnedHazardScorer(TEAM_LABELS, sentiment_model_sha256=classifier.manifest["files"]["model.onnx"])
    tagger = LexiconHazardTagger.for_scorer(scorer, sentiment_label=label)
    return np.array([int("defect" in tagger.tag_sentence(record["text"])["tags"]) for record in records])


def added_flags(y, keyword, probability, praise, weight, rows, threshold) -> dict[str, float]:
    """Weighted tally of the flags the model adds: keywords silent, not praise."""
    added = rows & (keyword == 0) & ~praise & (probability >= threshold)
    return {
        "right": round(float(weight[added & (y == 1)].sum()), 1),
        "wrong": round(float(weight[added & (y == 0)].sum()), 1),
    }


def most_right_over_wrong(y, probability, weight, base: float) -> float:
    """The threshold, from `base` up, where right minus wrong flags is largest."""
    signed = weight * (2 * y - 1)
    best, best_net = base, float(signed.sum())
    for threshold in sorted(set(np.round(probability, 6).tolist())):
        net = float(signed[probability >= threshold].sum())
        if net > best_net + 1e-9:
            best, best_net = float(threshold), net
    return best


def high_star_bar(y, keyword, probability, praise, weight, stars, groups, base: float) -> tuple[float, dict[str, Any]]:
    """The bar for adding a flag in a 4-5 star review (HIGH_STAR_RULE)."""
    high, low = stars >= 4, (stars >= 1) & (stars < 4)
    in_question = (keyword == 0) & ~praise & (probability >= base)
    rows = high & in_question
    bar = most_right_over_wrong(y[rows], probability[rows], weight[rows], base)
    # How steady is that choice? Redo it on resampled reviews (scores are not refitted).
    members: dict[str, list[int]] = {}
    for i in np.flatnonzero(rows):
        members.setdefault(str(groups[i]), []).append(int(i))
    names = sorted(members)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(RESAMPLES):
        picked = np.array([i for name in rng.choice(names, size=len(names)) for i in members[name]])
        draws.append(most_right_over_wrong(y[picked], probability[picked], weight[picked], base))
    low_rows = low & in_question
    return bar, {
        "rule": HIGH_STAR_RULE,
        "ordinary_threshold": base,
        "bar": bar,
        "sentences_in_question": int(rows.sum()),
        "resampled_reviews": {
            "draws": RESAMPLES,
            "p10": round(float(np.percentile(draws, 10)), 4),
            "median": round(float(np.percentile(draws, 50)), 4),
            "p90": round(float(np.percentile(draws, 90)), 4),
        },
        "added_flags_in_4_5_star_reviews": {
            "at_the_ordinary_threshold": added_flags(y, keyword, probability, praise, weight, high, base),
            "at_the_bar": added_flags(y, keyword, probability, praise, weight, high, bar),
        },
        "added_flags_in_1_3_star_reviews": {
            "at_the_ordinary_threshold": added_flags(y, keyword, probability, praise, weight, low, base),
            "the_same_rule_would_set": most_right_over_wrong(y[low_rows], probability[low_rows], weight[low_rows], base),
        },
    }


def star_numbers(records: list[dict[str, Any]]) -> np.ndarray:
    """Each sentence's review rating as a number; 0 where the source gave none."""
    stars = []
    for record in records:
        try:
            stars.append(float(record["rating"]))
        except (TypeError, ValueError):
            stars.append(0.0)
    return np.array(stars)


def round_three_check(records, view, praise_all, feats, stars_all, weight_all) -> dict[str, Any]:
    """Train the same design without round 3, then score both checkers on the
    team's own round-3 defect answers. Round 3 is the one human-labelled round
    neither the team-label models nor this copy was trained on."""
    from defaketive_sentiment.hazard_learned import fit_category_model

    train = np.array([i for i, r in enumerate(records) if r["labels"]["defect"] is not None and r["round"] != "3"])
    y = np.array([int(records[i]["labels"]["defect"]) for i in train])
    groups = np.array([records[i]["review_id"] for i in train])
    keyword = view["flags"]["defect"].astype(int)
    out_of_fold = cross_validate("TK", "defect", train, y, groups, feats)
    threshold, _ = best_threshold(y, keyword[train], out_of_fold, praise_all[train], gated=True)
    model = fit_category_model("TK", list(view["text"][train]), view["kw"][train], None, y, category="defect", seed=SEED)
    test = np.array([i for i, r in enumerate(records) if r["round"] == "3" and r["team"].get("defect") is not None])
    truth = np.array([int(records[i]["team"]["defect"]) for i in test])
    probability = model.probability(list(view["text"][test]), view["kw"][test], None)
    bars: Any = threshold
    extra: dict[str, Any] = {}
    if ORDINARY:
        bar, _ = high_star_bar(
            y, keyword[train], out_of_fold, praise_all[train], weight_all[train], stars_all[train], groups, threshold
        )
        bars = np.where(stars_all[test] >= 4, bar, threshold)
        extra = {
            "high_star_bar_without_round_3": bar,
            "round_3_sentences_from_4_5_star_reviews": int((stars_all[test] >= 4).sum()),
            "this_design_without_the_bar": scores(truth, decisions(keyword[test], probability, praise_all[test], threshold, True)),
        }
    return {
        "sentences": int(len(test)),
        "threshold_without_round_3": threshold,
        "team_label_models": scores(truth, served_defect_flags([records[i] for i in test])),
        "this_design_without_round_3": scores(truth, decisions(keyword[test], probability, praise_all[test], bars, True)),
        **extra,
    }


def main() -> int:
    import joblib

    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, file_sha256, fit_category_model, keyword_file_hashes
    from defaketive_sentiment.hazard_model import SHIPPED_COUNTERFEIT_RULES
    from defaketive_sentiment.transformer_sentiment import TransformerSentimentClassifier

    records = load_five_label_records()
    if not ORDINARY:
        records = [record for record in records if record["round"] != "8"]
    view = keyword_view(records, KEYWORD_RULES)
    praise_all = sentiment_labels(records) == "positive"
    stars_all = star_numbers(records)
    weight_all = np.array([record["weight"] for record in records])
    feats = {"normalized": view["text"], "kw": view["kw"], "emb": None}
    idx = np.array([i for i, record in enumerate(records) if record["labels"]["defect"] is not None])
    y = np.array([int(records[i]["labels"]["defect"]) for i in idx])
    groups = np.array([records[i]["review_id"] for i in idx])
    rounds = np.array([records[i]["round"] for i in idx])
    gadget = np.array([records[i]["gadget"] for i in idx])
    keyword = view["flags"]["defect"][idx].astype(int)
    praise = praise_all[idx]

    out_of_fold = cross_validate("TK", "defect", idx, y, groups, feats)
    threshold, overall = best_threshold(y, keyword, out_of_fold, praise, gated=True)
    bar, bar_report = None, None
    bars: Any = threshold
    if ORDINARY:
        bar, bar_report = high_star_bar(y, keyword, out_of_fold, praise, weight_all[idx], stars_all[idx], groups, threshold)
        bars = np.where(stars_all[idx] >= 4, bar, threshold)
    predicted = decisions(keyword, out_of_fold, praise, bars, True)
    ai = np.isin(rounds, AI_ROUNDS)
    served = served_defect_flags([records[i] for i in idx[ai]])
    by_part = {
        "all_rounds": scores(y, predicted),
        "gadget_sentences": scores(y[gadget], predicted[gadget]),
        "rounds_5_7": scores(y[ai], predicted[ai]),
        "rounds_6_7_gadget_rounds": scores(y[np.isin(rounds, ["6", "7"])], predicted[np.isin(rounds, ["6", "7"])]),
        "round_7": scores(y[rounds == "7"], predicted[rounds == "7"]),
        "rounds_1_3_relabelled": scores(y[np.isin(rounds, ["1", "2", "3"])], predicted[np.isin(rounds, ["1", "2", "3"])]),
    }
    if ORDINARY:
        by_part["all_rounds_without_the_high_star_bar"] = overall
    print(f"defect: {len(idx)} sentences, {int(y.sum())} reports, threshold {threshold:.4f}", flush=True)
    if bar_report:
        print(f"  high-star bar {bar:.4f}: {json.dumps({k: v for k, v in bar_report.items() if k != 'rule'})}")
    for name, stats in by_part.items():
        print(f"  candidate, cross-validated, {name:36} {stats}")
    served_stats = {
        "rounds_5_7": scores(y[ai], served),
        "round_7": scores(y[ai][rounds[ai] == "7"], served[rounds[ai] == "7"]),
        "keywords_alone_rounds_5_7": scores(y[ai], keyword[ai]),
    }
    for name, stats in served_stats.items():
        print(f"  team-label models (v2b),     {name:36} {stats}")
    held_out = round_three_check(records, view, praise_all, feats, stars_all, weight_all)
    print(f"  round 3, team labels, held out for both: {held_out}")

    model = fit_category_model("TK", list(view["text"][idx]), view["kw"][idx], None, y, category="defect", seed=SEED)
    model.threshold = threshold
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    joblib.dump(model, OUT / "defect.joblib")
    team = json.loads((TEAM_LABELS / MANIFEST_NAME).read_bytes().decode("utf-8"))
    safety = team["categories"]["safety"]
    shutil.copyfile(TEAM_LABELS / safety["file"], OUT / safety["file"])
    classifier = TransformerSentimentClassifier()
    manifest = {
        "version": VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "keyword_rules": KEYWORD_RULES,
        # The label rules the app serves: the report shows the quality and
        # not-as-described labels, so picture and number disputes land there.
        "counterfeit_rules": SHIPPED_COUNTERFEIT_RULES,
        "categories": {
            "safety": {**safety, "from": "hazard_classifier_v2b (unchanged)"},
            "defect": {
                "file": "defect.joblib",
                "sha256": file_sha256(OUT / "defect.joblib"),
                "candidate": "TK",
                "threshold": threshold,
                **({"threshold_high_star": bar, "high_star": bar_report} if bar_report else {}),
                "combine": "or_gated",
                "gate": "sentence sentiment is not positive",
                "needs_embedding": False,
                "training_rows": int(len(idx)),
                "training_positives": int(y.sum()),
                "training_rows_by_round": dict(Counter(rounds.tolist())),
                "training_data": (
                    "rounds 1-3: the team's defect answers re-read by the AI assistant under guide 1.4 "
                    "(ai_relabel.csv); rounds 5-7" + (" and the mining listings of round 8" if ORDINARY else "")
                    + ": AI-proposed labels, accepted by the owner on 8 October 2026 without an audit"
                ),
            }
        },
        "rule_labels": ["counterfeit", "quality", "not_as_described"],
        "keyword_files": keyword_file_hashes(),
        "embedding_model": None,
        "sentiment_model": {"model_version": classifier.model_version, "model_sha256": classifier.manifest["files"]["model.onnx"]},
        "qualified_for_integration": False,
        "adoption": {
            "adopted": False,
            "status": "candidate; research/live_check.py records which folder serves and why",
            "round_3_team_labels_held_out_for_both": held_out,
        },
    }
    (OUT / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    REPORT.write_bytes(
        (json.dumps({
            "generated_at": manifest["trained_at"],
            "warning": "Agreement with AI-proposed labels (research/five_label_data.py), cross-validated. Not accuracy.",
            "cross_validation": {"folds": FOLDS, "grouped_by": "review_id", "seed": SEED},
            "defect": {
                "threshold": threshold, "sentences": int(len(idx)), "reports": int(y.sum()),
                **({"high_star_bar": bar_report} if bar_report else {}),
                "candidate": by_part,
                "team_label_models": served_stats, "round_3_team_labels_held_out_for_both": held_out,
            },
        }, indent=2) + "\n").encode("utf-8")
    )
    print(f"frozen {OUT}; wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
