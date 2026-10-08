"""Agreement of the checkers with the AI-labelled rounds (5 and 6), and a
dry run of retraining the defect model with those rounds added.

Rounds 5 and 6 were labelled by the AI assistant and accepted by the owner
unaudited, not ground truth (research/PREREGISTRATION_final_v6.md, departures
1 and 3). Every figure
here is agreement with those labels, never accuracy, and nothing is saved
except the report: no model file is written and nothing ships.

    python research/score_ai_rounds.py
"""

from __future__ import annotations

import csv
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

from gadget_scope import is_gadget  # noqa: E402
from train_defect_gate import sentiment_reader  # noqa: E402
from train_final_v6 import load_records  # noqa: E402
from train_hazard_classifier import FOLDS, SEED, scores  # noqa: E402

OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "research" / "ai_rounds_check.json"
CATEGORIES = ("safety", "defect", "counterfeit")
AI_ROUNDS = (5, 6, 7)
HUMAN_ROUNDS = ("1", "2", "3")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def ai_round(number: int) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    folder = OUTPUTS / f"labelling_queue_round_{number}"
    return read(folder / "ai_labels.csv"), {row["row_id"]: row for row in read(folder / "queue_master.csv")}


def agreement() -> dict[str, Any]:
    """The checkers as they run (shipped, and the v3 candidate), by subset."""
    from defaketive_sentiment.hazard_model import EXTRA_RULES, EXTRA_TAGS, LexiconHazardTagger
    from measure_hazard_tagger import final_configuration

    taggers = final_configuration()
    shipped = taggers["v6_shipped"]
    # The shipped checker with rules c2: the two new labels, and picture or
    # number disputes no longer counted as counterfeit.
    taggers["shipped_c2"] = LexiconHazardTagger(
        learned=shipped.learned, sentiment_label=shipped.sentiment_label, counterfeit_rules=EXTRA_RULES
    )
    out: dict[str, Any] = {}
    for number in AI_ROUNDS:
        rows, master = ai_round(number)
        gadget = np.array([is_gadget(master[row["row_id"]]) for row in rows])
        lazada = np.array([master[row["row_id"]].get("cohort") == "own_scrape" for row in rows])
        subsets = {"all": np.ones(len(rows), dtype=bool), "gadget": gadget, "not_gadget": ~gadget, "lazada": lazada, "shopee": ~lazada}
        result: dict[str, Any] = {"sentences": {name: int(mask.sum()) for name, mask in subsets.items()}}
        for name, tagger in taggers.items():
            tagged = [tagger.tag_sentence(row["sentence_text"]) for row in rows]
            tags = [set(item["tags"]) | set(item.get("extra_tags") or []) for item in tagged]
            result[name] = {}
            for category in CATEGORIES + (EXTRA_TAGS if name == "shipped_c2" else ()):
                y = np.array([int(row[f"ai_{category}"] == "yes") for row in rows])
                flagged = np.array([int(category in found) for found in tags])
                sure = np.array([category not in row["ai_unsure"].split("|") for row in rows])
                by = {key: scores(y[mask], flagged[mask]) for key, mask in subsets.items() if mask.any()}
                by["sure_only"] = scores(y[sure], flagged[sure])
                result[name][category] = by
        out[f"round_{number}"] = result
    return out


def defect_dry_run() -> dict[str, Any]:
    """Would the gated defect model gain from the AI-labelled rounds?"""
    from sklearn.model_selection import StratifiedGroupKFold

    from defaketive_sentiment.hazard_learned import fit_category_model, keyword_features
    from defaketive_sentiment.hazard_model import LexiconHazardTagger

    records = load_records()
    master: dict[str, dict[str, str]] = {}
    for number in (1, 2, 3):
        master.update({row["row_id"]: row for row in read(OUTPUTS / f"labelling_queue_round_{number}" / "queue_master.csv")})
    for number in (5, 6):
        rows, rows_master = ai_round(number)
        master.update(rows_master)
        records += [
            {"row_id": row["row_id"], "review_id": row["review_id"], "text": row["sentence_text"],
             "round": str(number), "labels": {"defect": row["ai_defect"] == "yes"}}
            for row in rows
        ]
    records = [record for record in records if record["labels"]["defect"] is not None]
    gadget = np.array([is_gadget(master[record["row_id"]]) for record in records])
    rounds = np.array([record["round"] for record in records])
    y = np.array([int(record["labels"]["defect"]) for record in records])
    groups = np.array([record["review_id"] for record in records])
    tagger = LexiconHazardTagger(counterfeit_rules=None)
    tagged = [tagger.tag_sentence(record["text"]) for record in records]
    text = np.array([item["text_normalized"] for item in tagged], dtype=object)
    kw = np.vstack([keyword_features(item) for item in tagged])
    flag = np.array(["defect" in item["tags"] for item in tagged])
    read_sentiment, _ = sentiment_reader()
    praise = np.array([read_sentiment(record["text"]) == "positive" for record in records])

    def fit(idx: np.ndarray):
        return fit_category_model("TK", list(text[idx]), kw[idx], None, y[idx], category="defect", seed=SEED)

    def out_of_fold(idx: np.ndarray, always: np.ndarray | None = None) -> np.ndarray:
        probability = np.zeros(len(idx))
        folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for train, valid in folds.split(np.zeros(len(idx)), y[idx], groups[idx]):
            fit_on = idx[train] if always is None else np.concatenate([idx[train], always])
            probability[valid] = fit(fit_on).probability(list(text[idx[valid]]), kw[idx[valid]], None)
        return probability

    def decide(idx: np.ndarray, probability: np.ndarray, threshold: float) -> np.ndarray:
        return (flag[idx] | ((probability >= threshold) & ~praise[idx])).astype(int)

    def threshold_for(idx: np.ndarray, probability: np.ndarray) -> float:
        best = None
        for threshold in sorted(set(np.round(probability, 6))):
            f1 = scores(y[idx], decide(idx, probability, threshold))["f1"]
            if best is None or f1 >= best[1]:
                best = (float(threshold), f1)
        return best[0]

    human = np.where(np.isin(rounds, HUMAN_ROUNDS))[0]
    five, six = np.where(rounds == "5")[0], np.where(rounds == "6")[0]
    human_five = np.concatenate([human, five])
    everything = np.concatenate([human_five, six])
    human_oof = out_of_fold(human)
    t_human = threshold_for(human, human_oof)
    t_five = threshold_for(human_five, out_of_fold(human_five))
    all_oof = out_of_fold(everything)
    t_all = threshold_for(everything, all_oof)
    all_pred = decide(everything, all_oof, t_all)
    is_human = np.isin(rounds[everything], HUMAN_ROUNDS)
    return {
        "design": "TK, keywords OR (model >= threshold AND not praise); threshold = best out-of-fold F1; r1b rule not simulated",
        "reports": {"rounds_1_3": [int(y[human].sum()), len(human)], "round_5": [int(y[five].sum()), len(five)], "round_6": [int(y[six].sum()), len(six)]},
        "scored_on_round_6": {
            "keywords_alone": scores(y[six], flag[six].astype(int)),
            "trained_rounds_1_3": {"threshold": t_human, **scores(y[six], decide(six, fit(human).probability(list(text[six]), kw[six], None), t_human))},
            "trained_rounds_1_3_5": {"threshold": t_five, **scores(y[six], decide(six, fit(human_five).probability(list(text[six]), kw[six], None), t_five))},
            "trained_rounds_1_3_5_6_cross_validated": {"threshold": t_all, **scores(y[six], decide(six, out_of_fold(six, human_five), t_all))},
        },
        "scored_on_human_gadget_sentences": {
            "trained_rounds_1_3_cross_validated": scores(y[human][gadget[human]], decide(human, human_oof, t_human)[gadget[human]]),
            "trained_all_rounds_cross_validated": scores(y[everything][is_human & gadget[everything]], all_pred[is_human & gadget[everything]]),
        },
        "scored_on_human_sentences": {
            "trained_rounds_1_3_cross_validated": scores(y[human], decide(human, human_oof, t_human)),
            "trained_all_rounds_cross_validated": scores(y[everything][is_human], all_pred[is_human]),
        },
    }


def five_label_dry_run(relabelled: bool = False) -> dict[str, Any]:
    """Cross-validated dry run of a learned model for each of the five labels.

    The two new labels have no human-labelled data, so they are trained and
    scored on the AI-labelled rounds only. Nothing is saved.

    With `relabelled`, the team's defect and counterfeit answers for rounds 1-3
    are replaced by ai_relabel.csv (guide 1.4 applied by the assistant)."""
    from sklearn.model_selection import StratifiedGroupKFold

    from defaketive_sentiment.hazard_learned import fit_category_model, keyword_features
    from defaketive_sentiment.hazard_model import LexiconHazardTagger
    from label_round5_ai import CATEGORIES as FIVE_LABELS

    records = load_records()
    # The two new labels for the team's rounds come from the assistant
    # (research/label_previous_rounds_ai.py); the team's three are untouched.
    extra: dict[str, dict[str, bool]] = {}
    for number in (1, 2, 3, 4):
        for row in read(OUTPUTS / f"labelling_queue_round_{number}" / "ai_extra_labels.csv"):
            extra[row["row_id"]] = {label: row[f"ai_{label}"] == "yes" for label in ("quality", "not_as_described")}
    for record in records:
        record["labels"] = {**{label: None for label in FIVE_LABELS}, **record["labels"], **extra[record["row_id"]]}
    if relabelled:
        answer = {"yes": True, "no": False, "": None}
        moved = {}
        for number in (1, 2, 3):
            for row in read(OUTPUTS / f"labelling_queue_round_{number}" / "ai_relabel.csv"):
                moved[row["row_id"]] = {"defect": answer[row["relabel_defect"]], "counterfeit": answer[row["relabel_counterfeit"]]}
        for record in records:
            record["labels"].update(moved[record["row_id"]])
    # Round 4 was labelled for counterfeit only, and only its new labels are used.
    for row in read(OUTPUTS / "labelling_queue_round_4" / "queue_master.csv"):
        records.append({
            "review_id": row["review_id"], "text": row["sentence_text"], "round": "4",
            "labels": {**{label: None for label in FIVE_LABELS}, **extra[row["row_id"]]},
        })
    for number in AI_ROUNDS:
        rows, _ = ai_round(number)
        records += [
            {"review_id": row["review_id"], "text": row["sentence_text"], "round": str(number),
             "labels": {label: row[f"ai_{label}"] == "yes" for label in FIVE_LABELS}}
            for row in rows
        ]
    rounds = np.array([record["round"] for record in records])
    groups = np.array([record["review_id"] for record in records])
    plain, with_rules = LexiconHazardTagger(counterfeit_rules=None), LexiconHazardTagger()
    tagged = [plain.tag_sentence(record["text"]) for record in records]
    text = np.array([item["text_normalized"] for item in tagged], dtype=object)
    kw = np.vstack([keyword_features(item) for item in tagged])
    flags = {
        "safety": np.array(["safety" in item["tags"] for item in tagged]),
        "defect": np.array(["defect" in item["tags"] for item in tagged]),
        "counterfeit": np.array(["counterfeit" in with_rules.tag_sentence(record["text"])["tags"] for record in records]),
    }
    read_sentiment, _ = sentiment_reader()
    praise = np.array([read_sentiment(record["text"]) == "positive" for record in records])
    six, gadget_rounds, ai = rounds == "6", np.isin(rounds, ["6", "7"]), np.isin(rounds, [str(n) for n in AI_ROUNDS])
    out: dict[str, Any] = {}
    for label in FIVE_LABELS:
        idx = np.array([i for i, record in enumerate(records) if record["labels"][label] is not None])
        y = np.array([int(records[i]["labels"][label]) for i in idx])
        probability = np.zeros(len(idx))
        folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        for train, valid in folds.split(np.zeros(len(idx)), y, groups[idx]):
            model = fit_category_model("TK", list(text[idx[train]]), kw[idx[train]], None, y[train], category=label, seed=SEED)
            probability[valid] = model.probability(list(text[idx[valid]]), kw[idx[valid]], None)
        flag = flags[label][idx] if label in flags else np.zeros(len(idx), dtype=bool)
        gate = ~praise[idx] if label == "defect" else np.ones(len(idx), dtype=bool)

        def combine(threshold: float, with_keywords: bool) -> np.ndarray:
            add = (probability >= threshold) & gate
            return ((flag | add) if with_keywords else (probability >= threshold)).astype(int)

        def best(with_keywords: bool) -> tuple[float, np.ndarray]:
            top = None
            for threshold in sorted(set(np.round(probability, 6))):
                f1 = scores(y, combine(threshold, with_keywords))["f1"]
                if top is None or f1 >= top[1]:
                    top = (float(threshold), f1)
            return top[0], combine(top[0], with_keywords)

        subsets = {"round_6_gadgets": six[idx], "rounds_6_7_gadgets": gadget_rounds[idx], "ai_rounds": ai[idx]}
        subsets["earlier_rounds"] = ~ai[idx]
        entry: dict[str, Any] = {"training_sentences": int(len(idx)), "training_reports": int(y.sum())}
        if label not in flags:
            # What the same model reached with the AI-labelled rounds 5-7 alone.
            only = np.where(ai[idx])[0]
            before = np.zeros(len(only))
            for train, valid in folds.split(np.zeros(len(only)), y[only], groups[idx][only]):
                model = fit_category_model("TK", list(text[idx[only[train]]]), kw[idx[only[train]]], None, y[only[train]], category=label, seed=SEED)
                before[valid] = model.probability(list(text[idx[only[valid]]]), kw[idx[only[valid]]], None)
            best_f1 = max(scores(y[only], (before >= t).astype(int))["f1"] for t in sorted(set(np.round(before, 6))))
            entry["rounds_5_7_only_best_f1"] = best_f1
        variants = {"model_alone": best(False)}
        if label in flags:
            variants["keywords_alone"] = (None, flag.astype(int))
            variants["keywords_or_model"] = best(True)
        for name, (threshold, predicted) in variants.items():
            entry[name] = {"threshold": threshold, **{key: scores(y[mask], predicted[mask]) for key, mask in subsets.items()}}
        out[label] = entry
    return out


def main() -> int:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Rounds 5 and 6 are AI-labelled and were accepted unaudited. Agreement with those labels, not accuracy.",
        "agreement": agreement(),
        "defect_training_dry_run": defect_dry_run(),
        "five_label_dry_run": five_label_dry_run(),
        "five_label_dry_run_relabelled": five_label_dry_run(relabelled=True),
    }
    REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))

    def line(stats: dict[str, Any]) -> str:
        return f"P {stats['precision']:.2f} R {stats['recall']:.2f} F1 {stats['f1']:.3f} (caught {stats['tp']}, false {stats['fp']}, missed {stats['fn']})"

    for round_name, result in report["agreement"].items():
        print(f"{round_name}: {result['sentences']}")
        for name in ("v6_shipped", "v6_candidate", "shipped_c2"):
            for category in result[name]:
                parts = "  |  ".join(f"{key} {line(stats)}" for key, stats in result[name][category].items() if key in ("all", "gadget"))
                print(f"  {name:13} {category:16} {parts}")
    dry = report["defect_training_dry_run"]
    print("defect dry run, scored on round 6:")
    for key, stats in dry["scored_on_round_6"].items():
        print(f"  {key:40} {line(stats)}")
    for block in ("scored_on_human_gadget_sentences", "scored_on_human_sentences"):
        for key, stats in dry[block].items():
            print(f"  {block[10:]:26} {key:36} {line(stats)}")
    for title, block in (("team labels for rounds 1-3", "five_label_dry_run"), ("rounds 1-3 relabelled", "five_label_dry_run_relabelled")):
      print(f"five-label dry run, {title} (cross-validated):")
      for label, entry in report[block].items():
          alone = entry.get("rounds_5_7_only_best_f1")
          note = f" (rounds 5-7 alone: best F1 {alone:.3f})" if alone is not None else ""
          print(f"  {label}: {entry['training_reports']} reports in {entry['training_sentences']} sentences{note}")
          for name in ("keywords_alone", "model_alone", "keywords_or_model"):
              if name in entry:
                  parts = "  |  ".join(f"{key} {line(stats)}" for key, stats in entry[name].items() if key != "threshold")
                  print(f"    {name:18} {parts}")
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
