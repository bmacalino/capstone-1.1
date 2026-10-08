"""A model for every label, trained on every labelled round.

The owner asked (8 October 2026) for the models to be trained on the
assistant's labels from all the rounds. The defect model already is
(hazard_classifier_v5). This does the same for the other labels and checks,
one round at a time, whether a learned model beats what the app serves.

For each label, each of these is left out in turn and used only for scoring:
rounds 2, 3, 5, 6, 7, and round 8's live-check listings (ordinary reviews no
model was trained on). The design in use (TF-IDF and keyword features,
logistic regression) is trained on everything else and combined with the
label's keywords or rules in three ways:

  decide  the model alone
  add     rules OR model (never on praise, for defect and quality)
  filter  rules AND model

Each threshold is the best out-of-fold F1 on the training rounds, so the
left-out round takes no part in any choice.

Truth is the team's agreed answer where one exists (safety, defect and
counterfeit in rounds 2-3) and the assistant's label otherwise. So only those
team lines are measurements; the rest is agreement with AI labels that the
owner accepted unaudited (PREREGISTRATION_final_v6.md, departure 3).

Rule, written before the first run. A design replaces what the app serves
for a label only if, summed over the left-out rounds, its F1 is at least 0.02
above the served checker's, and - for safety and counterfeit - it loses no
real report the served checker finds in the team's rounds 2 and 3. Of the
designs that pass, the one with the highest summed F1 is taken. Defect is
reported but not decided here: the served defect model was itself trained on
these rounds, so its figures on them are not comparable.

    python research/train_all_rounds.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from five_label_data import LABELS, keyword_view, load_five_label_records, sentiment_labels  # noqa: E402
from ordinary_review_pool import checker  # noqa: E402
from train_hazard_classifier import FOLDS, SEED, cross_validate, scores  # noqa: E402

REPORT = ROOT / "research" / "all_rounds_training.json"
SERVED = "hazard_classifier_v5"
LEFT_OUT = ("2", "3", "5", "6", "7", "8_live")
TEAM_ROUNDS = ("2", "3")
TEAM_LABELS = ("safety", "defect", "counterfeit")
LEARNED_NOW = ("safety", "defect")
GATED = ("defect", "quality")
DESIGNS = ("decide", "add", "filter")
MARGIN = 0.02


def split_of(record: dict[str, Any]) -> str:
    return "8_live" if record["part"] == "live_check" else record["round"]


def truth_for(record: dict[str, Any], label: str, split: str) -> bool | None:
    if split in TEAM_ROUNDS and label in TEAM_LABELS:
        return record["team"].get(label)
    return record["labels"].get(label)


def combine(design: str, flag: np.ndarray, score: np.ndarray, gate: np.ndarray, threshold: float) -> np.ndarray:
    over = score >= threshold
    if design == "decide":
        return over
    if design == "add":
        return flag | (over & gate)
    return flag & over


def pick_threshold(design: str, y: np.ndarray, flag: np.ndarray, score: np.ndarray, gate: np.ndarray) -> float:
    best = None
    for threshold in sorted(set(np.round(score, 6))):
        f1 = scores(y, combine(design, flag, score, gate, threshold).astype(int))["f1"]
        if best is None or f1 >= best[1]:
            best = (float(threshold), f1)
    return best[0]


def total(counts: Counter) -> dict[str, Any]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


LENIENT = {"filter_keeping_every_report": 0.0, "filter_keeping_95_in_100": 0.05}


def lenient_safety() -> int:
    """A further safety variant, tried after the first result (the owner asked
    for the best model to be used): a filter set so loosely that it keeps
    every real report the keywords flag in the training rounds, or 95 in 100
    of them. Same left-out rounds, same rule. Added to the report."""
    from defaketive_sentiment.hazard_learned import fit_category_model

    label = "safety"
    records = load_five_label_records(include_live_check=True)
    view = keyword_view(records)
    feats = {"normalized": view["text"], "kw": view["kw"], "emb": None}
    splits = np.array([split_of(record) for record in records])
    served = checker(SERVED)
    now = np.array([label in served.tag_sentence(record["text"], record["rating"])["tags"] for record in records])
    rules = view["flags"][label].astype(bool)
    sums = {name: Counter() for name in ("served", *LENIENT)}
    lost: Counter = Counter()
    rows: dict[str, Any] = {}
    for left in LEFT_OUT:
        train = np.array([
            i for i, record in enumerate(records)
            if splits[i] != left and record["part"] != "live_check" and record["labels"][label] is not None
        ])
        test = np.array([i for i, record in enumerate(records) if splits[i] == left and truth_for(record, label, left) is not None])
        y = np.array([int(records[i]["labels"][label]) for i in train])
        truth = np.array([int(truth_for(records[i], label, left)) for i in test])
        groups = np.array([records[i]["review_id"] for i in train])
        out_of_fold = cross_validate("TK", label, train, y, groups, feats)
        model = fit_category_model("TK", list(view["text"][train]), view["kw"][train], None, y, category=label, seed=SEED)
        probability = model.probability(list(view["text"][test]), view["kw"][test], None)
        row: dict[str, Any] = {"reports": int(truth.sum()), "served": scores(truth, now[test].astype(int))}
        sums["served"].update({key: row["served"][key] for key in ("tp", "fp", "fn")})
        flagged_reports = np.sort(out_of_fold[rules[train] & (y == 1)])
        for name, share in LENIENT.items():
            threshold = float(flagged_reports[int(np.floor(share * len(flagged_reports)))]) if len(flagged_reports) else 0.0
            predicted = rules[test] & (probability >= threshold)
            row[name] = {**scores(truth, predicted.astype(int)), "threshold": threshold}
            sums[name].update({key: row[name][key] for key in ("tp", "fp", "fn")})
            if left in TEAM_ROUNDS:
                lost[name] += int((now[test] & (truth == 1) & ~predicted).sum())
        rows[left] = row
        print(f"safety left out {left:7} reports {row['reports']:3} | " + " | ".join(
            f"{name} {row[name]['tp']}/{row[name]['fp']}" for name in ("served", *LENIENT)
        ), flush=True)
    totals = {name: total(counts) for name, counts in sums.items()}
    passing = [name for name in LENIENT if totals[name]["f1"] >= totals["served"]["f1"] + MARGIN and lost[name] == 0]
    for name, value in totals.items():
        print(f"  SUMMED {name:28} caught {value['tp']} false {value['fp']} F1 {value['f1']:.3f} lost on team rounds {lost.get(name, 0)}")
    print(f"  passes the rule: {passing or 'none'}")
    report = json.loads(REPORT.read_bytes().decode("utf-8"))
    report["labels"]["safety"]["lenient_filters"] = {
        "tried_after_the_first_result": True, "left_out_rounds": rows, "summed": totals,
        "served_reports_lost_on_team_rounds": dict(lost), "passes_the_rule": passing,
    }
    REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(f"added to {REPORT.name}")
    return 0


def main() -> int:
    if "--lenient-safety" in sys.argv:
        return lenient_safety()
    from defaketive_sentiment.hazard_learned import fit_category_model

    records = load_five_label_records(include_live_check=True)
    view = keyword_view(records)
    praise = sentiment_labels(records) == "positive"
    feats = {"normalized": view["text"], "kw": view["kw"], "emb": None}
    splits = np.array([split_of(record) for record in records])
    served = checker(SERVED)
    served_tags = [set(served.tag_sentence(record["text"], record["rating"])["tags"]) for record in records]

    report: dict[str, Any] = {}
    for label in LABELS:
        rules = view["flags"][label].astype(bool)
        now = np.array([label in tags for tags in served_tags]) if label in LEARNED_NOW else rules
        sums = {name: Counter() for name in ("served", "rules", *DESIGNS)}
        lost = Counter()
        by_round: dict[str, Any] = {}
        for left in LEFT_OUT:
            train = np.array([
                i for i, record in enumerate(records)
                if splits[i] != left and record["part"] != "live_check" and record["labels"][label] is not None
            ])
            test = np.array([i for i, record in enumerate(records) if splits[i] == left and truth_for(record, label, left) is not None])
            if not len(test):
                continue
            y = np.array([int(records[i]["labels"][label]) for i in train])
            truth = np.array([int(truth_for(records[i], label, left)) for i in test])
            groups = np.array([records[i]["review_id"] for i in train])
            out_of_fold = cross_validate("TK", label, train, y, groups, feats)
            model = fit_category_model("TK", list(view["text"][train]), view["kw"][train], None, y, category=label, seed=SEED)
            probability = model.probability(list(view["text"][test]), view["kw"][test], None)
            open_gate = np.ones(len(records), dtype=bool)
            gate = ~praise if label in GATED else open_gate
            team = left in TEAM_ROUNDS and label in TEAM_LABELS
            row: dict[str, Any] = {
                "sentences": int(len(test)), "reports": int(truth.sum()), "truth": "team" if team else "assistant",
                "served": scores(truth, now[test].astype(int)), "rules": scores(truth, rules[test].astype(int)),
            }
            for name in ("served", "rules"):
                sums[name].update({key: row[name][key] for key in ("tp", "fp", "fn")})
            for design in DESIGNS:
                threshold = pick_threshold(design, y, rules[train], out_of_fold, gate[train])
                predicted = combine(design, rules[test], probability, gate[test], threshold)
                row[design] = {**scores(truth, predicted.astype(int)), "threshold": threshold}
                sums[design].update({key: row[design][key] for key in ("tp", "fp", "fn")})
                if team:
                    row[design]["served_reports_lost"] = int((now[test] & (truth == 1) & ~predicted).sum())
                    lost[design] += row[design]["served_reports_lost"]
            by_round[left] = row
            print(f"{label:17} left out {left:7} reports {row['reports']:4} ({row['truth']:9}) | " + " | ".join(
                f"{name} {row[name]['tp']}/{row[name]['fp']}" for name in ("served", "rules", *DESIGNS)
            ), flush=True)
        totals = {name: total(counts) for name, counts in sums.items()}
        passing = [
            design for design in DESIGNS
            if totals[design]["f1"] >= totals["served"]["f1"] + MARGIN and (label not in ("safety", "counterfeit") or lost[design] == 0)
        ]
        decided = label != "defect"
        choice = max(passing, key=lambda design: totals[design]["f1"]) if decided and passing else None
        report[label] = {
            "left_out_rounds": by_round, "summed": totals,
            "served_reports_lost_on_team_rounds": dict(lost) if label in ("safety", "counterfeit") else None,
            "decided_here": decided, "passes_the_rule": passing if decided else None, "choice": choice,
        }
        print(f"{label:17} SUMMED " + " | ".join(
            f"{name} caught {value['tp']} false {value['fp']} F1 {value['f1']:.3f}" for name, value in totals.items()
        ) + f" | lost on team rounds {dict(lost)} | choice: {choice if decided else 'not decided here'}", flush=True)
    REPORT.write_bytes((json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Only lines with truth 'team' are measured against human labels. The rest is agreement with AI labels.",
        "served": SERVED, "margin": MARGIN, "cross_validation": {"folds": FOLDS, "grouped_by": "review_id", "seed": SEED},
        "labels": report,
    }, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
