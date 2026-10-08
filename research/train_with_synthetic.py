"""Do the written (synthetic) training sentences make any label better?

Same protocol as research/train_all_rounds.py: for each label, each of
rounds 2, 3, 5, 6, 7 and round 8's live-check listings is left out in turn
and used only for scoring. The difference is the training set: the real
rounds PLUS every sentence of research/synthetic/ (written by the AI
assistant, research/synthetic_data.py). Synthetic sentences are never scored
and never used to set a threshold: thresholds come from out-of-fold scores
of the real training sentences only.

Rule, written before the first run. A design trained with the synthetic
sentences replaces what the app serves for a label only if, summed over the
left-out real rounds:
  1. its F1 is at least 0.02 above the served checker's;
  2. for safety and counterfeit, it loses no real report the served checker
     finds in the team's rounds 2 and 3;
  3. of the flags it raises that the served checker does not, at least 70 in
     100 are right. (Added after research/train_all_rounds.py showed that F1
     alone let through a model whose additions were half wrong.)
Of the designs that pass, the one with the highest summed F1 is taken.
Defect is reported but decided separately (research/train_candidate_v4.py
--synthetic and research/live_check.py), because the served defect model was
trained on these rounds.

    python research/train_with_synthetic.py
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
from synthetic_data import MANIFEST, load_synthetic_records  # noqa: E402
from train_all_rounds import (  # noqa: E402
    DESIGNS, GATED, LEARNED_NOW, LEFT_OUT, MARGIN, SERVED, TEAM_LABELS, TEAM_ROUNDS,
    combine, pick_threshold, split_of, total, truth_for,
)
from train_hazard_classifier import FOLDS, SEED, scores  # noqa: E402

REPORT = ROOT / "research" / "synthetic_training.json"
REAL_ONLY = ROOT / "research" / "all_rounds_training.json"
ADDITIONS_RIGHT = 0.70


def out_of_fold_with_extra(label, train, y, groups, view, extra, extra_y) -> np.ndarray:
    """Out-of-fold scores of the real training sentences. The synthetic ones
    join every training fold and are never in a validation fold."""
    from sklearn.model_selection import StratifiedGroupKFold

    from defaketive_sentiment.hazard_learned import fit_category_model

    out = np.zeros(len(train))
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fit, held in folds.split(np.zeros(len(train)), y, groups):
        rows = np.concatenate([train[fit], extra])
        model = fit_category_model(
            "TK", list(view["text"][rows]), view["kw"][rows], None, np.concatenate([y[fit], extra_y]), category=label, seed=SEED
        )
        out[held] = model.probability(list(view["text"][train[held]]), view["kw"][train[held]], None)
    return out


def serious_safety_flags(records) -> np.ndarray:
    """True where a safety keyword stronger than mild heat counted: fire, smoke,
    shock, swelling. The filter is never allowed to remove such a flag
    (manifest key filter_spares_severity_above)."""
    from defaketive_sentiment.hazard_model import MILD_HEAT_MAX_SEVERITY, SHIPPED_RULES, LexiconHazardTagger

    plain = LexiconHazardTagger(rules=SHIPPED_RULES, counterfeit_rules=None)
    strongest = []
    for record in records:
        terms = plain.tag_sentence(record["text"])["terms"]
        strongest.append(max(
            [term["severity"] for term in terms if term["category"] == "safety" and not term["negated"] and not term.get("excluded")]
            or [0]
        ))
    return np.array(strongest) > MILD_HEAT_MAX_SEVERITY


def lenient_safety() -> int:
    """The two looser safety filters of research/train_all_rounds.py
    (keep every real report the keywords flag in the training rounds, or 95
    in 100), now with the synthetic sentences in training. Same left-out
    rounds and the same rule. Tried after the first result, where the
    best-F1 filter kept every report in the team's rounds and missed the F1
    margin by 0.004. One safeguard was then added, before anything was
    served: the filter may not remove a flag stronger than mild heat
    (serious_safety_flags), so a fire, smoke or shock report always stays."""
    from defaketive_sentiment.hazard_learned import fit_category_model
    from train_all_rounds import LENIENT

    label = "safety"
    real = load_five_label_records(include_live_check=True)
    synthetic = load_synthetic_records()
    records = real + synthetic
    view = keyword_view(records)
    splits = np.array([split_of(record) for record in real] + ["synthetic"] * len(synthetic))
    extra = np.arange(len(real), len(records))
    extra_y = np.array([int(records[i]["labels"][label]) for i in extra])
    served = checker(SERVED)
    now = np.array([label in served.tag_sentence(record["text"], record["rating"])["tags"] for record in real] + [False] * len(synthetic))
    rules = view["flags"][label].astype(bool)
    serious = serious_safety_flags(records)
    sums = {name: Counter() for name in ("served", *LENIENT)}
    lost: Counter = Counter()
    rows_out: dict[str, Any] = {}
    for left in LEFT_OUT:
        train = np.array([
            i for i, record in enumerate(real)
            if splits[i] != left and record["part"] != "live_check" and record["labels"][label] is not None
        ])
        test = np.array([i for i, record in enumerate(real) if splits[i] == left and truth_for(record, label, left) is not None])
        y = np.array([int(real[i]["labels"][label]) for i in train])
        truth = np.array([int(truth_for(real[i], label, left)) for i in test])
        groups = np.array([real[i]["review_id"] for i in train])
        out_of_fold = out_of_fold_with_extra(label, train, y, groups, view, extra, extra_y)
        rows = np.concatenate([train, extra])
        model = fit_category_model(
            "TK", list(view["text"][rows]), view["kw"][rows], None, np.concatenate([y, extra_y]), category=label, seed=SEED
        )
        probability = model.probability(list(view["text"][test]), view["kw"][test], None)
        row: dict[str, Any] = {"reports": int(truth.sum()), "served": scores(truth, now[test].astype(int))}
        sums["served"].update({key: row["served"][key] for key in ("tp", "fp", "fn")})
        flagged_reports = np.sort(out_of_fold[rules[train] & (y == 1)])
        for name, share in LENIENT.items():
            threshold = float(flagged_reports[int(np.floor(share * len(flagged_reports)))]) if len(flagged_reports) else 0.0
            predicted = rules[test] & ((probability >= threshold) | serious[test])
            row[name] = {**scores(truth, predicted.astype(int)), "threshold": threshold}
            sums[name].update({key: row[name][key] for key in ("tp", "fp", "fn")})
            if left in TEAM_ROUNDS:
                lost[name] += int((now[test] & (truth == 1) & ~predicted).sum())
        rows_out[left] = row
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
        "tried_after_the_first_result": True, "never_removes_a_flag_stronger_than_mild_heat": True,
        "left_out_rounds": rows_out, "summed": totals,
        "served_reports_lost_on_team_rounds": dict(lost), "passes_the_rule": passing,
    }
    REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(f"added to {REPORT.name}")
    return 0


KEPT_SHARE = 0.05  # the filter that met the rule: keeps 95 in 100 flagged real reports, out of fold
OUT_FOLDER = "hazard_classifier_v6"


def freeze_safety() -> int:
    """Freeze the safety filter that met the rule into hazard_classifier_v6.

    Trained on every real round (the live-check listings stay out, as always)
    plus the synthetic sentences. It is a filter: a keyword safety flag
    stands only if the model agrees, the model never adds a flag, and it may
    not remove a flag stronger than mild heat. The
    defect model is hazard_classifier_v5's, copied unchanged."""
    import shutil

    import joblib

    from defaketive_sentiment.hazard_learned import MANIFEST_NAME, MODELS_ROOT, file_sha256, fit_category_model

    label = "safety"
    real = load_five_label_records()
    synthetic = load_synthetic_records()
    records = real + synthetic
    view = keyword_view(records)
    train = np.array([i for i, record in enumerate(real) if record["labels"][label] is not None])
    y = np.array([int(real[i]["labels"][label]) for i in train])
    groups = np.array([real[i]["review_id"] for i in train])
    extra = np.arange(len(real), len(records))
    extra_y = np.array([int(records[i]["labels"][label]) for i in extra])
    rules = view["flags"][label].astype(bool)
    serious = serious_safety_flags(records)
    out_of_fold = out_of_fold_with_extra(label, train, y, groups, view, extra, extra_y)
    flagged_reports = np.sort(out_of_fold[rules[train] & (y == 1)])
    threshold = float(flagged_reports[int(np.floor(KEPT_SHARE * len(flagged_reports)))])
    cross_validated = {
        "keywords_alone": scores(y, rules[train].astype(int)),
        "filter": scores(y, (rules[train] & ((out_of_fold >= threshold) | serious[train])).astype(int)),
    }
    rows = np.concatenate([train, extra])
    model = fit_category_model(
        "TK", list(view["text"][rows]), view["kw"][rows], None, np.concatenate([y, extra_y]), category=label, seed=SEED
    )
    model.threshold = threshold

    source, out = MODELS_ROOT / SERVED, MODELS_ROOT / OUT_FOLDER
    manifest = json.loads((source / MANIFEST_NAME).read_bytes().decode("utf-8"))
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    defect = manifest["categories"]["defect"]
    shutil.copyfile(source / defect["file"], out / defect["file"])
    joblib.dump(model, out / "safety.joblib")
    evidence = json.loads(REPORT.read_bytes().decode("utf-8"))["labels"][label]["lenient_filters"]
    manifest.update({
        "version": "hazard-learned-6.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "categories": {
            "safety": {
                "file": "safety.joblib", "sha256": file_sha256(out / "safety.joblib"), "candidate": "TK",
                "threshold": threshold, "combine": "and", "needs_embedding": False,
                # A flag stronger than mild heat (fire, smoke, shock) is never the model's to drop.
                "filter_spares_severity_above": 2,
                "threshold_rule": "keeps 95 in 100 of the real reports the keywords flag, out of fold (research/train_with_synthetic.py)",
                "training_rows": int(len(train)), "training_positives": int(y.sum()),
                "synthetic_rows": int(len(extra)), "synthetic_positives": int(extra_y.sum()),
                "training_data": (
                    "every labelled round (team safety answers for rounds 1-3, AI labels for rounds 5-8, accepted "
                    "unaudited) plus sentences written by the AI assistant (research/synthetic/), used for training only"
                ),
                "cross_validated_on_real_sentences": cross_validated,
            },
            "defect": {**defect, "from": f"{SERVED} (unchanged)"},
        },
        "synthetic_training_data": json.loads(MANIFEST.read_bytes().decode("utf-8")),
        "qualified_for_integration": False,
        "adoption": {
            "adopted": True,
            "adopted_by": "owner instruction",
            "instruction": "adviser approved AI labels and synthetic data, go ahead (8 October 2026)",
            "not_a_preregistered_test": "research/PREREGISTRATION_final_v6.md, departures 2 and 4",
            "rule": "research/train_with_synthetic.py",
            "safety_summed_over_left_out_real_rounds": {
                "served_before": evidence["summed"]["served"],
                "this_filter": evidence["summed"]["filter_keeping_95_in_100"],
                "served_reports_lost_on_team_rounds": evidence["served_reports_lost_on_team_rounds"]["filter_keeping_95_in_100"],
            },
            "fallback": "DEFAKETIVE_HAZARD_MODEL=team_labels",
        },
    })
    (out / MANIFEST_NAME).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"safety filter: threshold {threshold:.4f}; real sentences {len(train)}, reports {int(y.sum())}; synthetic {len(extra)}")
    print(f"  cross-validated on real sentences: {cross_validated}")
    print(f"frozen {out}")
    return 0


def main() -> int:
    if "--lenient-safety" in sys.argv:
        return lenient_safety()
    if "--freeze-safety" in sys.argv:
        return freeze_safety()
    from defaketive_sentiment.hazard_learned import fit_category_model

    real = load_five_label_records(include_live_check=True)
    synthetic = load_synthetic_records()
    records = real + synthetic
    view = keyword_view(records)
    # The praise gate is only ever applied to real sentences.
    praise = np.concatenate([sentiment_labels(real) == "positive", np.zeros(len(synthetic), dtype=bool)])
    splits = np.array([split_of(record) for record in real] + ["synthetic"] * len(synthetic))
    extra = np.arange(len(real), len(records))
    served = checker(SERVED)
    served_tags = [set(served.tag_sentence(record["text"], record["rating"])["tags"]) for record in real] + [set()] * len(synthetic)
    before = json.loads(REAL_ONLY.read_bytes().decode("utf-8"))["labels"]

    report: dict[str, Any] = {}
    for label in LABELS:
        rules = view["flags"][label].astype(bool)
        now = np.array([label in tags for tags in served_tags]) if label in LEARNED_NOW else rules
        extra_y = np.array([int(records[i]["labels"][label]) for i in extra])
        sums = {name: Counter() for name in ("served", *DESIGNS)}
        lost: Counter = Counter()
        added = {design: Counter() for design in DESIGNS}
        by_round: dict[str, Any] = {}
        for left in LEFT_OUT:
            train = np.array([
                i for i, record in enumerate(real)
                if splits[i] != left and record["part"] != "live_check" and record["labels"][label] is not None
            ])
            test = np.array([i for i, record in enumerate(real) if splits[i] == left and truth_for(record, label, left) is not None])
            if not len(test):
                continue
            y = np.array([int(real[i]["labels"][label]) for i in train])
            truth = np.array([int(truth_for(real[i], label, left)) for i in test])
            groups = np.array([real[i]["review_id"] for i in train])
            out_of_fold = out_of_fold_with_extra(label, train, y, groups, view, extra, extra_y)
            rows = np.concatenate([train, extra])
            model = fit_category_model(
                "TK", list(view["text"][rows]), view["kw"][rows], None, np.concatenate([y, extra_y]), category=label, seed=SEED
            )
            probability = model.probability(list(view["text"][test]), view["kw"][test], None)
            gate = ~praise if label in GATED else np.ones(len(records), dtype=bool)
            team = left in TEAM_ROUNDS and label in TEAM_LABELS
            row: dict[str, Any] = {"reports": int(truth.sum()), "truth": "team" if team else "assistant", "served": scores(truth, now[test].astype(int))}
            sums["served"].update({key: row["served"][key] for key in ("tp", "fp", "fn")})
            for design in DESIGNS:
                threshold = pick_threshold(design, y, rules[train], out_of_fold, gate[train])
                predicted = combine(design, rules[test], probability, gate[test], threshold)
                row[design] = {**scores(truth, predicted.astype(int)), "threshold": threshold}
                sums[design].update({key: row[design][key] for key in ("tp", "fp", "fn")})
                extra_flags = predicted & ~now[test]
                added[design].update({"right": int((extra_flags & (truth == 1)).sum()), "wrong": int((extra_flags & (truth == 0)).sum())})
                if team:
                    lost[design] += int((now[test] & (truth == 1) & ~predicted).sum())
            by_round[left] = row
            print(f"{label:17} left out {left:7} reports {row['reports']:4} ({row['truth']:9}) | " + " | ".join(
                f"{name} {row[name]['tp']}/{row[name]['fp']}" for name in ("served", *DESIGNS)
            ), flush=True)
        totals = {name: total(counts) for name, counts in sums.items()}
        checks = {}
        for design in DESIGNS:
            extra_total = added[design]["right"] + added[design]["wrong"]
            checks[design] = {
                "f1_gain": round(totals[design]["f1"] - totals["served"]["f1"], 4),
                "served_reports_lost_on_team_rounds": int(lost[design]) if label in ("safety", "counterfeit") else None,
                "added_flags": dict(added[design]),
                "added_flags_right_share": round(added[design]["right"] / extra_total, 4) if extra_total else None,
            }
            checks[design]["passes"] = bool(
                checks[design]["f1_gain"] >= MARGIN
                and (label not in ("safety", "counterfeit") or lost[design] == 0)
                and (not extra_total or added[design]["right"] / extra_total >= ADDITIONS_RIGHT)
            )
        decided = label != "defect"
        passing = [design for design in DESIGNS if checks[design]["passes"]]
        choice = max(passing, key=lambda design: totals[design]["f1"]) if decided and passing else None
        report[label] = {
            "left_out_rounds": by_round, "summed": totals, "checks": checks,
            "summed_without_synthetic": before[label]["summed"], "decided_here": decided, "choice": choice,
        }
        print(f"{label:17} SUMMED served {totals['served']['tp']}/{totals['served']['fp']} F1 {totals['served']['f1']:.3f}", flush=True)
        for design in DESIGNS:
            old = before[label]["summed"][design]
            print(f"   {design:7} with synthetic {totals[design]['tp']}/{totals[design]['fp']} F1 {totals[design]['f1']:.3f}"
                  f" (without {old['tp']}/{old['fp']} F1 {old['f1']:.3f}) | added {dict(added[design])} | lost {lost[design]}"
                  f" | passes {checks[design]['passes']}", flush=True)
        print(f"   choice: {choice if decided else 'decided separately'}", flush=True)
    REPORT.write_bytes((json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Scored on real sentences only. Lines with truth 'team' are against human labels; the rest is agreement with AI labels.",
        "synthetic": json.loads(MANIFEST.read_bytes().decode("utf-8")),
        "served": SERVED, "margin": MARGIN, "additions_right_at_least": ADDITIONS_RIGHT, "labels": report,
    }, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
