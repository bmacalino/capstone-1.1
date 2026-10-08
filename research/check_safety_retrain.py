"""Would the safety model gain from the rounds labelled since it was trained?

The served safety model learned from round 1 only (30 reports). This trains
the same design (TK, the model decides, highest threshold with out-of-fold
recall >= 0.80) on more rounds and scores it on the team's round-3 safety
answers, which neither it nor the served model was trained on:

  team_rounds_1_2   the team's labels only
  all_but_round_3   the same plus rounds 5-7 and round 8's mining listings,
                    whose labels were proposed by the AI assistant

Rule, written before the first run: a variant may replace the served safety
model only if on round 3 it finds at least as many of the reports and raises
fewer false flags; of two that qualify, the one trained on the team's labels
alone is preferred. Otherwise the served model stays.

    python research/check_safety_retrain.py
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

from five_label_data import KEYWORD_RULES, keyword_view, load_five_label_records  # noqa: E402
from ordinary_review_pool import checker  # noqa: E402
from train_hazard_classifier import SEED, choose_threshold, cross_validate, scores  # noqa: E402

REPORT = ROOT / "research" / "safety_retrain_check.json"
SERVED = "hazard_classifier_v2b"
VARIANTS = {"team_rounds_1_2": ("1", "2"), "all_but_round_3": ("1", "2", "5", "6", "7", "8")}


def main() -> int:
    from defaketive_sentiment.hazard_learned import fit_category_model

    records = load_five_label_records(include_live_check=True)
    view = keyword_view(records, KEYWORD_RULES)
    feats = {"normalized": view["text"], "kw": view["kw"], "emb": None}
    test = np.array([i for i, r in enumerate(records) if r["round"] == "3" and r["team"].get("safety") is not None])
    truth = np.array([int(records[i]["team"]["safety"]) for i in test])
    ordinary = np.array([i for i, r in enumerate(records) if r["part"] == "live_check"])
    ordinary_truth = np.array([int(records[i]["labels"]["safety"]) for i in ordinary])
    served = checker(SERVED)

    def served_flags(rows: np.ndarray) -> np.ndarray:
        return np.array([int("safety" in served.tag_sentence(records[i]["text"])["tags"]) for i in rows])

    results: dict[str, Any] = {
        "served_round_1_only": {
            "round_3_team_labels": scores(truth, served_flags(test)),
            "round_8_live_check_ai_labels": scores(ordinary_truth, served_flags(ordinary)),
        },
        "keywords_alone": {
            "round_3_team_labels": scores(truth, view["flags"]["safety"][test].astype(int)),
            "round_8_live_check_ai_labels": scores(ordinary_truth, view["flags"]["safety"][ordinary].astype(int)),
        },
    }
    for name, rounds in VARIANTS.items():
        train = np.array([
            i for i, r in enumerate(records)
            if r["round"] in rounds and r["part"] != "live_check" and r["labels"]["safety"] is not None
        ])
        y = np.array([int(records[i]["labels"]["safety"]) for i in train])
        groups = np.array([records[i]["review_id"] for i in train])
        out_of_fold = cross_validate("TK", "safety", train, y, groups, feats)
        threshold = choose_threshold(y, out_of_fold, "safety")
        model = fit_category_model("TK", list(view["text"][train]), view["kw"][train], None, y, category="safety", seed=SEED)

        def flags(rows: np.ndarray) -> np.ndarray:
            return (model.probability(list(view["text"][rows]), view["kw"][rows], None) >= threshold).astype(int)

        results[name] = {
            "training_sentences": int(len(train)), "training_reports": int(y.sum()), "threshold": threshold,
            "cross_validated": scores(y, (out_of_fold >= threshold).astype(int)),
            "round_3_team_labels": scores(truth, flags(test)),
            "round_8_live_check_ai_labels": scores(ordinary_truth, flags(ordinary)),
        }
    base = results["served_round_1_only"]["round_3_team_labels"]
    for name in VARIANTS:
        got = results[name]["round_3_team_labels"]
        results[name]["qualifies"] = bool(got["tp"] >= base["tp"] and got["fp"] < base["fp"])
    for name, body in results.items():
        print(name, json.dumps(body))
    REPORT.write_bytes((json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "replace the served safety model only if on round 3 a variant finds at least as many reports and raises fewer false flags",
        "round_3_sentences": int(len(test)), "round_3_reports": int(truth.sum()),
        "results": results,
    }, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
