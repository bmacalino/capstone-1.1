"""Does more data raise the three weak labels on the held-out third?

The yardstick is the held-out third of rounds 1-8
(research/improve_label_rules.py): 1,748 real sentences that were never read
while anything was built and that no later round can join. Everything else
is training material, tried in steps so each source's share is visible:

  rounds 1-8        the dev two thirds only
  + rounds 9..N     the real sentences labelled since (research/build_round9.py),
                    one batch at a time: the learning curve
  + written         the sentences written by the assistant (research/synthetic/):
                    the first set, then the second, and the written ones
                    without the new real rounds, so each source's share shows

For each step and label the design in use (TF-IDF and keyword features,
logistic regression) is trained on four fifths of the real training
sentences plus any written ones. The other fifth sets the threshold: the
model may add a flag to the rules only above the lowest score at which at
least 80 in 100 of its additions are right there.

Rule for switching anything on, written before any result: on the held-out
third, F1 at least 0.02 above the rules alone, and at least 70 in 100 of the
added flags right.

What happened (8 October 2026; research/more_data_training.json and
research/RESULTS_TABLES.md, Table 11j, step 10). Round 9 (989 sentences) and
two written sets were tried. The model on its own improves with each, most
for not as advertised. As an add-on to the rules nothing reached the rule:
the best was poor quality with everything, 0.729 to 0.746, 5 additions right
and 2 wrong. Nothing was switched on, and rounds 10 and 11 were not labelled.

    python research/train_with_more_data.py
"""

from __future__ import annotations

import hashlib
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

from finetune_label_transformer import additions_threshold, best_f1_threshold, validation  # noqa: E402
from five_label_data import keyword_view, load_five_label_records  # noqa: E402
from improve_label_rules import held_out  # noqa: E402
from synthetic_data import FILES as WRITTEN_FILES, load_synthetic_records  # noqa: E402
from train_hazard_classifier import SEED, scores  # noqa: E402

REPORT = ROOT / "research" / "more_data_training.json"
TARGETS = ("quality", "counterfeit", "not_as_described")
MARGIN, ADDITIONS_RIGHT = 0.02, 0.70


def main() -> int:
    from defaketive_sentiment.hazard_learned import fit_category_model

    real = load_five_label_records(include_live_check=True)
    written = load_synthetic_records(tuple(name for name, path in WRITTEN_FILES.items() if path.is_file()))
    records = real + written
    view = keyword_view(records)
    yardstick = np.array([held_out(record) for record in records])
    is_real = np.array([i < len(real) for i in range(len(records))])
    rounds = np.array([str(record["round"]) for record in records])
    new_rounds = sorted({r for r in rounds[is_real] if r.isdigit() and int(r) >= 9}, key=int)
    steps: list[tuple[str, np.ndarray]] = [("rounds 1-8", is_real & ~np.isin(rounds, new_rounds))]
    for index in range(len(new_rounds)):
        steps.append((f"+ rounds 9-{new_rounds[index]}" if index else "+ round 9", is_real & ~np.isin(rounds, new_rounds[index + 1:])))
    second = np.array([str(record["row_id"]).startswith("syn2-") for record in records])
    if written:
        everything = steps[-1][0]
        steps.append(("rounds 1-8 + written", steps[0][1] | ~is_real))
        steps.append((everything + " + written 1", ~second))
        if second.any():
            steps.append((everything + " + written 1, 2", np.ones(len(records), dtype=bool)))

    result: dict[str, Any] = {}
    for label in TARGETS:
        labelled = np.array([record["labels"].get(label) is not None for record in records])
        truth = np.array([int(bool(record["labels"].get(label))) for record in records])
        rules = view["flags"][label].astype(bool)
        test = yardstick & labelled
        base = scores(truth[test], rules[test].astype(int))
        rows: dict[str, Any] = {}
        print(f"{label}: held-out reports {int(truth[test].sum())} | rules alone {base['tp']}/{base['fp']} F1 {base['f1']:.3f}", flush=True)
        for name, pool in steps:
            usable = pool & labelled & ~yardstick
            holdback = np.array([usable[i] and is_real[i] and validation(records[i]) for i in range(len(records))])
            train = np.flatnonzero(usable & ~holdback)
            model = fit_category_model(
                "TK", list(view["text"][train]), view["kw"][train], None, truth[train], category=label, seed=SEED
            )
            score = model.probability(list(view["text"]), view["kw"], None)
            threshold = additions_threshold(truth[holdback], score[holdback], rules[holdback])
            combined = rules[test] | (score[test] >= threshold) if threshold is not None else rules[test]
            extra = combined & ~rules[test]
            stats = scores(truth[test], combined.astype(int))
            right, wrong = int((extra & (truth[test] == 1)).sum()), int((extra & (truth[test] == 0)).sum())
            alone = best_f1_threshold(truth[holdback], score[holdback])
            rows[name] = {
                "training_sentences": int(len(train)), "training_reports": int(truth[train].sum()),
                "validation_reports": int(truth[holdback].sum()), "threshold": threshold,
                "rules_or_model": stats, "added_right": right, "added_wrong": wrong,
                "model_alone": scores(truth[test], (score[test] >= alone).astype(int)),
                "passes": bool(stats["f1"] >= base["f1"] + MARGIN and (right + wrong) and right / (right + wrong) >= ADDITIONS_RIGHT),
            }
            print(f"   {name:30} trained on {len(train):5} ({int(truth[train].sum()):4} reports) | rules OR model {stats['tp']}/{stats['fp']}"
                  f" F1 {stats['f1']:.3f} (added {right} right, {wrong} wrong) | model alone F1 {rows[name]['model_alone']['f1']:.3f}"
                  f" | passes {rows[name]['passes']}", flush=True)
        result[label] = {"held_out_reports": int(truth[test].sum()), "rules_alone": base, "steps": rows}
    REPORT.write_bytes((json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Agreement with AI labels on the held-out third of rounds 1-8. Later rounds and written sentences are training only.",
        "held_out_sentences": int(yardstick.sum()),
        # The same value before and after new data shows the yardstick did not move.
        "held_out_sha1": hashlib.sha1(
            "\n".join(sorted(str(records[i]["row_id"]) for i in np.flatnonzero(yardstick))).encode("utf-8")
        ).hexdigest(),
        "written_sentences": len(written), "margin": MARGIN, "additions_right_at_least": ADDITIONS_RIGHT,
        "labels": result,
    }, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
