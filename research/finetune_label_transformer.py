"""Experiment: can a fine-tuned transformer find the reports the rules miss?

The rules for poor quality, possible fake and not as advertised are precise
and miss many reports, and neither a TF-IDF model nor new patterns closed
the gap (research/RESULTS_TABLES.md, Table 11j). This fine-tunes the
multilingual encoder the sentiment model is built on (XLM-RoBERTa, the local
checkpoint) as one model with five yes/no outputs.

Same split as research/improve_label_rules.py: one review in three is held
out and only scored. The rest ("dev") is split again by review: four fifths
train the model, together with the synthetic sentences, and one fifth sets
the thresholds. For each label two things are scored on the held-out third:

  model alone      threshold with the best F1 on the validation fifth
  rules OR model   the model may add a flag only above the lowest threshold
                   at which, on the validation fifth, at least 80 in 100 of
                   its additions are right

An experiment only: nothing is saved for the app. Serving it would need a
second transformer run on every sentence.

    python research/finetune_label_transformer.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from five_label_data import LABELS, keyword_view, load_five_label_records  # noqa: E402
from improve_label_rules import held_out  # noqa: E402
from synthetic_data import load_synthetic_records  # noqa: E402
from train_hazard_classifier import scores  # noqa: E402

CHECKPOINT = ROOT / "sentiment-analysis" / "models" / "transformers" / "cardiffnlp__twitter-xlm-roberta-base-sentiment"
# "--tag round9" writes label_transformer_experiment_round9.json, so a rerun
# with more labelled rounds does not replace the first report.
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
REPORT = ROOT / "research" / (f"label_transformer_experiment_{TAG}.json" if TAG else "label_transformer_experiment.json")
SEED, EPOCHS, BATCH, MAX_LENGTH, RATE = 42, 4, 16, 64, 2e-5
# "--seed 7" repeats a run with another seed, to see whether a result is luck;
# "--written v1,v2" trains with both written sets (the first alone by default).
if "--seed" in sys.argv:
    SEED = int(sys.argv[sys.argv.index("--seed") + 1])
WRITTEN = tuple(sys.argv[sys.argv.index("--written") + 1].split(",")) if "--written" in sys.argv else ("v1",)
ADDITIONS_RIGHT = 0.80


def validation(record: dict[str, Any]) -> bool:
    return int(hashlib.sha1(("v" + str(record["review_id"])).encode("utf-8")).hexdigest(), 16) % 5 == 0


def best_f1_threshold(truth: np.ndarray, score: np.ndarray) -> float:
    best = (0.5, -1.0)
    for threshold in np.unique(np.round(score, 3)):
        f1 = scores(truth, (score >= threshold).astype(int))["f1"]
        if f1 > best[1]:
            best = (float(threshold), f1)
    return best[0]


def additions_threshold(truth: np.ndarray, score: np.ndarray, rules: np.ndarray) -> float | None:
    """The lowest threshold at which at least 80 in 100 of the flags the model
    adds to the rules are right on the validation part; None if there is none."""
    chosen = None
    for threshold in sorted(np.unique(np.round(score, 3)), reverse=True):
        added = (score >= threshold) & ~rules
        if added.sum() and (truth[added] == 1).mean() >= ADDITIONS_RIGHT:
            chosen = float(threshold)
        elif added.sum() >= 5 and (truth[added] == 1).mean() < ADDITIONS_RIGHT - 0.15:
            break
    return chosen


def main() -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = load_five_label_records(include_live_check=True)
    synthetic = load_synthetic_records(WRITTEN)
    view = keyword_view(real)
    held = np.array([held_out(record) for record in real])
    valid = np.array([not held[i] and validation(record) for i, record in enumerate(real)])
    train_rows = [record for i, record in enumerate(real) if not held[i] and not valid[i]] + synthetic

    def targets(records):
        y = np.array([[float(bool(r["labels"].get(label))) for label in LABELS] for r in records], dtype=np.float32)
        mask = np.array([[r["labels"].get(label) is not None for label in LABELS] for r in records], dtype=np.float32)
        return y, mask

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=len(LABELS), problem_type="multi_label_classification", ignore_mismatched_sizes=True
    ).to(device)
    y_train, mask_train = targets(train_rows)
    positives = (y_train * mask_train).sum(axis=0)
    negatives = mask_train.sum(axis=0) - positives
    weight = torch.tensor(np.minimum(10.0, negatives / np.maximum(positives, 1.0)), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=RATE, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(train_rows) / BATCH))
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=RATE, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    texts = [record["text"] for record in train_rows]
    order = np.random.default_rng(SEED)
    started = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for batch in np.array_split(order.permutation(len(texts)), int(np.ceil(len(texts) / BATCH))):
            encoded = tokenizer([texts[i] for i in batch], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(**encoded).logits.float()
            mask = torch.tensor(mask_train[batch], device=device)
            loss = (loss_fn(logits, torch.tensor(y_train[batch], device=device)) * mask).sum() / mask.sum().clamp(min=1)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += float(loss)
        print(f"epoch {epoch + 1}: mean loss {total / int(np.ceil(len(texts) / BATCH)):.4f} ({time.time() - started:.0f}s)", flush=True)

    def predict(records) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(records), 64):
                chunk = [record["text"] for record in records[start:start + 64]]
                encoded = tokenizer(chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(device)
                out.append(torch.sigmoid(model(**encoded).logits.float()).cpu().numpy())
        return np.vstack(out)

    scores_all = predict(real)
    result: dict[str, Any] = {}
    for column, label in enumerate(LABELS):
        labelled = np.array([record["labels"].get(label) is not None for record in real])
        truth = np.array([int(bool(record["labels"].get(label))) for record in real])
        rules = view["flags"][label].astype(bool)
        v, h = valid & labelled, held & labelled
        alone = best_f1_threshold(truth[v], scores_all[v, column])
        added = additions_threshold(truth[v], scores_all[v, column], rules[v])
        combined = rules[h] | (scores_all[h, column] >= added) if added is not None else rules[h]
        extra = combined & ~rules[h]
        result[label] = {
            "held_out_sentences": int(h.sum()), "held_out_reports": int(truth[h].sum()),
            "rules": scores(truth[h], rules[h].astype(int)),
            "model_alone": {**scores(truth[h], (scores_all[h, column] >= alone).astype(int)), "threshold": alone},
            "rules_or_model": {
                **scores(truth[h], combined.astype(int)), "threshold": added,
                "added_right": int((extra & (truth[h] == 1)).sum()), "added_wrong": int((extra & (truth[h] == 0)).sum()),
            },
        }
        r = result[label]
        print(f"{label:17} held-out reports {r['held_out_reports']:4} | rules {r['rules']['tp']}/{r['rules']['fp']} F1 {r['rules']['f1']:.3f}"
              f" | model alone {r['model_alone']['tp']}/{r['model_alone']['fp']} F1 {r['model_alone']['f1']:.3f}"
              f" | rules OR model {r['rules_or_model']['tp']}/{r['rules_or_model']['fp']} F1 {r['rules_or_model']['f1']:.3f}"
              f" (added {r['rules_or_model']['added_right']} right, {r['rules_or_model']['added_wrong']} wrong)", flush=True)
    REPORT.write_bytes((json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "An experiment. Agreement with AI labels on a held-out third; nothing was saved for the app.",
        "model": CHECKPOINT.name, "seed": SEED, "written_sets": list(WRITTEN), "epochs": EPOCHS, "batch": BATCH, "max_length": MAX_LENGTH, "learning_rate": RATE,
        "training_sentences": len(train_rows), "of_which_synthetic": len(synthetic),
        "validation_sentences": int(valid.sum()), "seconds": round(time.time() - started),
        "note": "Safety and defect rows compare with the keywords alone: the served models were trained on these sentences.",
        "labels": result,
    }, indent=2) + "\n").encode("utf-8"))
    print(f"wrote {REPORT.name}")
    if TAG:
        # A tagged run keeps its trained model (outputs/ is not versioned), so a
        # run that meets the rule can be slimmed and served as the model that
        # was scored, not as a retrained look-alike.
        folder = ROOT / "outputs" / "label_transformer_runs" / TAG
        folder.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(folder / "checkpoint")
        tokenizer.save_pretrained(folder / "checkpoint")
        np.save(folder / "scores.npy", scores_all)
        (folder / "rows.json").write_bytes(json.dumps([str(record["row_id"]) for record in real]).encode("utf-8"))
        (folder / "report.json").write_bytes(REPORT.read_bytes())
        print(f"saved the model in {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
