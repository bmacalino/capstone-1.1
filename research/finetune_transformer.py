"""Fine-tune a transformer encoder on the cleaned FiReCS training split.

The existing comparison ran pretrained transformers off the shelf on a task
they were never trained for, which is not a fair test of the architecture.
This script trains them properly on the same rows the incumbent learned from.

Discipline matches train_firecs.py: the cleaned training split is reused, a
stratified slice of it is held out for choosing the epoch, and the official
test labels are touched only once the epoch has been chosen. The resulting
artifact is a comparison arm and is written with qualified_for_integration
set to false; nothing here is promoted into the application.

Memory notes for small GPUs. XLM-R carries a 250k-token vocabulary, so its
embedding matrix alone is about 69% of its parameters. Freezing it (the
default) cuts optimizer state by roughly the same proportion and is what makes
this fit in 4GB. Reviews are short, so the default sequence length is 128.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SENTIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SENTIMENT_ROOT))

from evaluate_firecs import load_firecs
from research_utils import evaluate_labels
from train_firecs import (
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    audit_key,
    audit_training_records,
    normalized_text,
)

LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DEFAULT_MODEL = "xlm-roberta-base"
DEFAULT_OUTPUT_ROOT = SENTIMENT_ROOT / "models" / "transformers"
SEED = 42


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Install it with:\n"
            "  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
        ) from exc
    return torch


def resolve_device(requested: str):
    torch = require_torch()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but this PyTorch build cannot see a GPU.\n"
                f"Installed build: {torch.__version__}\n"
                "A build tagged '+cpu' has no CUDA support; reinstall the CUDA build."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def raw_firecs_rows(
    train_records: list[dict[str, str]], test_records: list[dict[str, str]]
) -> tuple[list[str], list[str]]:
    """The rows audit_training_records keeps, as the raw review text.

    Same duplicate and cross-split rules, same order, so the row set is
    identical; only the text differs. The first version of this script trained
    on the normalized, marker-appended text and was then scored on raw text.
    """
    test_keys = {audit_key(record["review"]) for record in test_records}
    seen: set[str] = set()
    texts: list[str] = []
    labels: list[str] = []
    for record in train_records:
        key = audit_key(record["review"])
        if key in seen:
            continue
        seen.add(key)
        if key in test_keys:
            continue
        texts.append(record["review"])
        labels.append(record["label"])
    return texts, labels


def load_extra_train(path: Path) -> list[dict[str, str]]:
    """Weak-label rows from research/build_sentiment_training_set.py."""
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        if row["label"] not in LABEL_TO_ID or not row.get("product_id"):
            raise SystemExit(f"{path.name}: every row needs a known label and a product_id")
    return rows


def product_holdout(
    rows: list[dict[str, str]], fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Hold out whole products, so validation never sees a product it trained on."""
    products = sorted({row["product_id"] for row in rows})
    random.Random(seed).shuffle(products)
    held = set(products[: max(1, round(len(products) * fraction))])
    fit = [row for row in rows if row["product_id"] not in held]
    validation = [row for row in rows if row["product_id"] in held]
    return fit, validation, sorted(held)


def stratified_validation_split(
    texts: list[str], labels: list[str], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices by label so the validation slice keeps the class balance."""
    from sklearn.model_selection import train_test_split

    indices = list(range(len(texts)))
    fit_indices, validation_indices = train_test_split(
        indices, test_size=fraction, random_state=seed, stratify=labels
    )
    return sorted(fit_indices), sorted(validation_indices)


def encode(
    tokenizer,
    texts: list[str],
    labels: list[str] | None,
    max_length: int,
    weights: list[float] | None = None,
):
    torch = require_torch()
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    tensors = [encoded["input_ids"], encoded["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor([LABEL_TO_ID[label] for label in labels], dtype=torch.long))
    if weights is not None:
        tensors.append(torch.tensor(weights, dtype=torch.float32))
    return torch.utils.data.TensorDataset(*tensors)


def predict(model, loader, device, use_amp: bool) -> list[str]:
    torch = require_torch()
    model.eval()
    predicted: list[str] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predicted.extend(LABELS[index] for index in logits.argmax(dim=-1).tolist())
    return predicted


def freeze_embeddings(model) -> int:
    """Freeze the input embedding matrix and report how many parameters that froze."""
    embeddings = model.get_input_embeddings()
    frozen = 0
    for parameter in embeddings.parameters():
        parameter.requires_grad = False
        frozen += parameter.numel()
    return frozen


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch = require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    use_amp = args.fp16 and device.type == "cuda"

    train_records = load_firecs(args.train)
    test_records = load_firecs(args.test)
    clean_texts, clean_labels, audit = audit_training_records(train_records, test_records)
    if args.text == "raw":
        clean_texts, raw_labels = raw_firecs_rows(train_records, test_records)
        if raw_labels != clean_labels:
            raise SystemExit("Raw and normalized FiReCS row sets differ; the leakage rules drifted.")

    fit_indices, validation_indices = stratified_validation_split(
        clean_texts, clean_labels, args.validation_fraction, args.seed
    )
    fit_texts = [clean_texts[index] for index in fit_indices]
    fit_labels = [clean_labels[index] for index in fit_indices]
    fit_weights = [1.0] * len(fit_texts)
    validation_texts = [clean_texts[index] for index in validation_indices]
    validation_labels = [clean_labels[index] for index in validation_indices]
    validation_sources = ["firecs"] * len(validation_texts)
    composition: dict[str, Any] = {
        "text": args.text,
        "firecs_fit_rows": len(fit_texts),
        "firecs_validation_rows": len(validation_texts),
    }

    if args.extra_train:
        extra = load_extra_train(args.extra_train)
        extra_fit, extra_validation, held_products = product_holdout(
            extra, args.validation_fraction, args.seed
        )
        fit_texts += [row["text"] for row in extra_fit]
        fit_labels += [row["label"] for row in extra_fit]
        fit_weights += [args.extra_weight] * len(extra_fit)
        validation_texts += [row["text"] for row in extra_validation]
        validation_labels += [row["label"] for row in extra_validation]
        validation_sources += ["extra"] * len(extra_validation)
        composition.update(
            {
                "extra_train_file": args.extra_train.name,
                "extra_train_sha256": hashlib.sha256(args.extra_train.read_bytes()).hexdigest(),
                "extra_fit_rows": len(extra_fit),
                "extra_validation_rows": len(extra_validation),
                "extra_validation_products": held_products,
                "extra_label_type": "weak: reviewer star rating (1-2 negative, 4-5 positive)",
                "extra_loss_weight": args.extra_weight,
            }
        )
    composition["fit_label_counts"] = dict(sorted(Counter(fit_labels).items()))
    test_texts = [record["review"] for record in test_records]
    if args.text == "normalized" and args.extra_train:
        # FiReCS rows arrive normalized from audit_training_records; match them.
        extra_start = composition["firecs_fit_rows"]
        fit_texts[extra_start:] = [normalized_text(text) for text in fit_texts[extra_start:]]
        firecs_val = composition["firecs_validation_rows"]
        validation_texts[firecs_val:] = [normalized_text(text) for text in validation_texts[firecs_val:]]
    if args.text == "normalized":
        test_texts = [normalized_text(text) for text in test_texts]
    test_labels = [record["label"] for record in test_records]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=dict(LABEL_TO_ID),
    ).to(device)

    frozen_parameters = freeze_embeddings(model) if args.freeze_embeddings else 0
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    fit_loader = torch.utils.data.DataLoader(
        encode(
            tokenizer,
            fit_texts,
            fit_labels,
            args.max_length,
            # Weights only when they differ from 1, so unweighted runs keep the
            # model's own loss exactly as before.
            fit_weights if args.extra_weight != 1.0 else None,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        encode(tokenizer, validation_texts, validation_labels, args.max_length),
        batch_size=args.eval_batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        encode(tokenizer, test_texts, None, args.max_length),
        batch_size=args.eval_batch_size,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    steps_per_epoch = max(1, len(fit_loader) // args.gradient_accumulation)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    output_dir = args.output or (DEFAULT_OUTPUT_ROOT / args.model.replace("/", "__"))
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    epoch_history: list[dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(fit_loader, start=1):
            input_ids, attention_mask, targets, *rest = (item.to(device) for item in batch)
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=targets
                )
                batch_loss = outputs.loss
                if rest:
                    # Weighted mean cross-entropy: a weak-label row counts
                    # extra_weight times as much as a human-labelled one.
                    weights = rest[0]
                    per_row = torch.nn.functional.cross_entropy(
                        outputs.logits.float(), targets, reduction="none"
                    )
                    batch_loss = (per_row * weights).sum() / weights.sum()
                loss = batch_loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += float(batch_loss.detach())
            if step % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()

        # Epoch selection uses the validation slice only; test labels stay untouched.
        validation_predictions = predict(model, validation_loader, device, use_amp)
        validation_metrics = evaluate_labels(validation_labels, validation_predictions)
        by_source = {
            source: evaluate_labels(
                [label for label, origin in zip(validation_labels, validation_sources) if origin == source],
                [label for label, origin in zip(validation_predictions, validation_sources) if origin == source],
            )["f1"]
            for source in sorted(set(validation_sources))
        }
        epoch_history.append(
            {
                "epoch": epoch,
                "mean_train_loss": round(running_loss / max(1, len(fit_loader)), 4),
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["f1"],
                "validation_macro_f1_by_source": by_source,
            }
        )
        print(
            f"epoch {epoch}: loss={running_loss / max(1, len(fit_loader)):.4f} "
            f"validation macro F1={validation_metrics['f1']:.4f}",
            flush=True,
        )
        if validation_metrics["f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["f1"]
            best_epoch = epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    train_seconds = round(time.perf_counter() - started, 2)

    # Reload the epoch validation chose, then score the held-out test split once.
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_predictions = predict(model, test_loader, device, use_amp)
    test_metrics = evaluate_labels(test_labels, test_predictions)

    report = {
        "model_id": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(output_dir),
        "deployment_status": "comparison_candidate",
        "qualified_for_integration": False,
        "training": {
            "device": device.type,
            "mixed_precision": use_amp,
            "epochs": args.epochs,
            "selected_epoch": best_epoch,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "embeddings_frozen": bool(frozen_parameters),
            "frozen_parameters": frozen_parameters,
            "trainable_parameters": sum(item.numel() for item in trainable),
            "seed": args.seed,
            "train_seconds": train_seconds,
            "fit_rows": len(fit_texts),
            "validation_rows": len(validation_texts),
            "composition": composition,
        },
        "training_audit": audit,
        "epoch_history": epoch_history,
        "test": test_metrics,
    }

    (output_dir / "predictions.json").write_bytes(
        json.dumps({"test_file": args.test.name, "predictions": test_predictions}, indent=2).encode("utf-8")
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument(
        "--freeze-embeddings",
        action="store_true",
        default=True,
        help="Freeze the token embedding matrix. Required to fit XLM-R in 4GB.",
    )
    parser.add_argument(
        "--no-freeze-embeddings", dest="freeze_embeddings", action="store_false"
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--text",
        choices=("raw", "normalized"),
        default="raw",
        help="Train and test on raw review text (what the app serves) or the normalized form.",
    )
    parser.add_argument(
        "--extra-train",
        type=Path,
        help="Weak-label CSV from research/build_sentiment_training_set.py (needs product_id).",
    )
    parser.add_argument(
        "--extra-weight",
        type=float,
        default=1.0,
        help="Loss weight of each --extra-train row relative to a FiReCS row.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = train(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    report_path = args.report or (ROOT / "research" / f"finetuned_{args.model.replace('/', '__')}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes((rendered + "\n").encode("utf-8"))
    print(
        f"\nTest accuracy={report['test']['accuracy']:.4f} "
        f"macro F1={report['test']['f1']:.4f} (epoch {report['training']['selected_epoch']})"
    )
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
