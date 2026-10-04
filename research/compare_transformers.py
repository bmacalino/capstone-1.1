"""Compare pretrained Hugging Face sentiment models on the locked FiReCS test set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


DEFAULT_TEST = SENTIMENT_ROOT / "data" / "firecs" / "FiReCS_test_set.csv"
MODEL_LABELS = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    },
    "finiteautomata/bertweet-base-sentiment-analysis": {
        "NEG": "negative",
        "NEU": "neutral",
        "POS": "positive",
    },
    "ldlazaro/sentiment_roberta_taglish": {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    },
}


PASSTHROUGH_LABELS = {
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


def resolve_device(requested: str) -> int:
    """Map a device choice to the integer index the transformers pipeline wants."""
    if requested == "cpu":
        return -1
    try:
        import torch
    except ImportError:
        return -1
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA was requested but this PyTorch build cannot see a GPU "
                f"(installed build: {torch.__version__})."
            )
        return 0
    return 0 if torch.cuda.is_available() else -1


def label_mapping(model_id: str) -> dict[str, str]:
    """Return the label mapping for a registered model or a local checkpoint.

    Checkpoints fine-tuned by research/finetune_transformer.py carry their own
    id2label, so their raw labels are already the project's label names.
    """
    if model_id in MODEL_LABELS:
        return {key.casefold(): value for key, value in MODEL_LABELS[model_id].items()}
    if Path(model_id).exists():
        return PASSTHROUGH_LABELS
    raise SystemExit(
        f"Unknown model {model_id!r}. Register it in MODEL_LABELS or pass a local checkpoint path."
    )


def predict(
    model_id: str, texts: list[str], batch_size: int, device: int
) -> tuple[list[str], list[float | None]]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install transformer comparison dependencies with: "
            "python -m pip install transformers torch"
        ) from exc

    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        device=device,
    )
    outputs = classifier(texts, batch_size=batch_size, truncation=True, max_length=256)
    mapping = label_mapping(model_id)
    labels = []
    confidence = []
    for output in outputs:
        raw_label = str(output["label"]).casefold()
        if raw_label not in mapping:
            raise ValueError(f"Unsupported label {raw_label!r} from {model_id}")
        labels.append(mapping[raw_label])
        confidence.append(round(float(output.get("score")), 6) if output.get("score") is not None else None)
    return labels, confidence


def compare(
    test_path: Path, model_ids: list[str], batch_size: int, device: int
) -> dict[str, Any]:
    records = load_firecs(test_path)
    texts = [record["review"] for record in records]
    actual = [record["label"] for record in records]
    results: dict[str, Any] = {}
    for model_id in model_ids:
        predictions, confidence = predict(model_id, texts, batch_size, device)
        metrics = evaluate_labels(actual, predictions)
        metrics.update(
            {
                "model_id": model_id,
                "mean_confidence": round(sum(confidence) / len(confidence), 6),
                "max_length": 256,
                "test_file": test_path.name,
                "device": "cuda" if device >= 0 else "cpu",
                "fine_tuned": model_id not in MODEL_LABELS,
            }
        )
        results[model_id] = metrics
        print(
            f"{model_id}: accuracy={metrics['accuracy']:.4f}, "
            f"macro F1={metrics['f1']:.4f}",
            flush=True,
        )
    return {"dataset": "FiReCS", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--models",
        nargs="+",
        default=sorted(MODEL_LABELS),
        help="Registered model ids, or paths to locally fine-tuned checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(
        args.test.resolve(), args.models, args.batch_size, resolve_device(args.device)
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()