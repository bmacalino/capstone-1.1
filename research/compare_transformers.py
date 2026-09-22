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


def predict(model_id: str, texts: list[str], batch_size: int) -> tuple[list[str], list[float | None]]:
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
        device=-1,
    )
    outputs = classifier(texts, batch_size=batch_size, truncation=True, max_length=256)
    mapping = MODEL_LABELS[model_id]
    labels = []
    confidence = []
    for output in outputs:
        raw_label = str(output["label"])
        if raw_label not in mapping:
            raise ValueError(f"Unsupported label {raw_label!r} from {model_id}")
        labels.append(mapping[raw_label])
        confidence.append(round(float(output.get("score")), 6) if output.get("score") is not None else None)
    return labels, confidence


def compare(test_path: Path, model_ids: list[str], batch_size: int) -> dict[str, Any]:
    records = load_firecs(test_path)
    texts = [record["review"] for record in records]
    actual = [record["label"] for record in records]
    results: dict[str, Any] = {}
    for model_id in model_ids:
        predictions, confidence = predict(model_id, texts, batch_size)
        metrics = evaluate_labels(actual, predictions)
        metrics.update(
            {
                "model_id": model_id,
                "mean_confidence": round(sum(confidence) / len(confidence), 6),
                "max_length": 256,
                "test_file": test_path.name,
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
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_LABELS), default=sorted(MODEL_LABELS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.test.resolve(), args.models, args.batch_size)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()