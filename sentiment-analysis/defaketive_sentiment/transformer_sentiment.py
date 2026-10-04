"""Serve the fine-tuned XLM-RoBERTa sentiment model on CPU through ONNX Runtime.

WHY THIS REPLACES THE TF-IDF CLASSIFIER
The TF-IDF model scores 0.814 macro F1 on FiReCS but 0.563 on the project's
human-labelled Lazada/Temu reviews, and reads real complaints as praise. The
fine-tuned transformer holds up far better on marketplace text (see
research/RESULTS_TABLES.md). It is served slimmed - vocabulary pruned, exported
to ONNX, weights quantized - so it fits a free CPU host without PyTorch.

WHY IT REFUSES TO LOAD RATHER THAN FALLING BACK
The model folder is built by research/slim_sentiment_model.py and is not in
git. A manifest beside it records the SHA-256 of every file and whether the
model passed the project's gate. If the folder is missing, a file was changed,
or the model did not qualify, loading raises TransformerModelError with the
command that rebuilds it. Scoring reviews with a different model than the one
the report names is exactly what registry.py is designed to prevent.

INPUT
A model receives text in the form it was trained on, recorded in its manifest
as "text": either the review as written ("raw", stripped of surrounding
whitespace), or "normalized" - the shared normalizer's output with the
feature-marker words appended, exactly train_firecs.normalized_text().
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .trained import feature_text

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models" / "sentiment_onnx"
# Chosen by the rule in research/PREREGISTRATION_xlmr_firecs_lazada_norm.md:
# the first fine-tuned XLM-R (FiReCS, normalized training text), served on raw
# text as it was measured (0.719 marketplace macro F1).
DEFAULT_MODEL_NAME = "xlmr_firecs_v1"
SOURCE_CHECKPOINT = "cardiffnlp__twitter-xlm-roberta-base-sentiment"
DEFAULT_MODEL_DIR = MODELS_ROOT / DEFAULT_MODEL_NAME
MANIFEST_NAME = "manifest.json"
EMBEDDING_OUTPUT = "hidden_states"
REQUIRED_FILES = ("model.onnx", "tokenizer.json")

REBUILD_HINT = (
    "Build it from the fine-tuned checkpoint with:\n"
    "  python research/slim_sentiment_model.py "
    f"--checkpoint sentiment-analysis/models/transformers/{SOURCE_CHECKPOINT} "
    f"--name {DEFAULT_MODEL_NAME} --text raw\n"
    "or copy the folder sentiment-analysis/models/sentiment_onnx/"
    f"{DEFAULT_MODEL_NAME}/ from the machine that built it."
)


class TransformerModelError(RuntimeError):
    """Raised when the served sentiment model is missing, altered or unqualified."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(model_dir: Path) -> dict[str, Any]:
    """Return the manifest after checking every file against its recorded hash."""
    manifest_path = model_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise TransformerModelError(
            f"The sentiment model is not installed: {manifest_path} is missing.\n{REBUILD_HINT}"
        )
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise TransformerModelError(f"Could not read {manifest_path}: {exc}\n{REBUILD_HINT}") from exc

    recorded = manifest.get("files") or {}
    for name in REQUIRED_FILES:
        path = model_dir / name
        if not path.is_file():
            raise TransformerModelError(f"The sentiment model is incomplete: {path} is missing.\n{REBUILD_HINT}")
        if recorded.get(name) != file_sha256(path):
            raise TransformerModelError(
                f"{path} does not match the SHA-256 in its manifest, so it is not the "
                f"model that was measured.\n{REBUILD_HINT}"
            )
    if not manifest.get("qualified_for_integration"):
        raise TransformerModelError(
            f"The sentiment model in {model_dir} did not pass the integration gate "
            f"({manifest.get('gate', 'no gate recorded')})."
        )
    if not manifest.get("labels") or not manifest.get("model_version"):
        raise TransformerModelError(f"{manifest_path} has no labels or model_version.")
    if manifest.get("text", "raw") not in ("raw", "normalized"):
        raise TransformerModelError(f"{manifest_path} names an unknown text form {manifest.get('text')!r}.")
    return manifest


class TransformerSentimentClassifier:
    """Same interface as trained.TrainedSentimentClassifier, on ONNX Runtime.

    Each review is scored on its own. The int8 model quantizes activations with
    a scale computed over the whole input, so in a batch a review's label could
    depend on which other reviews - and how much padding - share the batch.
    Scoring one at a time makes a review's label a function of its text alone.
    """

    label_source = "xlmr_finetuned_classifier"

    def __init__(self, model_dir: str | Path | None = None, *, threads: int = 1):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.manifest = read_manifest(self.model_dir)
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise TransformerModelError(
                "The sentiment model needs onnxruntime and tokenizers. Run: "
                ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # A free host has ~512 MB. The arena and memory-pattern planner trade
        # memory for a little speed; measured, disabling them costs little.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=int(self.manifest.get("max_length") or 128))
        # DefaketiveSentimentModel reads this to decide which text to pass in.
        self.wants_raw_text = self.manifest.get("text", "raw") == "raw"
        self.labels = [str(label) for label in self.manifest["labels"]]
        self.model_version = str(self.manifest["model_version"])
        self.metadata = self.manifest
        # research/add_embedding_output.py adds the encoder's last hidden layer
        # as a second output; the hazard checker's learned models read it.
        self.has_embedding = EMBEDDING_OUTPUT in {item.name for item in self.session.get_outputs()}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = np.exp(logits - logits.max())
        return shifted / shifted.sum()

    def predict(self, text: str) -> dict[str, Any]:
        """Score one review. A normalized-text model expects the normalizer's
        output (what DefaketiveSentimentModel passes) and adds the markers."""
        text = str(text or "").strip()
        encoding = self.tokenizer.encode(text if self.wants_raw_text else feature_text(text))
        logits = self.session.run(
            ["logits"],
            {
                "input_ids": np.array([encoding.ids], dtype=np.int64),
                "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            },
        )[0][0]
        values = self._softmax(logits.astype(np.float64))
        index = int(values.argmax())
        return {
            "label": self.labels[index],
            "confidence": round(float(values[index]), 6),
            "probabilities": {
                label: round(float(value), 6) for label, value in zip(self.labels, values)
            },
            # The interface shows no per-word explanation for sentiment, and a
            # transformer has no faithful linear one to give.
            "evidence": [],
        }

    def predict_batch(self, texts: Iterable[str]) -> list[dict[str, Any]]:
        return [self.predict(text) for text in texts]

    def embed(self, text: str) -> np.ndarray:
        """Mean of the last hidden layer over one unpadded text (768 floats).

        Given the same text form as `predict`, so research and app embed alike.
        """
        if not self.has_embedding:
            raise TransformerModelError(
                f"{self.model_dir / 'model.onnx'} has no {EMBEDDING_OUTPUT!r} output. "
                "Run: python research/add_embedding_output.py --backup-dir <folder>"
            )
        text = str(text or "").strip()
        encoding = self.tokenizer.encode(text if self.wants_raw_text else feature_text(text))
        hidden = self.session.run(
            [EMBEDDING_OUTPUT],
            {
                "input_ids": np.array([encoding.ids], dtype=np.int64),
                "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            },
        )[0][0]
        return hidden.astype(np.float32).mean(axis=0)
