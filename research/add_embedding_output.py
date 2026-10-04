"""Expose the sentiment model's last hidden layer as a second ONNX output.

The hazard checker's learned candidates (research/PREREGISTRATION_hazard_v4.md)
use a sentence embedding: the mean of the XLM-R encoder's last hidden layer.
Reusing the shipped sentiment model for it costs no extra memory, and computing
it with the served file means research and app use the same numbers.

The output is added without touching any node, so the sentiment logits are
computed exactly as before. This script checks that on both test sets before
it replaces the shipped file, then updates the manifest (new SHA-256, so the
model version changes and saved analyses are redone once).

    python research/add_embedding_output.py [--model-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

DEFAULT_MODEL_DIR = SENTIMENT_ROOT / "models" / "sentiment_onnx" / "xlmr_firecs_v1"
OUTPUT_NAME = "hidden_states"


def final_hidden_state(graph) -> str:
    """The tensor the classifier head reads: the last encoder LayerNorm's output."""
    gathers = [node for node in graph.node if node.op_type == "Gather" and "classifier" in node.name]
    if len(gathers) != 1:
        raise SystemExit(f"Expected one classifier Gather node, found {len(gathers)}.")
    return gathers[0].input[0]


def add_hidden_state_output(src: Path, dst: Path) -> str:
    """Write a copy of `src` with an extra float output `hidden_states`."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(src))
    graph = model.graph
    if any(output.name == OUTPUT_NAME for output in graph.output):
        raise SystemExit(f"{src} already has a {OUTPUT_NAME!r} output.")
    tensor = final_hidden_state(graph)
    graph.node.append(helper.make_node("Identity", [tensor], [OUTPUT_NAME], name="hidden_states_out"))
    graph.output.append(
        helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, ["batch", "sequence", 768])
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def compare(old: Path, new: Path, tokenizer_path: Path, texts: list[str]) -> dict:
    """Logits of every text, one at a time, from both files."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=128)
    before, after = session(old), session(new)
    max_diff, label_changes, width = 0.0, 0, None
    for text in texts:
        encoding = tokenizer.encode(text.strip())
        feed = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        a = before.run(["logits"], feed)[0][0]
        b, hidden = after.run(["logits", OUTPUT_NAME], feed)
        b = b[0]
        width = hidden.shape[-1]
        max_diff = max(max_diff, float(np.abs(a - b).max()))
        label_changes += int(a.argmax() != b.argmax())
    return {"texts": len(texts), "max_logit_difference": max_diff, "label_changes": label_changes, "hidden_width": width}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--backup-dir", type=Path, required=True, help="Where to keep the original model.onnx.")
    args = parser.parse_args()

    from evaluate_firecs import load_firecs
    from evaluate_marketplace_shift import load as load_marketplace
    from train_firecs import DEFAULT_TEST

    model_dir = args.model_dir.resolve()
    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    original = model_dir / "model.onnx"
    if sha256(original) != manifest["files"]["model.onnx"]:
        raise SystemExit("model.onnx does not match its manifest; refusing to modify it.")

    with tempfile.TemporaryDirectory() as folder:
        candidate = Path(folder) / "model.onnx"
        tensor = add_hidden_state_output(original, candidate)
        texts = [r["review"] for r in load_firecs(Path(DEFAULT_TEST))]
        texts += [r["text"] for r in load_marketplace(ROOT / "complete_manual_marketplace_reviews (1).csv")]
        check = compare(original, candidate, model_dir / "tokenizer.json", texts)
        print(json.dumps(check))
        if check["label_changes"] or check["max_logit_difference"] > 1e-4:
            raise SystemExit("Sentiment output changed; the shipped model was NOT replaced.")

        args.backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, args.backup_dir / "model_logits_only.onnx")
        shutil.copyfile(candidate, original)

    new_hash = sha256(original)
    previous = manifest["files"]["model.onnx"]
    manifest["files"]["model.onnx"] = new_hash
    manifest["model_version"] = manifest["model_version"].rsplit("-", 1)[0] + "-" + new_hash[:8]
    manifest["outputs"] = ["logits", OUTPUT_NAME]
    manifest["embedding"] = {
        "output": OUTPUT_NAME,
        "source_tensor": tensor,
        "pooling": "mean over all tokens of one unpadded text",
        "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "previous_model_sha256": previous,
        "sentiment_unchanged_check": check,
    }
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"replaced {original.name}; model_version {manifest['model_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
