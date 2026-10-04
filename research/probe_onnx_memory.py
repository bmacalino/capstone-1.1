"""Measure what an ONNX sentiment model costs in memory, and nothing else.

Imports only what serving needs (numpy, tokenizers, onnxruntime) so the figure
is not inflated by the research tooling. Prints one JSON line.

    python research/probe_onnx_memory.py MODEL_DIR [--no-arena]

MODEL_DIR holds model.onnx and its (pruned) tokenizer.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import psutil


def main() -> int:
    model_dir = Path(sys.argv[1])
    no_arena = "--no-arena" in sys.argv
    process = psutil.Process()
    baseline = process.memory_info().rss

    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if no_arena:
        # Trade a little speed for memory: no pre-allocated arena, no reuse plan.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
    model_path = model_dir / "model.onnx"
    if not model_path.exists():
        model_path = next(model_dir.glob("model_int8.onnx"))
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=128)
    loaded = process.memory_info().rss

    # The app scores one review at a time; a review long enough to reach the
    # 128-token limit is the worst case for activation memory.
    review = " ".join(["Maganda ang quality pero medyo matagal dumating ang parcel, sulit pa rin."] * 12)
    for _ in range(3):
        encoding = tokenizer.encode(review)
        session.run(
            ["logits"],
            {
                "input_ids": np.array([encoding.ids], dtype=np.int64),
                "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            },
        )
    info = process.memory_info()
    print(json.dumps({
        "model": model_path.name,
        "no_arena": no_arena,
        "python_baseline_mb": round(baseline / 1e6, 1),
        "after_load_mb": round(loaded / 1e6, 1),
        "steady_mb": round(info.rss / 1e6, 1),
        "peak_mb": round((getattr(info, "peak_wset", None) or info.rss) / 1e6, 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
