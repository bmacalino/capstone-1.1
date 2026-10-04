"""Slim a fine-tuned sentiment checkpoint for a free CPU host, measuring each stage.

The fine-tuned XLM-RoBERTa is 1.1 GB and needs PyTorch. Most of that is a
250,002-piece vocabulary for 100 languages, of which Filipino/English product
reviews use under 5%. Four stages, each scored on both test sets:

  full       the checkpoint as trained, PyTorch fp32
  pruned     vocabulary trimmed to pieces seen in NON-test text, PyTorch fp32
  onnx       the pruned model exported to ONNX, fp32
  onnx_int8  the ONNX model with dynamic int8 weight quantization

Vocabulary is built from FiReCS training text and the weak-label training set
only; neither contains test text, so pruning leaks nothing.

The TOKENIZER is pruned to the same pieces, in the same order, so it emits the
pruned model's ids directly. The original plan kept the tokenizer and remapped
ids; measured, the 250,002-piece tokenizer alone took 262 MB of RAM, more than
the int8 model. Pruning a Unigram vocabulary cannot change the best
segmentation of any text whose best segmentation uses only kept pieces -- a
maximum over a subset that still contains the argmax is unchanged -- so
training-like text segments identically (checked on the whole vocabulary
corpus, and the script refuses to continue otherwise), and unseen words fall
back to smaller kept pieces instead of <unk>.

Acceptance rule, fixed before any stage was measured: ship the smallest ONNX
stage whose marketplace macro F1 is within 0.02 of the full model and whose
McNemar test against the full model is not significant (p >= 0.05). Latency and
memory are measured single-threaded, the conservative case for a free host.

Every stage scores each review on its own, exactly as the app serves it. In a
batch, the int8 model computes its activation scale over the whole padded
input, so a review's label depends on its batch-mates: measured on the
cardiffnlp checkpoint, 18 of 470 marketplace labels changed between batched
and one-at-a-time scoring (fp32 ONNX: none). Batching was also slower, because
short reviews were padded to the longest in the batch.

Integration gate, applied to the chosen stage before the app will load it
(the transformer_sentiment manifest records the outcome): FiReCS test macro F1
no lower than the incumbent TF-IDF model's (the same "no worse than the
incumbent" rule train_firecs.py applies), and marketplace macro F1 at least
0.02 above the incumbent's, since fixing the marketplace is the reason for the
replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENTIMENT_ROOT = ROOT / "sentiment-analysis"
for candidate in (ROOT, SENTIMENT_ROOT, ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_firecs import load_firecs  # noqa: E402
from evaluate_marketplace_shift import load as load_marketplace  # noqa: E402
from finetune_transformer import raw_firecs_rows  # noqa: E402
from research_utils import bootstrap_macro_f1_ci, evaluate_labels, mcnemar_test  # noqa: E402
from train_firecs import DEFAULT_TEST, DEFAULT_TRAIN, audit_training_records, normalized_text  # noqa: E402

DEFAULT_WEAK = SENTIMENT_ROOT / "data" / "lazada_star_labels" / "lazada_star_labels_v1.csv"
DEFAULT_SHIP_ROOT = SENTIMENT_ROOT / "models" / "sentiment_onnx"
MAX_LENGTH = 128
# Long enough to hit the 128-token limit: the memory probe's worst case.
LONG_REVIEW = " ".join(["Maganda ang quality pero medyo matagal dumating ang parcel, sulit pa rin."] * 12)
STAGES = ("full", "pruned", "onnx", "onnx_int8")
TOLERANCE = 0.02
MARKETPLACE_MARGIN = 0.02


def onnx_file(stage: str) -> str:
    return "model_int8.onnx" if stage == "onnx_int8" else "model_fp32.onnx"


def vocabulary_corpus(weak_csv: Path, text_mode: str = "raw") -> list[str]:
    """Training text only, in the form the model was trained on: FiReCS
    training rows plus the weak-label set, raw or normalized."""
    train, test = load_firecs(Path(DEFAULT_TRAIN)), load_firecs(Path(DEFAULT_TEST))
    with weak_csv.open("r", encoding="utf-8-sig", newline="") as source:
        weak = [row["text"] for row in csv.DictReader(source)]
    if text_mode == "normalized":
        return audit_training_records(train, test)[0] + [normalized_text(item) for item in weak]
    return raw_firecs_rows(train, test)[0] + weak


def kept_token_ids(checkpoint: Path, special_ids: list[int], corpus: list[str]) -> list[int]:
    """Every piece used anywhere in the corpus, from UNtruncated text.

    Built from truncated text, a word straddling the 128-token cut kept only
    its first piece, and the pruned tokenizer then split that word differently.
    The raw `tokenizers` object is used because it was measured to produce ids
    identical to transformers' AutoTokenizer on all 12,360 corpus texts.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    ids = set(special_ids)
    for item in tokenizer.encode_batch(corpus):
        ids.update(item.ids)
    return sorted(ids)


def build_remap(kept: list[int], full_size: int, unk_id: int) -> np.ndarray:
    """Old id -> new id; ids that were not kept point at <unk>."""
    remap = np.full(full_size, -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)
    remap[remap < 0] = remap[unk_id]
    return remap


def prune_tokenizer(tokenizer_json: Path, kept: list[int], remap: np.ndarray) -> None:
    """Rewrite a Unigram tokenizer.json so it contains only the kept pieces."""
    data = json.loads(tokenizer_json.read_bytes().decode("utf-8"))
    model = data["model"]
    if model.get("type") != "Unigram":
        raise SystemExit(f"expected a Unigram tokenizer, found {model.get('type')}")
    model["vocab"] = [model["vocab"][index] for index in kept]
    model["unk_id"] = int(remap[model["unk_id"]])
    for token in data.get("added_tokens") or []:
        token["id"] = int(remap[token["id"]])
    specials = (data.get("post_processor") or {}).get("special_tokens") or {}
    for entry in specials.values():
        entry["ids"] = [int(remap[index]) for index in entry.get("ids", [])]
    tokenizer_json.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def segmentation_check(original_dir: Path, pruned_dir: Path, remap: np.ndarray, texts: list[str]) -> dict[str, Any]:
    """Compare pruned tokenization with the original mapped into pruned ids."""
    from tokenizers import Tokenizer

    original = Tokenizer.from_file(str(original_dir / "tokenizer.json"))
    pruned = Tokenizer.from_file(str(pruned_dir / "tokenizer.json"))
    for tokenizer in (original, pruned):
        tokenizer.enable_truncation(max_length=MAX_LENGTH)
        # The fine-tuned checkpoint's tokenizer.json carries fixed padding to 128
        # from training; compare the pieces themselves, not the padding.
        tokenizer.no_padding()
    original_unk = original.token_to_id("<unk>")
    pruned_unk = pruned.token_to_id("<unk>")
    changed = unk = total = 0
    for before, after in zip(original.encode_batch(texts), pruned.encode_batch(texts)):
        changed += list(remap[np.array(before.ids, dtype=np.int64)]) != after.ids
        unk += after.ids.count(pruned_unk) - before.ids.count(original_unk)
        total += len(after.ids)
    return {
        "texts": len(texts),
        "segmentation_changed_share": round(changed / len(texts), 5) if texts else 0.0,
        "unk_tokens_introduced_share": round(unk / total, 5) if total else 0.0,
    }


def prune(checkpoint: Path, corpus: list[str], out_dir: Path) -> dict[str, Any]:
    """Write a vocabulary-pruned copy of the checkpoint with a pruned tokenizer."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    kept = kept_token_ids(checkpoint, tokenizer.all_special_ids, corpus)
    full_size = model.get_input_embeddings().weight.shape[0]
    remap = build_remap(kept, full_size, tokenizer.unk_token_id)
    # RoBERTa derives position ids from the padding id, so it must not move.
    # Special tokens hold the lowest ids and survive pruning in place.
    for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, name)
        if token_id is not None and remap[token_id] != token_id:
            raise SystemExit(f"{name} {token_id} would move to {remap[token_id]}; refusing to prune")

    old = model.get_input_embeddings()
    new = torch.nn.Embedding(len(kept), old.weight.shape[1], padding_idx=old.padding_idx)
    new.weight.data = old.weight.data[kept].clone()
    model.set_input_embeddings(new)
    model.config.vocab_size = len(kept)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    np.save(out_dir / "id_remap.npy", remap)  # kept only for the segmentation audit
    prune_tokenizer(out_dir / "tokenizer.json", kept, remap)
    check = segmentation_check(checkpoint, out_dir, remap, corpus)
    if check["segmentation_changed_share"] != 0.0:
        raise SystemExit(f"pruned tokenizer changed segmentation of training text: {check}")
    return {"kept_tokens": len(kept), "full_tokens": full_size, "training_corpus_check": check}


def export_onnx(pruned_dir: Path, onnx_path: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(pruned_dir).eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = torch.ones((2, 16), dtype=torch.long)
    torch.onnx.export(
        LogitsOnly(model),
        (sample, torch.ones_like(sample)),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )


def quantize(onnx_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_path), str(int8_path), weight_type=QuantType.QInt8)


# ----------------------------------------------------------------- predictors ---

def fast_tokenizer(tokenizer_dir: Path):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")
    return tokenizer


def torch_predictor(model_dir: Path) -> Callable[[list[str]], list[str]]:
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.set_num_threads(1)
    tokenizer = fast_tokenizer(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    labels = [model.config.id2label[index] for index in range(model.config.num_labels)]

    def predict(texts: list[str]) -> list[str]:
        batch = tokenizer.encode_batch(texts)
        ids = torch.tensor([item.ids for item in batch], dtype=torch.long)
        mask = torch.tensor([item.attention_mask for item in batch], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits
        return [labels[index] for index in logits.argmax(-1).tolist()]

    return predict


def onnx_predictor(onnx_path: Path, tokenizer_dir: Path, labels: list[str]) -> Callable[[list[str]], list[str]]:
    """The exact path the app will serve: pruned tokenizer + onnxruntime."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    tokenizer = fast_tokenizer(tokenizer_dir)

    def predict(texts: list[str]) -> list[str]:
        batch = tokenizer.encode_batch(texts)
        ids = np.array([item.ids for item in batch], dtype=np.int64)
        mask = np.array([item.attention_mask for item in batch], dtype=np.int64)
        logits = session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        return [labels[index] for index in logits.argmax(-1).tolist()]

    return predict


def stage_predictor(stage: str, work: Path):
    labels = json.loads((work / "labels.json").read_bytes().decode("utf-8"))
    if stage == "full":
        return torch_predictor(Path((work / "source.txt").read_bytes().decode("utf-8").strip()))
    if stage == "pruned":
        return torch_predictor(work / "pruned")
    return onnx_predictor(work / onnx_file(stage), work / "pruned", labels)


def score_each(predict, texts: list[str]) -> list[str]:
    """One review per call, as served (see the module docstring)."""
    return [predict([text])[0] for text in texts]


def latency(predict, texts: list[str]) -> dict[str, float]:
    singles = []
    for text in texts[:100]:
        started = time.perf_counter()
        predict([text])
        singles.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms_per_review": round(statistics.median(singles), 2),
        "seconds_per_100_reviews": round(sum(singles) / 1000, 2),
        "threads": 1,
    }


def peak_ram_mb(stage: str, work: Path) -> float:
    """Load one stage in a fresh process and report its peak memory.

    ONNX stages use research/probe_onnx_memory.py, which imports only what
    serving needs; the PyTorch stages are not serving candidates and are probed
    through this script, research imports included.
    """
    if stage.startswith("onnx"):
        probe_dir = work / f"probe_{stage}"
        probe_dir.mkdir(exist_ok=True)
        shutil.copyfile(work / onnx_file(stage), probe_dir / "model.onnx")
        shutil.copyfile(work / "pruned" / "tokenizer.json", probe_dir / "tokenizer.json")
        command = [sys.executable, str(Path(__file__).with_name("probe_onnx_memory.py")), str(probe_dir), "--no-arena"]
        output = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout
        return float(json.loads(output.strip().splitlines()[-1])["peak_mb"])
    command = [sys.executable, str(Path(__file__)), "--ram-probe", stage, "--work-dir", str(work)]
    output = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace", check=True).stdout
    return float(output.strip().splitlines()[-1])


def ram_probe(stage: str, work: Path) -> None:
    import psutil

    predict = stage_predictor(stage, work)
    for _ in range(3):
        predict([LONG_REVIEW])
    info = psutil.Process().memory_info()
    print(round((getattr(info, "peak_wset", None) or info.rss) / 1e6, 1))


def stage_size(stage: str, work: Path, checkpoint: Path) -> int:
    if stage == "full":
        return sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))
    if stage == "pruned":
        return sum(path.stat().st_size for path in (work / "pruned").glob("*.safetensors"))
    return (work / onnx_file(stage)).stat().st_size


# ------------------------------------------------------------------- driver ---

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def incumbent_scores() -> dict[str, float]:
    """The deployed TF-IDF model's recorded scores on the same two test sets."""
    comparison = json.loads((ROOT / "research" / "sentiment_model_comparison.json").read_bytes().decode("utf-8"))
    shift = json.loads((ROOT / "research" / "marketplace_shift.json").read_bytes().decode("utf-8"))
    return {
        "firecs_test_macro_f1": comparison["results"]["tfidf_logistic_regression_incumbent"]["f1"],
        "marketplace_macro_f1": shift["results"]["deployed_tfidf"]["f1"],
    }


def integration_gate(stage_results: dict[str, Any], incumbent: dict[str, float]) -> dict[str, Any]:
    firecs = stage_results["firecs_test"]["macro_f1"]
    market = stage_results["marketplace"]["macro_f1"]
    checks = {
        "firecs_not_worse_than_incumbent": firecs >= incumbent["firecs_test_macro_f1"],
        "marketplace_beats_incumbent_by_margin": market >= incumbent["marketplace_macro_f1"] + MARKETPLACE_MARGIN,
    }
    return {
        "rule": (
            "FiReCS test macro F1 >= incumbent TF-IDF; marketplace macro F1 >= "
            f"incumbent + {MARKETPLACE_MARGIN}; stage passed the slimming acceptance rule"
        ),
        "scores": {"firecs_test_macro_f1": firecs, "marketplace_macro_f1": market},
        "incumbent": incumbent,
        "checks": checks,
        "qualified_for_integration": all(checks.values()),
    }


def evaluate_stage(predict, sets: dict[str, dict[str, list[str]]]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    metrics, predictions = {}, {}
    for name, data in sets.items():
        predicted = score_each(predict, data["texts"])
        predictions[name] = predicted
        scored = evaluate_labels(data["labels"], predicted)
        metrics[name] = {
            "accuracy": scored["accuracy"],
            "macro_f1": scored["f1"],
            "per_class_f1": {label: values["f1"] for label, values in scored["per_class"].items()},
            "macro_f1_ci": bootstrap_macro_f1_ci(data["labels"], predicted, iterations=1000, seed=42),
        }
    return metrics, predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--name", default="xlmr_sentiment")
    parser.add_argument(
        "--text",
        choices=("raw", "normalized"),
        default="raw",
        help="The text form the checkpoint was trained on; test text is fed in the same form.",
    )
    parser.add_argument("--weak-train", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Score only the first N reviews of each set (smoke tests).")
    parser.add_argument("--no-ship", action="store_true", help="Measure only; do not copy the chosen stage.")
    parser.add_argument("--ram-probe", choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.ram_probe:
        ram_probe(args.ram_probe, args.work_dir)
        return 0

    checkpoint = args.checkpoint.resolve()
    work = (args.work_dir or (SENTIMENT_ROOT / "models" / "slimming" / args.name)).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "source.txt").write_bytes(str(checkpoint).encode("utf-8"))
    config = json.loads((checkpoint / "config.json").read_bytes().decode("utf-8"))
    labels = [config["id2label"][str(index)] for index in range(len(config["id2label"]))]
    (work / "labels.json").write_bytes(json.dumps(labels).encode("utf-8"))

    print("pruning vocabulary and tokenizer...", flush=True)
    pruning = prune(checkpoint, vocabulary_corpus(args.weak_train, args.text), work / "pruned")
    print(f"  kept {pruning['kept_tokens']:,} of {pruning['full_tokens']:,} tokens; "
          f"training text segments identically: {pruning['training_corpus_check']}", flush=True)
    print("exporting ONNX...", flush=True)
    export_onnx(work / "pruned", work / "model_fp32.onnx")
    print("quantizing int8...", flush=True)
    quantize(work / "model_fp32.onnx", work / "model_int8.onnx")

    firecs = load_firecs(Path(DEFAULT_TEST))
    market = load_marketplace(ROOT / "complete_manual_marketplace_reviews (1).csv")
    sets = {
        "firecs_test": {"texts": [r["review"] for r in firecs], "labels": [r["label"] for r in firecs]},
        "marketplace": {"texts": [r["text"] for r in market], "labels": [r["label"] for r in market]},
    }
    if args.text == "normalized":
        for data in sets.values():
            data["texts"] = [normalized_text(item) for item in data["texts"]]
    if args.limit:
        sets = {name: {key: values[: args.limit] for key, values in data.items()} for name, data in sets.items()}
    remap = np.load(work / "pruned" / "id_remap.npy")
    segmentation = {
        name: segmentation_check(checkpoint, work / "pruned", remap, data["texts"]) for name, data in sets.items()
    }

    results: dict[str, Any] = {}
    predictions: dict[str, dict[str, list[str]]] = {}
    for stage in STAGES:
        print(f"scoring {stage}...", flush=True)
        predict = stage_predictor(stage, work)
        results[stage], predictions[stage] = evaluate_stage(predict, sets)
        results[stage]["latency"] = latency(predict, sets["marketplace"]["texts"])
        results[stage]["size_mb"] = round(stage_size(stage, work, checkpoint) / 1e6, 1)
        del predict
        results[stage]["peak_ram_mb"] = peak_ram_mb(stage, work)
        if stage != "full":
            for name, data in sets.items():
                results[stage][name]["mcnemar_vs_full"] = mcnemar_test(
                    data["labels"], predictions["full"][name], predictions[stage][name]
                )
                results[stage][name]["agreement_with_full"] = round(
                    sum(a == b for a, b in zip(predictions["full"][name], predictions[stage][name]))
                    / len(data["labels"]),
                    4,
                )
        print(
            f"  {stage}: marketplace F1 {results[stage]['marketplace']['macro_f1']} | "
            f"FiReCS F1 {results[stage]['firecs_test']['macro_f1']} | "
            f"{results[stage]['size_mb']} MB | peak {results[stage]['peak_ram_mb']} MB",
            flush=True,
        )

    full_f1 = results["full"]["marketplace"]["macro_f1"]
    passing = [
        stage
        for stage in STAGES[1:]
        if full_f1 - results[stage]["marketplace"]["macro_f1"] <= TOLERANCE
        and results[stage]["marketplace"]["mcnemar_vs_full"]["p_value"] >= 0.05
    ]
    servable = [stage for stage in passing if stage.startswith("onnx")]
    chosen = min(servable, key=lambda stage: results[stage]["size_mb"]) if servable else None
    gate = integration_gate(results[chosen], incumbent_scores()) if chosen else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        "acceptance_rule": (
            f"smallest ONNX stage with marketplace macro F1 within {TOLERANCE} of full "
            "and McNemar p >= 0.05 against full"
        ),
        "vocabulary": {
            **pruning,
            "corpus": f"{args.text} FiReCS training rows + weak-label training set (no test text)",
            "tokenizer": "pruned to the kept pieces, in id order",
            "test_set_segmentation": segmentation,
        },
        "limit": args.limit,
        "results": results,
        "passing_stages": passing,
        "chosen_stage": chosen,
        "integration_gate": gate,
        # A --limit run scores a sample, which qualifies nothing.
        "qualified_for_integration": bool(gate and gate["qualified_for_integration"] and not args.limit),
    }
    out = ROOT / "research" / f"slimming_{args.name}.json"
    out.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"passing: {passing} | chosen: {chosen} | wrote {out.name}")

    if chosen and not args.no_ship and not args.limit:
        ship = DEFAULT_SHIP_ROOT / args.name
        ship.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / onnx_file(chosen), ship / "model.onnx")
        shutil.copyfile(work / "pruned" / "tokenizer.json", ship / "tokenizer.json")
        files = {name: sha256(ship / name) for name in ("model.onnx", "tokenizer.json")}
        manifest = {
            # The hash in the version means a retrained model is a new version,
            # so saved analyses are redone even if the name is reused.
            "model_version": f"defaketive-{args.name}-{chosen}-{files['model.onnx'][:8]}".replace("_", "-"),
            "name": args.name,
            "stage": chosen,
            "labels": labels,
            "max_length": MAX_LENGTH,
            "text": args.text,
            "expects_raw_text": args.text == "raw",
            "files": files,
            "source_checkpoint": report["checkpoint"],
            "slimming_report": f"research/{out.name}",
            "gate": gate["rule"],
            "scores": gate["scores"],
            "incumbent": gate["incumbent"],
            "qualified_for_integration": report["qualified_for_integration"],
            "built_at": report["generated_at"],
        }
        (ship / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        stale = ship / "labels.json"
        if stale.exists():
            stale.unlink()
        print(f"shipped {chosen} to {ship} as {manifest['model_version']} "
              f"(qualified: {manifest['qualified_for_integration']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
