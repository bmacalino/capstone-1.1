"""Every labelled sentence with the five labels of tagging guide 1.4.

One loader for the comparison and training scripts, so they cannot drift:

  rounds 1-3  the team's safety answers; defect and counterfeit as re-read under
              guide 1.4 (ai_relabel.csv); quality and not_as_described proposed
              by the assistant (ai_extra_labels.csv)
  round 4     the fake-only round: counterfeit as labelled by the assistant and
              adjudicated by the owner (labels_adjudicated.csv), re-read under
              guide 1.4; quality and not_as_described (ai_extra_labels.csv)
  rounds 5-7  all five labels proposed by the assistant (ai_labels.csv)
  round 8     the same, on sentences the defect models flag in ordinary unused
              reviews. Only its "mining" listings are returned by default; the
              "live_check" listings are kept out of every training set and
              come back only with include_live_check=True

Only the team's safety answers here are human ground truth. The rest is
AI-proposed, and the owner accepted it on 8 Oct 2026 without auditing it
(PREREGISTRATION_final_v6.md, departure 3), so anything fitted to it is development work,
and no figure computed from it is an accuracy.

Sentiment readings and sentence embeddings are cached under outputs/cache/,
keyed by the sentiment model's file hash, because recomputing them dominated
every earlier script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "sentiment-analysis", ROOT / "research"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

OUTPUTS = ROOT / "outputs"
CACHE = OUTPUTS / "cache"
LABELS = ("safety", "defect", "quality", "counterfeit", "not_as_described")
KEYWORD_RULES = "0.4"
ANSWER = {"yes": True, "no": False, "": None}
# Round 4 was labelled under guide 1.2, where a picture mismatch counted as a
# fake. Its nine fake answers that are also not-as-described were re-read under
# guide 1.4 (8 Oct 2026): these two dispute the photos and never the brand.
ROUND_4_NOT_FAKE_UNDER_GUIDE_1_4 = frozenset({"r4-00003", "r4-00079"})


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_five_label_records(include_live_check: bool = False) -> list[dict[str, Any]]:
    from gadget_scope import is_gadget
    from train_final_v6 import load_records

    records: list[dict[str, Any]] = []
    master: dict[str, dict[str, str]] = {}
    for number in range(1, 9):
        master.update({row["row_id"]: row for row in _read(OUTPUTS / f"labelling_queue_round_{number}" / "queue_master.csv")})
    extra = {
        row["row_id"]: {label: row[f"ai_{label}"] == "yes" for label in ("quality", "not_as_described")}
        for number in (1, 2, 3, 4)
        for row in _read(OUTPUTS / f"labelling_queue_round_{number}" / "ai_extra_labels.csv")
    }
    moved = {
        row["row_id"]: {"defect": ANSWER[row["relabel_defect"]], "counterfeit": ANSWER[row["relabel_counterfeit"]]}
        for number in (1, 2, 3)
        for row in _read(OUTPUTS / f"labelling_queue_round_{number}" / "ai_relabel.csv")
    }
    for record in load_records():
        team = record["labels"]
        records.append({
            "row_id": record["row_id"], "review_id": record["review_id"], "text": record["text"], "round": record["round"],
            "team": dict(team),
            "labels": {"safety": team["safety"], **moved[record["row_id"]], **extra[record["row_id"]]},
        })
    fake = {
        row["row_id"]: row["final_counterfeit"] == "yes" and row["row_id"] not in ROUND_4_NOT_FAKE_UNDER_GUIDE_1_4
        for row in _read(OUTPUTS / "labelling_queue_round_4" / "labels_adjudicated.csv")
    }
    for row in _read(OUTPUTS / "labelling_queue_round_4" / "queue_master.csv"):
        records.append({
            "row_id": row["row_id"], "review_id": row["review_id"], "text": row["sentence_text"], "round": "4", "team": {},
            "labels": {"safety": None, "defect": None, "counterfeit": fake.get(row["row_id"]), **extra[row["row_id"]]},
        })
    for number in (5, 6, 7, 8):
        for row in _read(OUTPUTS / f"labelling_queue_round_{number}" / "ai_labels.csv"):
            part = master[row["row_id"]].get("part") or "round"
            if part == "live_check" and not include_live_check:
                continue
            records.append({
                "row_id": row["row_id"], "review_id": row["review_id"], "text": row["sentence_text"], "round": str(number),
                "team": {}, "labels": {label: row[f"ai_{label}"] == "yes" for label in LABELS},
            })
    for record in records:
        record["gadget"] = is_gadget(master[record["row_id"]])
        record["part"] = master[record["row_id"]].get("part") or "round"
        record["stratum"] = master[record["row_id"]].get("stratum") or ""
        record["rating"] = master[record["row_id"]].get("rating") or ""
        # Round 8 sampled its strata at different rates: how many pool sentences a row stands for.
        sampled = master[record["row_id"]] if record["round"] == "8" else {}
        record["weight"] = float(sampled["pool"]) / float(sampled["sampled"]) if sampled else 1.0
    return records


def keyword_view(records: list[dict[str, Any]], rules: str = KEYWORD_RULES) -> dict[str, Any]:
    """Normalized text, keyword features and what the rules alone flag."""
    from defaketive_sentiment.hazard_learned import keyword_features
    from defaketive_sentiment.hazard_model import EXTRA_RULES, LexiconHazardTagger

    plain = LexiconHazardTagger(rules=rules, counterfeit_rules=None)
    full = LexiconHazardTagger(rules=rules, counterfeit_rules=EXTRA_RULES)
    tagged = [plain.tag_sentence(record["text"]) for record in records]
    with_rules = [full.tag_sentence(record["text"]) for record in records]
    flags = {
        "safety": np.array(["safety" in item["tags"] for item in tagged]),
        "defect": np.array(["defect" in item["tags"] for item in tagged]),
        "counterfeit": np.array(["counterfeit" in item["tags"] for item in with_rules]),
        "quality": np.array(["quality" in item["extra_tags"] for item in with_rules]),
        "not_as_described": np.array(["not_as_described" in item["extra_tags"] for item in with_rules]),
    }
    return {
        "text": np.array([item["text_normalized"] for item in tagged], dtype=object),
        "kw": np.vstack([keyword_features(item) for item in tagged]),
        "flags": flags,
    }


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _classifier():
    from train_defect_gate import sentiment_reader

    read, classifier = sentiment_reader()
    return read, classifier, classifier.manifest["files"]["model.onnx"][:16]


def sentiment_labels(records: list[dict[str, Any]]) -> np.ndarray:
    """The app's sentiment reading of each sentence, cached on disk."""
    read, _, version = _classifier()
    path = CACHE / f"sentiment_{version}.json"
    known = json.loads(path.read_bytes().decode("utf-8")) if path.is_file() else {}
    fresh = False
    for record in records:
        key = _key(record["text"])
        if key not in known:
            known[key] = read(record["text"])
            fresh = True
    if fresh:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(known).encode("utf-8"))
    return np.array([known[_key(record["text"])] for record in records], dtype=object)


def embeddings(records: list[dict[str, Any]]) -> np.ndarray:
    """Mean-pooled sentence embeddings from the shipped sentiment encoder, cached."""
    _, classifier, version = _classifier()
    path = CACHE / f"embeddings_{version}.npz"
    known: dict[str, np.ndarray] = dict(np.load(path)) if path.is_file() else {}
    fresh = False
    for record in records:
        key = _key(record["text"])
        if key not in known:
            known[key] = np.asarray(classifier.embed(record["text"]), dtype=np.float32).reshape(-1)
            fresh = True
    if fresh:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **known)
    return np.vstack([known[_key(record["text"])] for record in records])
