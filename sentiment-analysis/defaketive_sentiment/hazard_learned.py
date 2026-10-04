"""Learned per-category hazard models (hazard checker 0.4 candidates).

The keyword checker (`hazard_model.LexiconHazardTagger`) finds a report only
when it uses a listed word. These models learn from sentences people labelled
(research/PREREGISTRATION_hazard_v4.md). Four candidate designs, all small
enough for a CPU host:

  T   TF-IDF word and character n-grams -> logistic regression
  TK  T plus keyword features
  E   sentence embedding from the shipped sentiment model -> logistic regression
  EK  E plus keyword features

"Keyword features" are the keyword checker's own output for the sentence: how
many listed terms counted for each category, and its three tags. They are
always computed from the keyword result BEFORE any learned decision is applied,
exactly as in training.

Training and selection live in research/train_hazard_classifier.py; this module
holds what both the research script and the app need, so the two cannot drift.

A model folder carries a manifest with the SHA-256 of every file it depends on,
including the keyword lists and the sentiment model that produce its features.
If any of them changed, loading refuses: the model would be fed features it was
not trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .product_badges import CATEGORIES

MODELS_ROOT = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_DIR = MODELS_ROOT / "hazard_classifier_v1"
MANIFEST_NAME = "manifest.json"
CANDIDATES = ("T", "TK", "E", "EK")
# How a category's learned model combines with the keyword checker:
#   decide    the learned model makes the final yes/no (checker 0.4)
#   or_gated  keywords' flag stands; the learned model may only ADD a flag,
#             and only to a sentence the sentiment model does not read as
#             positive (research/PREREGISTRATION_defect_v5.md)
COMBINE_MODES = ("decide", "or_gated")
LEXICON_DIR = Path(__file__).with_name("lexicons")
# Files whose content shapes the keyword features.
KEYWORD_FILES = (
    "risk_lexicon.tsv",
    "safety_false_friends_v1.tsv",
    "hazard_phrase_exclusions_v1.tsv",
)


class LearnedModelError(RuntimeError):
    """Raised when a learned hazard model is missing, altered or mismatched."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def keyword_file_hashes() -> dict[str, str]:
    return {name: file_sha256(LEXICON_DIR / name) for name in KEYWORD_FILES}


def keyword_features(tagged: Mapping[str, Any]) -> np.ndarray:
    """Six numbers from one keyword-tagged sentence.

    Per category: the number of listed terms that counted (not negated, not
    excluded), then the keyword checker's tag (1/0) for each category.
    """
    counts = []
    for category in CATEGORIES:
        counts.append(
            sum(
                1
                for term in tagged.get("terms") or []
                if term.get("category") == category
                and not term.get("negated")
                and not term.get("excluded")
            )
        )
    tags = set(tagged.get("tags") or [])
    return np.array(counts + [1.0 if category in tags else 0.0 for category in CATEGORIES], dtype=np.float64)


@dataclass
class LearnedCategoryModel:
    """One fitted candidate for one category, with its frozen threshold."""

    code: str
    classifier: Any
    word: Any = None
    char: Any = None
    scaler: Any = None
    threshold: float = 0.5
    category: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_embedding(self) -> bool:
        return self.code.startswith("E")

    @property
    def uses_keywords(self) -> bool:
        return self.code.endswith("K")

    def _matrix(self, normalized: Sequence[str], kw: np.ndarray | None, emb: np.ndarray | None):
        from scipy.sparse import csr_matrix, hstack

        parts = []
        if self.code.startswith("T"):
            parts += [self.word.transform(normalized), self.char.transform(normalized)]
        dense = []
        if self.uses_keywords:
            dense.append(np.asarray(kw, dtype=np.float64))
        if self.needs_embedding:
            dense.append(np.asarray(emb, dtype=np.float64))
        if dense:
            parts.append(csr_matrix(self.scaler.transform(np.hstack(dense))))
        return hstack(parts).tocsr()

    def probability(self, normalized, kw=None, emb=None) -> np.ndarray:
        return self.classifier.predict_proba(self._matrix(normalized, kw, emb))[:, 1]


def fit_category_model(
    code: str,
    normalized: Sequence[str],
    kw: np.ndarray | None,
    emb: np.ndarray | None,
    labels: Sequence[int],
    *,
    category: str = "",
    seed: int = 42,
) -> LearnedCategoryModel:
    """Fit one candidate with the pre-registered, untuned hyper-parameters."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if code not in CANDIDATES:
        raise ValueError(f"unknown candidate {code!r}")
    model = LearnedCategoryModel(
        code=code,
        classifier=LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed),
        category=category,
    )
    if code.startswith("T"):
        model.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        model.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        model.word.fit(normalized)
        model.char.fit(normalized)
    dense = []
    if model.uses_keywords:
        dense.append(np.asarray(kw, dtype=np.float64))
    if model.needs_embedding:
        dense.append(np.asarray(emb, dtype=np.float64))
    if dense:
        model.scaler = StandardScaler().fit(np.hstack(dense))
    model.classifier.fit(model._matrix(normalized, kw, emb), np.asarray(labels, dtype=int))
    # stop_words_ (terms dropped by min_df) is a set kept only for inspection;
    # its pickled order varies between runs, which made identical models save
    # as different files. scikit-learn documents it as safe to remove.
    for vectorizer in (model.word, model.char):
        if vectorizer is not None and hasattr(vectorizer, "stop_words_"):
            vectorizer.stop_words_ = None
    return model


class LearnedHazardScorer:
    """The adopted learned models, loaded behind the manifest gate."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        embedding_model_sha256: str | None = None,
        sentiment_model_sha256: str | None = None,
    ):
        import joblib

        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        manifest_path = self.model_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise LearnedModelError(
                f"The learned hazard model is not installed: {manifest_path} is missing. "
                "Build it with: python research/train_hazard_classifier.py"
            )
        self.manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
        if self.manifest.get("keyword_files") != keyword_file_hashes():
            raise LearnedModelError(
                "The keyword lists changed since the learned hazard model was trained, "
                "so its keyword features would be wrong. Retrain it."
            )
        self.models: dict[str, LearnedCategoryModel] = {}
        self.combine: dict[str, str] = {}
        for category, entry in (self.manifest.get("categories") or {}).items():
            mode = entry.get("combine", "decide")
            if mode not in COMBINE_MODES:
                raise LearnedModelError(f"Unknown combine mode {mode!r} for {category}.")
            self.combine[category] = mode
            path = self.model_dir / entry["file"]
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                raise LearnedModelError(f"{path} is missing or does not match its manifest.")
            model = joblib.load(path)
            model.threshold = float(entry["threshold"])
            self.models[category] = model
        expected = (self.manifest.get("embedding_model") or {}).get("model_sha256")
        if self.needs_embedding and embedding_model_sha256 is not None and embedding_model_sha256 != expected:
            raise LearnedModelError(
                "The sentiment model changed since the learned hazard model was trained, "
                "so its sentence embeddings would be different. Retrain it."
            )
        gate_sha = (self.manifest.get("sentiment_model") or {}).get("model_sha256")
        if self.needs_sentiment and sentiment_model_sha256 is not None and sentiment_model_sha256 != gate_sha:
            raise LearnedModelError(
                "The sentiment model changed since the gated hazard model's threshold was set "
                "with it. Retrain it."
            )
        # The file hashes are part of the version, so a retrained model is a new
        # version and saved analyses are redone.
        digest = hashlib.sha256(
            "|".join(f"{c}:{e['sha256']}" for c, e in sorted(self.manifest["categories"].items())).encode("utf-8")
        ).hexdigest()[:8]
        self.version = f"{self.manifest.get('version') or 'hazard-learned'}-{'-'.join(self.categories)}-{digest}"

    @property
    def categories(self) -> list[str]:
        return list(self.models)

    @property
    def needs_embedding(self) -> bool:
        return any(model.needs_embedding for model in self.models.values())

    @property
    def needs_sentiment(self) -> bool:
        return any(mode == "or_gated" for mode in self.combine.values())

    def decide(
        self, normalized: str, kw: np.ndarray, embedding: np.ndarray | None
    ) -> dict[str, tuple[bool, float]]:
        """{category: (tagged, probability)} for one sentence."""
        out = {}
        for category, model in self.models.items():
            probability = float(
                model.probability([normalized], kw.reshape(1, -1), None if embedding is None else embedding.reshape(1, -1))[0]
            )
            out[category] = (probability >= model.threshold, probability)
        return out


Embedder = Callable[[str], np.ndarray]


def embed_all(embedder: Embedder, texts: Iterable[str]) -> np.ndarray:
    return np.vstack([embedder(text) for text in texts])
