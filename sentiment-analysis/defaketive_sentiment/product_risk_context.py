"""Learned context features for product-only risk candidates.

Signals are classifier inputs, never hard risk labels or severity weights.
"""

import re
import unicodedata

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def clean_risk_text(value):
    """FROZEN at v1. Do not modify.

    This function is referenced by module path inside pickled `RiskTextCleaner`
    steps in the committed artifacts under `sentiment-analysis/models/`. Editing
    it would retroactively change the behaviour of models that have already been
    fitted, evaluated and reported, without any version number changing to say
    so. Every recorded metric in `sentiment-analysis/evaluation/` assumes the
    current behaviour.

    It is also known to disagree with the runtime normalizer - notably it maps a
    bare "d" to "hindi" and the runtime path does not. That inconsistency is the
    reason `text_norm.py` exists. New work uses `text_norm.normalize_text`; this
    stays exactly as it is so old artifacts keep loading and old results keep
    meaning what they said.

    `test_product_risk_context.py` hashes this function's output over a fixture
    so an edit fails the suite rather than silently invalidating past results.
    """
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("’", "'")
    text = re.sub(r"\bfeature_\w+\b", " ", text)
    replacements = {
        r"\b(?:doesn't|doesnt)\b": "does not",
        r"\b(?:don't|dont)\b": "do not",
        r"\b(?:didn't|didnt)\b": "did not",
        r"\b(?:isn't|isnt)\b": "is not",
        r"\b(?:can't|cant|cannot)\b": "can not",
        r"\b(?:couldn't|couldnt)\b": "could not",
        r"\b(?:won't|wont)\b": "will not",
        r"\b(?:haven't|havent)\b": "have not",
        r"\b(?:hnd|hindi|di|d)\b": "hindi",
        r"\b(?:deffective|defectiv)\b": "defective",
        r"\b(?:mlobat|malobat|malowbat)\b": "lowbat",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


class RiskTextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_risk_text(text) for text in X]


PATTERNS = {
    "functional_failure": r"\b(?:(?:not|never)\s+(?:\w+\s+){0,2}(?:work\w*|function\w*|charg\w*|connect\w*|pair\w*|turn\w*|light\w*)|(?:hindi|ayaw|walang)\s+(?:\w+\s+){0,2}(?:gumana|gumagana|nagana|umandar|magcharge|macharge|magconnect|connect|nailaw)|malfunction\w*|unrecognized|disconnect\w*|reconnecting)\b",
    "performance_problem": r"\b(?:mahina|humihina|mabagal|blurry|blurred|lagging|overheat\w*|distort\w*|static noise|weak|barely heard|not clear|not loud|very slow)\b|\b(?:hina|hirap)\s+(?:\w+\s+){0,2}(?:sound|tunog|connect|iconnect)\b",
    "battery_problem": r"\b(?:mabilis|madali|quickly|fast)\b.{0,25}\b(?:lowbat|drain\w*|discharg\w*)\b|\b(?:battery|charge)\b.{0,25}\b(?:not last|not long enough|short life)\b|\b(?:cannot|can not) fully charge\b",
    "physical_problem": r"\b(?:broken|broke|defect\w*|damag\w*|sira|sirang|basag|crack\w*|leaking|naputol|nasira|sirain)\b",
    "positive_function": r"\b(?:working (?:well|fine|properly|okay)|works (?:well|fine|great)|gumagana naman|nagana naman|clear and crisp|malinaw|good sound|easy to use|madaling gamitin)\b",
    "positive_quality": r"\b(?:good quality|great quality|matibay|durable|excellent|perfect|maganda|sulit)\b",
    "not_tested": r"\b(?:not (?:yet )?(?:test\w*|tried|used)|have not (?:test\w*|tried|used)|hindi (?:ko )?pa (?:na ?try|natest|nagamit)|hopefully|sana magtagal|hope it (?:will|would) last|not sure)\b",
    "order_issue": r"\b(?:wrong (?:item|product|colou?r|design)|mali\w* (?:ang |ung |yung )?(?:item|pinadala|order)|missing item|hindi dumating|not delivered|late delivery|no delivery)\b",
    "delivery_context": r"\b(?:delivery|deliver\w*|shipping|courier|rider|dumating|nadeliver)\b",
    "packaging_context": r"\b(?:box|packaging|package|balot|kahon|parcel)\b",
    "service_context": r"\b(?:seller|agent\w*|customer service|refund|chat|response)\b",
    "contrast": r"\b(?:but|pero|kaso|however|although)\b",
    "poor_product_aspect": r"\b(?:sound quality|functionality|performance|compatibility|battery life)\s*:\s*(?:not good|bad|poor|0(?:\b|/)|ang hina)\b",
}
COMPILED = {name: re.compile(pattern) for name, pattern in PATTERNS.items()}
NEGATED_DAMAGE = re.compile(r"\b(?:no|not|without|walang|wala|hindi)\s+(?:\w+\s+){0,2}(?:broken|defect\w*|damag\w*|sira|sirang|basag|crack\w*|nasira)\b")
PACKAGE_DAMAGE = re.compile(r"\b(?:box|packaging|package|balot|kahon|parcel)\b.{0,30}\b(?:damag\w*|broken|basag|yupi|dented|crush\w*)\b|\b(?:damag\w*|broken|basag|yupi|dented|crush\w*)\b.{0,30}\b(?:box|packaging|package|balot|kahon|parcel)\b")


def context_signals(text):
    text = clean_risk_text(text)
    damage_text = NEGATED_DAMAGE.sub(" ", text)
    product_damage_text = PACKAGE_DAMAGE.sub(" ", damage_text)
    signals = {name: float(bool(pattern.search(text))) for name, pattern in COMPILED.items()}
    signals["physical_problem"] = float(bool(COMPILED["physical_problem"].search(product_damage_text)))
    signals["negated_damage"] = float(bool(NEGATED_DAMAGE.search(text)))
    signals["packaging_damage"] = float(bool(PACKAGE_DAMAGE.search(damage_text)))
    concrete = any(signals[k] for k in ["physical_problem", "functional_failure", "performance_problem", "battery_problem", "poor_product_aspect"])
    signals["concrete_product_problem"] = float(concrete)
    signals["complaint_with_praise"] = float(concrete and (signals["positive_function"] or signals["positive_quality"]))
    signals["service_or_order_without_product_failure"] = float(not concrete and (signals["order_issue"] or signals["packaging_damage"]))
    signals["untested_without_failure"] = float(signals["not_tested"] and not concrete)
    return signals


class ProductContextSignals(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [context_signals(text) for text in X]


def context_pipeline():
    return Pipeline([
        ("clean", RiskTextCleaner()),
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.995, max_features=12000, sublinear_tf=True, dtype=np.float32)),
            ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=24000, sublinear_tf=True, dtype=np.float32)),
            ("context", Pipeline([("signals", ProductContextSignals()), ("vectorize", DictVectorizer())])),
        ])),
        ("classifier", LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000, solver="lbfgs")),
    ])
