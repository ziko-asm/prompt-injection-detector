"""
Detectors. Every detector exposes the same interface:

    fit(texts, labels) -> self
    score(texts)       -> np.ndarray of P(injection) in [0, 1]

keyword : a rule built automatically from training data. It selects the
          word n-grams whose frequency differs most between injection and
          benign text (smoothed log-odds) and flags a text that contains any
          of them. No phrases are written by hand.
tfidf   : word + character n-gram TF-IDF with logistic regression.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline


class KeywordRule:
    name = "keyword"

    def __init__(self, k: int = 30, ngram_range=(1, 3), min_df: int = 3):
        self.k, self.ngram_range, self.min_df = k, ngram_range, min_df

    def fit(self, texts, labels):
        labels = np.asarray(labels)
        vec = CountVectorizer(ngram_range=self.ngram_range, min_df=self.min_df,
                              lowercase=True, binary=True)
        X = vec.fit_transform(texts)
        pos = np.asarray(X[labels == 1].sum(0)).ravel() + 1.0
        neg = np.asarray(X[labels == 0].sum(0)).ravel() + 1.0
        n_pos, n_neg = (labels == 1).sum() + 2.0, (labels == 0).sum() + 2.0
        log_odds = np.log(pos / n_pos) - np.log(neg / n_neg)
        top = np.argsort(-log_odds)[: self.k]
        vocab = vec.get_feature_names_out()
        self.terms_ = [vocab[i] for i in top]
        self.vec_ = CountVectorizer(vocabulary=self.terms_, ngram_range=self.ngram_range,
                                    lowercase=True, binary=True)
        return self

    def score(self, texts):
        hits = np.asarray(self.vec_.transform(texts).sum(1)).ravel()
        return (hits > 0).astype(float)


class TfidfLR:
    name = "tfidf"

    def __init__(self, C: float = 4.0):
        self.C = C

    def fit(self, texts, labels):
        feats = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                     sublinear_tf=True, max_features=200_000)),
        ])
        self.model_ = make_pipeline(feats, LogisticRegression(C=self.C, max_iter=2000,
                                                              class_weight="balanced"))
        self.model_.fit(list(texts), np.asarray(labels))
        return self

    def score(self, texts):
        return self.model_.predict_proba(list(texts))[:, 1]


DETECTORS = {"keyword": KeywordRule, "tfidf": TfidfLR}


def make_detector(name: str):
    """'keyword', 'tfidf', 'transformer', or 'ollama:<model>' (e.g. ollama:llama3.2:3b)."""
    if name.startswith("ollama:"):
        from src.llm_judge import OllamaJudge  # needs a running Ollama only when used

        return OllamaJudge(model=name.split(":", 1)[1])
    if name == "transformer":
        from src.finetune import FineTunedTransformer  # needs torch + transformers

        return FineTunedTransformer()
    if name == "cascade":
        from src.cascade import Cascade  # transformer + ollama:llama3.1:8b

        return Cascade()
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}; use {list(DETECTORS)} or ollama:<model>")
    return DETECTORS[name]()


def is_zero_shot(name: str) -> bool:
    return name.startswith("ollama:")


def timed_score(det, texts) -> tuple[np.ndarray, float]:
    """Scores and mean latency in milliseconds per text. Detectors that cache
    their answers report the latency measured on the first, uncached call."""
    t0 = time.perf_counter()
    s = det.score(texts)
    ms = 1000 * (time.perf_counter() - t0) / max(len(texts), 1)
    return s, getattr(det, "ms_per_text_", ms)
