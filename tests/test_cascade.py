"""Cascade routing with stand-in detectors (no GPU or Ollama needed)."""

import numpy as np

from src.cascade import Cascade


class Fixed:
    def __init__(self, scores, ms=0.0):
        self.scores, self.seen = np.asarray(scores, float), []
        self.ms_per_text_ = ms

    def fit(self, texts, labels):
        return self

    def score(self, texts):
        self.seen.append(list(texts))
        return self.scores[: len(texts)]


def test_only_unsure_texts_go_to_the_judge():
    fast = Fixed([0.01, 0.5, 0.99, 0.3])
    judge = Fixed([0.9, 0.1], ms=1000.0)
    c = Cascade(fast=fast, judge=judge).fit(["a", "b", "c", "d"], [0, 1, 1, 0])
    s = c.score(["a", "b", "c", "d"])
    assert judge.seen == [["b", "d"]]
    assert list(s) == [0.01, 0.9, 0.99, 0.1]
    assert c.llm_share_ == 0.5
    assert c.ms_per_text_ >= 500.0  # half the texts paid the judge's latency


def test_confident_fast_detector_never_calls_judge():
    judge = Fixed([0.5])
    c = Cascade(fast=Fixed([0.0, 1.0]), judge=judge).fit(["a", "b"], [0, 1])
    c.score(["a", "b"])
    assert judge.seen == [] and c.llm_share_ == 0.0
