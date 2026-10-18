"""
Cascade: a fast detector screens every text; only texts it is unsure about are
sent to the slow LLM judge.

    fast detector score p
      p <= low   -> benign     (decided in milliseconds)
      p >= high  -> injection  (decided in milliseconds)
      otherwise  -> ask the LLM judge (seconds)

The interesting numbers are how much of the LLM's accuracy the cascade keeps and
what share of texts still needs the LLM (reported as llm_share).
"""

from __future__ import annotations

import time

import numpy as np


class Cascade:
    name = "cascade"

    def __init__(self, fast=None, judge=None, low: float = 0.05, high: float = 0.95,
                 judge_model: str = "llama3.1:8b"):
        self.fast, self.judge = fast, judge
        self.low, self.high, self.judge_model = low, high, judge_model

    def fit(self, texts, labels):
        if self.fast is None:
            from src.finetune import FineTunedTransformer

            self.fast = FineTunedTransformer()
        if self.judge is None:
            from src.llm_judge import OllamaJudge

            self.judge = OllamaJudge(self.judge_model)
        self.fast.fit(texts, labels)
        return self

    def score(self, texts) -> np.ndarray:
        texts = list(texts)
        t0 = time.perf_counter()
        p = np.asarray(self.fast.score(texts), dtype=float).copy()
        fast_ms = 1000 * (time.perf_counter() - t0) / max(len(texts), 1)

        unsure = (p > self.low) & (p < self.high)
        self.llm_share_ = float(unsure.mean()) if len(texts) else 0.0
        llm_ms = 0.0
        if unsure.any():
            idx = np.flatnonzero(unsure)
            p[idx] = self.judge.score([texts[i] for i in idx])
            llm_ms = getattr(self.judge, "ms_per_text_", 0.0) * len(idx) / len(texts)
        self.ms_per_text_ = fast_ms + llm_ms
        return p
