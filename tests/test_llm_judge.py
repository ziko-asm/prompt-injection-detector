"""LLM judge tests with a fake Ollama server (no model or GPU needed)."""

import json

import pandas as pd
import pytest

from src import llm_judge
from src.evaluate import merge, run
from src.llm_judge import OllamaJudge, parse_probability


@pytest.mark.parametrize("reply,expected", [
    ('{"injection_probability": 90}', 0.9),
    ('{"injection_probability": 0}', 0.0),
    ('{"injection_probability": 250}', 1.0),
    ("probability: 30", 0.3),
    ("no number here", None),
])
def test_parse_probability(reply, expected):
    assert parse_probability(reply) == expected


class FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


@pytest.fixture
def fake_ollama(monkeypatch, tmp_path):
    calls = []

    def post(url, json=None, timeout=None):
        text = json["messages"][1]["content"]
        calls.append(text)
        prob = 95 if "omega" in text else 5
        return FakeResponse(f'{{"injection_probability": {prob}}}')

    monkeypatch.setattr(llm_judge.requests, "post", post)
    monkeypatch.setattr(llm_judge, "CACHE_DIR", tmp_path)
    return calls


def test_judge_scores_and_caches(fake_ollama):
    texts = ["alpha report", "omega summary", "beta travel"]
    s1 = OllamaJudge("fake").score(texts)
    assert list(s1) == [0.05, 0.95, 0.05]
    assert len(fake_ollama) == 3
    s2 = OllamaJudge("fake").score(texts)  # new instance reads the cache file
    assert list(s2) == list(s1) and len(fake_ollama) == 3


def test_judge_wraps_text_as_data(fake_ollama):
    OllamaJudge("fake").score(["alpha"])
    assert fake_ollama[0].startswith("<text>") and fake_ollama[0].rstrip().endswith("</text>")


def test_zero_shot_rows_in_benchmark(fake_ollama):
    df = pd.DataFrame({"text": ["omega a", "alpha b", "omega c", "beta d"],
                       "label": [1, 0, 1, 0], "split": "test"})
    rows = run({"x": df, "y": df}, ["ollama:fake"])
    assert len(rows) == 2 and {r["setting"] for r in rows} == {"zero-shot"}
    assert rows[0]["f1"] == 1.0 and rows[0]["parse_failures"] == 0
    assert json.dumps(rows)  # serialisable


def test_merge_replaces_only_rerun_detectors():
    old = [{"detector": "tfidf", "v": 1}, {"detector": "ollama:m", "v": 1}]
    new = [{"detector": "ollama:m", "v": 2}]
    assert merge(old, new) == [{"detector": "tfidf", "v": 1}, {"detector": "ollama:m", "v": 2}]
