"""Pipeline tests on a synthetic, neutral two-class corpus (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from src.data import clean
from src.detectors import DETECTORS
from src.evaluate import metrics, run


def toy_corpus(n=200, seed=0):
    rng = np.random.default_rng(seed)
    common = ["report", "weather", "meeting", "invoice", "summary", "travel"]
    pos_words, neg_words = ["zeta", "omega", "sigma"], ["alpha", "beta", "gamma"]
    rows = []
    for i in range(n):
        lbl = int(i % 3 == 0)
        words = list(rng.choice(common, 5)) + list(rng.choice(pos_words if lbl else neg_words, 2))
        rng.shuffle(words)
        rows.append({"text": " ".join(words) + f" item{i}", "label": lbl, "split": "train"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frames():
    return {"a": clean(toy_corpus(240, 0)), "b": clean(toy_corpus(240, 1))}


def test_clean_creates_disjoint_test_split():
    df = clean(toy_corpus(100))
    assert set(df["split"]) == {"train", "test"}
    assert not set(df[df.split == "train"].text) & set(df[df.split == "test"].text)
    assert 0 < df[df.split == "test"].label.mean() < 1


def test_clean_drops_duplicates_and_empty():
    df = toy_corpus(20)
    df = pd.concat([df, df.iloc[:5], pd.DataFrame([{"text": "  ", "label": 0, "split": "train"}])])
    out = clean(df)
    assert len(out) == 20


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detectors_learn_and_score_in_range(name, frames):
    df = frames["a"]
    tr, te = df[df.split == "train"], df[df.split == "test"]
    det = DETECTORS[name]().fit(tr.text.tolist(), tr.label.to_numpy())
    s = det.score(te.text.tolist())
    assert s.shape == (len(te),) and (s >= 0).all() and (s <= 1).all()
    assert metrics(te.label, s)["f1"] > 0.8


def test_keyword_rule_learns_terms_from_data(frames):
    df = frames["a"]
    tr = df[df.split == "train"]
    det = DETECTORS["keyword"](k=5).fit(tr.text.tolist(), tr.label.to_numpy())
    assert set(det.terms_) & {"zeta", "omega", "sigma"}


def test_metrics_perfect_and_fpr():
    m = metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.7])
    assert m["recall"] == 1.0 and m["false_positive_rate"] == 0.5


def test_run_produces_in_domain_cross_and_combined_rows(frames):
    rows = run(frames, ["tfidf"])
    assert len(rows) == 6
    assert {r["setting"] for r in rows} == {"in-domain", "cross-dataset", "combined"}
