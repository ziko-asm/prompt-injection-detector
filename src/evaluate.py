"""
Benchmark.

Trained detectors (keyword, tfidf, transformer, cascade) are trained on each
dataset and tested on the test split of every dataset (in-domain and
cross-dataset), then trained once on the union of all training splits
(combined). Zero-shot LLM judges (ollama:<model>) are not trained and are
tested on every test split.

    python -m src.evaluate                                     # keyword + tfidf
    python -m src.evaluate --detectors ollama:llama3.2:3b      # adds the LLM rows
    python -m src.evaluate --detectors transformer cascade     # GPU + Ollama

Rows for the detectors you run replace their old rows in results/results.json;
rows of other detectors are kept, so the LLM can be added later.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.data import SOURCES, describe, load
from src.detectors import DETECTORS, is_zero_shot, make_detector, timed_score


def metrics(y, scores, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(scores) >= threshold).astype(int)
    neg = y == 0
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(pred[neg].mean()) if neg.any() else float("nan"),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def _row(det, det_name, train_name, test_name, setting, te) -> dict:
    scores, ms = timed_score(det, te["text"].tolist())
    row = {"detector": det_name, "train": train_name, "test": test_name, "setting": setting,
           "n_test": len(te), "ms_per_text": round(ms, 3),
           **{k: round(v, 4) for k, v in metrics(te["label"], scores).items()}}
    if hasattr(det, "parse_failures_"):
        row["parse_failures"] = det.parse_failures_
    if hasattr(det, "llm_share_"):
        row["llm_share"] = round(det.llm_share_, 4)
    return row


def run(frames: dict, detector_names=None) -> list[dict]:
    detector_names = detector_names or list(DETECTORS)
    tests = {n: df[df["split"] == "test"] for n, df in frames.items()}
    rows = []
    for det_name in detector_names:
        if is_zero_shot(det_name):
            det = make_detector(det_name).fit()
            for test_name, te in tests.items():
                rows.append(_row(det, det_name, "none", test_name, "zero-shot", te))
            continue
        for train_name, train_df in frames.items():
            tr = train_df[train_df["split"] == "train"]
            det = make_detector(det_name).fit(tr["text"].tolist(), tr["label"].to_numpy())
            for test_name, te in tests.items():
                setting = "in-domain" if train_name == test_name else "cross-dataset"
                rows.append(_row(det, det_name, train_name, test_name, setting, te))
        if len(frames) > 1:
            tr = pd.concat([df[df["split"] == "train"] for df in frames.values()])
            det = make_detector(det_name).fit(tr["text"].tolist(), tr["label"].to_numpy())
            for test_name, te in tests.items():
                rows.append(_row(det, det_name, "all", test_name, "combined", te))
    return rows


def table(rows) -> str:
    head = "| Detector | Train | Test | Setting | Precision | Recall | F1 | FPR | AUC | ms/text |"
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lines.append(f"| {r['detector']} | {r['train']} | {r['test']} | {r['setting']} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                     f"{r['false_positive_rate']:.3f} | {r['roc_auc']:.3f} | {r['ms_per_text']} |")
    return "\n".join(lines)


def merge(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    replaced = {r["detector"] for r in new_rows}
    return [r for r in old_rows if r["detector"] not in replaced] + new_rows


def main():
    p = argparse.ArgumentParser(description="Prompt injection detector benchmark")
    p.add_argument("--detectors", nargs="*", default=None,
                   help=f"any of {list(DETECTORS)}, transformer, cascade, or ollama:<model>")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    names = a.detectors or list(DETECTORS)
    for n in names:
        if not is_zero_shot(n):
            make_detector(n)  # fail fast on a typo before loading data

    frames = {name: load(name) for name in SOURCES}
    new_rows = run(frames, names)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    path = out / "results.json"
    old_rows = json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else []
    rows = merge(old_rows, new_rows)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "datasets": [describe(n, df) for n, df in frames.items()],
            "python": platform.python_version(), "threshold": 0.5}
    path.write_text(json.dumps({"meta": meta, "results": rows}, indent=1), encoding="utf-8")
    t = table(rows)
    (out / "summary_table.md").write_text(t + "\n", encoding="utf-8")
    print(t)


if __name__ == "__main__":
    main()
