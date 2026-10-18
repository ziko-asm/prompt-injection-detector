"""
Public, labelled datasets of benign vs. prompt-injection text.

Nothing in this repository writes attack text. All examples come from datasets
that researchers have already published on the Hugging Face Hub; they are
downloaded on first use and cached in data/ (not committed).

    python -m src.data --inspect        # schema and label balance, no texts printed
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("data")


@dataclass(frozen=True)
class Source:
    hf_id: str
    text_col: str
    label_col: str
    positive: object          # value of label_col that means "injection"
    config: str | None = None


# Two independent sources, so we can test whether a detector trained on one
# generalises to the other. Column names are checked at load time.
SOURCES = {
    "deepset": Source("deepset/prompt-injections", "text", "label", 1),
    "jackhhao": Source("jackhhao/jailbreak-classification", "prompt", "type", "jailbreak"),
}


def _download(name: str) -> pd.DataFrame:
    from datasets import load_dataset  # imported lazily: tests do not need it

    src = SOURCES[name]
    ds = load_dataset(src.hf_id, src.config)
    frames = []
    for split, part in ds.items():
        df = part.to_pandas()
        missing = {src.text_col, src.label_col} - set(df.columns)
        if missing:
            raise KeyError(f"{src.hf_id}/{split}: columns {sorted(missing)} not found, "
                           f"available: {list(df.columns)}")
        frames.append(pd.DataFrame({
            "text": df[src.text_col].astype(str),
            "label": (df[src.label_col] == src.positive).astype(int),
            "split": split,
        }))
    return pd.concat(frames, ignore_index=True)


def load(name: str, refresh: bool = False) -> pd.DataFrame:
    """Columns: text, label (1 = injection), split (train / test)."""
    path = CACHE / f"{name}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        df = _download(name)
        CACHE.mkdir(exist_ok=True)
        df.to_parquet(path, index=False)
    return clean(df)


def clean(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Drop empty and duplicate texts, keep one label per text, and make sure
    a held-out test split exists that shares no text with train."""
    df = df[df["text"].str.strip().str.len() > 0].copy()
    df["key"] = df["text"].str.strip().str.lower()
    df = df.drop_duplicates("key")
    if "test" not in set(df["split"]):
        rng = np.random.default_rng(seed)
        df["split"] = "train"
        for lbl in (0, 1):
            idx = df.index[df["label"] == lbl].to_numpy().copy()
            rng.shuffle(idx)
            df.loc[idx[: int(round(0.25 * len(idx)))], "split"] = "test"
    df.loc[~df["split"].isin(["train", "test"]), "split"] = "train"
    return df.drop(columns="key").reset_index(drop=True)


def describe(name: str, df: pd.DataFrame) -> dict:
    out = {"dataset": name, "rows": len(df)}
    for split in ("train", "test"):
        part = df[df["split"] == split]
        out[f"{split}_rows"] = len(part)
        out[f"{split}_injection_share"] = round(float(part["label"].mean()), 3) if len(part) else None
    out["median_chars"] = int(df["text"].str.len().median())
    return out


def main():
    p = argparse.ArgumentParser(description="Download and inspect the datasets")
    p.add_argument("--inspect", action="store_true", help="print statistics (default; no texts)")
    p.add_argument("--refresh", action="store_true", help="re-download")
    a = p.parse_args()
    for name in SOURCES:
        df = load(name, refresh=a.refresh)
        print(describe(name, df))


if __name__ == "__main__":
    main()
