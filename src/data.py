"""Load the prepared splits and build the noisy robustness test set."""

import random
from pathlib import Path

import pandas as pd

from common import ROOT


def load_split(cfg, name: str) -> pd.DataFrame:
    path = ROOT / cfg["data"]["dir"] / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/data_download.py")
    return pd.read_csv(path).fillna("")


def load_all(cfg):
    return {name: load_split(cfg, name) for name in ("train", "val", "test", "oos_val", "oos_test")}


def add_typos(text: str, rate: float, rng: random.Random) -> str:
    """Corrupt about `rate` of the words with a typo (delete, swap, double or replace a letter)."""
    words = text.split()
    for i, w in enumerate(words):
        if len(w) < 3 or rng.random() >= rate:
            continue
        j = rng.randrange(1, len(w) - 1)
        op = rng.random()
        if op < 0.25:
            w = w[:j] + w[j + 1:]
        elif op < 0.5:
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        elif op < 0.75:
            w = w[:j] + w[j] + w[j:]
        else:
            w = w[:j] + rng.choice("abcdefghijklmnopqrstuvwxyz") + w[j + 1:]
        words[i] = w
    return " ".join(words)


def noisy_copy(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy()
    out["text"] = [add_typos(t, rate, rng) for t in out["text"]]
    return out
