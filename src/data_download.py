"""
Download the datasets into data/ and create the train/validation/test splits.

    python src/data_download.py

1. Bitext customer-support dataset (Hugging Face) - 26,872 customer messages
   labelled with 27 intents and an example reply. Used for training.
2. CLINC150 "out-of-scope" queries (Hugging Face, fallback: GitHub) - questions
   that do not belong to customer support. Used ONLY to test out-of-scope
   detection, never for training.

Outputs (all inside data/, which is git-ignored):
    bitext_raw.csv, train.csv, val.csv, test.csv, oos_val.csv, oos_test.csv
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config  # noqa: E402

CLINC_GITHUB = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_oos_plus.json"
CLINC_OOS_LABEL = 42   # label id of "oos" in the Hugging Face version


def download_bitext(cfg, out_dir: Path) -> pd.DataFrame:
    from huggingface_hub import HfApi, hf_hub_download

    repo, fname = cfg["data"]["bitext_repo"], cfg["data"]["bitext_file"]
    print(f"Downloading Bitext dataset from Hugging Face ({repo}) ...")
    try:
        path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
    except Exception:
        # file name changed upstream: take the first CSV in the repo
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".csv")]
        if not files:
            raise
        path = hf_hub_download(repo_id=repo, filename=files[0], repo_type="dataset")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "instruction" not in df.columns and "utterance" in df.columns:
        df = df.rename(columns={"utterance": "instruction"})
    df = df.dropna(subset=["instruction", "intent"])
    df.to_csv(out_dir / "bitext_raw.csv", index=False)
    print(f"  {len(df):,} rows, {df['intent'].nunique()} intents")
    return df


def download_clinc_oos(cfg, out_dir: Path):
    """Return (oos_val, oos_test) lists of out-of-scope queries."""
    repo = cfg["data"]["clinc_repo"]
    try:
        from huggingface_hub import HfApi, hf_hub_download
        print(f"Downloading CLINC150 out-of-scope queries ({repo}) ...")
        files = HfApi().list_repo_files(repo, repo_type="dataset")
        parts = {}
        for split in ("validation", "test"):
            match = [f for f in files if f.startswith("plus/") and split in f and f.endswith(".parquet")]
            df = pd.read_parquet(hf_hub_download(repo_id=repo, filename=match[0], repo_type="dataset"))
            parts[split] = df.loc[df["intent"] == CLINC_OOS_LABEL, "text"].tolist()
        oos_val, oos_test = parts["validation"], parts["test"]
    except Exception as e:
        print(f"  Hugging Face download failed ({e}); trying GitHub ...")
        with urllib.request.urlopen(CLINC_GITHUB, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        oos_val = [t for t, _ in data["oos_val"]]
        oos_test = [t for t, _ in data["oos_test"]]
    print(f"  out-of-scope: {len(oos_val)} validation, {len(oos_test)} test")
    return oos_val, oos_test


def make_splits(df: pd.DataFrame, cfg, out_dir: Path):
    seed = cfg["seed"]
    val_size, test_size = cfg["data"]["val_size"], cfg["data"]["test_size"]
    df = df[["instruction", "category", "intent", "response"]].rename(columns={"instruction": "text"})
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    train, rest = train_test_split(df, test_size=val_size + test_size,
                                   stratify=df["intent"], random_state=seed)
    val, test = train_test_split(rest, test_size=test_size / (val_size + test_size),
                                 stratify=rest["intent"], random_state=seed)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Splits: train {len(train):,} | val {len(val):,} | test {len(test):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = ROOT / cfg["data"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df = download_bitext(cfg, out_dir)
    make_splits(df, cfg, out_dir)
    oos_val, oos_test = download_clinc_oos(cfg, out_dir)
    pd.DataFrame({"text": oos_val}).to_csv(out_dir / "oos_val.csv", index=False)
    pd.DataFrame({"text": oos_test}).to_csv(out_dir / "oos_test.csv", index=False)
    print(f"Done. Files saved in {out_dir}")


if __name__ == "__main__":
    main()
