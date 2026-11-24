"""Shared helpers: paths, config, seeding, text normalisation, MLflow setup."""

import json
import random
import re
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
BEST_DIR = MODELS_DIR / "best"


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def normalize(text: str) -> str:
    """Lowercase, unify quotes, collapse whitespace."""
    text = str(text).lower().replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer used by the from-scratch models."""
    return TOKEN_RE.findall(normalize(text))


def setup_mlflow(cfg: dict):
    """Use a local file store in ./mlruns so the runs can be committed to GitHub."""
    import os
    # Newer MLflow versions refuse the ./mlruns folder store unless this is set (older ones ignore it).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    tracking_dir = ROOT / cfg["mlflow"]["tracking_dir"]
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    return mlflow


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
