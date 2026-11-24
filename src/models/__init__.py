"""
All intent classifiers share one small interface so train.py / predict.py can treat them the same:

    model = build_model(kind, params, labels)
    model.fit(train_texts, train_y, val_texts, val_y)  -> dict history (may be empty)
    model.logits(texts)                                -> np.ndarray [n, n_classes]
    model.save(dir) / load_model(dir)

`logits` are raw scores before softmax. They are what the out-of-scope detector uses
(softmax probability, temperature scaling, energy score).
"""

from pathlib import Path

from common import load_json

MODEL_KINDS = {
    "tfidf_logreg": ("models.tfidf_logreg", "TfidfLogReg"),
    "naive_bayes_scratch": ("models.naive_bayes_scratch", "NaiveBayesScratch"),
    "transformer_scratch": ("models.transformer_scratch", "TransformerScratch"),
    "distilbert_finetune": ("models.distilbert_finetune", "DistilBertFinetune"),
}


def _cls(kind):
    import importlib
    module, name = MODEL_KINDS[kind]
    return getattr(importlib.import_module(module), name)


def build_model(kind, params, labels):
    return _cls(kind)(params=params, labels=labels)


def load_model(model_dir):
    model_dir = Path(model_dir)
    info = load_json(model_dir / "model_info.json")
    return _cls(info["kind"]).load(model_dir), info
