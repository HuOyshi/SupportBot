"""Baseline: TF-IDF (word + character n-grams) + Logistic Regression (scikit-learn)."""

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import normalize, save_json, load_json


class TfidfLogReg:
    def __init__(self, params, labels):
        self.params, self.labels = params, list(labels)
        self.word_vec = TfidfVectorizer(preprocessor=normalize, ngram_range=tuple(params["word_ngrams"]),
                                        max_features=params["max_features"], sublinear_tf=True)
        self.char_vec = TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                        ngram_range=tuple(params["char_ngrams"]),
                                        max_features=params["max_features"], sublinear_tf=True)
        self.clf = LogisticRegression(C=params["C"], max_iter=2000)

    def _features(self, texts, fit=False):
        if fit:
            return hstack([self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)]).tocsr()
        return hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()

    def fit(self, X, y, X_val=None, y_val=None, log_fn=None):
        self.clf.fit(self._features(X, fit=True), y)
        return {}

    def logits(self, texts):
        return self.clf.decision_function(self._features(list(texts))).astype(np.float32)

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.word_vec, self.char_vec, self.clf), d / "tfidf_logreg.joblib", compress=3)
        save_json({"params": self.params, "labels": self.labels}, d / "weights_meta.json")

    @classmethod
    def load(cls, d):
        meta = load_json(Path(d) / "weights_meta.json")
        m = cls(meta["params"], meta["labels"])
        m.word_vec, m.char_vec, m.clf = joblib.load(Path(d) / "tfidf_logreg.joblib")
        return m
