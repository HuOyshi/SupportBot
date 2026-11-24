"""
Multinomial Naive Bayes written from scratch (no sklearn classifier).

Idea:  P(intent | words)  is proportional to  P(intent) * product over words of P(word | intent)
We work in log space so the product becomes a sum:

    score(c) = log P(c) + sum_w count(w) * log P(w | c)

P(w | c) is estimated with Lidstone smoothing:
    P(w | c) = (count of w in class c + alpha) / (total words in class c + alpha * vocabulary size)
alpha > 0 stops unseen words from giving probability 0.
"""

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from common import tokenize, save_json, load_json


def ngrams(tokens, n_max):
    feats = list(tokens)
    for n in range(2, n_max + 1):
        feats += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return feats


class NaiveBayesScratch:
    def __init__(self, params, labels):
        self.params, self.labels = params, list(labels)
        self.alpha = params["alpha"]
        self.ngram_max = params["ngram_max"]
        self.min_df = params["min_df"]
        self.vocab = {}

    # ---- own bag-of-ngrams vectorizer -------------------------------------
    def _build_vocab(self, texts):
        df = Counter()
        for t in texts:
            df.update(set(ngrams(tokenize(t), self.ngram_max)))
        kept = sorted(f for f, c in df.items() if c >= self.min_df)
        self.vocab = {f: i for i, f in enumerate(kept)}

    def _vectorize(self, texts):
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            counts = Counter(f for f in ngrams(tokenize(t), self.ngram_max) if f in self.vocab)
            for f, c in counts.items():
                rows.append(r); cols.append(self.vocab[f]); vals.append(c)
        return csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.vocab)), dtype=np.float32)

    # ---- training: just counting ------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, log_fn=None):
        self._build_vocab(X)
        Xm = self._vectorize(X)
        y = np.asarray(y)
        k, v = len(self.labels), len(self.vocab)
        class_counts = np.bincount(y, minlength=k)
        self.log_prior = np.log(class_counts / class_counts.sum()).astype(np.float32)
        word_counts = np.zeros((k, v), dtype=np.float64)
        for c in range(k):
            word_counts[c] = np.asarray(Xm[y == c].sum(axis=0)).ravel()
        smoothed = word_counts + self.alpha
        self.log_likelihood = np.log(smoothed / smoothed.sum(axis=1, keepdims=True)).astype(np.float32)
        return {}

    def logits(self, texts):
        """Log joint probability log P(c) + log P(words | c) for every class."""
        Xm = self._vectorize(list(texts))
        return np.asarray(Xm @ self.log_likelihood.T + self.log_prior, dtype=np.float32)

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "naive_bayes.npz", log_prior=self.log_prior, log_likelihood=self.log_likelihood)
        save_json({"params": self.params, "labels": self.labels, "vocab": self.vocab}, d / "weights_meta.json")

    @classmethod
    def load(cls, d):
        meta = load_json(Path(d) / "weights_meta.json")
        m = cls(meta["params"], meta["labels"])
        m.vocab = meta["vocab"]
        w = np.load(Path(d) / "naive_bayes.npz")
        m.log_prior, m.log_likelihood = w["log_prior"], w["log_likelihood"]
        return m
