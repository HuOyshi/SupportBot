"""
Retrieval-based answer generation.

After the classifier predicts the intent, we look at the training examples of THAT intent and
pick the reply whose customer message is most similar (TF-IDF cosine similarity) to the new
message. Bitext replies contain slots like {{Order Number}}; we show them as [Order Number] so a
human agent / CRM system can fill them in.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from common import normalize

SLOT_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")
OOS_REPLY = ("I'm sorry, I can only help with customer-support topics such as orders, payments, "
             "refunds, delivery, invoices and your account. Could you rephrase your question, "
             "or would you like me to connect you with a human agent?")
BANK_FILE = "response_bank.csv.gz"


def fill_slots(text):
    return SLOT_RE.sub(lambda m: f"[{m.group(1)}]", str(text))


class Responder:
    def __init__(self, bank: pd.DataFrame):
        self.bank = bank.reset_index(drop=True)
        self.vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform(self.bank["text"])
        self.by_intent = {i: np.flatnonzero(self.bank["intent"].values == i) for i in self.bank["intent"].unique()}
        self.category = dict(zip(self.bank["intent"], self.bank["category"]))

    @classmethod
    def build(cls, train_df, per_intent=150, seed=42):
        shuffled = train_df.sample(frac=1.0, random_state=seed)
        bank = shuffled.groupby("intent").head(per_intent)   # up to `per_intent` examples per intent
        return cls(bank[["text", "intent", "category", "response"]])

    def reply(self, text, intent):
        idx = self.by_intent.get(intent)
        if idx is None or len(idx) == 0:
            return OOS_REPLY
        sims = (self.X[idx] @ self.vec.transform([text]).T).toarray().ravel()
        return fill_slots(self.bank.at[idx[int(sims.argmax())], "response"])

    def save(self, d):
        self.bank.to_csv(Path(d) / BANK_FILE, index=False, compression="gzip")

    @classmethod
    def load(cls, d):
        return cls(pd.read_csv(Path(d) / BANK_FILE).fillna(""))
